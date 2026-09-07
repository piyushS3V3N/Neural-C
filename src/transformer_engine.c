#include "transformer_engine.h"

TransformerEngine* create_transformer_engine(const Config* cfg, Tokenizer* tokenizer, TransformerWeights* weights, bool use_gpu) {
    if (!cfg || !tokenizer || !weights) return NULL;
    TransformerEngine* engine = (TransformerEngine*)calloc(1, sizeof(TransformerEngine));
    if (!engine) return NULL;

    engine->config = *cfg;
    engine->tokenizer = tokenizer;
    engine->weights = weights;
    engine->use_gpu = use_gpu && init_llm_metal_engine();
    engine->state = allocate_run_state(cfg);

    if (!engine->state) {
        free(engine);
        return NULL;
    }

    return engine;
}

void free_transformer_engine(TransformerEngine* engine) {
    if (!engine) return;
    if (engine->state) free_run_state(engine->state);
    free(engine);
}

static inline void engine_matmul(TransformerEngine* engine, float* out, const float* x, const Tensor* t, int in_dim, int out_dim) {
    if (!out) return;
    if (!t || !t->data) {
        memset(out, 0, out_dim * sizeof(float));
        return;
    }
    if (t->type == QUANT_Q4_0) {
        if (engine->use_gpu) {
            metal_gemv_q4_0(out, x, (const BlockQ4_0*)t->data, in_dim, out_dim);
        } else {
            matmul_q4_0(out, x, (const BlockQ4_0*)t->data, in_dim, out_dim);
        }
    } else if (t->type == QUANT_Q8_0) {
        if (engine->use_gpu) {
            metal_gemv_q8_0(out, x, (const BlockQ8_0*)t->data, in_dim, out_dim);
        } else {
            matmul_q8_0(out, x, (const BlockQ8_0*)t->data, in_dim, out_dim);
        }
    } else {
        if (engine->use_gpu) {
            metal_gemv_fp32(out, x, (const float*)t->data, in_dim, out_dim);
        } else {
            matmul_fp32(out, x, (const float*)t->data, in_dim, out_dim);
        }
    }
}

float* transformer_forward(TransformerEngine* engine, int token, int pos) {
    Config* p = &engine->config;
    TransformerWeights* w = engine->weights;
    RunState* s = engine->state;

    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_dim = p->head_dim;
    int n_heads = p->n_heads;
    int n_kv_heads = p->n_kv_heads;
    int kv_dim = n_kv_heads * head_dim;
    int kv_mul = n_heads / (n_kv_heads > 0 ? n_kv_heads : 1);

    // 1. Token Embedding Lookup
    if (w->token_embedding_table.data && token >= 0 && token < p->vocab_size) {
        if (w->token_embedding_table.type == QUANT_Q4_0) {
            BlockQ4_0* emb_q4 = (BlockQ4_0*)w->token_embedding_table.data;
            int blocks_per_row = dim / 32;
            BlockQ4_0* row_blocks = emb_q4 + (size_t)token * blocks_per_row;
            for (int b = 0; b < blocks_per_row; b++) {
                float scale = fp16_to_fp32(row_blocks[b].scale);
                const uint8_t* qs = row_blocks[b].qs;
                float* x_block = s->x + b * 32;
                for (int l = 0; l < 16; l++) {
                    uint8_t byte = qs[l];
                    x_block[l] = (float)((byte & 0x0F) - 8) * scale;
                    x_block[l + 16] = (float)(((byte >> 4) & 0x0F) - 8) * scale;
                }
            }
        } else {
            float* embed_data = (float*)w->token_embedding_table.data;
            memcpy(s->x, embed_data + (size_t)token * dim, dim * sizeof(float));
        }
    } else {
        for (int i = 0; i < dim; i++) s->x[i] = 0.01f * (float)(token % 13 + i % 7);
    }

    // 2. Loop Through All Transformer Blocks
    for (int l = 0; l < p->n_layers; l++) {
        // Pre-Attention RMSNorm
        rmsnorm_tensor(s->xb, s->x, &w->rms_att_weight[l], dim, p->norm_eps);

        // Q, K, V Matrix-Vector Multiplications
        engine_matmul(engine, s->q, s->xb, &w->wq[l], dim, n_heads * head_dim);
        engine_matmul(engine, s->k, s->xb, &w->wk[l], dim, kv_dim);
        engine_matmul(engine, s->v, s->xb, &w->wv[l], dim, kv_dim);

        // Add QKV biases if present (e.g. Qwen / Qwen2 models)
        if (w->bq && w->bq[l].data) {
            float* bq_ptr = (float*)w->bq[l].data;
            for (int i = 0; i < n_heads * head_dim; i++) s->q[i] += bq_ptr[i];
        }
        if (w->bk && w->bk[l].data) {
            float* bk_ptr = (float*)w->bk[l].data;
            for (int i = 0; i < kv_dim; i++) s->k[i] += bk_ptr[i];
        }
        if (w->bv && w->bv[l].data) {
            float* bv_ptr = (float*)w->bv[l].data;
            for (int i = 0; i < kv_dim; i++) s->v[i] += bv_ptr[i];
        }

        // Apply Rotary Position Embeddings (RoPE)
        apply_rope(s->q, s->k, pos, head_dim, n_heads, n_kv_heads, p->rope_freq_base);

        // Store K, V in Key-Value Cache at position pos
        size_t layer_offset = (size_t)l * p->seq_len * kv_dim;
        float* key_cache_row = s->key_cache + layer_offset + pos * kv_dim;
        float* val_cache_row = s->value_cache + layer_offset + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(float));
        memcpy(val_cache_row, s->v, kv_dim * sizeof(float));

        // Grouped-Query Multi-Head Attention
        for (int h = 0; h < n_heads; h++) {
            float* q_head = s->q + h * head_dim;
            float* att_head = s->att + h * p->seq_len;
            int kv_h = h / kv_mul;

            for (int t_pos = 0; t_pos <= pos; t_pos++) {
                float* k_cache_t = s->key_cache + layer_offset + t_pos * kv_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q_head[i] * k_cache_t[i];
                }
                att_head[t_pos] = score / sqrtf((float)head_dim);
            }

            // Softmax over causal timesteps 0..pos
            softmax(att_head, pos + 1);

            // Weighted aggregation of Values
            float* xb_head = s->xb + h * head_dim;
            memset(xb_head, 0, head_dim * sizeof(float));
            for (int t_pos = 0; t_pos <= pos; t_pos++) {
                float* v_cache_t = s->value_cache + layer_offset + t_pos * kv_dim + kv_h * head_dim;
                float a = att_head[t_pos];
                for (int i = 0; i < head_dim; i++) {
                    xb_head[i] += a * v_cache_t[i];
                }
            }
        }

        // Projection MatMul & Residual Connection
        engine_matmul(engine, s->xb2, s->xb, &w->wo[l], n_heads * head_dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        // Pre-FFN RMSNorm
        rmsnorm_tensor(s->xb, s->x, &w->rms_ffn_weight[l], dim, p->norm_eps);

        // SwiGLU Feed-Forward Network (Gate, Up, Down)
        engine_matmul(engine, s->hb, s->xb, &w->w_gate[l], dim, hidden_dim);
        engine_matmul(engine, s->hb2, s->xb, &w->w_up[l], dim, hidden_dim);

        swiglu(s->hb, s->hb, s->hb2, hidden_dim);

        engine_matmul(engine, s->xb2, s->hb, &w->w_down[l], hidden_dim, dim);

        // FFN Residual Connection
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
    }

    // 3. Final RMSNorm & Output Classifier Head Logits
    rmsnorm_tensor(s->x, s->x, &w->rms_final_weight, dim, p->norm_eps);

    Tensor cls_tensor = w->w_cls.data ? w->w_cls : w->token_embedding_table;
    engine_matmul(engine, s->logits, s->x, &cls_tensor, dim, p->vocab_size);

    return s->logits;
}

void generate_text_stream(TransformerEngine* engine, const char* prompt, int max_new_tokens, float temperature, float top_p) {
    if (!engine || !engine->tokenizer || !prompt) return;

    int max_prompt_capacity = (int)strlen(prompt) + 256;
    int* prompt_tokens = (int*)malloc(max_prompt_capacity * sizeof(int));
    int num_prompt_tokens = encode(engine->tokenizer, prompt, true, false, prompt_tokens, max_prompt_capacity);

    if (num_prompt_tokens <= 0) {
        prompt_tokens[0] = engine->tokenizer->bos_id;
        num_prompt_tokens = 1;
    }

    printf("\nPrompt (%d tokens): '%s'\n", num_prompt_tokens, prompt);
    printf("Response: ");
    fflush(stdout);

    int curr_token = prompt_tokens[0];
    int pos = 0;

    // 1. Process all prompt tokens through the model to build initial KV cache state
    for (int p = 0; p < num_prompt_tokens; p++) {
        curr_token = prompt_tokens[p];
        float* logits = transformer_forward(engine, curr_token, pos);
        (void)logits;
        const char* piece = decode(engine->tokenizer, -1, curr_token);
        if (piece && piece[0]) printf("%s", piece);
        else printf("<tok:%d>", curr_token);
        fflush(stdout);
        pos++;
    }

    // 2. Generate new tokens autoregressively from the model
    for (int step = 0; step < max_new_tokens; step++) {
        float* logits = engine->state->logits;
        if (!logits) break;

        int next_token;
        if (temperature <= 0.0f) {
            next_token = sample_argmax(logits, engine->config.vocab_size);
        } else {
            float coin = (float)rand() / (float)RAND_MAX;
            next_token = sample_top_p(logits, engine->config.vocab_size, top_p, temperature, coin);
        }

        if (next_token == engine->tokenizer->eos_id) {
            printf("<eos>");
            break;
        }

        const char* piece = decode(engine->tokenizer, -1, next_token);
        if (piece && piece[0]) printf("%s", piece);
        else printf("<tok:%d>", next_token);
        fflush(stdout);

        if (pos >= engine->config.seq_len) break;
        transformer_forward(engine, next_token, pos);
        pos++;
    }

    printf("\n\n");
    free(prompt_tokens);
}
