#include "transformer_engine.h"

TransformerEngine* create_transformer_engine(const Config* cfg, Tokenizer* tokenizer, TransformerWeights* weights, bool use_gpu) {
    if (!cfg || !tokenizer || !weights) return NULL;
    // Validate config so one bad GGUF header can't cause OOB everywhere
    if (cfg->dim <= 0 || cfg->hidden_dim <= 0 || cfg->n_layers <= 0 ||
        cfg->n_heads <= 0 || cfg->n_kv_heads <= 0 || cfg->head_dim <= 0 ||
        cfg->vocab_size <= 0 || cfg->seq_len <= 0) return NULL;
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
    if (!out || out_dim <= 0) return;
    if (in_dim <= 0) { memset(out, 0, (size_t)out_dim * sizeof(float)); return; }
    if (!x) { memset(out, 0, (size_t)out_dim * sizeof(float)); return; }
    if (!t || !t->data) {
        memset(out, 0, (size_t)out_dim * sizeof(float));
        return;
    }
    static bool g_warned_shape = false;
    if (t->n_dims >= 2 && t->shape[0] != in_dim) {
        if (!g_warned_shape) {
            printf("[WARN] GGUF tensor shape mismatch: shape[0]=%d != in_dim=%d (layout orientation check)\n", t->shape[0], in_dim);
            g_warned_shape = true;
        }
    }
    // If loader filled numel, verify rows match. Mismatched GGUF (e.g. tied
    // weights, transposed layout, vocab override) must not over-read mmap.
    if (t->numel > 0 && t->type == QUANT_FP32) {
        if (t->numel < (size_t)in_dim * (size_t)out_dim) {
            memset(out, 0, (size_t)out_dim * sizeof(float));
            return;
        }
    } else if (t->numel > 0 && t->type == QUANT_FP16) {
        if (t->numel < (size_t)in_dim * (size_t)out_dim) {
            memset(out, 0, (size_t)out_dim * sizeof(float));
            return;
        }
    }
    if (t->type == QUANT_Q4_0) {
        if (in_dim % 32 != 0) { memset(out, 0, (size_t)out_dim * sizeof(float)); return; }
        if (engine && engine->use_gpu) {
            metal_gemv_q4_0(out, x, (const BlockQ4_0*)t->data, in_dim, out_dim);
        } else {
            matmul_q4_0(out, x, (const BlockQ4_0*)t->data, in_dim, out_dim);
        }
    } else if (t->type == QUANT_Q8_0) {
        if (in_dim % 32 != 0) { memset(out, 0, (size_t)out_dim * sizeof(float)); return; }
        if (engine && engine->use_gpu) {
            metal_gemv_q8_0(out, x, (const BlockQ8_0*)t->data, in_dim, out_dim);
        } else {
            matmul_q8_0(out, x, (const BlockQ8_0*)t->data, in_dim, out_dim);
        }
    } else if (t->type == QUANT_Q6_K) {
        matmul_q6_k(out, x, (const BlockQ6_K*)t->data, in_dim, out_dim);
    } else if (t->type == QUANT_FP16) {
        // No Metal FP16 kernel — always CPU dequant path
        matmul_fp16(out, x, (const uint16_t*)t->data, in_dim, out_dim);
    } else if (t->type == QUANT_FP32) {
        if (engine && engine->use_gpu) {
            metal_gemv_fp32(out, x, (const float*)t->data, in_dim, out_dim);
        } else {
            matmul_fp32(out, x, (const float*)t->data, in_dim, out_dim);
        }
    } else {
        memset(out, 0, (size_t)out_dim * sizeof(float));
    }
}

static inline void add_bias_vec(float* acc, const Tensor* bias, int n) {
    if (!acc || !bias || !bias->data || n <= 0) return;
    if (bias->type == QUANT_FP16) {
        const uint16_t* h = (const uint16_t*)bias->data;
        for (int i = 0; i < n; i++) acc[i] += fp16_to_fp32(h[i]);
    } else if (bias->type == QUANT_FP32) {
        const float* f = (const float*)bias->data;
        for (int i = 0; i < n; i++) acc[i] += f[i];
    }
    // Quantized bias never occurs; ignore other types rather than misread.
}

static inline void embed_row_to(float* dst, const Tensor* emb, int token, int dim) {
    if (!dst || !emb || !emb->data || dim <= 0) return;
    if (emb->type == QUANT_Q4_0) {
        if (dim % 32 != 0) return;
        int blocks_per_row = dim / 32;
        const BlockQ4_0* emb_q4 = (const BlockQ4_0*)emb->data;
        const BlockQ4_0* row_blocks = emb_q4 + (size_t)token * blocks_per_row;
        for (int b = 0; b < blocks_per_row; b++) {
            float scale = fp16_to_fp32(row_blocks[b].scale);
            const uint8_t* qs = row_blocks[b].qs;
            float* x_block = dst + (size_t)b * 32;
            for (int l = 0; l < 16; l++) {
                uint8_t byte = qs[l];
                x_block[l] = (float)((byte & 0x0F) - 8) * scale;
                x_block[l + 16] = (float)(((byte >> 4) & 0x0F) - 8) * scale;
            }
        }
    } else if (emb->type == QUANT_Q8_0) {
        if (dim % 32 != 0) return;
        int blocks_per_row = dim / 32;
        const BlockQ8_0* emb_q8 = (const BlockQ8_0*)emb->data;
        const BlockQ8_0* row_blocks = emb_q8 + (size_t)token * blocks_per_row;
        for (int b = 0; b < blocks_per_row; b++) {
            float scale = fp16_to_fp32(row_blocks[b].scale);
            const int8_t* qs = row_blocks[b].qs;
            float* x_block = dst + (size_t)b * 32;
            for (int l = 0; l < 32; l++) x_block[l] = (float)qs[l] * scale;
        }
    } else if (emb->type == QUANT_Q6_K) {
        if (dim % 256 != 0) return;
        int blocks_per_row = dim / 256;
        const BlockQ6_K* emb_q6 = (const BlockQ6_K*)emb->data;
        const BlockQ6_K* row_blocks = emb_q6 + (size_t)token * blocks_per_row;
        for (int b = 0; b < blocks_per_row; b++) {
            const BlockQ6_K* block = &row_blocks[b];
            float d = fp16_to_fp32(block->d);
            const uint8_t* ql = block->ql;
            const uint8_t* qh = block->qh;
            const int8_t*  sc = block->scales;
            float* x_block = dst + (size_t)b * 256;

            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    uint8_t qh_val = qh[l];
                    int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | ((qh_val & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh_val >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh_val >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh_val >> 6) & 3) << 4)) - 32;

                    x_block[l +  0] = d * (float)sc[is + 0] * (float)q1;
                    x_block[l + 32] = d * (float)sc[is + 2] * (float)q2;
                    x_block[l + 64] = d * (float)sc[is + 4] * (float)q3;
                    x_block[l + 96] = d * (float)sc[is + 6] * (float)q4;
                }
                x_block += 128; ql += 64; qh += 32; sc += 8;
            }
        }
    } else if (emb->type == QUANT_FP16) {
        const uint16_t* emb_h = (const uint16_t*)emb->data;
        const uint16_t* row = emb_h + (size_t)token * dim;
        for (int i = 0; i < dim; i++) dst[i] = fp16_to_fp32(row[i]);
    } else if (emb->type == QUANT_FP32) {
        const float* emb_f = (const float*)emb->data;
        memcpy(dst, emb_f + (size_t)token * dim, (size_t)dim * sizeof(float));
    }
    // Unsupported (Q4_K etc.): leave fallback pattern in dst.
}

float* transformer_forward(TransformerEngine* engine, int token, int pos) {
    if (!engine || !engine->weights || !engine->state) return NULL;
    Config* p = &engine->config;
    TransformerWeights* w = engine->weights;
    RunState* s = engine->state;

    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_dim = p->head_dim;
    int n_heads = p->n_heads;
    int n_kv_heads = p->n_kv_heads;
    if (dim <= 0 || hidden_dim <= 0 || head_dim <= 0 || n_heads <= 0 ||
        n_kv_heads <= 0 || p->n_layers <= 0 || p->seq_len <= 0 || p->vocab_size <= 0)
        return NULL;
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k ||
        !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache)
        return NULL;
    // Position guard: prevents KV-cache / att-buffer overflow that crashed
    // long prompts and over-length generation on small-context models.
    if (pos < 0 || pos >= p->seq_len) return s->logits;

    int kv_dim = n_kv_heads * head_dim;
    if (kv_dim <= 0) return NULL;
    int q_dim = n_heads * head_dim;
    // Attention output (q_dim) is staged in xb (dim). Layout requires q_dim<=dim.
    if (q_dim <= 0 || q_dim > dim) return NULL;
    // GQA factor with zero-guard + clamp (was SIGFPE / OOB when
    // n_kv_heads==0 or n_kv_heads>n_heads on some arch configs)
    int kv_mul = (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 0;
    if (kv_mul <= 0) kv_mul = 1;

    // 1. Token Embedding Lookup (was Q4_0/FP32 only -> crash on F16/Q8_0 embeds)
    bool emb_ok = false;
    if (w->token_embedding_table.data && token >= 0 && token < p->vocab_size) {
        QuantType et = w->token_embedding_table.type;
        if (et == QUANT_Q4_0 || et == QUANT_Q8_0 || et == QUANT_FP16 || et == QUANT_FP32) {
            // Validate row exists when loader supplied numel
            bool shape_ok = true;
            if (w->token_embedding_table.numel > 0) {
                size_t need = (size_t)(token + 1) * (size_t)dim;
                if (et == QUANT_FP32 || et == QUANT_FP16) {
                    if (w->token_embedding_table.numel < need) shape_ok = false;
                }
            }
            if (shape_ok) { embed_row_to(s->x, &w->token_embedding_table, token, dim); emb_ok = true; }
        }
    }
    if (!emb_ok) {
        for (int i = 0; i < dim; i++) s->x[i] = 0.01f * (float)(token % 13 + i % 7);
    }

    // 2. Loop Through All Transformer Blocks
    for (int l = 0; l < p->n_layers; l++) {
        // Pre-Attention RMSNorm
        rmsnorm_tensor(s->xb, s->x, &w->rms_att_weight[l], dim, p->norm_eps);

        // Q, K, V Matrix-Vector Multiplications
        engine_matmul(engine, s->q, s->xb, &w->wq[l], dim, q_dim);
        engine_matmul(engine, s->k, s->xb, &w->wk[l], dim, kv_dim);
        engine_matmul(engine, s->v, s->xb, &w->wv[l], dim, kv_dim);

        // Add QKV biases if present (e.g. Qwen / Qwen2 models, FP32 or FP16)
        if (w->bq) add_bias_vec(s->q, &w->bq[l], q_dim);
        if (w->bk) add_bias_vec(s->k, &w->bk[l], kv_dim);
        if (w->bv) add_bias_vec(s->v, &w->bv[l], kv_dim);

        // Apply Rotary Position Embeddings (RoPE)
        apply_rope(s->q, s->k, pos, head_dim, n_heads, n_kv_heads, p->rope_freq_base);

        // Store K, V in Key-Value Cache at position pos
        size_t layer_offset = (size_t)l * (size_t)p->seq_len * (size_t)kv_dim;
        float* key_cache_row = s->key_cache + layer_offset + (size_t)pos * (size_t)kv_dim;
        float* val_cache_row = s->value_cache + layer_offset + (size_t)pos * (size_t)kv_dim;
        memcpy(key_cache_row, s->k, (size_t)kv_dim * sizeof(float));
        memcpy(val_cache_row, s->v, (size_t)kv_dim * sizeof(float));

        // Grouped-Query Multi-Head Attention
        float attn_scale = (head_dim > 0) ? 1.0f / sqrtf((float)head_dim) : 0.0f;
        for (int h = 0; h < n_heads; h++) {
            float* q_head = s->q + (size_t)h * head_dim;
            float* att_head = s->att + (size_t)h * p->seq_len;
            // Clamp group index: protects misconfigured GQA factors
            int kv_h = h / kv_mul;
            if (kv_h < 0) kv_h = 0;
            if (kv_h >= n_kv_heads) kv_h = n_kv_heads - 1;

            for (int t_pos = 0; t_pos <= pos; t_pos++) {
                float* k_cache_t = s->key_cache + layer_offset + (size_t)t_pos * kv_dim + (size_t)kv_h * head_dim;
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q_head[i] * k_cache_t[i];
                }
                att_head[t_pos] = score * attn_scale;
            }

            // Softmax over causal timesteps 0..pos
            softmax(att_head, pos + 1);

            // Weighted aggregation of Values
            float* xb_head = s->xb + (size_t)h * head_dim;
            memset(xb_head, 0, (size_t)head_dim * sizeof(float));
            for (int t_pos = 0; t_pos <= pos; t_pos++) {
                float* v_cache_t = s->value_cache + layer_offset + (size_t)t_pos * kv_dim + (size_t)kv_h * head_dim;
                float a = att_head[t_pos];
                for (int i = 0; i < head_dim; i++) {
                    xb_head[i] += a * v_cache_t[i];
                }
            }
        }

        // Projection MatMul & Residual Connection
        engine_matmul(engine, s->xb2, s->xb, &w->wo[l], q_dim, dim);
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

    // Classifier must have vocab_size rows; tied fallback may share the
    // embedding table. Validate before matmul to avoid mmap over-read when
    // config vocab and weight rows disagree across models.
    const Tensor* cls = NULL;
    if (w->w_cls.data) cls = &w->w_cls;
    else if (w->token_embedding_table.data) cls = &w->token_embedding_table;
    if (cls) {
        bool cls_ok = true;
        if (cls->numel > 0) {
            if (cls->type == QUANT_FP32 || cls->type == QUANT_FP16) {
                if (cls->numel < (size_t)dim * (size_t)p->vocab_size) cls_ok = false;
            }
        }
        if (cls_ok) engine_matmul(engine, s->logits, s->x, cls, dim, p->vocab_size);
        else memset(s->logits, 0, (size_t)p->vocab_size * sizeof(float));
    }

    return s->logits;
}

void generate_text_stream(TransformerEngine* engine, const char* prompt, int max_new_tokens, float temperature, float top_p) {
    generate_text_stream_full(engine, prompt, max_new_tokens, temperature, top_p, 40, 1.1f, 0.0f, 0.0f, 64, 0);
}

void generate_text_stream_ex(TransformerEngine* engine, const char* prompt, int max_new_tokens,
                             float temperature, float top_p, float repeat_penalty,
                             int repeat_last_n, int use_chat_template) {
    generate_text_stream_full(engine, prompt, max_new_tokens, temperature, top_p, 40, repeat_penalty, 0.0f, 0.0f, repeat_last_n, use_chat_template);
}

void generate_text_stream_full(TransformerEngine* engine, const char* prompt, int max_new_tokens,
                               float temperature, float top_p, int top_k, float repeat_penalty,
                               float frequency_penalty, float presence_penalty,
                               int repeat_last_n, int use_chat_template) {
    if (!engine || !engine->tokenizer || !engine->state || !prompt) return;
    if (engine->config.seq_len <= 0 || engine->config.vocab_size <= 0) return;
    if (max_new_tokens <= 0) max_new_tokens = 1;
    if (!(repeat_penalty > 1.0f)) repeat_penalty = 1.0f;
    if (repeat_last_n < 0) repeat_last_n = 0;
    if (repeat_last_n > 512) repeat_last_n = 512;

    char* eff_prompt = (char*)prompt;
    char* owned = NULL;
    if (use_chat_template) {
        size_t need = strlen(prompt) + 2048;
        owned = (char*)malloc(need);
        if (owned && render_chat_template(engine->tokenizer, prompt, owned, need)) {
            eff_prompt = owned;
        }
    }

    int max_prompt_capacity = (int)strlen(eff_prompt) + 256;
    if (max_prompt_capacity <= 0) { free(owned); return; }
    int* prompt_tokens = (int*)malloc((size_t)max_prompt_capacity * sizeof(int));
    if (!prompt_tokens) { free(owned); return; }

    int num_prompt_tokens = encode(engine->tokenizer, eff_prompt,
        engine->tokenizer->add_bos_token, engine->tokenizer->add_eos_token,
        prompt_tokens, max_prompt_capacity);

    if (num_prompt_tokens <= 0) {
        prompt_tokens[0] = engine->tokenizer->bos_id;
        num_prompt_tokens = 1;
    }
    if (num_prompt_tokens > engine->config.seq_len) {
        int drop = num_prompt_tokens - engine->config.seq_len;
        memmove(prompt_tokens, prompt_tokens + drop, (size_t)engine->config.seq_len * sizeof(int));
        num_prompt_tokens = engine->config.seq_len;
    }

    printf("\nPrompt (%d tokens): '%s'\n", num_prompt_tokens, prompt);
    printf("Response: ");
    fflush(stdout);

    int pos = 0;

    for (int p = 0; p < num_prompt_tokens; p++) {
        if (pos >= engine->config.seq_len) break;
        int curr_token = prompt_tokens[p];
        transformer_forward(engine, curr_token, pos);
        const char* piece = decode(engine->tokenizer, -1, curr_token);
        if (piece && piece[0]) printf("%s", piece);
        else printf("<tok:%d>", curr_token);
        fflush(stdout);
        pos++;
    }

    int nl_id = tokenizer_find_token(engine->tokenizer, "\xc4\x8a");
    int hist_cap = repeat_last_n > 0 ? repeat_last_n : 0;
    int* history = NULL;
    int hist_len = 0;
    if (hist_cap > 0) {
        history = (int*)malloc((size_t)hist_cap * sizeof(int));
        if (!history) hist_cap = 0;
        else {
            int tail = num_prompt_tokens < hist_cap ? num_prompt_tokens : hist_cap;
            for (int i = 0; i < tail; i++) {
                int id = prompt_tokens[num_prompt_tokens - tail + i];
                bool structural = (id == nl_id) || (id == engine->tokenizer->bos_id) ||
                                  (id == engine->tokenizer->eos_id) || (id == engine->tokenizer->pad_id);
                if (!structural && hist_len < hist_cap) history[hist_len++] = id;
            }
        }
    }
    float* work = NULL;
    if (repeat_penalty > 1.0f || frequency_penalty != 0.0f || presence_penalty != 0.0f || temperature > 0.0f) {
        work = (float*)malloc((size_t)engine->config.vocab_size * sizeof(float));
        if (!work && repeat_penalty > 1.0f) repeat_penalty = 1.0f;
    }

    for (int step = 0; step < max_new_tokens; step++) {
        float* logits = engine->state->logits;
        if (!logits) break;

        float* sample_buf = logits;
        if (work) {
            memcpy(work, logits, (size_t)engine->config.vocab_size * sizeof(float));
            sample_buf = work;
        }
        if ((repeat_penalty > 1.0f || frequency_penalty != 0.0f || presence_penalty != 0.0f) && hist_cap > 0 && hist_len > 0)
            apply_repetition_penalty_ex(sample_buf, engine->config.vocab_size, history, hist_len, repeat_penalty, frequency_penalty, presence_penalty);

        if (getenv("NEURAL_C_DEBUG_TOPK")) {
            int V = engine->config.vocab_size;
            float *dbg = (float*)malloc((size_t)V * sizeof(float));
            if (dbg) {
                memcpy(dbg, sample_buf, (size_t)V * sizeof(float));
                softmax(dbg, V);
                fprintf(stderr, "\n[TOPK step %d pos %d]", step, pos);
                for (int kk = 0; kk < 5; kk++) {
                    int bi = 0; float bp = -1.0f;
                    for (int ii = 0; ii < V; ii++)
                        if (dbg[ii] > bp) { bp = dbg[ii]; bi = ii; }
                    const char *pt = decode(engine->tokenizer, -1, bi);
                    fprintf(stderr, " #%d id=%d p=%.3f '%s'", kk + 1, bi, bp, pt ? pt : "?");
                    dbg[bi] = -1.0f;
                }
                fprintf(stderr, "\n");
                free(dbg);
            }
        }

        int next_token;
        if (temperature <= 0.0f) {
            next_token = sample_argmax(sample_buf, engine->config.vocab_size);
        } else {
            float coin = (float)rand() / (float)RAND_MAX;
            next_token = sample_top_k_top_p(sample_buf, engine->config.vocab_size, top_k, top_p, temperature, coin);
        }
        if (next_token < 0 || next_token >= engine->config.vocab_size)
            next_token = engine->tokenizer->eos_id;

        if (next_token == engine->tokenizer->eos_id) {
            printf("<eos>");
            break;
        }

        const char* piece = decode(engine->tokenizer, -1, next_token);
        if (piece && piece[0]) printf("%s", piece);
        else printf("<tok:%d>", next_token);
        fflush(stdout);
        if (getenv("NEURAL_C_DEBUG_IDS")) fprintf(stderr, "[id=%d]", next_token);

        if (history && hist_cap > 0) {
            bool structural = (next_token == nl_id) || (next_token == engine->tokenizer->bos_id) ||
                              (next_token == engine->tokenizer->eos_id) || (next_token == engine->tokenizer->pad_id);
            if (!structural) {
                if (hist_len < hist_cap) history[hist_len++] = next_token;
                else { memmove(history, history + 1, (size_t)(hist_cap - 1) * sizeof(int)); history[hist_cap - 1] = next_token; }
            }
        }

        if (pos >= engine->config.seq_len) break;
        transformer_forward(engine, next_token, pos);
        pos++;
    }

    printf("\n\n");
    free(work);
    free(history);
    free(prompt_tokens);
    free(owned);
}
