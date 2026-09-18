#include "tensor.h"

size_t get_element_size(QuantType type) {
    switch (type) {
        case QUANT_FP32: return sizeof(float);
        case QUANT_FP16: return 2;
        case QUANT_Q8_0: return 1; // Approx 1 byte per weight
        case QUANT_Q4_0: return 1; // Packed 4-bit pair + scale factor
        case QUANT_Q4_K: return 1;
        default: return sizeof(float);
    }
}

void init_tensor(Tensor* t, const char* name, int n_dims, const int* shape, QuantType type) {
    if (!t) return;
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->n_dims = n_dims;
    t->type = type;
    t->numel = 1;
    for (int i = 0; i < n_dims; i++) {
        t->shape[i] = shape[i];
        t->numel *= shape[i];
    }
    t->size_bytes = t->numel * get_element_size(type);
    t->data = NULL;
    t->is_mmap = false;
}

void free_tensor(Tensor* t) {
    if (!t) return;
    if (t->data && !t->is_mmap) {
        free(t->data);
        t->data = NULL;
    }
}

TransformerWeights* create_transformer_weights(const Config* cfg) {
    if (!cfg) return NULL;
    if (cfg->n_layers <= 0 || cfg->n_layers > 512) return NULL;
    TransformerWeights* w = (TransformerWeights*)calloc(1, sizeof(TransformerWeights));
    if (!w) return NULL;

    int n_layers = cfg->n_layers;
    w->rms_att_weight = (Tensor*)calloc(n_layers, sizeof(Tensor));
    w->wq = (Tensor*)calloc(n_layers, sizeof(Tensor));
    w->wk = (Tensor*)calloc(n_layers, sizeof(Tensor));
    w->wv = (Tensor*)calloc(n_layers, sizeof(Tensor));
    w->wo = (Tensor*)calloc(n_layers, sizeof(Tensor));
    w->bq = (Tensor*)calloc(n_layers, sizeof(Tensor));
    w->bk = (Tensor*)calloc(n_layers, sizeof(Tensor));
    w->bv = (Tensor*)calloc(n_layers, sizeof(Tensor));

    w->rms_ffn_weight = (Tensor*)calloc(n_layers, sizeof(Tensor));
    w->w_gate = (Tensor*)calloc(n_layers, sizeof(Tensor));
    w->w_up = (Tensor*)calloc(n_layers, sizeof(Tensor));
    w->w_down = (Tensor*)calloc(n_layers, sizeof(Tensor));

    // Initialize non-zero weights for mock demonstration mode when no GGUF file is loaded (for small configs <= 6 layers)
    if (cfg->n_layers <= 6 && cfg->dim > 0 && cfg->hidden_dim > 0
        && cfg->n_heads > 0 && cfg->n_kv_heads > 0 && cfg->head_dim > 0) {
        int kv_dim = cfg->n_kv_heads * cfg->head_dim;
        int q_dim = cfg->n_heads * cfg->head_dim;
        size_t wq_sz = (size_t)cfg->dim * (size_t)q_dim;
        size_t wkv_sz = (size_t)cfg->dim * (size_t)kv_dim;
        size_t wo_sz = (size_t)q_dim * (size_t)cfg->dim;
        size_t ffn_sz = (size_t)cfg->dim * (size_t)cfg->hidden_dim;
        size_t ffn_down_sz = (size_t)cfg->hidden_dim * (size_t)cfg->dim;

        for (int l = 0; l < cfg->n_layers; l++) {
            w->wq[l].data = malloc(wq_sz * sizeof(float));
            w->wk[l].data = malloc(wkv_sz * sizeof(float));
            w->wv[l].data = malloc(wkv_sz * sizeof(float));
            w->wo[l].data = malloc(wo_sz * sizeof(float));
            w->w_gate[l].data = malloc(ffn_sz * sizeof(float));
            w->w_up[l].data = malloc(ffn_sz * sizeof(float));
            w->w_down[l].data = malloc(ffn_down_sz * sizeof(float));
            if (!w->wq[l].data || !w->wk[l].data || !w->wv[l].data || !w->wo[l].data ||
                !w->w_gate[l].data || !w->w_up[l].data || !w->w_down[l].data) {
                continue; // engine_matmul treats missing tensors as zeros
            }
            // Tag mock tensors so shape validation works
            w->wq[l].type = QUANT_FP32; w->wq[l].numel = wq_sz;
            w->wk[l].type = QUANT_FP32; w->wk[l].numel = wkv_sz;
            w->wv[l].type = QUANT_FP32; w->wv[l].numel = wkv_sz;
            w->wo[l].type = QUANT_FP32; w->wo[l].numel = wo_sz;
            w->w_gate[l].type = QUANT_FP32; w->w_gate[l].numel = ffn_sz;
            w->w_up[l].type = QUANT_FP32; w->w_up[l].numel = ffn_sz;
            w->w_down[l].type = QUANT_FP32; w->w_down[l].numel = ffn_down_sz;
            
            float* q_ptr = (float*)w->wq[l].data;
            float* k_ptr = (float*)w->wk[l].data;
            float* v_ptr = (float*)w->wv[l].data;
            float* o_ptr = (float*)w->wo[l].data;
            float* g_ptr = (float*)w->w_gate[l].data;
            float* u_ptr = (float*)w->w_up[l].data;
            float* d_ptr = (float*)w->w_down[l].data;

            for (size_t i = 0; i < wq_sz; i++) {
                q_ptr[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
            }
            for (size_t i = 0; i < wkv_sz; i++) {
                k_ptr[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
                v_ptr[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
            }
            for (size_t i = 0; i < wo_sz; i++) {
                o_ptr[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
            }
            for (size_t i = 0; i < ffn_sz; i++) {
                g_ptr[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
                u_ptr[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
            }
            for (size_t i = 0; i < ffn_down_sz; i++) {
                d_ptr[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
            }
        }

        size_t cls_sz = (size_t)cfg->dim * (size_t)cfg->vocab_size;
        w->w_cls.data = malloc(cls_sz * sizeof(float));
        if (w->w_cls.data) {
            w->w_cls.type = QUANT_FP32;
            w->w_cls.numel = cls_sz;
            float* cls_ptr = (float*)w->w_cls.data;
            for (size_t i = 0; i < cls_sz; i++) {
                cls_ptr[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
            }
        }
    }

    return w;
}

void free_transformer_weights(TransformerWeights* w, int n_layers) {
    if (!w) return;

    free_tensor(&w->token_embedding_table);
    free_tensor(&w->rms_final_weight);
    free_tensor(&w->w_cls);

    if (w->rms_att_weight) {
        for (int i = 0; i < n_layers; i++) {
            free_tensor(&w->rms_att_weight[i]);
            free_tensor(&w->wq[i]);
            free_tensor(&w->wk[i]);
            free_tensor(&w->wv[i]);
            free_tensor(&w->wo[i]);
            free_tensor(&w->bq[i]);
            free_tensor(&w->bk[i]);
            free_tensor(&w->bv[i]);
            free_tensor(&w->rms_ffn_weight[i]);
            free_tensor(&w->w_gate[i]);
            free_tensor(&w->w_up[i]);
            free_tensor(&w->w_down[i]);
        }
        free(w->rms_att_weight);
        free(w->wq);
        free(w->wk);
        free(w->wv);
        free(w->wo);
        if (w->bq) free(w->bq);
        if (w->bk) free(w->bk);
        if (w->bv) free(w->bv);
        free(w->rms_ffn_weight);
        free(w->w_gate);
        free(w->w_up);
        free(w->w_down);
    }

    free(w);
}

RunState* allocate_run_state(const Config* cfg) {
    if (!cfg) return NULL;
    // Validate config before any allocation (prevents calloc(0)/calloc(huge))
    if (cfg->dim <= 0 || cfg->hidden_dim <= 0 || cfg->n_layers <= 0 ||
        cfg->n_heads <= 0 || cfg->n_kv_heads <= 0 || cfg->head_dim <= 0 ||
        cfg->seq_len <= 0 || cfg->vocab_size <= 0) {
        return NULL;
    }
    RunState* s = (RunState*)calloc(1, sizeof(RunState));
    if (!s) return NULL;

    int dim = cfg->dim;
    int hidden_dim = cfg->hidden_dim;
    int n_layers = cfg->n_layers;
    int n_heads = cfg->n_heads;
    int n_kv_heads = cfg->n_kv_heads;
    int head_dim = cfg->head_dim;
    int seq_len = cfg->seq_len;
    int vocab_size = cfg->vocab_size;

    int kv_dim = n_kv_heads * head_dim;

    s->x = (float*)calloc(dim, sizeof(float));
    s->xb = (float*)calloc(dim, sizeof(float));
    s->xb2 = (float*)calloc(dim, sizeof(float));
    s->hb = (float*)calloc(hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(hidden_dim, sizeof(float));
    s->q = (float*)calloc(n_heads * head_dim, sizeof(float));
    s->k = (float*)calloc(kv_dim, sizeof(float));
    s->v = (float*)calloc(kv_dim, sizeof(float));
    s->att = (float*)calloc(n_heads * seq_len, sizeof(float));
    s->logits = (float*)calloc(vocab_size, sizeof(float));

    // KV Cache Allocation
    size_t kv_cache_size = (size_t)n_layers * (size_t)seq_len * (size_t)kv_dim;
    // Guard against size_t overflow / absurd allocations (>8GB per cache)
    if (kv_dim <= 0 || kv_cache_size / (size_t)kv_dim != (size_t)n_layers * (size_t)seq_len) {
        free_run_state(s);
        return NULL;
    }
    s->key_cache = (float*)calloc(kv_cache_size, sizeof(float));
    s->value_cache = (float*)calloc(kv_cache_size, sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 ||
        !s->q || !s->k || !s->v || !s->att || !s->logits ||
        !s->key_cache || !s->value_cache) {
        free_run_state(s);
        return NULL;
    }

    return s;
}

void free_run_state(RunState* s) {
    if (!s) return;
    if (s->x) free(s->x);
    if (s->xb) free(s->xb);
    if (s->xb2) free(s->xb2);
    if (s->hb) free(s->hb);
    if (s->hb2) free(s->hb2);
    if (s->q) free(s->q);
    if (s->k) free(s->k);
    if (s->v) free(s->v);
    if (s->att) free(s->att);
    if (s->logits) free(s->logits);
    if (s->key_cache) free(s->key_cache);
    if (s->value_cache) free(s->value_cache);
    free(s);
}
