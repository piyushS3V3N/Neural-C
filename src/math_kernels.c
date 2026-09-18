#include "math_kernels.h"

// Helper: Convert FP16 uint16 bit pattern to FP32 float
float fp16_to_fp32(uint16_t h) {
    uint32_t w = (uint32_t)(h & 0x7FFF) << 13;
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    if (exp == 0x1F) w |= 0x7F800000;
    else if (exp != 0) w += (127 - 15) << 23;
    uint32_t result = sign | w;
    float val;
    memcpy(&val, &result, sizeof(float));
    return val;
}

void rmsnorm(float* out, const float* x, const float* weight, int size, float eps) {
    if (!out || !x || size <= 0) return;
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf((sum_sq / (float)size) + eps);
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * scale * (weight ? weight[i] : 1.0f);
    }
}

void rmsnorm_tensor(float* out, const float* x, const Tensor* weight, int size, float eps) {
    if (!out || !x || size <= 0) return;
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf((sum_sq / (float)size) + eps);
    if (!weight || !weight->data) {
        for (int i = 0; i < size; i++) out[i] = x[i] * scale;
        return;
    }
    if (weight->type == QUANT_FP16) {
        const uint16_t* h_w = (const uint16_t*)weight->data;
        for (int i = 0; i < size; i++) {
            out[i] = x[i] * scale * fp16_to_fp32(h_w[i]);
        }
    } else if (weight->type == QUANT_FP32) {
        const float* f_w = (const float*)weight->data;
        for (int i = 0; i < size; i++) {
            out[i] = x[i] * scale * f_w[i];
        }
    } else {
        // Norm weights are never quantized (Q4_0/Q8_0/Q4_K). A quantized
        // type here means a loader mismatch — do NOT reinterpret the bytes
        // as float (OOB/garbage). Fall back to unweighted norm.
        for (int i = 0; i < size; i++) out[i] = x[i] * scale;
    }
}

void softmax(float* x, int size) {
    if (!x || size <= 0) return;
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    if (!(sum > 0.0f) || !isfinite(sum)) {
        float inv = 1.0f / (float)size;
        for (int i = 0; i < size; i++) x[i] = inv;
        return;
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

void swiglu(float* out, const float* gate, const float* up, int size) {
    if (!out || !gate || !up || size <= 0) return;
    for (int i = 0; i < size; i++) {
        float g = gate[i];
        float swish = g * (1.0f / (1.0f + expf(-g))); // Swish(z) = z * sigmoid(z)
        out[i] = swish * up[i];
    }
}

void apply_rope(float* q, float* k, int pos, int head_dim, int n_heads, int n_kv_heads, float rope_freq_base) {
    if (!q || head_dim <= 0 || n_heads <= 0) return;
    if (pos < 0) pos = 0;
    if (rope_freq_base <= 0.0f) rope_freq_base = 10000.0f;
    int half_dim = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float* q_head = q + (size_t)h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(rope_freq_base, (float)(2 * i) / (float)head_dim);
            float val = (float)pos * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);

            float q0 = q_head[i];
            float q1 = q_head[i + half_dim];
            q_head[i]            = q0 * cos_val - q1 * sin_val;
            q_head[i + half_dim] = q0 * sin_val + q1 * cos_val;
        }
    }
    // Rotate K heads independently (handles MQA/GQA where n_kv_heads != n_heads)
    if (k && n_kv_heads > 0) {
        for (int h = 0; h < n_kv_heads; h++) {
            float* k_head = k + (size_t)h * head_dim;
            for (int i = 0; i < half_dim; i++) {
                float freq = 1.0f / powf(rope_freq_base, (float)(2 * i) / (float)head_dim);
                float val = (float)pos * freq;
                float cos_val = cosf(val);
                float sin_val = sinf(val);

                float k0 = k_head[i];
                float k1 = k_head[i + half_dim];
                k_head[i]            = k0 * cos_val - k1 * sin_val;
                k_head[i + half_dim] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
}

void matmul_fp32(float* out, const float* x, const float* w, int in_dim, int out_dim) {
    if (!out || out_dim <= 0) return;
    if (!w || !x || in_dim <= 0) {
        memset(out, 0, (size_t)out_dim * sizeof(float));
        return;
    }
    for (int j = 0; j < out_dim; j++) {
        float sum = 0.0f;
        const float* row = w + (size_t)j * in_dim;
        for (int i = 0; i < in_dim; i++) {
            sum += x[i] * row[i];
        }
        out[j] = sum;
    }
}

void matmul_fp16(float* out, const float* x, const uint16_t* w_fp16, int in_dim, int out_dim) {
    if (!out) return;
    if (!w_fp16 || !x || in_dim <= 0 || out_dim <= 0) {
        if (out_dim > 0) memset(out, 0, (size_t)out_dim * sizeof(float));
        return;
    }
    for (int j = 0; j < out_dim; j++) {
        float sum = 0.0f;
        const uint16_t* row = w_fp16 + (size_t)j * in_dim;
        for (int i = 0; i < in_dim; i++) {
            sum += x[i] * fp16_to_fp32(row[i]);
        }
        out[j] = isfinite(sum) ? sum : 0.0f;
    }
}

void matmul_q4_0(float* out, const float* x, const BlockQ4_0* w_q4, int in_dim, int out_dim) {
    if (!out || out_dim <= 0) return;
    if (!w_q4 || !x || in_dim <= 0) {
        memset(out, 0, (size_t)out_dim * sizeof(float));
        return;
    }
    if (in_dim % 32 != 0) {
        // Block layout requires multiples of 32; refuse silent truncation
        memset(out, 0, (size_t)out_dim * sizeof(float));
        return;
    }
    int blocks_per_row = in_dim / 32;
    for (int j = 0; j < out_dim; j++) {
        float sum = 0.0f;
        const BlockQ4_0* row_blocks = w_q4 + j * blocks_per_row;
        
        for (int b = 0; b < blocks_per_row; b++) {
            float scale = fp16_to_fp32(row_blocks[b].scale);
            const uint8_t* qs = row_blocks[b].qs;
            const float* x_block = x + b * 32;

            for (int l = 0; l < 16; l++) {
                uint8_t byte = qs[l];
                int v0 = (byte & 0x0F) - 8;
                int v1 = ((byte >> 4) & 0x0F) - 8;

                sum += (float)v0 * scale * x_block[l];
                sum += (float)v1 * scale * x_block[l + 16];
            }
        }
        if (isnan(sum) || isinf(sum)) sum = 0.0f;
        out[j] = sum;
    }
}

void matmul_q8_0(float* out, const float* x, const BlockQ8_0* w_q8, int in_dim, int out_dim) {
    if (!out || out_dim <= 0) return;
    if (!w_q8 || !x || in_dim <= 0) {
        memset(out, 0, (size_t)out_dim * sizeof(float));
        return;
    }
    if (in_dim % 32 != 0) {
        memset(out, 0, (size_t)out_dim * sizeof(float));
        return;
    }
    int blocks_per_row = in_dim / 32;
    for (int j = 0; j < out_dim; j++) {
        float sum = 0.0f;
        const BlockQ8_0* row_blocks = w_q8 + j * blocks_per_row;
        
        for (int b = 0; b < blocks_per_row; b++) {
            float scale = fp16_to_fp32(row_blocks[b].scale);
            const int8_t* qs = row_blocks[b].qs;
            const float* x_block = x + b * 32;

            for (int l = 0; l < 32; l++) {
                sum += (float)qs[l] * scale * x_block[l];
            }
        }
        if (isnan(sum) || isinf(sum)) sum = 0.0f;
        out[j] = sum;
    }
}

void matmul_q6_k(float* out, const float* x, const BlockQ6_K* w_q6, int in_dim, int out_dim) {
    if (!out || out_dim <= 0) return;
    if (!w_q6 || !x || in_dim <= 0) {
        memset(out, 0, (size_t)out_dim * sizeof(float));
        return;
    }
    if (in_dim % 256 != 0) {
        memset(out, 0, (size_t)out_dim * sizeof(float));
        return;
    }
    int blocks_per_row = in_dim / 256;
    for (int j = 0; j < out_dim; j++) {
        float sum = 0.0f;
        const BlockQ6_K* row_blocks = w_q6 + (size_t)j * blocks_per_row;

        for (int b = 0; b < blocks_per_row; b++) {
            const BlockQ6_K* block = &row_blocks[b];
            float d = fp16_to_fp32(block->d);
            const uint8_t* ql = block->ql;
            const uint8_t* qh = block->qh;
            const int8_t*  sc = block->scales;
            const float* x_block = x + b * 256;

            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    uint8_t qh_val = qh[l];
                    int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | ((qh_val & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh_val >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh_val >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh_val >> 6) & 3) << 4)) - 32;

                    sum += d * (float)sc[is + 0] * (float)q1 * x_block[l +  0];
                    sum += d * (float)sc[is + 2] * (float)q2 * x_block[l + 32];
                    sum += d * (float)sc[is + 4] * (float)q3 * x_block[l + 64];
                    sum += d * (float)sc[is + 6] * (float)q4 * x_block[l + 96];
                }
                x_block += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        if (isnan(sum) || isinf(sum)) sum = 0.0f;
        out[j] = sum;
    }
}

int sample_argmax(const float* probabilities, int size) {
    if (!probabilities || size <= 0) return 0;
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < size; i++) {
        if (probabilities[i] > max_p) {
            max_p = probabilities[i];
            max_i = i;
        }
    }
    if (max_i < 0) return 0;
    if (max_i >= size) return size - 1;
    return max_i;
}

typedef struct { float p; int idx; } ProbIdx;

static void quickselect_partition(ProbIdx* arr, int left, int right, int k) {
    if (left >= right || k <= left || k > right + 1) return;
    int pivot_idx = left + (right - left) / 2;
    float pivot_p = arr[pivot_idx].p;
    
    ProbIdx tmp = arr[pivot_idx]; arr[pivot_idx] = arr[right]; arr[right] = tmp;
    int store = left;
    for (int i = left; i < right; i++) {
        if (arr[i].p > pivot_p) {
            tmp = arr[i]; arr[i] = arr[store]; arr[store] = tmp;
            store++;
        }
    }
    tmp = arr[store]; arr[store] = arr[right]; arr[right] = tmp;
    
    if (store == k - 1) return;
    if (store < k - 1) quickselect_partition(arr, store + 1, right, k);
    else quickselect_partition(arr, left, store - 1, k);
}

static int cmp_prob_desc(const void* a, const void* b) {
    float pa = ((const ProbIdx*)a)->p, pb = ((const ProbIdx*)b)->p;
    return (pa < pb) - (pa > pb);
}

int sample_top_p(float* probabilities, int size, float top_p, float temp, float coin_flip) {
    return sample_top_k_top_p(probabilities, size, 0, top_p, temp, coin_flip);
}

int sample_top_k_top_p(float* probabilities, int size, int top_k, float top_p, float temp, float coin_flip) {
    if (!probabilities || size <= 0) return 0;
    if (temp <= 0.0f) return sample_argmax(probabilities, size);

    for (int i = 0; i < size; i++) {
        probabilities[i] /= temp;
    }
    softmax(probabilities, size);

    if (!(coin_flip >= 0.0f) || coin_flip >= 1.0f) {
        coin_flip = 0.5f;
    }

    int K = size;
    if (top_k > 0 && top_k < K) K = top_k;

    ProbIdx* sorted = (ProbIdx*)malloc((size_t)size * sizeof(ProbIdx));
    if (!sorted) {
        float cumulative = 0.0f;
        for (int i = 0; i < size; i++) {
            cumulative += probabilities[i];
            if (coin_flip < cumulative) return i;
        }
        return size - 1;
    }

    for (int i = 0; i < size; i++) {
        sorted[i].p = probabilities[i];
        sorted[i].idx = i;
    }

    if (size > K) {
        quickselect_partition(sorted, 0, size - 1, K);
    }

    qsort(sorted, (size_t)K, sizeof(ProbIdx), cmp_prob_desc);

    int cutoff = K;

    if (top_p > 0.0f && top_p < 1.0f) {
        float cumsum = 0.0f;
        int p_cutoff = cutoff;
        for (int i = 0; i < cutoff; i++) {
            cumsum += sorted[i].p;
            if (cumsum >= top_p) { p_cutoff = i + 1; break; }
        }
        cutoff = p_cutoff;
    }
    if (cutoff < 1) cutoff = 1;

    float mass = 0.0f;
    for (int i = 0; i < cutoff; i++) mass += sorted[i].p;
    if (!(mass > 0.0f) || !isfinite(mass)) {
        int id = sorted[0].idx;
        free(sorted);
        return id;
    }

    float threshold = coin_flip * mass;
    float acc = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        acc += sorted[i].p;
        if (threshold < acc) {
            int id = sorted[i].idx;
            free(sorted);
            return id;
        }
    }
    int id = sorted[cutoff - 1].idx;
    free(sorted);
    return id;
}

void apply_repetition_penalty(float* logits, int vocab_size, const int* recent_tokens, int n_recent, float penalty) {
    apply_repetition_penalty_ex(logits, vocab_size, recent_tokens, n_recent, penalty, 0.0f, 0.0f);
}

void apply_repetition_penalty_ex(float* logits, int vocab_size, const int* recent_tokens, int n_recent,
                                float repeat_penalty, float frequency_penalty, float presence_penalty) {
    if (!logits || vocab_size <= 0 || !recent_tokens || n_recent <= 0) return;
    if (repeat_penalty <= 1.0f && frequency_penalty == 0.0f && presence_penalty == 0.0f) return;

    // Apply ONCE per unique token (llama.cpp semantics). The history may
    // hold the same id many times; penalizing per occurrence would apply
    // repeat_penalty^count and count*count*freq instead of once.
    for (int r = 0; r < n_recent; r++) {
        int tok = recent_tokens[r];
        if (tok < 0 || tok >= vocab_size) continue;
        bool seen = false;
        for (int k = 0; k < r; k++) {
            if (recent_tokens[k] == tok) { seen = true; break; }
        }
        if (seen) continue;

        int count = 0;
        for (int k = 0; k < n_recent; k++) {
            if (recent_tokens[k] == tok) count++;
        }

        float v = logits[tok];
        if (!isfinite(v)) continue;

        if (repeat_penalty > 1.0f) {
            if (v > 0.0f) v /= repeat_penalty;
            else if (v < 0.0f) v *= repeat_penalty;
        }

        if (count > 0) {
            v -= frequency_penalty * (float)count;
            v -= presence_penalty;
        }
        logits[tok] = v;
    }
}
