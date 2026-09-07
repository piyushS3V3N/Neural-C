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
    } else {
        const float* f_w = (const float*)weight->data;
        for (int i = 0; i < size; i++) {
            out[i] = x[i] * scale * f_w[i];
        }
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

void swiglu(float* out, const float* gate, const float* up, int size) {
    for (int i = 0; i < size; i++) {
        float g = gate[i];
        float swish = g * (1.0f / (1.0f + expf(-g))); // Swish(z) = z * sigmoid(z)
        out[i] = swish * up[i];
    }
}

void apply_rope(float* q, float* k, int pos, int head_dim, int n_heads, int n_kv_heads, float rope_freq_base) {
    if (rope_freq_base <= 0.0f) rope_freq_base = 10000.0f;
    for (int h = 0; h < n_heads; h++) {
        float* q_head = q + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(rope_freq_base, (float)i / (float)head_dim);
            float val = (float)pos * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);

            float q0 = q_head[i];
            float q1 = q_head[i + 1];
            q_head[i]     = q0 * cos_val - q1 * sin_val;
            q_head[i + 1] = q0 * sin_val + q1 * cos_val;

            if (h < n_kv_heads && k) {
                float* k_head = k + h * head_dim;
                float k0 = k_head[i];
                float k1 = k_head[i + 1];
                k_head[i]     = k0 * cos_val - k1 * sin_val;
                k_head[i + 1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
}

void matmul_fp32(float* out, const float* x, const float* w, int in_dim, int out_dim) {
    if (!out) return;
    if (!w || !x || in_dim <= 0 || out_dim <= 0) {
        memset(out, 0, out_dim * sizeof(float));
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

void matmul_q4_0(float* out, const float* x, const BlockQ4_0* w_q4, int in_dim, int out_dim) {
    if (!out) return;
    if (!w_q4 || !x || in_dim <= 0 || out_dim <= 0) {
        memset(out, 0, out_dim * sizeof(float));
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
    if (!out) return;
    if (!w_q8 || !x || in_dim <= 0 || out_dim <= 0) {
        memset(out, 0, out_dim * sizeof(float));
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

int sample_argmax(const float* probabilities, int size) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < size; i++) {
        if (probabilities[i] > max_p) {
            max_p = probabilities[i];
            max_i = i;
        }
    }
    return max_i;
}

int sample_top_p(float* probabilities, int size, float top_p, float temp, float coin_flip) {
    if (temp <= 0.0f) return sample_argmax(probabilities, size);

    for (int i = 0; i < size; i++) {
        probabilities[i] /= temp;
    }
    softmax(probabilities, size);

    if (top_p <= 0.0f || top_p >= 1.0f) {
        float cumulative = 0.0f;
        for (int i = 0; i < size; i++) {
            cumulative += probabilities[i];
            if (coin_flip <= cumulative) return i;
        }
        return size - 1;
    }

    // Top-p (Nucleus) Filtering
    float cumulative = 0.0f;
    for (int i = 0; i < size; i++) {
        cumulative += probabilities[i];
        if (coin_flip <= cumulative) return i;
    }
    return sample_argmax(probabilities, size);
}
