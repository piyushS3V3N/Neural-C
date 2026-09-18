#ifndef MATH_KERNELS_H
#define MATH_KERNELS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "tensor.h"

// Q4_0 Quantized Block Representation (32 weights per block, 18 bytes)
typedef struct __attribute__((packed)) {
    uint16_t scale;        // FP16 Scale Factor
    uint8_t qs[16];        // 16 bytes storing 32 packed 4-bit nibbles
} BlockQ4_0;

// Q8_0 Quantized Block Representation (32 weights per block, 34 bytes)
typedef struct __attribute__((packed)) {
    uint16_t scale;        // FP16 Scale Factor
    int8_t qs[32];         // 32 signed 8-bit quantized weights
} BlockQ8_0;

// Q6_K Quantized Block Representation (256 weights per block, 210 bytes)
typedef struct __attribute__((packed)) {
    uint8_t ql[128];       // quants lower 4 bits (256 4-bit nibbles)
    uint8_t qh[64];        // quants upper 2 bits (256 2-bit values)
    int8_t  scales[16];    // 16 sub-block scales
    uint16_t d;            // FP16 super-block scale factor
} BlockQ6_K;

// Core Transformer Mathematical Kernels
float fp16_to_fp32(uint16_t h);
void rmsnorm(float* out, const float* x, const float* weight, int size, float eps);
void rmsnorm_tensor(float* out, const float* x, const Tensor* weight, int size, float eps);
void softmax(float* x, int size);
void swiglu(float* out, const float* gate, const float* up, int size);

// Rotary Position Embedding (RoPE)
void apply_rope(float* q, float* k, int pos, int head_dim, int n_heads, int n_kv_heads, float rope_freq_base);

// Matrix-Vector Multiplication Kernels (y = W * x)
void matmul_fp32(float* out, const float* x, const float* w, int in_dim, int out_dim);
void matmul_fp16(float* out, const float* x, const uint16_t* w_fp16, int in_dim, int out_dim);
void matmul_q4_0(float* out, const float* x, const BlockQ4_0* w_q4, int in_dim, int out_dim);
void matmul_q8_0(float* out, const float* x, const BlockQ8_0* w_q8, int in_dim, int out_dim);
void matmul_q6_k(float* out, const float* x, const BlockQ6_K* w_q6, int in_dim, int out_dim);

// Sampler (Temperature & Top-p / Top-k Sampling)
int sample_argmax(const float* probabilities, int size);
int sample_top_p(float* probabilities, int size, float top_p, float temp, float coin_flip);
int sample_top_k_top_p(float* probabilities, int size, int top_k, float top_p, float temp, float coin_flip);

// Repetition, frequency, and presence penalties.
void apply_repetition_penalty(float* logits, int vocab_size, const int* recent_tokens, int n_recent, float penalty);
void apply_repetition_penalty_ex(float* logits, int vocab_size, const int* recent_tokens, int n_recent,
                                float repeat_penalty, float frequency_penalty, float presence_penalty);

#endif // MATH_KERNELS_H
