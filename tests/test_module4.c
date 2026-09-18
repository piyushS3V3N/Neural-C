#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "math_kernels.h"

int main() {
    printf("=========================================\n");
    printf("  Module 4 Test: Quantized Math Kernels   \n");
    printf("=========================================\n");

    // 1. Test RMSNorm Kernel
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];
    rmsnorm(out, x, w, 4, 1e-5f);
    
    // RMS = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) = 2.7386
    float expected_rms = sqrtf(7.5f);
    assert(fabsf(out[0] - (1.0f / expected_rms)) < 1e-3f);
    assert(fabsf(out[3] - (4.0f / expected_rms)) < 1e-3f);
    printf("[PASS] RMSNorm Layer Normalization Kernel\n");

    // 2. Test Softmax Kernel
    float logits[3] = {1.0f, 2.0f, 3.0f};
    softmax(logits, 3);
    float sum = logits[0] + logits[1] + logits[2];
    assert(fabsf(sum - 1.0f) < 1e-5f);
    assert(logits[2] > logits[1] && logits[1] > logits[0]);
    printf("[PASS] Softmax Probability Distribution Kernel\n");

    // 3. Test SwiGLU Activation Kernel
    float gate[2] = {0.0f, 2.0f};
    float up[2]   = {1.0f, 3.0f};
    float ffn_out[2];
    swiglu(ffn_out, gate, up, 2);
    // gate 0.0 -> Swish(0) = 0 -> ffn_out[0] = 0
    assert(fabsf(ffn_out[0]) < 1e-5f);
    assert(ffn_out[1] > 0.0f);
    printf("[PASS] SwiGLU Gated Feed-Forward Kernel\n");

    // 4. Test RoPE Position Rotation
    float q[4] = {1.0f, 0.0f, 1.0f, 0.0f};
    float k[4] = {1.0f, 0.0f, 1.0f, 0.0f};
    apply_rope(q, k, 1, 4, 1, 1, 10000.0f);
    // After half-split rotation at pos 1, norm of paired elements (q[0], q[2]) remains conserved (sqrt(2))
    float q_norm = sqrtf(q[0]*q[0] + q[2]*q[2]);
    assert(fabsf(q_norm - sqrtf(2.0f)) < 1e-4f);
    printf("[PASS] Rotary Position Embedding (RoPE) Kernel\n");

    // 5. Test Q4_0 Quantized Matrix-Vector Multiplication
    BlockQ4_0 q4_weights[2]; // 1 row, 64 columns (2 blocks)
    // Scale = 1.0f (0x3C00 in FP16)
    q4_weights[0].scale = 0x3C00;
    q4_weights[1].scale = 0x3C00;
    memset(q4_weights[0].qs, 0x88, 16); // Nibbles = 8 (val = 8 - 8 = 0)
    memset(q4_weights[1].qs, 0x88, 16);

    float in_vec[64];
    for (int i = 0; i < 64; i++) in_vec[i] = 1.0f;
    float mat_out[1];
    matmul_q4_0(mat_out, in_vec, q4_weights, 64, 1);
    assert(fabsf(mat_out[0]) < 1e-4f); // 0 * scale * 1 = 0
    printf("[PASS] Q4_0 Quantized MatMul Acceleration Kernel\n");

    // 6. Test Sampler
    float probs[4] = {0.1f, 0.7f, 0.1f, 0.1f};
    int sampled_idx = sample_top_p(probs, 4, 0.9f, 0.0f, 0.5f);
    assert(sampled_idx == 1);
    printf("[PASS] Temperature & Top-p Nucleus Sampler\n");

    printf("\n>>> MODULE 4 (QUANTIZED MATH & TRANSFORMER KERNELS) PASSED ALL TESTS SUCCESSFULLY! <<<\n");
    return 0;
}
