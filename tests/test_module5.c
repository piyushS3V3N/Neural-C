#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "llm_metal_backend.h"

int main() {
    printf("=========================================\n");
    printf("  Module 5 Test: Metal GPU Acceleration   \n");
    printf("=========================================\n");

    // 1. Initialize Metal GPU Compute Engine
    bool metal_available = init_llm_metal_engine();
    assert(metal_available == true);
    printf("[PASS] Metal GPU Engine Initialization: %s\n", get_llm_metal_device_name());

    // 2. Test Metal FP32 MatMul Kernel vs CPU Reference
    int in_dim = 64;
    int out_dim = 32;
    float* x = (float*)malloc(in_dim * sizeof(float));
    float* w = (float*)malloc(in_dim * out_dim * sizeof(float));
    float* gpu_out = (float*)malloc(out_dim * sizeof(float));
    float* cpu_out = (float*)malloc(out_dim * sizeof(float));

    for (int i = 0; i < in_dim; i++) x[i] = (float)i * 0.1f;
    for (int j = 0; j < out_dim * in_dim; j++) w[j] = (float)(j % 7) * 0.05f;

    // CPU Reference MatMul
    matmul_fp32(cpu_out, x, w, in_dim, out_dim);

    // GPU MatMul Execution
    metal_gemv_fp32(gpu_out, x, w, in_dim, out_dim);

    for (int j = 0; j < out_dim; j++) {
        assert(fabsf(gpu_out[j] - cpu_out[j]) < 1e-3f);
    }
    printf("[PASS] Metal GPU Parallel FP32 Matrix-Vector Multiplication\n");

    // 3. Test Metal Q4_0 Quantized GEMV Kernel
    BlockQ4_0* w_q4 = (BlockQ4_0*)calloc(out_dim * (in_dim / 32), sizeof(BlockQ4_0));
    for (int i = 0; i < out_dim * (in_dim / 32); i++) {
        w_q4[i].scale = 0x3C00; // FP16 1.0f
        memset(w_q4[i].qs, 0x88, 16);
    }
    float* gpu_q4_out = (float*)malloc(out_dim * sizeof(float));
    float* cpu_q4_out = (float*)malloc(out_dim * sizeof(float));

    matmul_q4_0(cpu_q4_out, x, w_q4, in_dim, out_dim);
    metal_gemv_q4_0(gpu_q4_out, x, w_q4, in_dim, out_dim);

    for (int j = 0; j < out_dim; j++) {
        assert(fabsf(gpu_q4_out[j] - cpu_q4_out[j]) < 1e-3f);
    }
    printf("[PASS] Metal GPU Parallel Q4_0 Quantized Matrix-Vector Multiplication\n");

    // Cleanup
    free(x); free(w); free(gpu_out); free(cpu_out);
    free(w_q4); free(gpu_q4_out); free(cpu_q4_out);
    printf("[PASS] Memory Cleanup\n");

    printf("\n>>> MODULE 5 (METAL GPU ACCELERATION ENGINE) PASSED ALL TESTS SUCCESSFULLY! <<<\n");
    return 0;
}
