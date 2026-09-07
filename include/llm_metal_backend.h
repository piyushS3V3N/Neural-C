#ifndef LLM_METAL_BACKEND_H
#define LLM_METAL_BACKEND_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "math_kernels.h"

// Metal Engine API
bool init_llm_metal_engine(void);
const char* get_llm_metal_device_name(void);

// GPU Accelerated Quantized Matrix-Vector Operations
void metal_gemv_q4_0(float* out, const float* x, const BlockQ4_0* w_q4, int in_dim, int out_dim);
void metal_gemv_q8_0(float* out, const float* x, const BlockQ8_0* w_q8, int in_dim, int out_dim);
void metal_gemv_fp32(float* out, const float* x, const float* w, int in_dim, int out_dim);

#endif // LLM_METAL_BACKEND_H
