#include <metal_stdlib>
using namespace metal;

struct BlockQ4_0_Metal {
    ushort scale_bits;
    uint8_t qs[16];
} __attribute__((packed));

inline float fp16_to_fp32_msl(ushort h) {
    uint w = (uint(h & 0x7FFF)) << 13;
    uint sign = (uint(h & 0x8000)) << 16;
    uint exp = (h >> 10) & 0x1F;
    if (exp == 0x1F) w |= 0x7F800000;
    else if (exp != 0) w += (127 - 15) << 23;
    uint result = sign | w;
    return as_type<float>(result);
}

// Metal MSL Kernel: Parallel Q4_0 Matrix-Vector Multiplication (GEMV)
kernel void gemv_q4_0(
    device const float* x [[buffer(0)]],
    device const BlockQ4_0_Metal* weights [[buffer(1)]],
    device float* outputs [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    uint thread_idx [[thread_position_in_grid]]
) {
    int row = thread_idx; // 1 Thread per Output Dimension Row
    if (row >= out_dim) return;

    int blocks_per_row = in_dim / 32;
    device const BlockQ4_0_Metal* row_blocks = weights + row * blocks_per_row;

    float sum = 0.0;
    for (int b = 0; b < blocks_per_row; b++) {
        float scale = fp16_to_fp32_msl(row_blocks[b].scale_bits);
        device const uint8_t* qs = row_blocks[b].qs;
        device const float* x_block = x + b * 32;

        for (int l = 0; l < 16; l++) {
            uint8_t byte = qs[l];
            int v0 = (byte & 0x0F) - 8;
            int v1 = ((byte >> 4) & 0x0F) - 8;

            sum += float(v0) * scale * x_block[l];
            sum += float(v1) * scale * x_block[l + 16];
        }
    }
    if (isnan(sum) || isinf(sum)) sum = 0.0;
    outputs[row] = sum;
}

struct BlockQ8_0_Metal {
    ushort scale_bits;
    int8_t qs[32];
} __attribute__((packed));

// Metal MSL Kernel: Parallel Q8_0 Matrix-Vector Multiplication (GEMV)
kernel void gemv_q8_0(
    device const float* x [[buffer(0)]],
    device const BlockQ8_0_Metal* weights [[buffer(1)]],
    device float* outputs [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    uint thread_idx [[thread_position_in_grid]]
) {
    int row = thread_idx;
    if (row >= out_dim) return;

    int blocks_per_row = in_dim / 32;
    device const BlockQ8_0_Metal* row_blocks = weights + row * blocks_per_row;

    float sum = 0.0;
    for (int b = 0; b < blocks_per_row; b++) {
        float scale = fp16_to_fp32_msl(row_blocks[b].scale_bits);
        device const int8_t* qs = row_blocks[b].qs;
        device const float* x_block = x + b * 32;

        for (int l = 0; l < 32; l++) {
            sum += float(qs[l]) * scale * x_block[l];
        }
    }
    if (isnan(sum) || isinf(sum)) sum = 0.0;
    outputs[row] = sum;
}

// Metal MSL Kernel: Parallel FP32 Matrix-Vector Multiplication (GEMV)
kernel void gemv_fp32(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* outputs [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    uint thread_idx [[thread_position_in_grid]]
) {
    int row = thread_idx;
    if (row >= out_dim) return;

    float sum = 0.0;
    device const float* row_weights = weights + row * in_dim;
    for (int i = 0; i < in_dim; i++) {
        sum += x[i] * row_weights[i];
    }
    outputs[row] = sum;
}
