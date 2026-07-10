#include <metal_stdlib>
using namespace metal;

// GEMM Kernel for Batched Forward Propagation
kernel void forward_pass_layer(
    device const float* inputs [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* biases [[buffer(2)]],
    device float* outputs [[buffer(3)]],
    constant int& input_size [[buffer(4)]],
    constant int& output_size [[buffer(5)]],
    constant int& use_sigmoid [[buffer(6)]],
    constant int& batch_size [[buffer(7)]],
    uint2 id [[thread_position_in_grid]]
) {
    int row = id.x; // Batch Index
    int col = id.y; // Output Neuron Index
    
    if (row >= batch_size || col >= output_size) return;
    
    float sum = biases[col];
    
    for (int i = 0; i < input_size; i++) {
        sum += inputs[row * input_size + i] * weights[i * output_size + col];
    }
    
    if (use_sigmoid == 1) {
        outputs[row * output_size + col] = 1.0 / (1.0 + exp(-sum));
    } else {
        outputs[row * output_size + col] = sum > 0.0 ? sum : 0.01 * sum;
    }
}
