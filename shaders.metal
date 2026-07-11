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

// Backward Pass: Compute Output Deltas
kernel void backward_delta_output(
    device const float* out [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* delta_out [[buffer(2)]],
    constant int& output_size [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    uint2 id [[thread_position_in_grid]]
) {
    int row = id.x; // Batch
    int col = id.y; // Output Neuron Index
    if (row >= batch_size || col >= output_size) return;
    delta_out[row * output_size + col] = out[row * output_size + col] - target[row * output_size + col];
}

// Backward Pass: Compute Hidden Deltas
kernel void backward_delta_hidden(
    device const float* delta_next [[buffer(0)]],
    device const float* weights_next [[buffer(1)]],
    device const float* activations [[buffer(2)]],
    device float* delta_curr [[buffer(3)]],
    constant int& curr_size [[buffer(4)]],
    constant int& next_size [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    uint2 id [[thread_position_in_grid]]
) {
    int row = id.x; // Batch
    int col = id.y; // Current Neuron Index
    if (row >= batch_size || col >= curr_size) return;
    
    float err = 0.0;
    for (int j = 0; j < next_size; j++) {
        err += delta_next[row * next_size + j] * weights_next[col * next_size + j];
    }
    
    float act = activations[row * curr_size + col];
    float deriv = act > 0.0 ? 1.0 : 0.01; // Leaky ReLU derivative
    delta_curr[row * curr_size + col] = err * deriv;
}

// Backward Pass: Update Weights and Biases
kernel void backward_update_weights(
    device const float* delta [[buffer(0)]],
    device const float* prev_activations [[buffer(1)]],
    device float* weights [[buffer(2)]],
    device float* biases [[buffer(3)]],
    constant int& prev_size [[buffer(4)]],
    constant int& curr_size [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    constant float& learning_rate [[buffer(7)]],
    uint2 id [[thread_position_in_grid]]
) {
    int prev_idx = id.x; // 0 to prev_size (prev_size is for bias)
    int curr_idx = id.y; // 0 to curr_size-1
    
    if (curr_idx >= curr_size || prev_idx > prev_size) return;
    
    float lr = learning_rate / (float)batch_size;
    
    if (prev_idx == prev_size) {
        // Update bias
        float grad_b = 0.0;
        for (int b = 0; b < batch_size; b++) {
            grad_b += delta[b * curr_size + curr_idx];
        }
        biases[curr_idx] -= lr * grad_b;
    } else {
        // Update weights
        float grad_w = 0.0;
        for (int b = 0; b < batch_size; b++) {
            grad_w += delta[b * curr_size + curr_idx] * prev_activations[b * prev_size + prev_idx];
        }
        weights[prev_idx * curr_size + curr_idx] -= lr * grad_w;
    }
}

// Backward Pass: Compute Input Deltas (for GAN Generator backprop)
kernel void backward_delta_input(
    device const float* delta_next [[buffer(0)]],
    device const float* weights_next [[buffer(1)]],
    device float* delta_curr [[buffer(2)]],
    constant int& curr_size [[buffer(3)]],
    constant int& next_size [[buffer(4)]],
    constant int& batch_size [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    int row = id.x; // Batch
    int col = id.y; // Current Neuron Index (Input)
    if (row >= batch_size || col >= curr_size) return;
    
    float err = 0.0;
    for (int j = 0; j < next_size; j++) {
        err += delta_next[row * next_size + j] * weights_next[col * next_size + j];
    }
    
    // No activation derivative for the very first input layer (it's passed to generator)
    delta_curr[row * curr_size + col] = err;
}
