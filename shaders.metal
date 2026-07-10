#include <metal_stdlib>
using namespace metal;

// GPU Kernel for performing Neural Network Forward Propagation
// This replaces the nested for-loops in C with massive parallel execution.
kernel void forward_pass_layer(
    device const float* inputs [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* biases [[buffer(2)]],
    device float* outputs [[buffer(3)]],
    constant int& input_size [[buffer(4)]],
    constant int& output_size [[buffer(5)]],
    constant int& use_sigmoid [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    // Boundary check
    if ((int)id >= output_size) return;
    
    // Each thread calculates exactly ONE output neuron simultaneously
    float sum = biases[id];
    
    // Perform dot product on this neuron's weights
    for (int i = 0; i < input_size; i++) {
        sum += inputs[i] * weights[i * output_size + id]; 
    }
    
    // Apply Activation Function
    if (use_sigmoid == 1) {
        outputs[id] = 1.0 / (1.0 + exp(-sum));
    } else {
        // LeakyReLU
        outputs[id] = sum > 0.0 ? sum : 0.01 * sum;
    }
}
