#ifndef METAL_BACKEND_H
#define METAL_BACKEND_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the Metal GPU Device and Command Queue
bool init_metal_engine(void);
const char* get_metal_device_name(void);

// Offloads a single layer's Forward Pass Matrix Multiplication to the GPU
// mathematically equivalent to: outputs = activation(inputs * weights + biases)
void metal_forward_layer(const float* inputs, 
                         const float* weights, 
                         const float* biases, 
                         float* outputs, 
                         int input_size, 
                         int output_size,
                         bool use_sigmoid);

#ifdef __cplusplus
}
#endif

#endif // METAL_BACKEND_H
