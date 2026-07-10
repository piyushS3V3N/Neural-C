#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    size_t input_size;
    size_t hidden1_size;
    size_t hidden2_size;
    size_t hidden3_size;
    size_t hidden4_size;
    size_t output_size;

    float *weights_input_hidden1;
    float *weights_hidden1_hidden2;
    float *weights_hidden2_hidden3;
    float *weights_hidden3_hidden4;
    float *weights_hidden4_output;

    float *bias_hidden1;
    float *bias_hidden2;
    float *bias_hidden3;
    float *bias_hidden4;
    float *bias_output;

    float *hidden1_activations;
    float *hidden2_activations;
    float *hidden3_activations;
    float *hidden4_activations;
    float *output_activations;
    
    bool use_gpu;
    
} NeuralNetwork;

NeuralNetwork *create_neural_network(size_t input_size, size_t hidden1_size, size_t hidden2_size, size_t hidden3_size, size_t hidden4_size, size_t output_size);
void forward_propagation(NeuralNetwork *nn, const float *input, float *output);
void backward_propagation(NeuralNetwork *nn, const float *input, const float *target, float learning_rate);
void free_neural_network(NeuralNetwork *nn);

#endif // NEURAL_NETWORK_H
