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
    size_t max_batch_size;

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

NeuralNetwork *create_neural_network(size_t max_batch_size, size_t input_size, size_t hidden1_size, size_t hidden2_size, size_t hidden3_size, size_t hidden4_size, size_t output_size);
void forward_propagation(NeuralNetwork *nn, const float *input, float *output);
void forward_propagation_batch(NeuralNetwork* nn, const float* inputs, float* outputs, int batch_size);
void backward_propagation_batch(NeuralNetwork *nn, const float *inputs, const float *targets, int batch_size, float learning_rate);
void backward_propagation_batch_with_deltas(NeuralNetwork *nn, const float *inputs, const float *delta_out, int batch_size, float learning_rate);
void get_input_gradients_batch(NeuralNetwork *nn, const float *delta_out, float *delta_in, int batch_size);
void backward_propagation_batch(NeuralNetwork *nn, const float *inputs, const float *targets, int batch_size, float learning_rate);
void free_neural_network(NeuralNetwork *nn);
bool save_weights(NeuralNetwork *nn, const char *filename);
bool load_weights(NeuralNetwork *nn, const char *filename);

#endif // NEURAL_NETWORK_H
