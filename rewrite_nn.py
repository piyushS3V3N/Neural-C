import re

with open('NeuralNetwork.h', 'r') as f:
    h_code = f.read()

h_code = h_code.replace('size_t hidden2_size;', 'size_t hidden2_size;\n    size_t hidden3_size;')
h_code = h_code.replace('float *weights_hidden2_output;', 'float *weights_hidden2_hidden3;\n    float *weights_hidden3_output;')
h_code = h_code.replace('float *bias_output;', 'float *bias_hidden3;\n    float *bias_output;')
h_code = h_code.replace('float *hidden2_activations;', 'float *hidden2_activations;\n    float *hidden3_activations;')
h_code = h_code.replace('create_neural_network(size_t input_size, size_t hidden1_size, size_t hidden2_size, size_t output_size);', 'create_neural_network(size_t input_size, size_t hidden1_size, size_t hidden2_size, size_t hidden3_size, size_t output_size);')

with open('NeuralNetwork.h', 'w') as f:
    f.write(h_code)


with open('NeuralNetwork.c', 'r') as f:
    c_code = f.read()

# I will write the full new NeuralNetwork.c since it's only 160 lines
new_c = """#include <stdlib.h>
#include <math.h>
#include "NeuralNetwork.h"
#include "metal_backend.h"

// LeakyReLU Activation Function
static float leaky_relu(float x) {
    return x > 0.0f ? x : 0.01f * x;
}
static float leaky_relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.01f;
}

// Random float between -1.0 and 1.0
static float random_float() {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

NeuralNetwork *create_neural_network(size_t input_size, size_t hidden1_size, size_t hidden2_size, size_t hidden3_size, size_t output_size) {
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    nn->input_size = input_size;
    nn->hidden1_size = hidden1_size;
    nn->hidden2_size = hidden2_size;
    nn->hidden3_size = hidden3_size;
    nn->output_size = output_size;
    nn->use_gpu = false;

    nn->weights_input_hidden1 = malloc(input_size * hidden1_size * sizeof(float));
    nn->weights_hidden1_hidden2 = malloc(hidden1_size * hidden2_size * sizeof(float));
    nn->weights_hidden2_hidden3 = malloc(hidden2_size * hidden3_size * sizeof(float));
    nn->weights_hidden3_output = malloc(hidden3_size * output_size * sizeof(float));

    nn->bias_hidden1 = malloc(hidden1_size * sizeof(float));
    nn->bias_hidden2 = malloc(hidden2_size * sizeof(float));
    nn->bias_hidden3 = malloc(hidden3_size * sizeof(float));
    nn->bias_output = malloc(output_size * sizeof(float));

    nn->hidden1_activations = malloc(hidden1_size * sizeof(float));
    nn->hidden2_activations = malloc(hidden2_size * sizeof(float));
    nn->hidden3_activations = malloc(hidden3_size * sizeof(float));
    nn->output_activations = malloc(output_size * sizeof(float));

    // He Initialization
    float he_input = sqrtf(2.0f / input_size);
    for (size_t i = 0; i < input_size * hidden1_size; i++) nn->weights_input_hidden1[i] = random_float() * he_input;

    float he_h1 = sqrtf(2.0f / hidden1_size);
    for (size_t i = 0; i < hidden1_size * hidden2_size; i++) nn->weights_hidden1_hidden2[i] = random_float() * he_h1;

    float he_h2 = sqrtf(2.0f / hidden2_size);
    for (size_t i = 0; i < hidden2_size * hidden3_size; i++) nn->weights_hidden2_hidden3[i] = random_float() * he_h2;

    float he_h3 = sqrtf(2.0f / hidden3_size);
    for (size_t i = 0; i < hidden3_size * output_size; i++) nn->weights_hidden3_output[i] = random_float() * he_h3;

    for (size_t i = 0; i < hidden1_size; i++) nn->bias_hidden1[i] = 0.0f;
    for (size_t i = 0; i < hidden2_size; i++) nn->bias_hidden2[i] = 0.0f;
    for (size_t i = 0; i < hidden3_size; i++) nn->bias_hidden3[i] = 0.0f;
    for (size_t i = 0; i < output_size; i++) nn->bias_output[i] = 0.0f;

    return nn;
}

void forward_propagation(NeuralNetwork* nn, const float* input, float* output) {
    if (nn->use_gpu) {
        metal_forward_layer(input, nn->weights_input_hidden1, nn->bias_hidden1, nn->hidden1_activations, nn->input_size, nn->hidden1_size, false);
        metal_forward_layer(nn->hidden1_activations, nn->weights_hidden1_hidden2, nn->bias_hidden2, nn->hidden2_activations, nn->hidden1_size, nn->hidden2_size, false);
        metal_forward_layer(nn->hidden2_activations, nn->weights_hidden2_hidden3, nn->bias_hidden3, nn->hidden3_activations, nn->hidden2_size, nn->hidden3_size, false);
        metal_forward_layer(nn->hidden3_activations, nn->weights_hidden3_output, nn->bias_output, nn->output_activations, nn->hidden3_size, nn->output_size, true);
        if (output) output[0] = nn->output_activations[0];
        return;
    }

    // 1. Input to Hidden1
    for (size_t h = 0; h < nn->hidden1_size; h++) {
        float sum = nn->bias_hidden1[h];
        for (size_t i = 0; i < nn->input_size; i++) sum += input[i] * nn->weights_input_hidden1[i * nn->hidden1_size + h];
        nn->hidden1_activations[h] = leaky_relu(sum);
    }

    // 2. Hidden1 to Hidden2
    for (size_t h2 = 0; h2 < nn->hidden2_size; h2++) {
        float sum = nn->bias_hidden2[h2];
        for (size_t i = 0; i < nn->hidden1_size; i++) sum += nn->hidden1_activations[i] * nn->weights_hidden1_hidden2[i * nn->hidden2_size + h2];
        nn->hidden2_activations[h2] = leaky_relu(sum);
    }
    
    // 3. Hidden2 to Hidden3
    for (size_t h3 = 0; h3 < nn->hidden3_size; h3++) {
        float sum = nn->bias_hidden3[h3];
        for (size_t i = 0; i < nn->hidden2_size; i++) sum += nn->hidden2_activations[i] * nn->weights_hidden2_hidden3[i * nn->hidden3_size + h3];
        nn->hidden3_activations[h3] = leaky_relu(sum);
    }

    // 4. Hidden3 to Output
    for (size_t o = 0; o < nn->output_size; o++) {
        float sum = nn->bias_output[o];
        for (size_t i = 0; i < nn->hidden3_size; i++) sum += nn->hidden3_activations[i] * nn->weights_hidden3_output[i * nn->output_size + o];
        nn->output_activations[o] = 1.0f / (1.0f + expf(-sum));
        if (output) output[o] = nn->output_activations[o];
    }
}

void backward_propagation(NeuralNetwork *nn, const float *input, const float *target, float learning_rate) {
    float *output_deltas = malloc(nn->output_size * sizeof(float));
    for (size_t i = 0; i < nn->output_size; i++) {
        output_deltas[i] = target[i] - nn->output_activations[i];
    }

    float *hidden3_deltas = malloc(nn->hidden3_size * sizeof(float));
    for (size_t i = 0; i < nn->hidden3_size; i++) {
        float error = 0.0f;
        for (size_t j = 0; j < nn->output_size; j++) error += output_deltas[j] * nn->weights_hidden3_output[i * nn->output_size + j];
        hidden3_deltas[i] = error * leaky_relu_derivative(nn->hidden3_activations[i]);
    }

    float *hidden2_deltas = malloc(nn->hidden2_size * sizeof(float));
    for (size_t i = 0; i < nn->hidden2_size; i++) {
        float error = 0.0f;
        for (size_t j = 0; j < nn->hidden3_size; j++) error += hidden3_deltas[j] * nn->weights_hidden2_hidden3[i * nn->hidden3_size + j];
        hidden2_deltas[i] = error * leaky_relu_derivative(nn->hidden2_activations[i]);
    }

    float *hidden1_deltas = malloc(nn->hidden1_size * sizeof(float));
    for (size_t i = 0; i < nn->hidden1_size; i++) {
        float error = 0.0f;
        for (size_t j = 0; j < nn->hidden2_size; j++) error += hidden2_deltas[j] * nn->weights_hidden1_hidden2[i * nn->hidden2_size + j];
        hidden1_deltas[i] = error * leaky_relu_derivative(nn->hidden1_activations[i]);
    }

    for (size_t i = 0; i < nn->hidden3_size; i++) {
        for (size_t j = 0; j < nn->output_size; j++) nn->weights_hidden3_output[i * nn->output_size + j] += learning_rate * output_deltas[j] * nn->hidden3_activations[i];
    }
    for (size_t j = 0; j < nn->output_size; j++) nn->bias_output[j] += learning_rate * output_deltas[j];

    for (size_t i = 0; i < nn->hidden2_size; i++) {
        for (size_t j = 0; j < nn->hidden3_size; j++) nn->weights_hidden2_hidden3[i * nn->hidden3_size + j] += learning_rate * hidden3_deltas[j] * nn->hidden2_activations[i];
    }
    for (size_t j = 0; j < nn->hidden3_size; j++) nn->bias_hidden3[j] += learning_rate * hidden3_deltas[j];

    for (size_t i = 0; i < nn->hidden1_size; i++) {
        for (size_t j = 0; j < nn->hidden2_size; j++) nn->weights_hidden1_hidden2[i * nn->hidden2_size + j] += learning_rate * hidden2_deltas[j] * nn->hidden1_activations[i];
    }
    for (size_t j = 0; j < nn->hidden2_size; j++) nn->bias_hidden2[j] += learning_rate * hidden2_deltas[j];

    for (size_t i = 0; i < nn->input_size; i++) {
        for (size_t j = 0; j < nn->hidden1_size; j++) nn->weights_input_hidden1[i * nn->hidden1_size + j] += learning_rate * hidden1_deltas[j] * input[i];
    }
    for (size_t j = 0; j < nn->hidden1_size; j++) nn->bias_hidden1[j] += learning_rate * hidden1_deltas[j];

    free(output_deltas);
    free(hidden3_deltas);
    free(hidden2_deltas);
    free(hidden1_deltas);
}

void free_neural_network(NeuralNetwork *nn) {
    if (!nn) return;
    free(nn->weights_input_hidden1);
    free(nn->weights_hidden1_hidden2);
    free(nn->weights_hidden2_hidden3);
    free(nn->weights_hidden3_output);
    free(nn->bias_hidden1);
    free(nn->bias_hidden2);
    free(nn->bias_hidden3);
    free(nn->bias_output);
    free(nn->hidden1_activations);
    free(nn->hidden2_activations);
    free(nn->hidden3_activations);
    free(nn->output_activations);
    free(nn);
}
"""

with open('NeuralNetwork.c', 'w') as f:
    f.write(new_c)

