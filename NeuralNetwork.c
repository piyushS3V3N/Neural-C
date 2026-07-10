#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "NeuralNetwork.h"
#include "metal_backend.h"

static float leaky_relu(float x) { return x > 0.0f ? x : 0.01f * x; }
static float leaky_relu_derivative(float x) { return x > 0.0f ? 1.0f : 0.01f; }
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float random_float() { return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; }

NeuralNetwork *create_neural_network(size_t max_batch_size, size_t input_size, size_t hidden1_size, size_t hidden2_size, size_t hidden3_size, size_t hidden4_size, size_t output_size) {
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    nn->input_size = input_size;
    nn->hidden1_size = hidden1_size;
    nn->hidden2_size = hidden2_size;
    nn->hidden3_size = hidden3_size;
    nn->hidden4_size = hidden4_size;
    nn->output_size = output_size;
    nn->use_gpu = false;
    nn->max_batch_size = max_batch_size;

    nn->weights_input_hidden1 = malloc(input_size * hidden1_size * sizeof(float));
    nn->weights_hidden1_hidden2 = malloc(hidden1_size * hidden2_size * sizeof(float));
    nn->weights_hidden2_hidden3 = malloc(hidden2_size * hidden3_size * sizeof(float));
    nn->weights_hidden3_hidden4 = malloc(hidden3_size * hidden4_size * sizeof(float));
    nn->weights_hidden4_output = malloc(hidden4_size * output_size * sizeof(float));

    nn->bias_hidden1 = malloc(hidden1_size * sizeof(float));
    nn->bias_hidden2 = malloc(hidden2_size * sizeof(float));
    nn->bias_hidden3 = malloc(hidden3_size * sizeof(float));
    nn->bias_hidden4 = malloc(hidden4_size * sizeof(float));
    nn->bias_output = malloc(output_size * sizeof(float));

    nn->hidden1_activations = malloc(max_batch_size * hidden1_size * sizeof(float));
    nn->hidden2_activations = malloc(max_batch_size * hidden2_size * sizeof(float));
    nn->hidden3_activations = malloc(max_batch_size * hidden3_size * sizeof(float));
    nn->hidden4_activations = malloc(max_batch_size * hidden4_size * sizeof(float));
    nn->output_activations = malloc(max_batch_size * output_size * sizeof(float));

    float he_input = sqrtf(2.0f / input_size);
    for (size_t i = 0; i < input_size * hidden1_size; i++) nn->weights_input_hidden1[i] = random_float() * he_input;

    float he_h1 = sqrtf(2.0f / hidden1_size);
    for (size_t i = 0; i < hidden1_size * hidden2_size; i++) nn->weights_hidden1_hidden2[i] = random_float() * he_h1;

    float he_h2 = sqrtf(2.0f / hidden2_size);
    for (size_t i = 0; i < hidden2_size * hidden3_size; i++) nn->weights_hidden2_hidden3[i] = random_float() * he_h2;

    float he_h3 = sqrtf(2.0f / hidden3_size);
    for (size_t i = 0; i < hidden3_size * hidden4_size; i++) nn->weights_hidden3_hidden4[i] = random_float() * he_h3;

    float he_h4 = sqrtf(2.0f / hidden4_size);
    for (size_t i = 0; i < hidden4_size * output_size; i++) nn->weights_hidden4_output[i] = random_float() * he_h4;

    for (size_t i = 0; i < hidden1_size; i++) nn->bias_hidden1[i] = 0.0f;
    for (size_t i = 0; i < hidden2_size; i++) nn->bias_hidden2[i] = 0.0f;
    for (size_t i = 0; i < hidden3_size; i++) nn->bias_hidden3[i] = 0.0f;
    for (size_t i = 0; i < hidden4_size; i++) nn->bias_hidden4[i] = 0.0f;
    for (size_t i = 0; i < output_size; i++) nn->bias_output[i] = 0.0f;

    return nn;
}

void forward_propagation(NeuralNetwork* nn, const float* input, float* output) {
    if (nn->use_gpu) {
        metal_forward_layer(input, nn->weights_input_hidden1, nn->bias_hidden1, nn->hidden1_activations, nn->input_size, nn->hidden1_size, false, 1);
        metal_forward_layer(nn->hidden1_activations, nn->weights_hidden1_hidden2, nn->bias_hidden2, nn->hidden2_activations, nn->hidden1_size, nn->hidden2_size, false, 1);
        metal_forward_layer(nn->hidden2_activations, nn->weights_hidden2_hidden3, nn->bias_hidden3, nn->hidden3_activations, nn->hidden2_size, nn->hidden3_size, false, 1);
        metal_forward_layer(nn->hidden3_activations, nn->weights_hidden3_hidden4, nn->bias_hidden4, nn->hidden4_activations, nn->hidden3_size, nn->hidden4_size, false, 1);
        metal_forward_layer(nn->hidden4_activations, nn->weights_hidden4_output, nn->bias_output, nn->output_activations, nn->hidden4_size, nn->output_size, true, 1);
        if (output) {
            for(int i=0; i<nn->output_size; i++) output[i] = nn->output_activations[i];
        }
        return;
    }

    for (size_t h = 0; h < nn->hidden1_size; h++) {
        float sum = nn->bias_hidden1[h];
        for (size_t i = 0; i < nn->input_size; i++) sum += input[i] * nn->weights_input_hidden1[i * nn->hidden1_size + h];
        nn->hidden1_activations[h] = leaky_relu(sum);
    }
    for (size_t h2 = 0; h2 < nn->hidden2_size; h2++) {
        float sum = nn->bias_hidden2[h2];
        for (size_t i = 0; i < nn->hidden1_size; i++) sum += nn->hidden1_activations[i] * nn->weights_hidden1_hidden2[i * nn->hidden2_size + h2];
        nn->hidden2_activations[h2] = leaky_relu(sum);
    }
    for (size_t h3 = 0; h3 < nn->hidden3_size; h3++) {
        float sum = nn->bias_hidden3[h3];
        for (size_t i = 0; i < nn->hidden2_size; i++) sum += nn->hidden2_activations[i] * nn->weights_hidden2_hidden3[i * nn->hidden3_size + h3];
        nn->hidden3_activations[h3] = leaky_relu(sum);
    }
    for (size_t h4 = 0; h4 < nn->hidden4_size; h4++) {
        float sum = nn->bias_hidden4[h4];
        for (size_t i = 0; i < nn->hidden3_size; i++) sum += nn->hidden3_activations[i] * nn->weights_hidden3_hidden4[i * nn->hidden4_size + h4];
        nn->hidden4_activations[h4] = leaky_relu(sum);
    }
    for (size_t o = 0; o < nn->output_size; o++) {
        float sum = nn->bias_output[o];
        for (size_t i = 0; i < nn->hidden4_size; i++) sum += nn->hidden4_activations[i] * nn->weights_hidden4_output[i * nn->output_size + o];
        nn->output_activations[o] = 1.0f / (1.0f + expf(-sum));
        if (output) output[o] = nn->output_activations[o];
    }
}

void backward_propagation(NeuralNetwork *nn, const float *input, const float *target, float learning_rate) {
    float *output_deltas = malloc(nn->output_size * sizeof(float));
    for (size_t i = 0; i < nn->output_size; i++) output_deltas[i] = target[i] - nn->output_activations[i];

    float *hidden4_deltas = malloc(nn->hidden4_size * sizeof(float));
    for (size_t i = 0; i < nn->hidden4_size; i++) {
        float error = 0.0f;
        for (size_t j = 0; j < nn->output_size; j++) error += output_deltas[j] * nn->weights_hidden4_output[i * nn->output_size + j];
        hidden4_deltas[i] = error * leaky_relu_derivative(nn->hidden4_activations[i]);
    }

    float *hidden3_deltas = malloc(nn->hidden3_size * sizeof(float));
    for (size_t i = 0; i < nn->hidden3_size; i++) {
        float error = 0.0f;
        for (size_t j = 0; j < nn->hidden4_size; j++) error += hidden4_deltas[j] * nn->weights_hidden3_hidden4[i * nn->hidden4_size + j];
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

    for (size_t i = 0; i < nn->hidden4_size; i++) {
        for (size_t j = 0; j < nn->output_size; j++) nn->weights_hidden4_output[i * nn->output_size + j] += learning_rate * output_deltas[j] * nn->hidden4_activations[i];
    }
    for (size_t j = 0; j < nn->output_size; j++) nn->bias_output[j] += learning_rate * output_deltas[j];

    for (size_t i = 0; i < nn->hidden3_size; i++) {
        for (size_t j = 0; j < nn->hidden4_size; j++) nn->weights_hidden3_hidden4[i * nn->hidden4_size + j] += learning_rate * hidden4_deltas[j] * nn->hidden3_activations[i];
    }
    for (size_t j = 0; j < nn->hidden4_size; j++) nn->bias_hidden4[j] += learning_rate * hidden4_deltas[j];

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
    free(hidden4_deltas);
    free(hidden3_deltas);
    free(hidden2_deltas);
    free(hidden1_deltas);
}

void free_neural_network(NeuralNetwork *nn) {
    if (!nn) return;
    free(nn->weights_input_hidden1);
    free(nn->weights_hidden1_hidden2);
    free(nn->weights_hidden2_hidden3);
    free(nn->weights_hidden3_hidden4);
    free(nn->weights_hidden4_output);
    free(nn->bias_hidden1);
    free(nn->bias_hidden2);
    free(nn->bias_hidden3);
    free(nn->bias_hidden4);
    free(nn->bias_output);
    free(nn->hidden1_activations);
    free(nn->hidden2_activations);
    free(nn->hidden3_activations);
    free(nn->hidden4_activations);
    free(nn->output_activations);
    free(nn);
}

bool save_weights(NeuralNetwork *nn, const char *filename) {
    if (!nn) return false;
    FILE *f = fopen(filename, "wb");
    if (!f) return false;

    fwrite(nn->weights_input_hidden1, sizeof(float), nn->input_size * nn->hidden1_size, f);
    fwrite(nn->weights_hidden1_hidden2, sizeof(float), nn->hidden1_size * nn->hidden2_size, f);
    fwrite(nn->weights_hidden2_hidden3, sizeof(float), nn->hidden2_size * nn->hidden3_size, f);
    fwrite(nn->weights_hidden3_hidden4, sizeof(float), nn->hidden3_size * nn->hidden4_size, f);
    fwrite(nn->weights_hidden4_output, sizeof(float), nn->hidden4_size * nn->output_size, f);

    fwrite(nn->bias_hidden1, sizeof(float), nn->hidden1_size, f);
    fwrite(nn->bias_hidden2, sizeof(float), nn->hidden2_size, f);
    fwrite(nn->bias_hidden3, sizeof(float), nn->hidden3_size, f);
    fwrite(nn->bias_hidden4, sizeof(float), nn->hidden4_size, f);
    fwrite(nn->bias_output, sizeof(float), nn->output_size, f);

    fclose(f);
    return true;
}

bool load_weights(NeuralNetwork *nn, const char *filename) {
    if (!nn) return false;
    FILE *f = fopen(filename, "rb");
    if (!f) return false;

    fread(nn->weights_input_hidden1, sizeof(float), nn->input_size * nn->hidden1_size, f);
    fread(nn->weights_hidden1_hidden2, sizeof(float), nn->hidden1_size * nn->hidden2_size, f);
    fread(nn->weights_hidden2_hidden3, sizeof(float), nn->hidden2_size * nn->hidden3_size, f);
    fread(nn->weights_hidden3_hidden4, sizeof(float), nn->hidden3_size * nn->hidden4_size, f);
    fread(nn->weights_hidden4_output, sizeof(float), nn->hidden4_size * nn->output_size, f);

    fread(nn->bias_hidden1, sizeof(float), nn->hidden1_size, f);
    fread(nn->bias_hidden2, sizeof(float), nn->hidden2_size, f);
    fread(nn->bias_hidden3, sizeof(float), nn->hidden3_size, f);
    fread(nn->bias_hidden4, sizeof(float), nn->hidden4_size, f);
    fread(nn->bias_output, sizeof(float), nn->output_size, f);

    fclose(f);
    return true;
}


void forward_propagation_batch(NeuralNetwork* nn, const float* inputs, float* outputs, int batch_size) {
    if (batch_size > nn->max_batch_size) batch_size = nn->max_batch_size;

    if (nn->use_gpu) {
        metal_forward_layer(inputs, nn->weights_input_hidden1, nn->bias_hidden1, nn->hidden1_activations, nn->input_size, nn->hidden1_size, false, batch_size);
        metal_forward_layer(nn->hidden1_activations, nn->weights_hidden1_hidden2, nn->bias_hidden2, nn->hidden2_activations, nn->hidden1_size, nn->hidden2_size, false, batch_size);
        metal_forward_layer(nn->hidden2_activations, nn->weights_hidden2_hidden3, nn->bias_hidden3, nn->hidden3_activations, nn->hidden2_size, nn->hidden3_size, false, batch_size);
        metal_forward_layer(nn->hidden3_activations, nn->weights_hidden3_hidden4, nn->bias_hidden4, nn->hidden4_activations, nn->hidden3_size, nn->hidden4_size, false, batch_size);
        metal_forward_layer(nn->hidden4_activations, nn->weights_hidden4_output, nn->bias_output, nn->output_activations, nn->hidden4_size, nn->output_size, true, batch_size);
        
        if (outputs) {
            for(int b=0; b<batch_size; b++) {
                for(int i=0; i<nn->output_size; i++) {
                    outputs[b * nn->output_size + i] = nn->output_activations[b * nn->output_size + i];
                }
            }
        }
        return;
    }

    // CPU Batched Forward Pass
    for (int b = 0; b < batch_size; b++) {
        // L1
        for (size_t h = 0; h < nn->hidden1_size; h++) {
            float sum = nn->bias_hidden1[h];
            for (size_t i = 0; i < nn->input_size; i++) sum += inputs[b * nn->input_size + i] * nn->weights_input_hidden1[i * nn->hidden1_size + h];
            nn->hidden1_activations[b * nn->hidden1_size + h] = leaky_relu(sum);
        }
        // L2
        for (size_t h = 0; h < nn->hidden2_size; h++) {
            float sum = nn->bias_hidden2[h];
            for (size_t i = 0; i < nn->hidden1_size; i++) sum += nn->hidden1_activations[b * nn->hidden1_size + i] * nn->weights_hidden1_hidden2[i * nn->hidden2_size + h];
            nn->hidden2_activations[b * nn->hidden2_size + h] = leaky_relu(sum);
        }
        // L3
        for (size_t h = 0; h < nn->hidden3_size; h++) {
            float sum = nn->bias_hidden3[h];
            for (size_t i = 0; i < nn->hidden2_size; i++) sum += nn->hidden2_activations[b * nn->hidden2_size + i] * nn->weights_hidden2_hidden3[i * nn->hidden3_size + h];
            nn->hidden3_activations[b * nn->hidden3_size + h] = leaky_relu(sum);
        }
        // L4
        for (size_t h = 0; h < nn->hidden4_size; h++) {
            float sum = nn->bias_hidden4[h];
            for (size_t i = 0; i < nn->hidden3_size; i++) sum += nn->hidden3_activations[b * nn->hidden3_size + i] * nn->weights_hidden3_hidden4[i * nn->hidden4_size + h];
            nn->hidden4_activations[b * nn->hidden4_size + h] = leaky_relu(sum);
        }
        // Output
        for (size_t o = 0; o < nn->output_size; o++) {
            float sum = nn->bias_output[o];
            for (size_t i = 0; i < nn->hidden4_size; i++) sum += nn->hidden4_activations[b * nn->hidden4_size + i] * nn->weights_hidden4_output[i * nn->output_size + o];
            nn->output_activations[b * nn->output_size + o] = sigmoid(sum);
            if (outputs) outputs[b * nn->output_size + o] = nn->output_activations[b * nn->output_size + o];
        }
    }
}


void backward_propagation_batch(NeuralNetwork *nn, const float *inputs, const float *targets, int batch_size, float learning_rate) {
    if (batch_size > nn->max_batch_size) batch_size = nn->max_batch_size;
    
    // Allocate Gradients
    float *grad_w_out = calloc(nn->hidden4_size * nn->output_size, sizeof(float));
    float *grad_b_out = calloc(nn->output_size, sizeof(float));
    float *grad_w_h4 = calloc(nn->hidden3_size * nn->hidden4_size, sizeof(float));
    float *grad_b_h4 = calloc(nn->hidden4_size, sizeof(float));
    float *grad_w_h3 = calloc(nn->hidden2_size * nn->hidden3_size, sizeof(float));
    float *grad_b_h3 = calloc(nn->hidden3_size, sizeof(float));
    float *grad_w_h2 = calloc(nn->hidden1_size * nn->hidden2_size, sizeof(float));
    float *grad_b_h2 = calloc(nn->hidden2_size, sizeof(float));
    float *grad_w_h1 = calloc(nn->input_size * nn->hidden1_size, sizeof(float));
    float *grad_b_h1 = calloc(nn->hidden1_size, sizeof(float));

    float *delta_out = malloc(nn->output_size * sizeof(float));
    float *delta_h4 = malloc(nn->hidden4_size * sizeof(float));
    float *delta_h3 = malloc(nn->hidden3_size * sizeof(float));
    float *delta_h2 = malloc(nn->hidden2_size * sizeof(float));
    float *delta_h1 = malloc(nn->hidden1_size * sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        // Output Deltas (BCE + Sigmoid derivative cancels out to: out - target)
        for (size_t i = 0; i < nn->output_size; i++) {
            delta_out[i] = nn->output_activations[b * nn->output_size + i] - targets[b * nn->output_size + i];
            grad_b_out[i] += delta_out[i];
            for (size_t j = 0; j < nn->hidden4_size; j++) {
                grad_w_out[j * nn->output_size + i] += delta_out[i] * nn->hidden4_activations[b * nn->hidden4_size + j];
            }
        }

        // Hidden 4 Deltas
        for (size_t i = 0; i < nn->hidden4_size; i++) {
            float err = 0.0f;
            for (size_t j = 0; j < nn->output_size; j++) err += delta_out[j] * nn->weights_hidden4_output[i * nn->output_size + j];
            delta_h4[i] = err * leaky_relu_derivative(nn->hidden4_activations[b * nn->hidden4_size + i]);
            grad_b_h4[i] += delta_h4[i];
            for (size_t j = 0; j < nn->hidden3_size; j++) {
                grad_w_h4[j * nn->hidden4_size + i] += delta_h4[i] * nn->hidden3_activations[b * nn->hidden3_size + j];
            }
        }

        // Hidden 3 Deltas
        for (size_t i = 0; i < nn->hidden3_size; i++) {
            float err = 0.0f;
            for (size_t j = 0; j < nn->hidden4_size; j++) err += delta_h4[j] * nn->weights_hidden3_hidden4[i * nn->hidden4_size + j];
            delta_h3[i] = err * leaky_relu_derivative(nn->hidden3_activations[b * nn->hidden3_size + i]);
            grad_b_h3[i] += delta_h3[i];
            for (size_t j = 0; j < nn->hidden2_size; j++) {
                grad_w_h3[j * nn->hidden3_size + i] += delta_h3[i] * nn->hidden2_activations[b * nn->hidden2_size + j];
            }
        }

        // Hidden 2 Deltas
        for (size_t i = 0; i < nn->hidden2_size; i++) {
            float err = 0.0f;
            for (size_t j = 0; j < nn->hidden3_size; j++) err += delta_h3[j] * nn->weights_hidden2_hidden3[i * nn->hidden3_size + j];
            delta_h2[i] = err * leaky_relu_derivative(nn->hidden2_activations[b * nn->hidden2_size + i]);
            grad_b_h2[i] += delta_h2[i];
            for (size_t j = 0; j < nn->hidden1_size; j++) {
                grad_w_h2[j * nn->hidden2_size + i] += delta_h2[i] * nn->hidden1_activations[b * nn->hidden1_size + j];
            }
        }

        // Hidden 1 Deltas
        for (size_t i = 0; i < nn->hidden1_size; i++) {
            float err = 0.0f;
            for (size_t j = 0; j < nn->hidden2_size; j++) err += delta_h2[j] * nn->weights_hidden1_hidden2[i * nn->hidden2_size + j];
            delta_h1[i] = err * leaky_relu_derivative(nn->hidden1_activations[b * nn->hidden1_size + i]);
            grad_b_h1[i] += delta_h1[i];
            for (size_t j = 0; j < nn->input_size; j++) {
                grad_w_h1[j * nn->hidden1_size + i] += delta_h1[i] * inputs[b * nn->input_size + j];
            }
        }
    }

    // Apply Gradients (Average over batch to guarantee stable convergence)
    float lr = (learning_rate / (float)batch_size);
    
    for (size_t i = 0; i < nn->input_size * nn->hidden1_size; i++) nn->weights_input_hidden1[i] -= lr * grad_w_h1[i];
    for (size_t i = 0; i < nn->hidden1_size; i++) nn->bias_hidden1[i] -= lr * grad_b_h1[i];
    
    for (size_t i = 0; i < nn->hidden1_size * nn->hidden2_size; i++) nn->weights_hidden1_hidden2[i] -= lr * grad_w_h2[i];
    for (size_t i = 0; i < nn->hidden2_size; i++) nn->bias_hidden2[i] -= lr * grad_b_h2[i];
    
    for (size_t i = 0; i < nn->hidden2_size * nn->hidden3_size; i++) nn->weights_hidden2_hidden3[i] -= lr * grad_w_h3[i];
    for (size_t i = 0; i < nn->hidden3_size; i++) nn->bias_hidden3[i] -= lr * grad_b_h3[i];
    
    for (size_t i = 0; i < nn->hidden3_size * nn->hidden4_size; i++) nn->weights_hidden3_hidden4[i] -= lr * grad_w_h4[i];
    for (size_t i = 0; i < nn->hidden4_size; i++) nn->bias_hidden4[i] -= lr * grad_b_h4[i];
    
    for (size_t i = 0; i < nn->hidden4_size * nn->output_size; i++) nn->weights_hidden4_output[i] -= lr * grad_w_out[i];
    for (size_t i = 0; i < nn->output_size; i++) nn->bias_output[i] -= lr * grad_b_out[i];

    free(grad_w_out); free(grad_b_out);
    free(grad_w_h4); free(grad_b_h4);
    free(grad_w_h3); free(grad_b_h3);
    free(grad_w_h2); free(grad_b_h2);
    free(grad_w_h1); free(grad_b_h1);
    free(delta_out); free(delta_h4); free(delta_h3); free(delta_h2); free(delta_h1);
}
