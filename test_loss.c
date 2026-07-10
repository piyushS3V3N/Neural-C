#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "NeuralNetwork.h"

void metal_forward_layer(const float* i, const float* w, const float* b, float* o, int is, int os, bool u) {}

int main() {
    float training_inputs[26][5];
    float training_outputs[26][1];
    for (int i=0; i<26; i++) {
        for (int j=0; j<5; j++) training_inputs[i][j] = (float)(rand()%2);
        training_outputs[i][0] = (float)(rand()%2);
    }
    
    NeuralNetwork* nn = create_neural_network(5, 10, 6, 1);
    nn->use_gpu = false;

    for (int epoch = 0; epoch < 1000; epoch++) {
        float total_loss = 0.0f;
        for (int i = 0; i < 26; i++) {
            float output[1];
            forward_propagation(nn, training_inputs[i], output);
            float p = output[0];
            if (p < 0.0001f) p = 0.0001f;
            if (p > 0.9999f) p = 0.9999f;
            float bce_error = -(training_outputs[i][0] * logf(p) + (1.0f - training_outputs[i][0]) * logf(1.0f - p));
            total_loss += bce_error;
            backward_propagation(nn, training_inputs[i], training_outputs[i], 0.05f);
        }
        if (epoch % 100 == 0) printf("Epoch %d: Loss %f\n", epoch, total_loss / 26.0f);
    }
    return 0;
}
