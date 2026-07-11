#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "raylib.h"
#include "NeuralNetwork.h"
#include <stdbool.h>

#define MAX_HISTORY 200

typedef struct {
    bool is_training;
    int epoch;
    float learning_rate;
    int test_index;
    
    float loss_history[MAX_HISTORY];
    int history_count;
    int history_index;
    
    // Dataset pointers
    float (*training_inputs)[5];
    float (*training_outputs)[1];
    char (*name_labels)[32]; // Store string labels from CSV
    int num_samples;
    
    // Interaction states
    bool is_trained;
    int retraining_frames;
    bool custom_mode;
    float custom_inputs[5];
    char typed_name[32];
    int typed_len;
    float current_loss;
    float initial_loss;
    bool using_gpu;
    char compute_device_name[64];
    bool viewing_discriminator;
    
    int screenWidth;
    int screenHeight;
    
    Font mainFont;
} AppState;

Color GetWeightColor(float weight);
void DrawLossGraph(Font font, AppState* state, int uiX);
void DrawVisualization(NeuralNetwork* generator, NeuralNetwork* discriminator, AppState* state, const float* current_input_g, const float* current_output_g, const float* current_input_d, const float* current_output_d);

#endif // VISUALIZER_H
