#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "raylib.h"
#include "NeuralNetwork.h"
#include "visualizer.h"
#include "metal_backend.h"

int main() {
    AppState state = {0};
    state.screenWidth = 1200;
    state.screenHeight = 800;
    state.learning_rate = 0.05f; // Slightly lower for DNN
    
    // Enable Anti-Aliasing for smooth lines (fixes "jaggy" graphics)
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_HIGHDPI);
    InitWindow(state.screenWidth, state.screenHeight, "Deep Neural Network Debugger");
    SetTargetFPS(60);

    // Load Monaco (a clean monospace terminal font) at high resolution
    state.mainFont = LoadFontEx("/System/Library/Fonts/Monaco.ttf", 48, 0, 0);
    SetTextureFilter(state.mainFont.texture, TEXTURE_FILTER_BILINEAR);

    // Initialize the Apple Metal GPU Backend
    bool metal_available = init_metal_engine();
    if (metal_available) {
        snprintf(state.compute_device_name, 64, "%s", get_metal_device_name());
        state.using_gpu = true; // Auto-detect and switch to GPU
    } else {
        snprintf(state.compute_device_name, 64, "CPU (Sequential)");
        state.using_gpu = false;
    }

    // Initialize Deep Neural Network: 5 inputs, 16(L1), 12(L2), 8(L3), 4(L4), 1 output
    NeuralNetwork* nn = create_neural_network(8000, 5, 16, 12, 8, 4, 1);
    nn->use_gpu = state.using_gpu;
    if (load_weights(nn, "brain.bin")) {
        // If brain.bin exists, it means the network is already trained!
        // We skip the initial 3000-epoch training on dataset.csv.
        state.epoch = 3000;
        state.is_trained = true;
    }
    
    // Dynamically load NLTK-style dataset from CSV
    float training_inputs[8000][5];
    float training_outputs[8000][1];
    char name_labels[8000][32];
    int loaded_samples = 0;
    
    FILE *file = fopen("dataset.csv", "r");
    if (file) {
        char line[256];
        fgets(line, sizeof(line), file); // Skip header
        while (fgets(line, sizeof(line), file) && loaded_samples < 8000) {
            char name[32];
            float f1, f2, f3, f4, f5, label;
            if (sscanf(line, "%31[^,],%f,%f,%f,%f,%f,%f", name, &f1, &f2, &f3, &f4, &f5, &label) == 7) {
                snprintf(name_labels[loaded_samples], 32, "%s", name);
                training_inputs[loaded_samples][0] = f1;
                training_inputs[loaded_samples][1] = f2;
                training_inputs[loaded_samples][2] = f3;
                training_inputs[loaded_samples][3] = f4;
                training_inputs[loaded_samples][4] = f5;
                training_outputs[loaded_samples][0] = label;
                loaded_samples++;
            }
        }
        fclose(file);
    }
    
    state.training_inputs = training_inputs;
    state.training_outputs = training_outputs;
    state.name_labels = name_labels;
    state.num_samples = loaded_samples;

    // Execution Loop
    while (!WindowShouldClose()) {
        
        // --- 1. LOGIC & CONTROLS ---
        if (IsKeyPressed(KEY_SPACE) && !state.is_trained) state.is_training = !state.is_training;
        if (IsKeyPressed(KEY_RIGHT)) { state.test_index = (state.test_index + 1) % state.num_samples; state.custom_mode = false; }
        if (IsKeyPressed(KEY_LEFT)) { state.test_index = (state.test_index - 1 + state.num_samples) % state.num_samples; state.custom_mode = false; }
        if (IsKeyPressed(KEY_R)) {
            free_neural_network(nn);
            nn = create_neural_network(8000, 5, 16, 12, 8, 4, 1);
            nn->use_gpu = state.using_gpu;
            state.epoch = 0;
            state.history_count = 0;
            state.history_index = 0;
            state.is_training = false;
            state.is_trained = false;
            state.custom_mode = false;
            state.initial_loss = 0.0f;
            state.current_loss = 0.0f;
        }

        // True Simulation: Dynamic Feature Extraction Sandbox
        int key = GetCharPressed();
        while (key > 0) {
            if (isalpha(key) && state.typed_len < 31) {
                state.typed_name[state.typed_len++] = (char)toupper(key);
                state.typed_name[state.typed_len] = '\0';
                state.custom_mode = true;
            }
            key = GetCharPressed();
        }
        if (IsKeyPressed(KEY_BACKSPACE) && state.typed_len > 0) {
            state.typed_len--;
            state.typed_name[state.typed_len] = '\0';
            if (state.typed_len == 0) state.custom_mode = false;
        }

        // Compute features on-the-fly for typed name
        if (state.custom_mode && state.typed_len > 0) {
            if (IsKeyPressed(KEY_ONE) || IsKeyPressed(KEY_ZERO)) {
                float target[1] = { IsKeyPressed(KEY_ONE) ? 1.0f : 0.0f };
                
                // Temporarily disable GPU to run thousands of sequential CPU loops in nanoseconds
                bool was_gpu = nn->use_gpu;
                nn->use_gpu = false;
                
                int max_retries = 10000;
                int retries = 0;
                
                while (retries < max_retries) {
                    float out[1];
                    forward_propagation(nn, state.custom_inputs, out);
                    
                    // Reverification: Stop if we are highly confident in the correct direction
                    bool is_confident = (target[0] > 0.5f) ? (out[0] >= 0.6f) : (out[0] <= 0.4f);
                    
                    if (is_confident && retries > 10) { 
                        // Require at least 10 epochs to ensure it solidly learned it
                        break;
                    }
                    
                    backward_propagation(nn, state.custom_inputs, target, state.learning_rate);
                    retries++;
                }
                
                nn->use_gpu = was_gpu; // Restore GPU
                
                save_weights(nn, "brain.bin");
                state.retraining_frames = 60; // Show LIVE RETRAINING for 60 frames (1 second)
            }

            char last = tolower(state.typed_name[state.typed_len - 1]);
            char first = tolower(state.typed_name[0]);
            
            bool ends_vowel = (last=='a'||last=='e'||last=='i'||last=='o'||last=='u'||last=='y');
            bool ends_a = (last=='a');
            bool len_gt5 = (state.typed_len > 5);
            bool starts_vowel = (first=='a'||first=='e'||first=='i'||first=='o'||first=='u'||first=='y');
            bool has_y = false;
            for (int i = 0; i < state.typed_len; i++) {
                if (tolower(state.typed_name[i]) == 'y') has_y = true;
            }
            
            state.custom_inputs[0] = ends_vowel ? 1.0f : 0.0f;
            state.custom_inputs[1] = ends_a ? 1.0f : 0.0f;
            state.custom_inputs[2] = len_gt5 ? 1.0f : 0.0f;
            state.custom_inputs[3] = starts_vowel ? 1.0f : 0.0f;
            state.custom_inputs[4] = has_y ? 1.0f : 0.0f;
        }

        if (!state.custom_mode) {
            for(int i=0; i<5; i++) state.custom_inputs[i] = state.training_inputs[state.test_index][i];
        }

        // --- 2. TRAINING ---
        if (state.is_training && !state.is_trained) {
            nn->use_gpu = state.using_gpu; // Phase 3 Complete: Metal GEMM handles Batched Forward Pass!
            for (int e = 0; e < 20; e++) {
                float total_loss = 0.0f;
                
                float *flat_inputs = (float *)state.training_inputs;
                float *flat_targets = (float *)state.training_outputs;
                float batch_outputs[8000]; // 32KB on stack
                
                // 1. Batched Forward Pass
                forward_propagation_batch(nn, flat_inputs, batch_outputs, state.num_samples);
                
                // 2. Compute Loss
                for (int i = 0; i < state.num_samples; i++) {
                    float p = batch_outputs[i];
                    if (p < 0.0001f) p = 0.0001f;
                    if (p > 0.9999f) p = 0.9999f;
                    total_loss += -(flat_targets[i] * logf(p) + (1.0f - flat_targets[i]) * logf(1.0f - p));
                }
                
                // 3. Batched Backward Pass (Gradient Accumulation + Average)
                backward_propagation_batch(nn, flat_inputs, flat_targets, state.num_samples, state.learning_rate);
                
                state.epoch++;
                
                // Learning Rate Decay
                if (state.epoch % 100 == 0) {
                    state.learning_rate *= 0.99f;
                }
                
                if (state.epoch % 20 == 0) {
                    float mse = total_loss / (float)state.num_samples;
                    state.loss_history[state.history_index] = mse;
                    state.history_index = (state.history_index + 1) % MAX_HISTORY;
                    if (state.history_count < MAX_HISTORY) state.history_count++;
                    
                    state.current_loss = mse;
                    if (state.initial_loss == 0.0f || state.epoch <= 20) state.initial_loss = mse;
                    
                    // Auto-Completion: Stop training automatically after 3000 Epochs
                    // For BCE on noisy NLP data, loss plateaus instead of hitting 0.
                    // Training for a fixed duration ensures it actually learns the data thoroughly.
                    if (state.epoch >= 3000) {
                        state.is_trained = true;
                        state.is_training = false;
                        save_weights(nn, "brain.bin");
                        break;
                    }
                }
            }
            nn->use_gpu = state.using_gpu; // Restore GPU for live sandbox
        }
        
        // --- 3. EVALUATION PASS ---
        float current_input[5];
        for(int i=0; i<5; i++) current_input[i] = state.custom_inputs[i];
        
        float current_output[1];
        forward_propagation(nn, current_input, current_output);

        // --- 4. RENDERING ---
        if (state.retraining_frames > 0) state.retraining_frames--;
        DrawVisualization(nn, &state, current_input, current_output);
    }

    // Cleanup
    UnloadFont(state.mainFont);
    free_neural_network(nn);
    CloseWindow();
    return 0;
}
