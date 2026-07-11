#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include "raylib.h"
#include "NeuralNetwork.h"
#include "visualizer.h"
#include "metal_backend.h"

// Helper to generate random noise
static float random_float_gen() { return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; }

static bool is_vowel(char c) {
    c = tolower(c);
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

int main() {
    srand(time(NULL));
    AppState state = {0};
    state.screenWidth = 1600;
    state.screenHeight = 850;
    state.learning_rate = 0.01f;
    state.viewing_discriminator = true;
    
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_HIGHDPI);
    InitWindow(state.screenWidth, state.screenHeight, "GAN Debugger");
    SetTargetFPS(60);

    state.mainFont = LoadFontEx("/System/Library/Fonts/Monaco.ttf", 48, 0, 0);
    SetTextureFilter(state.mainFont.texture, TEXTURE_FILTER_BILINEAR);

    bool metal_available = init_metal_engine();
    if (metal_available) {
        snprintf(state.compute_device_name, 64, "%s", get_metal_device_name());
        state.using_gpu = true;
    } else {
        snprintf(state.compute_device_name, 64, "CPU (Sequential)");
        state.using_gpu = false;
    }

    // Discriminator: 5 Features -> 2 Outputs (Real/Fake, Male/Female)
    NeuralNetwork* discriminator = create_neural_network(8000, 5, 16, 12, 8, 4, 2);
    discriminator->use_gpu = state.using_gpu;
    
    // Generator: 4 Inputs (3 noise, 1 target gender) -> 5 Outputs (Features)
    NeuralNetwork* generator = create_neural_network(8000, 4, 16, 12, 8, 4, 5);
    generator->use_gpu = state.using_gpu;

    // Load pre-trained models if they exist
    if (load_weights(generator, "generator.bin") && load_weights(discriminator, "discriminator.bin")) {
        state.is_trained = true;
    }
    
    float training_inputs[8000][5];
    float training_outputs[8000][1];
    char name_labels[8000][32];
    int loaded_samples = 0;
    
    FILE *file = fopen("dataset.csv", "r");
    if (file) {
        char line[256];
        fgets(line, sizeof(line), file);
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
                training_outputs[loaded_samples][0] = label; // 1 = Female, 0 = Male
                loaded_samples++;
            }
        }
        fclose(file);
    }
    
    state.training_inputs = training_inputs;
    state.training_outputs = training_outputs;
    state.name_labels = name_labels;
    state.num_samples = loaded_samples;

    float* d_real_out = malloc(loaded_samples * 2 * sizeof(float));
    float* d_fake_out = malloc(loaded_samples * 2 * sizeof(float));
    float* d_delta_out = malloc(loaded_samples * 2 * sizeof(float));
    float* g_out = malloc(loaded_samples * 5 * sizeof(float));
    float* g_noise = malloc(loaded_samples * 4 * sizeof(float));
    float* d_delta_in = malloc(loaded_samples * 5 * sizeof(float));

    while (!WindowShouldClose()) {
        
        if (IsKeyPressed(KEY_TAB)) state.viewing_discriminator = !state.viewing_discriminator;
        if (IsKeyPressed(KEY_SPACE) && !state.is_trained) state.is_training = !state.is_training;
        if (IsKeyPressed(KEY_RIGHT)) { state.test_index = (state.test_index + 1) % state.num_samples; state.custom_mode = false; }
        if (IsKeyPressed(KEY_LEFT)) { state.test_index = (state.test_index - 1 + state.num_samples) % state.num_samples; state.custom_mode = false; }
        
        // --- 1. LIVE SANDBOX TYPING ---
        int key = GetCharPressed();
        while (key > 0) {
            if ((key >= 32) && (key <= 125) && (state.typed_len < 31)) {
                state.typed_name[state.typed_len] = (char)key;
                state.typed_name[state.typed_len+1] = '\0';
                state.typed_len++;
                state.custom_mode = true;
            }
            key = GetCharPressed();
        }
        if (IsKeyPressed(KEY_BACKSPACE)) {
            if (state.typed_len > 0) {
                state.typed_len--;
                state.typed_name[state.typed_len] = '\0';
            }
            if (state.typed_len == 0) state.custom_mode = false;
        }

        if (state.custom_mode && state.typed_len > 0) {
            int len = state.typed_len;
            char last_c = tolower(state.typed_name[len-1]);
            char first_c = tolower(state.typed_name[0]);
            state.custom_inputs[0] = is_vowel(last_c) ? 1.0f : 0.0f;
            state.custom_inputs[1] = (last_c == 'a') ? 1.0f : 0.0f;
            state.custom_inputs[2] = (len > 5) ? 1.0f : 0.0f;
            state.custom_inputs[3] = is_vowel(first_c) ? 1.0f : 0.0f;
            bool has_y = false;
            for(int i=0; i<len; i++) if(tolower(state.typed_name[i]) == 'y') has_y = true;
            state.custom_inputs[4] = has_y ? 1.0f : 0.0f;
        }
        
        // Live Retraining
        if (state.custom_mode && (IsKeyPressed(KEY_ONE) || IsKeyPressed(KEY_ZERO))) {
            float target_gender = IsKeyPressed(KEY_ONE) ? 1.0f : 0.0f;
            float d_delta[2];
            float d_out[2];
            for (int iter = 0; iter < 150; iter++) {
                 forward_propagation_batch(discriminator, state.custom_inputs, d_out, 1);
                 d_delta[0] = d_out[0] - 1.0f; // Target: Real
                 d_delta[1] = d_out[1] - target_gender; // Target: Gender
                 backward_propagation_batch_with_deltas(discriminator, state.custom_inputs, d_delta, 1, state.learning_rate);
            }
            state.retraining_frames = 60;
        }
        if (state.retraining_frames > 0) state.retraining_frames--;

        if (!state.custom_mode) {
            for(int i=0; i<5; i++) state.custom_inputs[i] = state.training_inputs[state.test_index][i];
        }

        // --- 2. TRAINING (GAN) ---
        if (state.is_training && !state.is_trained) {
            discriminator->use_gpu = state.using_gpu;
            generator->use_gpu = state.using_gpu;
            
            for (int e = 0; e < 20; e++) {
                // 1. Generate Noise and Fake Data
                for(int i = 0; i < state.num_samples; i++) {
                    g_noise[i*4 + 0] = random_float_gen();
                    g_noise[i*4 + 1] = random_float_gen();
                    g_noise[i*4 + 2] = random_float_gen();
                    g_noise[i*4 + 3] = (float)(rand() % 2); // Random Gender Target
                }
                forward_propagation_batch(generator, g_noise, g_out, state.num_samples);
                
                // 2. Train Discriminator on Real Data (Target: Real(0.9) for Label Smoothing, Correct Gender)
                forward_propagation_batch(discriminator, (float*)state.training_inputs, d_real_out, state.num_samples);
                for(int i=0; i<state.num_samples; i++) {
                    d_delta_out[i*2 + 0] = d_real_out[i*2 + 0] - 0.9f; // Label Smoothing: 0.9 instead of 1.0
                    d_delta_out[i*2 + 1] = d_real_out[i*2 + 1] - state.training_outputs[i][0]; // Wants Gender
                }
                backward_propagation_batch_with_deltas(discriminator, (float*)state.training_inputs, d_delta_out, state.num_samples, state.learning_rate);
                
                // 3. Train Discriminator on Fake Data (Target: Fake(0.1) for Label Smoothing, Generated Gender)
                forward_propagation_batch(discriminator, g_out, d_fake_out, state.num_samples);
                for(int i=0; i<state.num_samples; i++) {
                    d_delta_out[i*2 + 0] = d_fake_out[i*2 + 0] - 0.1f; // Label Smoothing: 0.1 instead of 0.0
                    d_delta_out[i*2 + 1] = d_fake_out[i*2 + 1] - g_noise[i*4 + 3]; // Wants Gender
                }
                backward_propagation_batch_with_deltas(discriminator, g_out, d_delta_out, state.num_samples, state.learning_rate);
                
                // 4. Train Generator (Target for D output: Real(0.9), Correct Gender)
                // We forward pass again to get clean activations for backprop
                forward_propagation_batch(discriminator, g_out, d_fake_out, state.num_samples);
                float total_g_loss = 0.0f;
                for(int i=0; i<state.num_samples; i++) {
                    d_delta_out[i*2 + 0] = d_fake_out[i*2 + 0] - 0.9f; // G wants D to think it's Real!
                    d_delta_out[i*2 + 1] = d_fake_out[i*2 + 1] - g_noise[i*4 + 3]; // G wants D to predict the correct gender
                    
                    float p = d_fake_out[i*2 + 0];
                    if (p < 0.0001f) p = 0.0001f;
                    if (p > 0.9999f) p = 0.9999f;
                    total_g_loss += -logf(p);
                }
                
                // Backprop through D to get gradients w.r.t G's output
                get_input_gradients_batch(discriminator, d_delta_out, d_delta_in, state.num_samples);
                
                // IMPORTANT: Apply the Sigmoid Derivative!
                // G's output layer uses a Sigmoid activation, so dL/dZ = dL/dOut * Out * (1 - Out)
                for(int i=0; i<state.num_samples * 5; i++) {
                    d_delta_in[i] = d_delta_in[i] * (g_out[i] * (1.0f - g_out[i]));
                }
                
                // Backprop through G using those gradients (Give Generator a learning rate boost!)
                backward_propagation_batch_with_deltas(generator, g_noise, d_delta_in, state.num_samples, state.learning_rate * 2.0f);
                
                state.epoch++;
                
                if (state.epoch % 20 == 0) {
                    float mse = total_g_loss / (float)state.num_samples;
                    state.loss_history[state.history_index] = mse;
                    state.history_index = (state.history_index + 1) % MAX_HISTORY;
                    if (state.history_count < MAX_HISTORY) state.history_count++;
                    state.current_loss = mse;
                    if (state.epoch >= 6000) {
                        state.is_trained = true;
                        state.is_training = false;
                        save_weights(generator, "generator.bin");
                        save_weights(discriminator, "discriminator.bin");
                        break;
                    }
                }
            }
        }
        
        // --- 3. EVALUATION PASS (For UI) ---
        float current_input_g[4] = { random_float_gen(), random_float_gen(), random_float_gen(), (float)(rand() % 2) };
        float current_output_g[5];
        forward_propagation(generator, current_input_g, current_output_g);
        
        float current_input_d[5];
        if (state.custom_mode) {
            for(int i=0; i<5; i++) current_input_d[i] = state.custom_inputs[i];
        } else {
            for(int i=0; i<5; i++) current_input_d[i] = current_output_g[i];
        }
        
        float current_output_d[2];
        forward_propagation(discriminator, current_input_d, current_output_d);

        // --- 4. RENDERING ---
        DrawVisualization(generator, discriminator, &state, current_input_g, current_output_g, current_input_d, current_output_d);
    }

    if (state.is_trained) {
        save_weights(generator, "generator.bin");
        save_weights(discriminator, "discriminator.bin");
    }

    free(d_real_out); free(d_fake_out); free(d_delta_out); free(g_out); free(g_noise); free(d_delta_in);
    UnloadFont(state.mainFont);
    free_neural_network(discriminator);
    free_neural_network(generator);
    CloseWindow();
    return 0;
}
