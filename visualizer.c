#include "visualizer.h"
#include <math.h>
#include <stdio.h>

// --- Lean & Clean Monochromatic Theme ---
#define COL_BG       (Color){ 12, 12, 12, 255 }
#define COL_SIDEBAR  (Color){ 18, 18, 18, 255 }
#define COL_BORDER   (Color){ 50, 50, 50, 255 }
#define COL_TEXT     (Color){ 240, 240, 240, 255 }
#define COL_TEXT_DIM (Color){ 140, 140, 140, 255 }
#define COL_NODE     (Color){ 12, 12, 12, 255 }
#define COL_NODE_OUT (Color){ 100, 100, 100, 255 }
#define COL_LOSS     (Color){ 220, 220, 220, 255 }
#define COL_TEST     (Color){ 250, 204, 21, 255 }
#define COL_POS      (Color){ 34, 197, 94, 255 }
#define COL_NEG      (Color){ 239, 68, 68, 255 }
#define COL_ACTIVATE (Color){ 255, 255, 255, 255 }

void DrawTextSmooth(Font font, const char *text, int x, int y, int fontSize, Color color) {
    if (font.texture.id == 0) {
        DrawText(text, x, y, fontSize, color);
    } else {
        DrawTextEx(font, text, (Vector2){(float)x, (float)y}, (float)fontSize, 1.0f, color);
    }
}

Color GetWeightColor(float weight) {
    unsigned char a = (unsigned char)(fabsf(weight) * 255.0f);
    if (a < 20) a = 20; 
    if (weight > 0) return (Color){ COL_POS.r, COL_POS.g, COL_POS.b, a };
    else return (Color){ COL_NEG.r, COL_NEG.g, COL_NEG.b, a };
}

static void DrawWeights(float* weights, float p1x, int n1, float p2x, int n2, int screenHeight) {
    for (int i = 0; i < n1; i++) {
        float y1 = (float)screenHeight / (float)(n1 + 1) * (float)(i + 1);
        for (int j = 0; j < n2; j++) {
            float y2 = (float)screenHeight / (float)(n2 + 1) * (float)(j + 1);
            float weight = weights[i * n2 + j];
            DrawLineEx((Vector2){p1x, y1}, (Vector2){p2x, y2}, fabsf(weight) * 1.5f + 0.5f, GetWeightColor(weight));
        }
    }
}

void DrawLossGraph(Font font, AppState* state, int uiX) {
    if (state->history_count > 1) {
        float max_loss = 0.01f;
        for (int i = 0; i < state->history_count; i++) {
            if (state->loss_history[i] > max_loss) max_loss = state->loss_history[i];
        }
        for (int i = 0; i < state->history_count - 1; i++) {
            int start_idx = (state->history_index - state->history_count + MAX_HISTORY) % MAX_HISTORY;
            int idx1 = (start_idx + i) % MAX_HISTORY;
            int idx2 = (start_idx + i + 1) % MAX_HISTORY;
            float x1 = uiX + 25 + (float)i / (state->history_count - 1) * 300;
            float y1 = 770 - (state->loss_history[idx1] / max_loss) * 100;
            float x2 = uiX + 25 + (float)(i + 1) / (state->history_count - 1) * 300;
            float y2 = 770 - (state->loss_history[idx2] / max_loss) * 100;
            DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, 2.0f, COL_ACTIVATE);
        }
        DrawTextSmooth(font, TextFormat("Cur: %.4f", state->current_loss), uiX + 25, 670, 14, COL_TEXT);
        DrawTextSmooth(font, TextFormat("Max: %.4f", max_loss), uiX + 25, 690, 14, COL_TEXT_DIM);
    }
}

static void DrawNetworkInstance(NeuralNetwork* nn, AppState* state, const float* current_input, const float* current_output, float startX, float width, bool is_generator) {
    int layer_sizes[] = { (int)nn->input_size, (int)nn->hidden1_size, (int)nn->hidden2_size, (int)nn->hidden3_size, (int)nn->hidden4_size, (int)nn->output_size };
    int num_layers = 6;
    float layerSpacing = width / (float)(num_layers + 1);
    float radius = 14.0f; // Restored larger radius
    
    Vector2 positions[6];
    for (int l = 0; l < num_layers; l++) {
        positions[l] = (Vector2){ startX + layerSpacing * (float)(l + 1), 0 };
    }

    DrawWeights(nn->weights_input_hidden1, positions[0].x, layer_sizes[0], positions[1].x, layer_sizes[1], state->screenHeight);
    DrawWeights(nn->weights_hidden1_hidden2, positions[1].x, layer_sizes[1], positions[2].x, layer_sizes[2], state->screenHeight);
    DrawWeights(nn->weights_hidden2_hidden3, positions[2].x, layer_sizes[2], positions[3].x, layer_sizes[3], state->screenHeight);
    DrawWeights(nn->weights_hidden3_hidden4, positions[3].x, layer_sizes[3], positions[4].x, layer_sizes[4], state->screenHeight);
    DrawWeights(nn->weights_hidden4_output, positions[4].x, layer_sizes[4], positions[5].x, layer_sizes[5], state->screenHeight);

    for (int l = 0; l < num_layers; l++) {
        float x = positions[l].x;
        int neuronsInLayer = layer_sizes[l];
        float* activations = NULL;
        if (l == 0) activations = (float*)current_input;
        else if (l == 1) activations = nn->hidden1_activations;
        else if (l == 2) activations = nn->hidden2_activations;
        else if (l == 3) activations = nn->hidden3_activations;
        else if (l == 4) activations = nn->hidden4_activations;
        else activations = (float*)current_output;

        for (int n = 0; n < neuronsInLayer; n++) {
            float y = (float)state->screenHeight / (float)(neuronsInLayer + 1) * (float)(n + 1);
            float activation = activations[n];
            DrawCircle((int)x, (int)y, radius, COL_NODE);
            float visual_act = (activation < 0.0f) ? 0.0f : (activation > 1.0f ? 1.0f : activation);
            unsigned char glow = (unsigned char)(visual_act * 255.0f);
            DrawCircle((int)x, (int)y, radius - 2.0f, (Color){ COL_ACTIVATE.r, COL_ACTIVATE.g, COL_ACTIVATE.b, glow }); 
            DrawCircleLines((int)x, (int)y, radius, COL_NODE_OUT);
            
            // Draw numeric value
            DrawTextSmooth(state->mainFont, TextFormat("%.1f", activation), (int)x - 12, (int)y - 6, 12, (visual_act > 0.5f ? COL_BG : COL_TEXT));
            
            // Labels
            if (is_generator && l == num_layers - 1) {
                // Generator output features
                const char* featNames[] = {"EndsVow", "Ends 'A'", "Len > 5", "StrtVow", "Has 'Y'"};
                DrawTextSmooth(state->mainFont, featNames[n], (int)x + 25, (int)y - 8, 14, COL_TEXT_DIM);
            }
            if (!is_generator && l == 0) {
                // Discriminator input features
                const char* featNames[] = {"EndsVow", "Ends 'A'", "Len > 5", "StrtVow", "Has 'Y'"};
                DrawTextSmooth(state->mainFont, featNames[n], (int)x - 70, (int)y - 8, 14, COL_TEXT_DIM);
            }
            if (!is_generator && l == num_layers - 1) {
                // Discriminator Outputs
                const char* outNames[] = {"Real/Fake", "Gender"};
                DrawTextSmooth(state->mainFont, outNames[n], (int)x + 25, (int)y - 8, 16, COL_POS);
            }
        }
    }
}

void DrawVisualization(NeuralNetwork* generator, NeuralNetwork* discriminator, AppState* state, const float* current_input_g, const float* current_output_g, const float* current_input_d, const float* current_output_d) {
    BeginDrawing();
    ClearBackground(COL_BG);

    int networkAreaWidth = state->screenWidth - 350; // Use all available space up to sidebar
    
    // We want the 6 Generator layers and 6 Discriminator layers to look perfectly continuous.
    // Total 12 layers across the screen. 
    float spacing = (float)networkAreaWidth / 13.0f; 
    
    // Width must be 7 * spacing so that internal layerSpacing = spacing
    float genWidth = spacing * 7.0f;
    float discStartX = spacing * 6.0f; 
    float discWidth = spacing * 7.0f;
    
    DrawNetworkInstance(generator, state, current_input_g, current_output_g, 0.0f, genWidth, true);
    DrawNetworkInstance(discriminator, state, current_input_d, current_output_d, discStartX, discWidth, false);

    // Draw connecting lines between Generator Output and Discriminator Input
    float gOutX = spacing * 6.0f; // Generator output is the 6th layer
    float dInX = spacing * 7.0f;  // Discriminator input is the 7th layer
    for (int i = 0; i < 5; i++) {
        float gY = (float)state->screenHeight / 6.0f * (float)(i + 1);
        float dY = (float)state->screenHeight / 6.0f * (float)(i + 1);
        DrawLineEx((Vector2){gOutX + 15.0f, gY}, (Vector2){dInX - 15.0f, dY}, 2.0f, (Color){100, 200, 255, 100});
    }
    
    DrawTextSmooth(state->mainFont, "GENERATOR", spacing * 1.5f, 20, 24, COL_TEXT);
    DrawTextSmooth(state->mainFont, "DISCRIMINATOR", spacing * 8.5f, 20, 24, COL_TEXT);

    int uiX = state->screenWidth - 350;
    DrawRectangle(uiX, 0, 350, state->screenHeight, COL_SIDEBAR);
    DrawLine(uiX, 0, uiX, state->screenHeight, COL_BORDER);
    
    DrawTextSmooth(state->mainFont, "> GAN ARCHITECTURE", uiX + 20, 30, 18, COL_TEXT);
    if (state->retraining_frames > 0) {
        DrawTextSmooth(state->mainFont, ">>> LIVE RETRAINING... <<<", uiX + 20, 65, 16, COL_ACTIVATE);
    } else if (state->is_trained) {
        DrawTextSmooth(state->mainFont, ">>> SYSTEM FULLY TRAINED <<<", uiX + 20, 65, 16, COL_POS);
    } else {
        Color statusColor = state->is_training ? COL_ACTIVATE : COL_NEG;
        DrawTextSmooth(state->mainFont, TextFormat("> Status: %s", state->is_training ? "TRAINING" : "PAUSED"), uiX + 20, 65, 16, statusColor);
    }
    
    float progress = 0.0f;
    if (state->is_trained) progress = 1.0f;
    else {
        progress = (float)state->epoch / 6000.0f;
        if (progress < 0.0f) progress = 0.0f;
        if (progress > 1.0f) progress = 1.0f;
    }
    
    DrawTextSmooth(state->mainFont, "Progress:", uiX + 20, 95, 14, COL_TEXT_DIM);
    DrawRectangle(uiX + 100, 97, 180, 12, COL_BG);
    DrawRectangleLines(uiX + 100, 97, 180, 12, COL_BORDER);
    DrawRectangle(uiX + 101, 98, (int)(progress * 178), 10, COL_POS);
    
    DrawTextSmooth(state->mainFont, TextFormat("> Epochs: %d", state->epoch), uiX + 20, 125, 16, COL_TEXT);
    DrawTextSmooth(state->mainFont, "> Dataset: NLTK Names Corpus", uiX + 20, 150, 16, COL_TEXT);
    
    // --- Live Sandbox UI ---
    DrawTextSmooth(state->mainFont, "--- Live Sandbox ---", uiX + 20, 185, 16, COL_TEXT_DIM);
    DrawTextSmooth(state->mainFont, "Type Name:", uiX + 20, 210, 14, COL_TEXT);
    DrawRectangle(uiX + 110, 205, 220, 26, COL_BG);
    DrawRectangleLines(uiX + 110, 205, 220, 26, state->custom_mode ? COL_ACTIVATE : COL_BORDER);
    DrawTextSmooth(state->mainFont, TextFormat("%s_", state->typed_name), uiX + 118, 210, 16, COL_TEXT);
    
    DrawTextSmooth(state->mainFont, "Extracted Features:", uiX + 20, 245, 14, COL_TEXT_DIM);
    DrawTextSmooth(state->mainFont, TextFormat("Ends Vowel: %1.0f", state->custom_inputs[0]), uiX + 20, 265, 14, state->custom_inputs[0] ? COL_POS : COL_TEXT_DIM);
    DrawTextSmooth(state->mainFont, TextFormat("Ends in A:  %1.0f", state->custom_inputs[1]), uiX + 170, 265, 14, state->custom_inputs[1] ? COL_POS : COL_TEXT_DIM);
    DrawTextSmooth(state->mainFont, TextFormat("Length > 5: %1.0f", state->custom_inputs[2]), uiX + 20, 285, 14, state->custom_inputs[2] ? COL_POS : COL_TEXT_DIM);
    DrawTextSmooth(state->mainFont, TextFormat("Starts Vow: %1.0f", state->custom_inputs[3]), uiX + 170, 285, 14, state->custom_inputs[3] ? COL_POS : COL_TEXT_DIM);
    DrawTextSmooth(state->mainFont, TextFormat("Has 'Y':    %1.0f", state->custom_inputs[4]), uiX + 20, 305, 14, state->custom_inputs[4] ? COL_POS : COL_TEXT_DIM);

    float pred = current_output_d[1]; // Gender output is index 1
    float real_pred = current_output_d[0]; // Real/Fake is index 0
    const char* gender = pred > 0.5f ? "FEMALE" : "MALE";
    float conf = (pred > 0.5f ? pred : (1.0f - pred)) * 100.0f;
    DrawTextSmooth(state->mainFont, TextFormat("GENDER: %s (%.1f%%)", gender, conf), uiX + 20, 335, 16, pred > 0.5f ? COL_POS : COL_ACTIVATE);
    DrawTextSmooth(state->mainFont, TextFormat("REAL/FAKE: %s", real_pred > 0.5f ? "REAL" : "FAKE"), uiX + 20, 355, 16, real_pred > 0.5f ? COL_POS : COL_NEG);

    // --- Dataset UI ---
    DrawTextSmooth(state->mainFont, "--- NLTK Names Dataset ---", uiX + 20, 385, 16, COL_TEXT_DIM);
    int start_idx = state->test_index - 4;
    if (start_idx < 0) start_idx = 0;
    if (start_idx > state->num_samples - 9) start_idx = state->num_samples - 9;
    if (start_idx < 0) start_idx = 0;
    
    for (int i = 0; i < 9; i++) {
        int idx = start_idx + i;
        if (idx >= state->num_samples) break;
        
        Color rowColor = (!state->custom_mode && idx == state->test_index) ? COL_ACTIVATE : COL_TEXT;
        const char* name = state->name_labels[idx];
        const char* type = (state->training_outputs[idx][0] > 0.5f) ? "FEMALE" : "MALE  ";
        
        DrawTextSmooth(state->mainFont, TextFormat("%-10s (%d%d%d%d%d) = %s", name, 
            (int)state->training_inputs[idx][0], (int)state->training_inputs[idx][1], 
            (int)state->training_inputs[idx][2], (int)state->training_inputs[idx][3], 
            (int)state->training_inputs[idx][4], type), 
            uiX + 20, 410 + i * 18, 14, rowColor);
            
        if (!state->custom_mode && idx == state->test_index) DrawTextSmooth(state->mainFont, "<- TEST", uiX + 260, 410 + i * 18, 14, COL_TEST);
    }
    
    DrawTextSmooth(state->mainFont, "--- Compute Backend ---", uiX + 20, 570, 16, COL_TEXT_DIM);
    if (state->using_gpu) {
        DrawTextSmooth(state->mainFont, TextFormat("Compute: %s", state->compute_device_name), uiX + 20, 595, 14, COL_POS);
    } else {
        DrawTextSmooth(state->mainFont, "Compute: CPU (Sequential)", uiX + 20, 595, 14, COL_NEG);
    }
    
    DrawTextSmooth(state->mainFont, "--- Generator Loss ---", uiX + 20, 630, 16, COL_TEXT_DIM);
    DrawRectangleLines(uiX + 20, 655, 310, 130, COL_BORDER);
    DrawRectangle(uiX + 21, 656, 308, 128, COL_BG);
    DrawLossGraph(state->mainFont, state, uiX);

    EndDrawing();
}
