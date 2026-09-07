#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "raylib.h"
#include "NeuralNetwork.h"
#include "LLMClassifier.h"
#include "transformer_engine.h"
#include <stdbool.h>

#define MAX_HISTORY 200

typedef enum {
    MODE_TRANSFORMER_LLM = 0,
    MODE_LLM_CLASSIFIER = 1
} VisualizerMode;

typedef struct {
    VisualizerMode mode;
    
    // LLM Router & Safety Classifier Head State
    LLMClassifier* classifier;
    ClassifierResult clf_result;
    float classifier_inputs[6];
    bool is_training;
    int epoch;
    float learning_rate;
    int test_index;
    float loss_history[MAX_HISTORY];
    int history_count;
    int history_index;
    float (*training_inputs)[6];
    int* training_targets;
    char (*prompt_samples)[128];
    int num_samples;
    bool is_trained;
    int retraining_frames;
    bool custom_mode;
    char typed_prompt[128];
    int typed_len;
    float current_loss;

    // Transformer LLM State
    char prompt_text[128];
    int prompt_len;
    int prompt_tokens[64];
    int num_prompt_tokens;
    int gen_pos;
    int preset_prompt_index;
    bool is_generating;
    float temperature;
    float top_p;
    int top_predicted_tokens[5];
    float top_predicted_probs[5];

    // Live Streaming Chat Response State
    char generated_text[512];
    int generated_len;
    int generated_tokens[128];
    int num_generated_tokens;
    float step_timer;
    bool auto_generating;
    float chat_scroll_offset;

    // Compute Hardware State
    bool using_gpu;
    char compute_device_name[64];

    // Discovered Models State
    char available_models[20][128];
    int num_available_models;
    int current_model_index;
    char active_model_name[128];
    bool model_dropdown_open;
    int pending_model_index;

    // UI Interactive Button Trigger Flags
    bool request_model_change;
    bool request_generate;
    bool request_preset_prev;
    bool request_preset_next;
    bool request_mode_toggle;
    
    // UI Layout State
    int screenWidth;
    int screenHeight;
    Font mainFont;
} AppState;

Color GetWeightColor(float weight);
void DrawLossGraph(Font font, AppState* state, int uiX);
void DrawVisualization(LLMClassifier* clf, TransformerEngine* engine, AppState* state, const float* current_input, const float* current_output);

#endif // VISUALIZER_H
