#ifndef LLM_CLASSIFIER_H
#define LLM_CLASSIFIER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

typedef enum {
    INTENT_CODE = 0,
    INTENT_CREATIVE = 1,
    INTENT_QA_GENERAL = 2,
    INTENT_UNSAFE = 3
} PromptIntent;

typedef struct {
    float code_prob;
    float creative_prob;
    float qa_prob;
    float unsafe_prob;
    PromptIntent primary_intent;
    
    // Auto-tuned LLM Hyperparameters based on classification
    float recommended_temp;
    float recommended_top_p;
    bool is_safe;
} ClassifierResult;

typedef struct {
    size_t input_size;     // 6 Features
    size_t hidden1_size;   // 16
    size_t hidden2_size;   // 12
    size_t output_size;    // 4 Classes (Code, Creative, QA, Unsafe)

    float* w1; // (input_size * hidden1_size)
    float* b1; // (hidden1_size)
    float* w2; // (hidden1_size * hidden2_size)
    float* b2; // (hidden2_size)
    float* w3; // (hidden2_size * output_size)
    float* b3; // (output_size)

    float* h1_act;
    float* h2_act;
    float* out_act;
} LLMClassifier;

// API
LLMClassifier* create_llm_classifier(void);
void free_llm_classifier(LLMClassifier* clf);

void extract_prompt_features(const char* prompt, float features[6]);
ClassifierResult classify_prompt(LLMClassifier* clf, const char* prompt);

bool save_classifier_weights(LLMClassifier* clf, const char* filename);
bool load_classifier_weights(LLMClassifier* clf, const char* filename);
void train_classifier_step(LLMClassifier* clf, const float inputs[6], int target_class, float lr);

#endif // LLM_CLASSIFIER_H
