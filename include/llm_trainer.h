#ifndef LLM_TRAINER_H
#define LLM_TRAINER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "model.h"
#include "tensor.h"
#include "tokenizer.h"
#include "transformer_engine.h"

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float eps;
    size_t step;
} AdamWOptimizer;

typedef struct {
    TransformerEngine* engine;
    AdamWOptimizer opt;
    float last_loss;
} LLMTrainer;

// API
LLMTrainer* create_llm_trainer(TransformerEngine* engine, float lr);
void free_llm_trainer(LLMTrainer* trainer);

float train_llm_step(LLMTrainer* trainer, const int* tokens, int num_tokens);
void train_llm_epoch(LLMTrainer* trainer, const char* text_corpus, int epochs);
bool save_model_checkpoint(TransformerEngine* engine, const char* filename);
bool load_model_checkpoint(TransformerEngine* engine, const char* filename);

#endif // LLM_TRAINER_H
