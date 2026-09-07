#ifndef TRANSFORMER_ENGINE_H
#define TRANSFORMER_ENGINE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "model.h"
#include "tensor.h"
#include "tokenizer.h"
#include "math_kernels.h"
#include "llm_metal_backend.h"

typedef struct {
    Config config;
    Tokenizer* tokenizer;
    TransformerWeights* weights;
    RunState* state;
    bool use_gpu;
} TransformerEngine;

// Transformer Engine API
TransformerEngine* create_transformer_engine(const Config* cfg, Tokenizer* tokenizer, TransformerWeights* weights, bool use_gpu);
void free_transformer_engine(TransformerEngine* engine);

// Forward Pass for a Single Token Timestep
float* transformer_forward(TransformerEngine* engine, int token, int pos);

// Text Generation Stream Pipeline
void generate_text_stream(TransformerEngine* engine, const char* prompt, int max_new_tokens, float temperature, float top_p);

#endif // TRANSFORMER_ENGINE_H
