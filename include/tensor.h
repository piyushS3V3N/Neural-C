#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "model.h"

typedef enum {
    QUANT_FP32 = 0,
    QUANT_FP16 = 1,
    QUANT_Q8_0 = 2,
    QUANT_Q4_0 = 3,
    QUANT_Q4_K = 4
} QuantType;

typedef struct {
    char name[64];
    int shape[4];
    int n_dims;
    size_t numel;
    QuantType type;
    void* data;
    size_t size_bytes;
    bool is_mmap;
} Tensor;

typedef struct {
    // Global Weights
    Tensor token_embedding_table; // (vocab_size, dim)
    Tensor rms_final_weight;      // (dim)
    Tensor w_cls;                 // (vocab_size, dim) - Output Logits Head

    // Layer Weights (Arrays of size n_layers)
    Tensor* rms_att_weight;       // (n_layers, dim)
    Tensor* wq;                   // (n_layers, n_heads * head_dim, dim)
    Tensor* wk;                   // (n_layers, n_kv_heads * head_dim, dim)
    Tensor* wv;                   // (n_layers, n_kv_heads * head_dim, dim)
    Tensor* wo;                   // (n_layers, dim, n_heads * head_dim)
    Tensor* bq;                   // (n_layers, n_heads * head_dim) - Q Bias for Qwen2
    Tensor* bk;                   // (n_layers, n_kv_heads * head_dim) - K Bias for Qwen2
    Tensor* bv;                   // (n_layers, n_kv_heads * head_dim) - V Bias for Qwen2
    
    Tensor* rms_ffn_weight;       // (n_layers, dim)
    Tensor* w_gate;               // (n_layers, hidden_dim, dim)
    Tensor* w_up;                 // (n_layers, hidden_dim, dim)
    Tensor* w_down;               // (n_layers, dim, hidden_dim)
} TransformerWeights;

typedef struct {
    // Activation Buffers (for 1 token step forward pass)
    float* x;           // Activation vector at current timestep (dim)
    float* xb;          // Secondary activation vector (dim)
    float* xb2;         // Tertiary activation vector (dim)
    float* hb;          // Hidden state inside FFN (hidden_dim)
    float* hb2;         // Secondary hidden state inside FFN (hidden_dim)
    float* q;           // Query vector (n_heads * head_dim)
    float* k;           // Key vector (n_kv_heads * head_dim)
    float* v;           // Value vector (n_kv_heads * head_dim)
    float* att;         // Attention scores buffer (n_heads * seq_len)
    float* logits;      // Output vocabulary logits (vocab_size)

    // Key-Value Cache for Fast Auto-Regressive Decoding
    float* key_cache;   // (n_layers * seq_len * n_kv_heads * head_dim)
    float* value_cache; // (n_layers * seq_len * n_kv_heads * head_dim)
} RunState;

// Tensor Management API
void init_tensor(Tensor* t, const char* name, int n_dims, const int* shape, QuantType type);
void free_tensor(Tensor* t);
size_t get_element_size(QuantType type);

// Weight Array Management
TransformerWeights* create_transformer_weights(const Config* cfg);
void free_transformer_weights(TransformerWeights* w, int n_layers);

// RunState Activation Buffer Management
RunState* allocate_run_state(const Config* cfg);
void free_run_state(RunState* s);

#endif // TENSOR_H
