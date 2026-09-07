#ifndef MODEL_H
#define MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Hyperparameters for Decoder-Only Transformer (LLaMA / Qwen / Mistral Architecture)
typedef struct {
    int dim;             // Transformer hidden dimension (d_model, e.g. 4096 for 8B, 288 for test)
    int hidden_dim;      // FFN inner dimension (d_ffn, e.g. 11008 or 14336)
    int n_layers;        // Number of Transformer blocks (e.g. 32 for 8B, 6 for test)
    int n_heads;         // Number of Query attention heads (e.g. 32)
    int n_kv_heads;      // Number of Key/Value heads for Grouped-Query Attention (GQA, e.g. 8 or 32)
    int vocab_size;      // Vocabulary size (e.g. 32000 or 128256)
    int seq_len;         // Context window size (e.g. 2048, 4096, 8192)
    int head_dim;        // Dimension per head (dim / n_heads, e.g. 128)
    float norm_eps;      // RMSNorm epsilon constant (e.g. 1e-5f)
    float rope_freq_base;// RoPE base frequency (e.g. 10000.0f for LLaMA, 1000000.0f for Qwen2.5)
} Config;

static inline void print_config(const Config* cfg) {
    if (!cfg) return;
    printf("=========================================\n");
    printf("  Transformer Model Configuration         \n");
    printf("=========================================\n");
    printf("  d_model (dim)    : %d\n", cfg->dim);
    printf("  d_ffn (hidden)   : %d\n", cfg->hidden_dim);
    printf("  n_layers         : %d\n", cfg->n_layers);
    printf("  n_heads (Q)      : %d\n", cfg->n_heads);
    printf("  n_kv_heads (K/V) : %d (GQA factor: %d:1)\n", cfg->n_kv_heads, cfg->n_heads / (cfg->n_kv_heads > 0 ? cfg->n_kv_heads : 1));
    printf("  head_dim         : %d\n", cfg->head_dim);
    printf("  vocab_size       : %d\n", cfg->vocab_size);
    printf("  seq_len (context): %d\n", cfg->seq_len);
    printf("  norm_eps         : %.1e\n", cfg->norm_eps);
    printf("  rope_freq_base   : %.1f\n", cfg->rope_freq_base);
    printf("=========================================\n");
}

#endif // MODEL_H
