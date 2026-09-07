#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "model.h"
#include "tensor.h"

int main() {
    printf("=========================================\n");
    printf("  Module 2 Test: Tensor & Model Config   \n");
    printf("=========================================\n");

    // 1. Initialize 8B LLM Equivalent Config (LLaMA-3 8B Hyperparameters)
    Config cfg_8b = {
        .dim = 4096,
        .hidden_dim = 14336,
        .n_layers = 32,
        .n_heads = 32,
        .n_kv_heads = 8,  // Grouped-Query Attention (4:1 ratio)
        .vocab_size = 128256,
        .seq_len = 8192,
        .head_dim = 128,
        .norm_eps = 1e-5f
    };

    print_config(&cfg_8b);
    assert(cfg_8b.dim / cfg_8b.n_heads == cfg_8b.head_dim);
    printf("[PASS] LLaMA-3 8B Hyperparameter Configuration Validation\n");

    // 2. Test Tensor Initialization
    Tensor t_embed;
    int embed_shape[2] = {cfg_8b.vocab_size, cfg_8b.dim};
    init_tensor(&t_embed, "token_embd.weight", 2, embed_shape, QUANT_Q4_0);
    
    assert(strcmp(t_embed.name, "token_embd.weight") == 0);
    assert(t_embed.n_dims == 2);
    assert(t_embed.numel == (size_t)128256 * 4096);
    assert(t_embed.type == QUANT_Q4_0);
    printf("[PASS] Tensor Metadata Initialization\n");

    // 3. Test TransformerWeights Data Structure Allocation
    TransformerWeights* w = create_transformer_weights(&cfg_8b);
    assert(w != NULL);
    assert(w->rms_att_weight != NULL);
    assert(w->wq != NULL);
    assert(w->w_gate != NULL);
    printf("[PASS] Transformer Weights Layer Arrays Allocation\n");

    // 4. Test RunState Activation & KV-Cache Buffer Allocation
    // Test with a smaller sequence length for lightweight execution
    Config cfg_test = cfg_8b;
    cfg_test.seq_len = 512;

    RunState* s = allocate_run_state(&cfg_test);
    assert(s != NULL);
    assert(s->x != NULL);
    assert(s->q != NULL);
    assert(s->k != NULL);
    assert(s->v != NULL);
    assert(s->key_cache != NULL);
    assert(s->value_cache != NULL);
    printf("[PASS] RunState Activations & KV-Cache Buffer Allocation\n");

    // 5. Memory Cleanup
    free_run_state(s);
    free_transformer_weights(w, cfg_8b.n_layers);
    free_tensor(&t_embed);
    printf("[PASS] Memory Cleanup\n");

    printf("\n>>> MODULE 2 (TENSOR & MODEL ARCHITECTURE) PASSED ALL TESTS SUCCESSFULLY! <<<\n");
    return 0;
}
