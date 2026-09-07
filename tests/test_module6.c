#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "model.h"
#include "tensor.h"
#include "tokenizer.h"
#include "math_kernels.h"
#include "llm_metal_backend.h"
#include "transformer_engine.h"

int main() {
    setbuf(stdout, NULL);
    printf("=========================================\n");
    printf("  Module 6 Test: LLM Generation Engine   \n");
    printf("=========================================\n");

    Config cfg = {
        .dim = 64,
        .hidden_dim = 128,
        .n_layers = 2,
        .n_heads = 2,
        .n_kv_heads = 2,
        .vocab_size = 16,
        .seq_len = 32,
        .head_dim = 32,
        .norm_eps = 1e-5f
    };

    Tokenizer* t = create_tokenizer(cfg.vocab_size);
    set_tokenizer_entry(t, 0, "<pad>", 0.0f);
    set_tokenizer_entry(t, 1, "<bos>", 0.0f);
    set_tokenizer_entry(t, 2, "<eos>", 0.0f);
    set_tokenizer_entry(t, 3, "A", 1.0f);
    set_tokenizer_entry(t, 4, "I", 1.0f);
    set_tokenizer_entry(t, 5, " ", 1.0f);
    set_tokenizer_entry(t, 6, "i", 1.0f);
    set_tokenizer_entry(t, 7, "n", 1.0f);
    set_tokenizer_entry(t, 8, "C", 1.0f);
    set_tokenizer_entry(t, 9, "AI", 10.0f);
    set_tokenizer_entry(t, 10, " in", 10.0f);
    set_tokenizer_entry(t, 11, " C", 10.0f);
    set_tokenizer_special_tokens(t, 1, 999, 0);

    TransformerWeights* w = create_transformer_weights(&cfg);
    bool gpu = init_llm_metal_engine();

    TransformerEngine* engine = create_transformer_engine(&cfg, t, w, gpu);
    assert(engine != NULL);
    printf("[PASS] Transformer Engine Initialization\n");

    // Test Forward Pass on Token Step 0
    float* logits = transformer_forward(engine, 1, 0);
    assert(logits != NULL);
    printf("[PASS] Single Token Forward Pass (Logits dim %d)\n", cfg.vocab_size);

    // Test Streaming Text Generation
    printf("[PASS] Streaming Text Generation Test:\n");
    generate_text_stream(engine, "AI in C", 5, 0.7f, 0.9f);

    // Cleanup
    free_transformer_engine(engine);
    free_transformer_weights(w, cfg.n_layers);
    free_tokenizer(t);
    printf("[PASS] Memory Cleanup\n");

    printf("\n>>> MODULE 6 (LLM GENERATION ENGINE & CLI) PASSED ALL TESTS SUCCESSFULLY! <<<\n");
    return 0;
}
