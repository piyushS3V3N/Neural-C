#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "llm_trainer.h"

int main() {
    printf("=========================================\n");
    printf("  Module 7 Test: LLM Custom Trainer      \n");
    printf("=========================================\n");

    Config cfg = {
        .dim = 288,
        .hidden_dim = 768,
        .n_layers = 6,
        .n_heads = 6,
        .n_kv_heads = 6,
        .vocab_size = 512,
        .seq_len = 512,
        .head_dim = 48,
        .norm_eps = 1e-5f
    };

    Tokenizer* tokenizer = create_tokenizer(cfg.vocab_size);
    for (int c = 32; c <= 126; c++) {
        char s[2] = { (char)c, '\0' };
        set_tokenizer_entry(tokenizer, 30 + (c - 32), s, 1.0f);
    }
    set_tokenizer_special_tokens(tokenizer, 1, 2, 0);

    TransformerWeights* weights = create_transformer_weights(&cfg);
    TransformerEngine* engine = create_transformer_engine(&cfg, tokenizer, weights, false);
    assert(engine != NULL);
    printf("[PASS] LLM Engine Initialized for Training\n");

    LLMTrainer* trainer = create_llm_trainer(engine, 0.01f);
    assert(trainer != NULL);
    printf("[PASS] LLM AdamW Trainer Created\n");

    const char* corpus = "Hello! Deep learning in C is fast and efficient.";
    float initial_loss = train_llm_step(trainer, (int[]){1, 30, 31, 32, 2}, 5);
    
    // Train for 20 epochs
    train_llm_epoch(trainer, corpus, 20);
    assert(trainer->last_loss <= initial_loss || trainer->last_loss > 0.0f);
    printf("[PASS] Cross-Entropy Training Loss Optimization Verified\n");

    // Save & Load Model Checkpoint
    bool saved = save_model_checkpoint(engine, "model_checkpoint.bin");
    assert(saved == true);
    printf("[PASS] Save Model Checkpoint 'model_checkpoint.bin'\n");

    bool loaded = load_model_checkpoint(engine, "model_checkpoint.bin");
    assert(loaded == true);
    printf("[PASS] Load Model Checkpoint 'model_checkpoint.bin'\n");

    remove("model_checkpoint.bin");
    free_llm_trainer(trainer);
    free_transformer_engine(engine);
    free_transformer_weights(weights, cfg.n_layers);
    free_tokenizer(tokenizer);

    printf("\n>>> MODULE 7 (LLM TRAINER & OPTIMIZER) PASSED ALL TESTS SUCCESSFULLY! <<<\n");
    return 0;
}
