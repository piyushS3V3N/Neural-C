#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llm_trainer.h"

int main(int argc, char** argv) {
    printf("=====================================================\n");
    printf("  Neural-C LLM Trainer: Construct & Train Model       \n");
    printf("=====================================================\n");

    const char* corpus_file = (argc > 1) ? argv[1] : NULL;
    int epochs = (argc > 2) ? atoi(argv[2]) : 50;
    float lr = (argc > 3) ? atof(argv[3]) : 0.005f;

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
    set_tokenizer_entry(tokenizer, 0, "<pad>", 0.0f);
    set_tokenizer_entry(tokenizer, 1, "<bos>", 0.0f);
    set_tokenizer_entry(tokenizer, 2, "<eos>", 0.0f);
    for (int c = 32; c <= 126; c++) {
        char s[2] = { (char)c, '\0' };
        set_tokenizer_entry(tokenizer, 30 + (c - 32), s, 1.0f);
    }
    set_tokenizer_special_tokens(tokenizer, 1, 2, 0);

    TransformerWeights* weights = create_transformer_weights(&cfg);
    TransformerEngine* engine = create_transformer_engine(&cfg, tokenizer, weights, false);
    LLMTrainer* trainer = create_llm_trainer(engine, lr);

    char text[4096] = "Hello! Deep learning in C is fast and efficient. Let us build custom neural networks from scratch.";
    if (corpus_file) {
        FILE* f = fopen(corpus_file, "r");
        if (f) {
            size_t bytes = fread(text, 1, sizeof(text) - 1, f);
            text[bytes] = '\0';
            fclose(f);
            printf("[TRAINER] Loaded training corpus from '%s' (%zu bytes)\n", corpus_file, bytes);
        }
    } else {
        printf("[TRAINER] Using default demonstration training corpus.\n");
    }

    // Run Training Loop
    train_llm_epoch(trainer, text, epochs);

    // Save Model Weights Checkpoint
    save_model_checkpoint(engine, "custom_llm_model.bin");
    printf("[TRAINER] Model weights checkpoint saved to 'custom_llm_model.bin'\n");

    // Test text generation post-training
    printf("\n[GENERATION POST-TRAINING]\n");
    generate_text_stream(engine, "Hello!", 32, 0.7f, 0.9f);

    // Cleanup
    free_llm_trainer(trainer);
    free_transformer_engine(engine);
    free_transformer_weights(weights, cfg.n_layers);
    free_tokenizer(tokenizer);
    return 0;
}
