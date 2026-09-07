#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "tensor.h"
#include "tokenizer.h"
#include "math_kernels.h"
#include "llm_metal_backend.h"
#include "gguf_parser.h"
#include "transformer_engine.h"

#include <unistd.h>

int main(int argc, char** argv) {
    printf("=====================================================\n");
    printf("  Neural-C LLM Engine: C & Apple Metal GPU Inference \n");
    printf("=====================================================\n");

    const char* default_model = "qwen2.5-coder-0.5b-instruct-q4_0.gguf";
    if (access(default_model, F_OK) != 0) default_model = "stories15M-q4_0.gguf";
    const char* model_path = (argc > 1) ? argv[1] : (access(default_model, F_OK) == 0 ? default_model : NULL);
    const char* prompt = (argc > 2) ? argv[2] : "Write a C function to compute Fibonacci numbers.";
    float temperature = (argc > 3) ? (float)atof(argv[3]) : 0.7f;
    float top_p = 0.9f;
    int max_tokens = 64;

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

    GGUFContext* gguf_ctx = NULL;
    if (model_path) {
        printf("[GGUF] Opening model: %s\n", model_path);
        gguf_ctx = open_gguf_file(model_path);
        if (gguf_ctx) {
            cfg = gguf_ctx->config;
            print_config(&cfg);
        }
    } else {
        printf("[INFO] No GGUF model path specified. Running Transformer Engine with Config:\n");
        print_config(&cfg);
    }

    Tokenizer* tokenizer = create_tokenizer(cfg.vocab_size);
    if (gguf_ctx) {
        load_gguf_tokenizer(gguf_ctx, tokenizer);
    } else {
        set_tokenizer_entry(tokenizer, 0, "<pad>", 0.0f);
        set_tokenizer_entry(tokenizer, 1, "<bos>", 0.0f);
        set_tokenizer_entry(tokenizer, 2, "<eos>", 0.0f);
        set_tokenizer_entry(tokenizer, 3, "Hello", 10.0f);
        set_tokenizer_entry(tokenizer, 4, "!", 5.0f);
        set_tokenizer_entry(tokenizer, 5, " Tell", 8.0f);
        set_tokenizer_entry(tokenizer, 6, " me", 8.0f);
        set_tokenizer_entry(tokenizer, 7, " about", 8.0f);
        set_tokenizer_entry(tokenizer, 8, " deep", 8.0f);
        set_tokenizer_entry(tokenizer, 9, " learning", 8.0f);
        set_tokenizer_entry(tokenizer, 10, " in", 8.0f);
        set_tokenizer_entry(tokenizer, 11, " C", 8.0f);

        for (int c = 32; c <= 126; c++) {
            char s[2] = { (char)c, '\0' };
            set_tokenizer_entry(tokenizer, 30 + (c - 32), s, 1.0f);
        }
        set_tokenizer_special_tokens(tokenizer, 1, 2, 0);
    }

    TransformerWeights* weights = create_transformer_weights(&cfg);
    if (gguf_ctx) {
        load_gguf_weights(gguf_ctx, weights);
    }
    bool gpu_available = init_llm_metal_engine();

    TransformerEngine* engine = create_transformer_engine(&cfg, tokenizer, weights, gpu_available);
    if (!engine) {
        printf("[ERROR] Failed to initialize Transformer Engine.\n");
        return 1;
    }

    printf("[ENGINE] Metal GPU Mode: %s (%s)\n", engine->use_gpu ? "ACTIVE" : "CPU FALLBACK", get_llm_metal_device_name());

    // Execute Auto-Regressive Generation Stream
    generate_text_stream(engine, prompt, max_tokens, temperature, top_p);

    // Cleanup
    if (gguf_ctx) close_gguf_file(gguf_ctx);
    free_transformer_engine(engine);
    free_transformer_weights(weights, cfg.n_layers);
    free_tokenizer(tokenizer);

    printf("=====================================================\n");
    printf("  Execution Complete!                                 \n");
    printf("=====================================================\n");
    return 0;
}
