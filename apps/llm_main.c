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
    float top_p = (argc > 4) ? (float)atof(argv[4]) : 0.9f;
    int max_tokens = (argc > 5) ? atoi(argv[5]) : 256;
    float repeat_penalty = (argc > 6) ? (float)atof(argv[6]) : 1.1f;
    int use_chat = (argc > 7) ? atoi(argv[7]) : -1; // -1 = auto
    if (max_tokens <= 0) max_tokens = 256;
    if (max_tokens > 2048) max_tokens = 2048;
    if (!(repeat_penalty >= 1.0f)) repeat_penalty = 1.0f;
    if (repeat_penalty > 2.0f) repeat_penalty = 2.0f;
    if (use_chat == -1) {
        // Auto: instruct-tuned files expect the chat template; base files don't.
        use_chat = (model_path && (strstr(model_path, "instruct") || strstr(model_path, "it-"))) ? 1 : 0;
    }

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
    if (getenv("NEURAL_C_CPU")) {
        printf("[ENGINE] NEURAL_C_CPU set: forcing CPU fallback (GPU divergence check)\n");
        gpu_available = false;
    }

    TransformerEngine* engine = create_transformer_engine(&cfg, tokenizer, weights, gpu_available);
    if (!engine) {
        printf("[ERROR] Failed to initialize Transformer Engine.\n");
        return 1;
    }

    printf("[ENGINE] Metal GPU Mode: %s (%s)\n", engine->use_gpu ? "ACTIVE" : "CPU FALLBACK", get_llm_metal_device_name());

    // NEURAL_C_SELFTEST=1: patch verifier. Runs a fixed probe through the
    // GPU engine and a fresh CPU engine, compares final logits, checks
    // determinism, exits. No eyeballing two runs.
    if (getenv("NEURAL_C_SELFTEST")) {
        const char* probe = "def fibonacci(n):";
        int pt[64];
        int npt = encode(tokenizer, probe, tokenizer->add_bos_token, false, pt, 64);
        printf("[SELFTEST] probe tokens: %d\n", npt);
        TransformerEngine* cpu = create_transformer_engine(&cfg, tokenizer, weights, false);
        if (!cpu) { printf("[SELFTEST] FAIL: cpu engine alloc\n"); return 2; }
        float *lg = NULL, *lc = NULL, *lc2 = NULL;
        for (int i = 0; i < npt; i++) {
            lg = transformer_forward(engine, pt[i], i);
            lc = transformer_forward(cpu, pt[i], i);
        }
        // determinism: replay probe on cpu from scratch
        TransformerEngine* cpu2 = create_transformer_engine(&cfg, tokenizer, weights, false);
        if (cpu2) {
            for (int i = 0; i < npt; i++) lc2 = transformer_forward(cpu2, pt[i], i);
        }
        int V = cfg.vocab_size;
        double maxdg = 0, sumdg = 0, maxdd = 0;
        if (lg && lc) {
            for (int i = 0; i < V; i++) {
                double d = fabs((double)lg[i] - (double)lc[i]);
                if (d > maxdg) maxdg = d;
                sumdg += d;
            }
        }
        if (lc && lc2) {
            for (int i = 0; i < V; i++) {
                double d = fabs((double)lc[i] - (double)lc2[i]);
                if (d > maxdd) maxdd = d;
            }
        }
        printf("[SELFTEST] cpu/gpu max|dlogit|=%.6f mean=%.6f\n", maxdg, sumdg / (V > 0 ? V : 1));
        printf("[SELFTEST] cpu/cpu determinism max|d|=%.9f %s\n", maxdd, maxdd == 0.0 ? "(bit-identical)" : "(NONDETERMINISTIC!)");
        for (int side = 0; side < 2; side++) {
            float* L = side ? lc : lg;
            if (!L) continue;
            float* cp = (float*)malloc((size_t)V * sizeof(float));
            if (!cp) continue;
            memcpy(cp, L, (size_t)V * sizeof(float));
            softmax(cp, V);
            printf("[SELFTEST] top5 %s:", side ? "cpu" : "gpu");
            for (int k = 0; k < 5; k++) {
                int bi = 0; float bp = -1;
                for (int i = 0; i < V; i++) if (cp[i] > bp) { bp = cp[i]; bi = i; }
                const char* s = decode(tokenizer, -1, bi);
                printf(" (%d %.3f '%s')", bi, bp, s ? s : "?");
                cp[bi] = -1;
            }
            printf("\n");
            free(cp);
        }
        printf("[SELFTEST] verdict: %s\n", maxdg < 1e-2 ? "MATCH (gpu math ok, look at sampling/data)" : "DIVERGED (metal/host bug, paste this block)");
        free_transformer_engine(cpu);
        if (cpu2) free_transformer_engine(cpu2);
        if (gguf_ctx) close_gguf_file(gguf_ctx);
        free_transformer_engine(engine);
        free_transformer_weights(weights, cfg.n_layers);
        free_tokenizer(tokenizer);
        return 0;
    }

    printf("[DECODE] temp=%.2f top_p=%.2f max_tokens=%d repeat_penalty=%.2f chat_template=%s\n",
        temperature, top_p, max_tokens, repeat_penalty, use_chat ? "ON" : "OFF");

    // Execute Auto-Regressive Generation Stream
    generate_text_stream_ex(engine, prompt, max_tokens, temperature, top_p, repeat_penalty, 64, use_chat);

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
