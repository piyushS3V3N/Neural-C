#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "raylib.h"
#include "LLMClassifier.h"
#include "llm_trainer.h"
#include "visualizer.h"
#include "metal_backend.h"
#include "transformer_engine.h"
#include "gguf_parser.h"

static void prefill_prompt(TransformerEngine* engine, AppState* state) {
    if (!engine || state->num_prompt_tokens <= 0) return;

    float* logits = NULL;
    // Process all prompt tokens through the model to build full KV cache state
    for (int p = 0; p < state->num_prompt_tokens; p++) {
        logits = transformer_forward(engine, state->prompt_tokens[p], p);
    }
    state->gen_pos = state->num_prompt_tokens;

    if (!logits) return;

    // Compute top 5 candidates for the next token following the prompt
    float* probs = (float*)malloc(engine->config.vocab_size * sizeof(float));
    if (!probs) return;

    for (int i = 0; i < engine->config.vocab_size; i++) probs[i] = logits[i];
    softmax(probs, engine->config.vocab_size);

    for (int k = 0; k < 5; k++) {
        int best_i = 0;
        float best_p = -1.0f;
        for (int i = 0; i < engine->config.vocab_size; i++) {
            if (probs[i] > best_p) {
                bool already = false;
                for (int prev = 0; prev < k; prev++) {
                    if (state->top_predicted_tokens[prev] == i) already = true;
                }
                if (!already) {
                    best_p = probs[i];
                    best_i = i;
                }
            }
        }
        state->top_predicted_tokens[k] = best_i;
        state->top_predicted_probs[k] = best_p;
    }
    free(probs);
}

#include <dirent.h>

static void scan_available_models(AppState* state) {
    state->num_available_models = 0;
    DIR* dir = opendir(".");
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            const char* name = entry->d_name;
            size_t len = strlen(name);
            if (len > 5 && (strcmp(name + len - 5, ".gguf") == 0 || strcmp(name + len - 4, ".bin") == 0)) {
                if (strcmp(name, "mock_model.gguf") == 0) continue;
                if (state->num_available_models < 20) {
                    snprintf(state->available_models[state->num_available_models], 128, "%s", name);
                    state->num_available_models++;
                }
            }
        }
        closedir(dir);
    }

    dir = opendir("models");
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            const char* name = entry->d_name;
            size_t len = strlen(name);
            if (len > 5 && (strcmp(name + len - 5, ".gguf") == 0 || strcmp(name + len - 4, ".bin") == 0)) {
                if (state->num_available_models < 20) {
                    snprintf(state->available_models[state->num_available_models], 128, "models/%s", name);
                    state->num_available_models++;
                }
            }
        }
        closedir(dir);
    }
}

static bool load_selected_model(int model_idx, AppState* state, GGUFContext** out_gguf_ctx, TransformerEngine** out_engine, TransformerWeights** out_weights, Tokenizer** out_tokenizer) {
    if (model_idx < 0 || model_idx >= state->num_available_models) return false;

    const char* path = state->available_models[model_idx];
    printf("[VISUALIZER] Switching model [%d/%d] -> '%s'\n", model_idx + 1, state->num_available_models, path);

    // Free old resources safely
    if (*out_engine) {
        free_transformer_engine(*out_engine);
        *out_engine = NULL;
    }
    if (*out_weights) {
        int n_layers = (*out_gguf_ctx) ? (*out_gguf_ctx)->config.n_layers : 6;
        free_transformer_weights(*out_weights, n_layers);
        *out_weights = NULL;
    }
    if (*out_tokenizer) {
        free_tokenizer(*out_tokenizer);
        *out_tokenizer = NULL;
    }
    if (*out_gguf_ctx) {
        close_gguf_file(*out_gguf_ctx);
        *out_gguf_ctx = NULL;
    }

    GGUFContext* new_gguf = NULL;
    size_t path_len = strlen(path);
    if (path_len > 5 && strcmp(path + path_len - 5, ".gguf") == 0) {
        new_gguf = open_gguf_file(path);
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

    if (new_gguf) {
        cfg = new_gguf->config;
        *out_gguf_ctx = new_gguf;
    }

    Tokenizer* tok = create_tokenizer(cfg.vocab_size);
    if (new_gguf) {
        load_gguf_tokenizer(new_gguf, tok);
    } else {
        set_tokenizer_entry(tok, 0, "<pad>", 0.0f);
        set_tokenizer_entry(tok, 1, "<bos>", 0.0f);
        set_tokenizer_entry(tok, 2, "<eos>", 0.0f);
        for (int c = 32; c <= 126; c++) {
            char s[2] = { (char)c, '\0' };
            set_tokenizer_entry(tok, 30 + (c - 32), s, 1.0f);
        }
        set_tokenizer_special_tokens(tok, 1, 2, 0);
    }
    *out_tokenizer = tok;

    TransformerWeights* w = create_transformer_weights(&cfg);
    if (new_gguf) {
        load_gguf_weights(new_gguf, w);
    }
    *out_weights = w;

    TransformerEngine* eng = create_transformer_engine(&cfg, tok, w, state->using_gpu);
    *out_engine = eng;

    if (!new_gguf && path_len > 4 && strcmp(path + path_len - 4, ".bin") == 0) {
        load_model_checkpoint(eng, path);
    }

    state->current_model_index = model_idx;
    snprintf(state->active_model_name, sizeof(state->active_model_name), "%s", path);
    state->generated_text[0] = '\0';
    state->num_generated_tokens = 0;
    state->auto_generating = false;

    state->num_prompt_tokens = encode(tok, state->prompt_text, true, false, state->prompt_tokens, 64);
    prefill_prompt(eng, state);

    return true;
}

int main(int argc, char** argv) {
    AppState state = {0};
    state.screenWidth = 1200;
    state.screenHeight = 800;
    state.learning_rate = 0.05f;
    state.mode = MODE_TRANSFORMER_LLM; // Default to Transformer LLM Visualizer
    snprintf(state.prompt_text, sizeof(state.prompt_text), "Hello! Tell me about C.");
    state.prompt_len = strlen(state.prompt_text);
    state.temperature = 0.7f;
    state.top_p = 0.9f;

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_HIGHDPI);
    InitWindow(state.screenWidth, state.screenHeight, "Neural-C Engine & LLM Visualizer");
    SetTargetFPS(60);

    // Center window perfectly in monitor display
    int monitor = GetCurrentMonitor();
    SetWindowPosition((GetMonitorWidth(monitor) - state.screenWidth) / 2, (GetMonitorHeight(monitor) - state.screenHeight) / 2);

    state.mainFont = LoadFontEx("/System/Library/Fonts/Monaco.ttf", 36, 0, 0);
    if (state.mainFont.texture.id == 0) {
        state.mainFont = LoadFontEx("/System/Library/Fonts/Menlo.ttc", 36, 0, 0);
    }
    if (state.mainFont.texture.id != 0) {
        GenTextureMipmaps(&state.mainFont.texture);
        SetTextureFilter(state.mainFont.texture, TEXTURE_FILTER_TRILINEAR);
    }

    // Initialize Metal GPU Engine
    bool metal_available = init_metal_engine();
    if (metal_available) {
        snprintf(state.compute_device_name, 64, "%s", get_metal_device_name());
        state.using_gpu = true;
    } else {
        snprintf(state.compute_device_name, 64, "CPU (Sequential)");
        state.using_gpu = false;
    }

    // Scan folder for all available .gguf and .bin model files
    scan_available_models(&state);

    // Default model loading setup
    GGUFContext* gguf_ctx = NULL;
    TransformerEngine* llm_engine = NULL;
    TransformerWeights* llm_weights = NULL;
    Tokenizer* llm_tokenizer = NULL;

    const char* user_path = (argc > 1) ? argv[1] : NULL;
    int initial_idx = 0;
    if (user_path) {
        // If user passed explicit path on command line, add to available_models
        snprintf(state.available_models[state.num_available_models], 128, "%s", user_path);
        initial_idx = state.num_available_models;
        state.num_available_models++;
    }

    if (state.num_available_models > 0) {
        load_selected_model(initial_idx, &state, &gguf_ctx, &llm_engine, &llm_weights, &llm_tokenizer);
    } else {
        // Fallback default mock engine
        Config llm_cfg = {
            .dim = 288, .hidden_dim = 768, .n_layers = 6, .n_heads = 6,
            .n_kv_heads = 6, .vocab_size = 512, .seq_len = 512, .head_dim = 48, .norm_eps = 1e-5f
        };
        llm_tokenizer = create_tokenizer(llm_cfg.vocab_size);
        llm_weights = create_transformer_weights(&llm_cfg);
        llm_engine = create_transformer_engine(&llm_cfg, llm_tokenizer, llm_weights, state.using_gpu);
        snprintf(state.active_model_name, sizeof(state.active_model_name), "Mock Engine");
    }

    // Auto-load custom trained checkpoint if available
    bool custom_loaded = load_model_checkpoint(llm_engine, "custom_llm_model.bin");
    if (!custom_loaded) custom_loaded = load_model_checkpoint(llm_engine, "model_checkpoint.bin");
    if (custom_loaded) {
        printf("[VISUALIZER] Successfully loaded custom trained weights into LLM Engine!\n");
    }

    // --- Initialize LLM Prompt Router & Safety Head Classifier ---
    LLMClassifier* clf = create_llm_classifier();
    state.classifier = clf;

    // Load prompt dataset
    static float training_inputs[100][6];
    static int training_targets[100];
    static char prompt_samples[100][128];
    int loaded_samples = 0;

    FILE* file = fopen("data/llm_prompt_dataset.csv", "r");
    if (!file) file = fopen("llm_prompt_dataset.csv", "r");
    if (file) {
        char line[512];
        fgets(line, sizeof(line), file); // Header
        while (fgets(line, sizeof(line), file) && loaded_samples < 100) {
            char p[128];
            float f1, f2, f3, f4, f5, f6;
            int cat;
            if (sscanf(line, "\"%127[^\"]\",%f,%f,%f,%f,%f,%f,%d", p, &f1, &f2, &f3, &f4, &f5, &f6, &cat) == 8) {
                snprintf(prompt_samples[loaded_samples], 128, "%s", p);
                training_inputs[loaded_samples][0] = f1;
                training_inputs[loaded_samples][1] = f2;
                training_inputs[loaded_samples][2] = f3;
                training_inputs[loaded_samples][3] = f4;
                training_inputs[loaded_samples][4] = f5;
                training_inputs[loaded_samples][5] = f6;
                training_targets[loaded_samples] = cat;
                loaded_samples++;
            }
        }
        fclose(file);
    }

    state.training_inputs = training_inputs;
    state.training_targets = training_targets;
    state.prompt_samples = prompt_samples;
    state.num_samples = loaded_samples;

    // Quick offline training steps on dataset
    for (int epoch = 0; epoch < 200; epoch++) {
        for (int i = 0; i < loaded_samples; i++) {
            train_classifier_step(clf, training_inputs[i], training_targets[i], 0.05f);
        }
    }

    // Initial Tokenization & Forward Pass Predictions
    state.num_prompt_tokens = encode(llm_tokenizer, state.prompt_text, true, false, state.prompt_tokens, 64);
    prefill_prompt(llm_engine, &state);

    // Application Event Loop
    while (!WindowShouldClose()) {
        // Mode Switch via TAB Key
        if (IsKeyPressed(KEY_TAB)) {
            state.mode = (state.mode == MODE_TRANSFORMER_LLM) ? MODE_LLM_CLASSIFIER : MODE_TRANSFORMER_LLM;
        }

        // --- Transformer LLM Key Controls & Real-Time Chat ---
        if (state.mode == MODE_TRANSFORMER_LLM) {
            // 60 FPS Fast Typing without lag (only re-encodes tokens, defers prefill to ENTER/SPACE)
            int key = GetCharPressed();
            while (key > 0) {
                if (key >= 32 && key <= 126 && state.prompt_len < 100) {
                    state.prompt_text[state.prompt_len++] = (char)key;
                    state.prompt_text[state.prompt_len] = '\0';
                    state.num_prompt_tokens = encode(llm_tokenizer, state.prompt_text, true, false, state.prompt_tokens, 64);
                    state.generated_text[0] = '\0';
                    state.generated_len = 0;
                    state.num_generated_tokens = 0;
                    state.auto_generating = false;
                }
                key = GetCharPressed();
            }

            if (IsKeyPressed(KEY_BACKSPACE) && state.prompt_len > 0) {
                state.prompt_len--;
                state.prompt_text[state.prompt_len] = '\0';
                state.num_prompt_tokens = encode(llm_tokenizer, state.prompt_text, true, false, state.prompt_tokens, 64);
                state.generated_text[0] = '\0';
                state.generated_len = 0;
                state.num_generated_tokens = 0;
                state.auto_generating = false;
            }

            // Quick Preset Prompt Selection with UP / DOWN Arrow Keys
            static const char* PRESET_PROMPTS[] = {
                "def fibonacci(n):",
                "Write a C function for quicksort",
                "// Neural-C Matrix Multiplication",
                "Once upon a time in a small village",
                "What is a transformer neural network?",
                "Hello! Tell me about C programming."
            };
            int num_presets = 6;

            if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_DOWN)) {
                if (IsKeyPressed(KEY_UP)) {
                    state.preset_prompt_index = (state.preset_prompt_index - 1 + num_presets) % num_presets;
                } else {
                    state.preset_prompt_index = (state.preset_prompt_index + 1) % num_presets;
                }
                snprintf(state.prompt_text, sizeof(state.prompt_text), "%s", PRESET_PROMPTS[state.preset_prompt_index]);
                state.prompt_len = strlen(state.prompt_text);
                state.num_prompt_tokens = encode(llm_tokenizer, state.prompt_text, true, false, state.prompt_tokens, 64);
                state.generated_text[0] = '\0';
                state.generated_len = 0;
                state.num_generated_tokens = 0;
                state.auto_generating = false;
                prefill_prompt(llm_engine, &state);
            }

            // Handle GUI Button Requests (RUN PROMPT, NEXT PRESET, MODE TOGGLE)
            if (state.request_preset_next) {
                state.request_preset_next = false;
                state.preset_prompt_index = (state.preset_prompt_index + 1) % num_presets;
                snprintf(state.prompt_text, sizeof(state.prompt_text), "%s", PRESET_PROMPTS[state.preset_prompt_index]);
                state.prompt_len = strlen(state.prompt_text);
                state.num_prompt_tokens = encode(llm_tokenizer, state.prompt_text, true, false, state.prompt_tokens, 64);
                state.generated_text[0] = '\0';
                state.generated_len = 0;
                state.num_generated_tokens = 0;
                state.auto_generating = false;
                prefill_prompt(llm_engine, &state);
            }

            if (state.request_generate || IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_KP_ENTER)) {
                state.request_generate = false;
                prefill_prompt(llm_engine, &state);
                state.auto_generating = true;
                state.generated_text[0] = '\0';
                state.generated_len = 0;
                state.num_generated_tokens = 0;
                state.step_timer = 1.0f; // Force immediate step on first frame
            }

            // Step token generation manually (SPACE) or automatically (Timer tick)
            state.step_timer += GetFrameTime();
            bool should_step = IsKeyPressed(KEY_SPACE) || (state.auto_generating && state.step_timer >= 0.10f);

            if (should_step && state.num_prompt_tokens > 0) {
                state.step_timer = 0.0f;
                
                // Pick top token candidate (skipping pad/bos)
                int top_tok = state.top_predicted_tokens[0];
                for (int k = 0; k < 5; k++) {
                    int candidate = state.top_predicted_tokens[k];
                    if (candidate != llm_tokenizer->pad_id && candidate != llm_tokenizer->bos_id) {
                        top_tok = candidate;
                        break;
                    }
                }

                if (top_tok == llm_tokenizer->eos_id) {
                    state.auto_generating = false;
                } else if (state.num_generated_tokens < 128 && llm_engine && state.gen_pos < llm_engine->config.seq_len) {
                    const char* piece = decode(llm_tokenizer, -1, top_tok);
                    if (piece && piece[0] && strcmp(piece, "<pad>") != 0 && strcmp(piece, "<bos>") != 0 && strcmp(piece, "<eos>") != 0) {
                        strncat(state.generated_text, piece, sizeof(state.generated_text) - strlen(state.generated_text) - 1);
                    } else {
                        char tok_str[16];
                        snprintf(tok_str, sizeof(tok_str), "<tok:%d>", top_tok);
                        strncat(state.generated_text, tok_str, sizeof(state.generated_text) - strlen(state.generated_text) - 1);
                    }
                    state.generated_len = strlen(state.generated_text);
                    state.generated_tokens[state.num_generated_tokens++] = top_tok;

                    // Execute transformer forward pass for the new token at current gen_pos
                    float* logits = transformer_forward(llm_engine, top_tok, state.gen_pos);
                    state.gen_pos++;

                    if (logits) {
                        float* probs = (float*)malloc(llm_engine->config.vocab_size * sizeof(float));
                        if (probs) {
                            for (int i = 0; i < llm_engine->config.vocab_size; i++) probs[i] = logits[i];
                            softmax(probs, llm_engine->config.vocab_size);
                            for (int k = 0; k < 5; k++) {
                                int best_i = 0;
                                float best_p = -1.0f;
                                for (int i = 0; i < llm_engine->config.vocab_size; i++) {
                                    if (probs[i] > best_p) {
                                        bool already = false;
                                        for (int prev = 0; prev < k; prev++) {
                                            if (state.top_predicted_tokens[prev] == i) already = true;
                                        }
                                        if (!already) {
                                            best_p = probs[i];
                                            best_i = i;
                                        }
                                    }
                                }
                                state.top_predicted_tokens[k] = best_i;
                                state.top_predicted_probs[k] = best_p;
                            }
                            free(probs);
                        }
                    }
                } else {
                    state.auto_generating = false;
                }
            }

            if (state.request_model_change) {
                state.request_model_change = false;
                load_selected_model(state.pending_model_index, &state, &gguf_ctx, &llm_engine, &llm_weights, &llm_tokenizer);
            }
        }

        if (state.request_mode_toggle) {
            state.request_mode_toggle = false;
            state.mode = (state.mode == MODE_TRANSFORMER_LLM) ? MODE_LLM_CLASSIFIER : MODE_TRANSFORMER_LLM;
        }

        // --- LLM Router Classifier Key Controls ---
        if (state.mode == MODE_LLM_CLASSIFIER) {
            if (IsKeyPressed(KEY_RIGHT) && state.num_samples > 0) {
                state.test_index = (state.test_index + 1) % state.num_samples;
                state.custom_mode = false;
            }
            if (IsKeyPressed(KEY_LEFT) && state.num_samples > 0) {
                state.test_index = (state.test_index - 1 + state.num_samples) % state.num_samples;
                state.custom_mode = false;
            }
        }

        // --- Active Classification Evaluation ---
        const char* active_p = state.mode == MODE_TRANSFORMER_LLM ? state.prompt_text : (state.custom_mode ? state.typed_prompt : (state.prompt_samples ? state.prompt_samples[state.test_index] : ""));
        extract_prompt_features(active_p, state.classifier_inputs);
        state.clf_result = classify_prompt(clf, active_p);

        DrawVisualization(clf, llm_engine, &state, NULL, NULL);
    }

    // Cleanup
    UnloadFont(state.mainFont);
    free_llm_classifier(clf);
    if (llm_engine) {
        int n_layers = llm_engine->config.n_layers;
        free_transformer_engine(llm_engine);
        if (llm_weights) free_transformer_weights(llm_weights, n_layers);
    }
    if (llm_tokenizer) free_tokenizer(llm_tokenizer);
    if (gguf_ctx) close_gguf_file(gguf_ctx);
    CloseWindow();
    return 0;
}
