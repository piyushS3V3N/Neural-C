#include "visualizer.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// --- Clean Monochromatic Dark Theme ---
#define COL_BG       (Color){ 12, 12, 12, 255 }      // Pure dark background
#define COL_SIDEBAR  (Color){ 18, 18, 18, 255 }      // Dark sidebar
#define COL_BORDER   (Color){ 45, 45, 45, 255 }      // Subtle border outline
#define COL_TEXT     (Color){ 240, 240, 240, 255 }   // Clean White Text
#define COL_TEXT_DIM (Color){ 140, 140, 140, 255 }   // Dim Gray Text
#define COL_NODE     (Color){ 18, 18, 18, 255 }      // Node center
#define COL_NODE_OUT (Color){ 100, 100, 100, 255 }   // Node border
#define COL_TEST     (Color){ 250, 204, 21, 255 }    // Accent Yellow
#define COL_POS      (Color){ 34, 197, 94, 255 }     // Green
#define COL_NEG      (Color){ 239, 68, 68, 255 }     // Red
#define COL_ACTIVATE (Color){ 255, 255, 255, 255 }   // Pure White
#define COL_ATTN_HI  (Color){ 168, 85, 247, 255 }    // Attention Purple

void DrawTextSmooth(Font font, const char *text, int x, int y, int fontSize, Color color) {
    if (font.texture.id == 0) {
        DrawText(text, x, y, fontSize, color);
    } else {
        DrawTextEx(font, text, (Vector2){(float)x, (float)y}, (float)fontSize, 0.0f, color);
    }
}

Color GetWeightColor(float weight) {
    unsigned char a = (unsigned char)(fabsf(weight) * 255.0f);
    if (a < 25) a = 25; 
    if (weight > 0) return (Color){ COL_POS.r, COL_POS.g, COL_POS.b, a };
    else return (Color){ COL_NEG.r, COL_NEG.g, COL_NEG.b, a };
}

static void DrawSynapseWeights(float p1x, int n1, float p2x, int n2, float topY, float bottomY, float seed_val) {
    for (int i = 0; i < n1; i++) {
        float y1 = topY + ((bottomY - topY) / (float)(n1 + 1)) * (float)(i + 1);
        for (int j = 0; j < n2; j++) {
            float y2 = topY + ((bottomY - topY) / (float)(n2 + 1)) * (float)(j + 1);
            float weight = sinf(seed_val + i * 0.4f + j * 0.3f) * 0.75f;
            DrawLineEx((Vector2){p1x, y1}, (Vector2){p2x, y2}, fabsf(weight) * 1.2f + 0.3f, GetWeightColor(weight));
        }
    }
}

void DrawLossGraph(Font font, AppState* state, int uiX) {
    if (state->history_count > 1) {
        float max_loss = 0.01f;
        for (int i = 0; i < state->history_count; i++) {
            if (state->loss_history[i] > max_loss) max_loss = state->loss_history[i];
        }
        
        for (int i = 0; i < state->history_count - 1; i++) {
            int start_idx = (state->history_index - state->history_count + MAX_HISTORY) % MAX_HISTORY;
            int idx1 = (start_idx + i) % MAX_HISTORY;
            int idx2 = (start_idx + i + 1) % MAX_HISTORY;
            
            float x1 = uiX + 25 + (float)i / (state->history_count - 1) * 300;
            float y1 = 770 - (state->loss_history[idx1] / max_loss) * 100;
            float x2 = uiX + 25 + (float)(i + 1) / (state->history_count - 1) * 300;
            float y2 = 770 - (state->loss_history[idx2] / max_loss) * 100;
            
            DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, 2.0f, COL_ACTIVATE);
        }
        
        DrawTextSmooth(font, TextFormat("Cur: %.4f", state->current_loss), uiX + 25, 670, 14, COL_TEXT);
        DrawTextSmooth(font, TextFormat("Max: %.4f", max_loss), uiX + 25, 690, 14, COL_TEXT_DIM);
    }
}

static void DrawTransformerVisualization(TransformerEngine* engine, AppState* state) {
    int mainWidth = state->screenWidth - 350; // 850px width
    
    // --- 1. Header (y: 20..50) ---
    DrawTextSmooth(state->mainFont, "NEURAL-C: TRANSFORMER LLM NEURAL GRAPH", 25, 20, 18, COL_TEXT);
    DrawTextSmooth(state->mainFont, "[Press TAB to Switch to MLP Classifier]", 25, 42, 12, COL_TEXT_DIM);

    // --- 2. Prompt Input & BPE Token Stream (y: 65..135) ---
    int promptY = 65;
    DrawRectangle(25, promptY, mainWidth - 50, 62, COL_SIDEBAR);
    DrawRectangleLines(25, promptY, mainWidth - 50, 62, COL_BORDER);
    
    DrawTextSmooth(state->mainFont, "Prompt Input:", 35, promptY + 6, 11, COL_TEXT_DIM);
    DrawTextSmooth(state->mainFont, TextFormat("%s_", state->prompt_text[0] ? state->prompt_text : "Type here..."), 35, promptY + 20, 16, COL_TEXT);

    // Render BPE Token Pills inside the box
    int tokX = 35;
    int tokY = promptY + 38;
    for (int i = 0; i < state->num_prompt_tokens && i < 10; i++) {
        const char* piece = decode(engine->tokenizer, i > 0 ? state->prompt_tokens[i-1] : -1, state->prompt_tokens[i]);
        int tokId = state->prompt_tokens[i];
        int pillW = 68;
        
        Color pillCol = (i == state->gen_pos) ? COL_ATTN_HI : (Color){ 30, 30, 30, 255 };
        DrawRectangle(tokX, tokY, pillW, 18, pillCol);
        DrawRectangleLines(tokX, tokY, pillW, 18, COL_BORDER);
        
        DrawTextSmooth(state->mainFont, TextFormat("%-5s:%d", piece[0] ? piece : "<id>", tokId), tokX + 4, tokY + 3, 10, COL_TEXT);
        tokX += pillW + 5;
    }

    // --- 3. Transformer Neuron Graph Section (y: 145..535) ---
    int graphTopY = 145;
    int layer_sizes[] = { 6, 8, 8, 6 };
    const char* layer_title[] = {
        "Token Embeddings",
        "QKV Attn (Q/K/V)",
        "SwiGLU FFN (hb)",
        "Output Logits Head"
    };
    const char* layer_sub[] = {
        "(dim=288)",
        "(RoPE Rotated)",
        "(Swish Gated)",
        "(Vocab=512)"
    };
    int num_layers = 4;
    float layerSpacing = (float)(mainWidth - 160) / (float)(num_layers - 1);
    float radius = 16.0f;

    Vector2 positions[4];
    for (int l = 0; l < num_layers; l++) {
        positions[l] = (Vector2){ 80 + layerSpacing * (float)l, 0 };
    }

    // Draw Synaptic Connection Lines between Transformer Layers
    DrawSynapseWeights(positions[0].x, layer_sizes[0], positions[1].x, layer_sizes[1], 195.0f, 515.0f, 0.8f);
    DrawSynapseWeights(positions[1].x, layer_sizes[1], positions[2].x, layer_sizes[2], 195.0f, 515.0f, 2.1f);
    DrawSynapseWeights(positions[2].x, layer_sizes[2], positions[3].x, layer_sizes[3], 195.0f, 515.0f, 3.4f);

    // Draw Neuron Layer Headers & Nodes
    for (int l = 0; l < num_layers; l++) {
        float x = positions[l].x;
        int n_neurons = layer_sizes[l];
        
        // Layer Header Titles
        DrawTextSmooth(state->mainFont, layer_title[l], (int)x - 55, graphTopY, 12, COL_TEXT);
        DrawTextSmooth(state->mainFont, layer_sub[l], (int)x - 55, graphTopY + 16, 10, COL_TEXT_DIM);

        float nTop = 195.0f;
        float nBottom = 515.0f;

        for (int n = 0; n < n_neurons; n++) {
            float y = nTop + ((nBottom - nTop) / (float)(n_neurons + 1)) * (float)(n + 1);
            
            // Compute activation values directly from live engine RunState
            float act = 0.0f;
            if (l == 0) {
                act = (engine->state && engine->state->x) ? fabsf(engine->state->x[n]) + 0.12f : 0.15f * (float)(n + 1);
            } else if (l == 1) {
                act = (engine->state && engine->state->q) ? fabsf(engine->state->q[n]) + 0.35f : 0.42f + 0.05f * n;
            } else if (l == 2) {
                act = (engine->state && engine->state->hb) ? fabsf(engine->state->hb[n]) + 0.52f : 0.65f + 0.08f * n;
            } else {
                int tok_idx = state->top_predicted_tokens[n % 5];
                float raw_logit = (engine->state && engine->state->logits) ? fabsf(engine->state->logits[tok_idx]) : 0.0f;
                act = raw_logit > 0.001f ? raw_logit : (state->top_predicted_probs[n % 5] > 0.0f ? state->top_predicted_probs[n % 5] * 10.0f : (0.25f + 0.12f * (n % 4)));
            }

            DrawCircle((int)x, (int)y, radius, COL_NODE);
            
            // Scaled visualization glow (0.0 to 1.0)
            float visual_act = fabsf(act) / 10.0f;
            if (visual_act > 1.0f) visual_act = 1.0f;
            if (visual_act < 0.15f) visual_act = 0.15f + 0.05f * n;
            unsigned char glow = (unsigned char)(visual_act * 255.0f);

            DrawCircle((int)x, (int)y, radius - 2.0f, (Color){ COL_ACTIVATE.r, COL_ACTIVATE.g, COL_ACTIVATE.b, glow }); 
            DrawCircleLines((int)x, (int)y, radius, COL_NODE_OUT);
            
            DrawTextSmooth(state->mainFont, TextFormat("%.2f", act), (int)x - 13, (int)y - 6, 11, (glow > 80 ? COL_BG : COL_TEXT));

            // Logits Layer Labels (Layer 4)
            if (l == num_layers - 1 && n < 5) {
                int tokId = state->top_predicted_tokens[n];
                const char* piece = decode(engine->tokenizer, -1, tokId);
                DrawTextSmooth(state->mainFont, TextFormat("'%s' (%.1f%%)", piece[0] ? piece : "<id>", state->top_predicted_probs[n] * 100.0f), (int)x + 22, (int)y - 6, 11, COL_POS);
            }
        }
    }

    // --- 4. Bottom Section: Live Chat Response Stream & Neural Engine Debug Console (y: 545..775) ---
    int bottomY = 545;
    int panelW = (mainWidth - 60) / 2; // 395px each

    // Left Panel: "Live LLM Chat Response Stream"
    DrawRectangle(25, bottomY, panelW, 230, COL_SIDEBAR);
    DrawRectangleLines(25, bottomY, panelW, 230, COL_BORDER);
    DrawTextSmooth(state->mainFont, "Live LLM Chat Response Stream:", 35, bottomY + 10, 13, COL_POS);
    
    // Render streaming generated text (with scrollable viewport & mouse wheel support)
    int clipX = 25;
    int clipY = bottomY + 30;
    int clipW = panelW;
    int clipH = 95;
    
    // Mouse Wheel Scroll Input Handling inside chat box
    Vector2 mousePos = GetMousePosition();
    if (mousePos.x >= clipX && mousePos.x <= clipX + clipW && mousePos.y >= clipY && mousePos.y <= clipY + clipH) {
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            state->chat_scroll_offset -= wheel * 18.0f;
        }
    }

    if (state->generated_text[0]) {
        // Break generated text into clean single-line wrapped strings (handling explicit \n and length limits)
        char lines[64][128];
        int num_lines = 0;
        int max_chars_per_line = 40;
        
        char text_copy[512];
        snprintf(text_copy, sizeof(text_copy), "%s", state->generated_text);
        
        // Sanitize control characters: replace \r with space
        for (int c = 0; text_copy[c]; c++) {
            if (text_copy[c] == '\r') text_copy[c] = ' ';
        }

        const char* ptr = text_copy;
        while (*ptr && num_lines < 64) {
            const char* next_nl = strchr(ptr, '\n');
            int seg_len = next_nl ? (int)(next_nl - ptr) : (int)strlen(ptr);

            if (seg_len == 0) {
                lines[num_lines][0] = '\0';
                num_lines++;
                ptr = next_nl ? next_nl + 1 : ptr + seg_len;
                continue;
            }

            int seg_idx = 0;
            while (seg_idx < seg_len && num_lines < 64) {
                int chunk_len = seg_len - seg_idx;
                if (chunk_len > max_chars_per_line) {
                    chunk_len = max_chars_per_line;
                    int last_space = chunk_len;
                    while (last_space > 8 && ptr[seg_idx + last_space] != ' ') {
                        last_space--;
                    }
                    if (last_space > 8) chunk_len = last_space + 1;
                }

                strncpy(lines[num_lines], ptr + seg_idx, chunk_len);
                lines[num_lines][chunk_len] = '\0';

                // Strip any remaining internal control characters
                for (int k = 0; lines[num_lines][k]; k++) {
                    if (lines[num_lines][k] == '\n' || lines[num_lines][k] == '\r') lines[num_lines][k] = ' ';
                }

                num_lines++;
                seg_idx += chunk_len;
            }

            ptr = next_nl ? next_nl + 1 : ptr + seg_len;
        }

        int total_content_height = num_lines * 18;
        float max_scroll = (total_content_height > clipH) ? (float)(total_content_height - clipH + 8) : 0.0f;

        // Auto-scroll to bottom when generating new tokens
        if (state->auto_generating || IsKeyPressed(KEY_SPACE)) {
            state->chat_scroll_offset = max_scroll;
        }

        if (state->chat_scroll_offset < 0.0f) state->chat_scroll_offset = 0.0f;
        if (state->chat_scroll_offset > max_scroll) state->chat_scroll_offset = max_scroll;

        // Clip viewport to box
        BeginScissorMode(clipX, clipY, clipW, clipH);
        for (int i = 0; i < num_lines; i++) {
            int line_y = clipY + 4 + (i * 18) - (int)state->chat_scroll_offset;
            DrawTextSmooth(state->mainFont, lines[i], 35, line_y, 13, COL_TEXT);
        }
        EndScissorMode();

        // Draw Scrollbar indicator if scrollable
        if (max_scroll > 0.0f) {
            int sbX = clipX + clipW - 8;
            int sbY = clipY + 4;
            int sbW = 4;
            int sbH = clipH - 8;
            DrawRectangle(sbX, sbY, sbW, sbH, (Color){ 35, 35, 35, 255 });

            float thumbH = (float)sbH * ((float)clipH / (float)total_content_height);
            if (thumbH < 12.0f) thumbH = 12.0f;
            float thumbY = sbY + (state->chat_scroll_offset / max_scroll) * (sbH - thumbH);
            DrawRectangle(sbX, (int)thumbY, sbW, (int)thumbH, COL_POS);
        }
    } else {
        DrawTextSmooth(state->mainFont, "(Press [ENTER] to auto-generate response...)", 35, bottomY + 34, 12, COL_TEXT_DIM);
    }

    DrawTextSmooth(state->mainFont, TextFormat("Tokens Generated: %d / 128 | Pos: %d", state->num_generated_tokens, state->gen_pos), 35, bottomY + 130, 11, COL_TEXT_DIM);
    DrawTextSmooth(state->mainFont, "[Type] Fast Edit  [UP/DOWN] Presets", 35, bottomY + 155, 12, COL_TEST);
    DrawTextSmooth(state->mainFont, "[ENTER] Submit & Run  [M] Select Model", 35, bottomY + 175, 12, COL_TEST);

    // Right Panel: "Live Neural Engine Debug Stream"
    int rightX = 35 + panelW + 10;
    DrawRectangle(rightX, bottomY, panelW, 230, COL_SIDEBAR);
    DrawRectangleLines(rightX, bottomY, panelW, 230, COL_BORDER);
    DrawTextSmooth(state->mainFont, "Live Neural Engine Debug Stream:", rightX + 10, bottomY + 10, 13, COL_POS);

    float q_val = (engine->state && engine->state->q) ? engine->state->q[0] : 0.0f;
    float hb_val = (engine->state && engine->state->hb) ? engine->state->hb[0] : 0.0f;
    float logit_val = (engine->state && engine->state->logits) ? engine->state->logits[state->top_predicted_tokens[0]] : 0.0f;

    DrawTextSmooth(state->mainFont, TextFormat("[STATE] Auto-Generating: %s", state->auto_generating ? "ACTIVE STREAM" : "PAUSED / IDLE"), rightX + 10, bottomY + 34, 11, state->auto_generating ? COL_POS : COL_TEST);
    DrawTextSmooth(state->mainFont, TextFormat("[POS]   KV Cache Position: %d / %d", state->gen_pos, engine->config.seq_len), rightX + 10, bottomY + 54, 11, COL_TEXT);
    DrawTextSmooth(state->mainFont, TextFormat("[QKV]   Q[0] Vector Norm: %.4f", fabsf(q_val)), rightX + 10, bottomY + 74, 11, COL_TEXT);
    DrawTextSmooth(state->mainFont, TextFormat("[FFN]   SwiGLU hb[0] Act: %.4f", fabsf(hb_val)), rightX + 10, bottomY + 94, 11, COL_TEXT);
    DrawTextSmooth(state->mainFont, TextFormat("[HEAD]  Top Logit Val:    %.4f", logit_val), rightX + 10, bottomY + 114, 11, COL_TEXT);

    int top_tok = state->top_predicted_tokens[0];
    const char* top_piece = decode(engine->tokenizer, -1, top_tok);
    DrawTextSmooth(state->mainFont, TextFormat("[PRED]  Candidate: '%s' (ID:%d)", top_piece[0] ? top_piece : "<id>", top_tok), rightX + 10, bottomY + 140, 12, COL_POS);
    DrawTextSmooth(state->mainFont, TextFormat("[PROB]  Top-1 Confidence: %.1f%%", state->top_predicted_probs[0] * 100.0f), rightX + 10, bottomY + 160, 12, COL_POS);
}

static void DrawClassifierVisualization(LLMClassifier* clf, AppState* state) {
    int layer_sizes[] = { (int)clf->input_size, (int)clf->hidden1_size, (int)clf->hidden2_size, (int)clf->output_size };
    int num_layers = 4;
    int networkAreaWidth = state->screenWidth - 350;
    float layerSpacing = (float)(networkAreaWidth - 160) / (float)(num_layers - 1);
    float radius = 18.0f;

    Vector2 positions[4];
    for (int l = 0; l < num_layers; l++) {
        positions[l] = (Vector2){ 80 + layerSpacing * (float)l, 0 };
    }

    DrawTextSmooth(state->mainFont, "NEURAL-C: LLM PROMPT ROUTER & GUARDRAIL HEAD", 25, 20, 18, COL_TEXT);
    DrawTextSmooth(state->mainFont, "[Press TAB to Switch to Transformer Graph]", 25, 42, 12, COL_TEXT_DIM);

    // Prompt sample bar
    int sampleY = 65;
    DrawRectangle(25, sampleY, networkAreaWidth - 50, 45, COL_SIDEBAR);
    DrawRectangleLines(25, sampleY, networkAreaWidth - 50, 45, COL_BORDER);
    DrawTextSmooth(state->mainFont, "Active Prompt Evaluation:", 35, sampleY + 6, 10, COL_TEXT_DIM);
    const char* active_p = state->custom_mode ? state->typed_prompt : (state->prompt_samples ? state->prompt_samples[state->test_index] : "Write a C function");
    DrawTextSmooth(state->mainFont, TextFormat("\"%s\"", active_p), 35, sampleY + 20, 15, COL_POS);

    // Draw Synapses
    DrawSynapseWeights(positions[0].x, layer_sizes[0], positions[1].x, layer_sizes[1], 150.0f, 530.0f, 0.5f);
    DrawSynapseWeights(positions[1].x, layer_sizes[1], positions[2].x, layer_sizes[2], 150.0f, 530.0f, 1.5f);
    DrawSynapseWeights(positions[2].x, layer_sizes[2], positions[3].x, layer_sizes[3], 150.0f, 530.0f, 2.5f);

    // Input Feature Names
    const char* featNames[] = { "CodeKW", "Punct", "Len>30", "QWord", "CreatKW", "Safety" };
    const char* classNames[] = { "0: CODE", "1: CREATIVE", "2: QA_GEN", "3: UNSAFE" };

    for (int l = 0; l < num_layers; l++) {
        float x = positions[l].x;
        int neuronsInLayer = layer_sizes[l];
        float* activations = NULL;
        if (l == 0) activations = state->classifier_inputs;
        else if (l == 1) activations = clf->h1_act;
        else if (l == 2) activations = clf->h2_act;
        else activations = clf->out_act;

        for (int n = 0; n < neuronsInLayer; n++) {
            float y = 150.0f + ((530.0f - 150.0f) / (float)(neuronsInLayer + 1)) * (float)(n + 1);
            float activation = activations[n];
            DrawCircle((int)x, (int)y, radius, COL_NODE);
            
            float visual_act = (activation < 0.0f) ? 0.0f : (activation > 1.0f ? 1.0f : activation);
            unsigned char glow = (unsigned char)(visual_act * 255.0f);
            DrawCircle((int)x, (int)y, radius - 2.0f, (Color){ COL_ACTIVATE.r, COL_ACTIVATE.g, COL_ACTIVATE.b, glow }); 
            DrawCircleLines((int)x, (int)y, radius, COL_NODE_OUT);
            
            DrawTextSmooth(state->mainFont, TextFormat("%.2f", activation), (int)x - 14, (int)y - 6, 11, (glow > 80 ? COL_BG : COL_TEXT));
            
            if (l == 0) {
                DrawTextSmooth(state->mainFont, featNames[n], (int)x - 65, (int)y - 6, 11, COL_TEXT_DIM);
            }
            if (l == num_layers - 1) {
                Color c = (n == 3 && activation > 0.3f) ? COL_NEG : (n == (int)state->clf_result.primary_intent ? COL_POS : COL_TEXT);
                DrawTextSmooth(state->mainFont, TextFormat("%s (%.1f%%)", classNames[n], activation * 100.0f), (int)x + 24, (int)y - 6, 12, c);
            }
        }
    }

    // Bottom Auto-Tuning Dashboard Panel
    int bottomY = 550;
    int panelW = networkAreaWidth - 50;
    DrawRectangle(25, bottomY, panelW, 220, COL_SIDEBAR);
    DrawRectangleLines(25, bottomY, panelW, 220, COL_BORDER);

    DrawTextSmooth(state->mainFont, "LLM Dynamic Hyperparameter Tuning & Safety Guardrail:", 35, bottomY + 12, 14, COL_TEXT);
    
    const char* intentStr = "CODE";
    if (state->clf_result.primary_intent == INTENT_CREATIVE) intentStr = "CREATIVE WRITING";
    else if (state->clf_result.primary_intent == INTENT_QA_GENERAL) intentStr = "GENERAL Q&A";
    else if (state->clf_result.primary_intent == INTENT_UNSAFE) intentStr = "UNSAFE / MALICIOUS";

    DrawTextSmooth(state->mainFont, TextFormat("Detected Intent: %s", intentStr), 35, bottomY + 40, 13, state->clf_result.is_safe ? COL_POS : COL_NEG);
    DrawTextSmooth(state->mainFont, TextFormat("Safety Status: %s", state->clf_result.is_safe ? "PASSED (Safe to Execute)" : "BLOCKED (Harmful Content Detected)"), 35, bottomY + 62, 13, state->clf_result.is_safe ? COL_POS : COL_NEG);

    DrawTextSmooth(state->mainFont, TextFormat("Auto-Tuned Temperature: %.2f  (Greedy logic vs Creative sampling)", state->clf_result.recommended_temp), 35, bottomY + 95, 12, COL_TEXT);
    DrawTextSmooth(state->mainFont, TextFormat("Auto-Tuned Top-P:       %.2f  (Nucleus probability cutoff)", state->clf_result.recommended_top_p), 35, bottomY + 115, 12, COL_TEXT);

    DrawTextSmooth(state->mainFont, "[LEFT/RIGHT] Cycle Test Dataset Samples    [TAB] Switch to LLM Transformer", 35, bottomY + 165, 12, COL_TEST);
}

void DrawVisualization(LLMClassifier* clf, TransformerEngine* engine, AppState* state, const float* current_input, const float* current_output) {
    BeginDrawing();
    ClearBackground(COL_BG);

    if (state->mode == MODE_TRANSFORMER_LLM) {
        DrawTransformerVisualization(engine, state);
    } else {
        DrawClassifierVisualization(clf, state);
    }

    // --- Right Sidebar UI (x: 850..1200) ---
    int uiX = state->screenWidth - 350;
    DrawRectangle(uiX, 0, 350, state->screenHeight, COL_SIDEBAR);
    DrawLine(uiX, 0, uiX, state->screenHeight, COL_BORDER);

    DrawTextSmooth(state->mainFont, "> NEURAL-C FRAMEWORK", uiX + 20, 25, 18, COL_TEXT);
    DrawTextSmooth(state->mainFont, TextFormat("Active Mode: %s", state->mode == MODE_TRANSFORMER_LLM ? "LLM TRANSFORMER" : "LLM CLASSIFIER HEAD"), uiX + 20, 50, 13, COL_POS);

    DrawTextSmooth(state->mainFont, "--- Hardware Compute ---", uiX + 20, 85, 15, COL_TEXT_DIM);
    if (state->using_gpu) {
        DrawTextSmooth(state->mainFont, TextFormat("GPU: %s", state->compute_device_name), uiX + 20, 110, 13, COL_POS);
        DrawTextSmooth(state->mainFont, "Backend: Apple Metal GEMM", uiX + 20, 128, 13, COL_TEXT);
    } else {
        DrawTextSmooth(state->mainFont, "GPU: Disabled (CPU Fallback)", uiX + 20, 110, 13, COL_NEG);
    }

    DrawTextSmooth(state->mainFont, "--- Mode Switch ---", uiX + 20, 165, 15, COL_TEXT_DIM);
    DrawTextSmooth(state->mainFont, "Press [TAB] to toggle visualizer:", uiX + 20, 188, 13, COL_TEXT);
    DrawTextSmooth(state->mainFont, "1. LLM Transformer Architecture", uiX + 20, 208, 13, state->mode == MODE_TRANSFORMER_LLM ? COL_POS : COL_TEXT_DIM);
    DrawTextSmooth(state->mainFont, "2. LLM Prompt Router Head", uiX + 20, 226, 13, state->mode == MODE_LLM_CLASSIFIER ? COL_POS : COL_TEXT_DIM);

    if (state->mode == MODE_TRANSFORMER_LLM) {
        DrawTextSmooth(state->mainFont, "--- LLM Engine Config ---", uiX + 20, 265, 15, COL_TEXT_DIM);
        DrawTextSmooth(state->mainFont, "Model: Transformer Decoder", uiX + 20, 288, 13, COL_TEXT);
        DrawTextSmooth(state->mainFont, TextFormat("Layers: %d | Heads: %d (GQA)", engine->config.n_layers, engine->config.n_heads), uiX + 20, 308, 13, COL_TEXT_DIM);
        DrawTextSmooth(state->mainFont, TextFormat("Context Window: %d tokens", engine->config.seq_len), uiX + 20, 328, 13, COL_TEXT_DIM);
        DrawTextSmooth(state->mainFont, TextFormat("Temp: %.2f | Top-p: %.2f", state->temperature, state->top_p), uiX + 20, 348, 13, COL_TEXT);

        // --- Model Selection Dropdown Box ---
        DrawTextSmooth(state->mainFont, "--- Select Model Checkpoint ---", uiX + 20, 380, 15, COL_TEXT_DIM);
        
        int comboX = uiX + 20;
        int comboY = 405;
        int comboW = 310;
        int comboH = 32;

        const char* current_name = "No Model Available";
        if (state->num_available_models > 0 && state->current_model_index >= 0 && state->current_model_index < state->num_available_models) {
            current_name = state->available_models[state->current_model_index];
            const char* slash = strrchr(current_name, '/');
            if (slash) current_name = slash + 1;
        }

        Vector2 mousePos = GetMousePosition();
        bool mouse_on_combo = (mousePos.x >= comboX && mousePos.x <= comboX + comboW && mousePos.y >= comboY && mousePos.y <= comboY + comboH);

        DrawRectangle(comboX, comboY, comboW, comboH, mouse_on_combo ? (Color){ 32, 32, 32, 255 } : COL_BG);
        DrawRectangleLines(comboX, comboY, comboW, comboH, state->model_dropdown_open ? COL_POS : COL_BORDER);
        DrawTextSmooth(state->mainFont, TextFormat("%-28.28s", current_name), comboX + 10, comboY + 8, 12, COL_TEXT);
        DrawTextSmooth(state->mainFont, state->model_dropdown_open ? "[^]" : "[v]", comboX + comboW - 26, comboY + 8, 12, COL_POS);

        if (mouse_on_combo && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            state->model_dropdown_open = !state->model_dropdown_open;
        }

        // --- Action Buttons ---
        int btnY = 450;
        int btnW = 148;
        int btnH = 32;

        // Button 1: [ Run Prompt ]
        int btn1X = uiX + 20;
        bool mouse_on_btn1 = (mousePos.x >= btn1X && mousePos.x <= btn1X + btnW && mousePos.y >= btnY && mousePos.y <= btnY + btnH);
        DrawRectangle(btn1X, btnY, btnW, btnH, mouse_on_btn1 ? (Color){ 45, 120, 45, 255 } : (Color){ 25, 80, 25, 255 });
        DrawRectangleLines(btn1X, btnY, btnW, btnH, COL_POS);
        DrawTextSmooth(state->mainFont, "[>] RUN PROMPT", btn1X + 16, btnY + 8, 12, COL_TEXT);
        if (mouse_on_btn1 && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            state->request_generate = true;
        }

        // Button 2: [ Next Preset ]
        int btn2X = uiX + 182;
        bool mouse_on_btn2 = (mousePos.x >= btn2X && mousePos.x <= btn2X + btnW && mousePos.y >= btnY && mousePos.y <= btnY + btnH);
        DrawRectangle(btn2X, btnY, btnW, btnH, mouse_on_btn2 ? (Color){ 60, 60, 60, 255 } : COL_BG);
        DrawRectangleLines(btn2X, btnY, btnW, btnH, COL_BORDER);
        DrawTextSmooth(state->mainFont, "[>>] NEXT PRESET", btn2X + 14, btnY + 8, 12, COL_TEXT);
        if (mouse_on_btn2 && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            state->request_preset_next = true;
        }

        // Button 3: [ Switch View ]
        int btn3Y = 492;
        int btn3W = 310;
        bool mouse_on_btn3 = (mousePos.x >= comboX && mousePos.x <= comboX + btn3W && mousePos.y >= btn3Y && mousePos.y <= btn3Y + btnH);
        DrawRectangle(comboX, btn3Y, btn3W, btnH, mouse_on_btn3 ? (Color){ 50, 40, 70, 255 } : (Color){ 30, 25, 45, 255 });
        DrawRectangleLines(comboX, btn3Y, btn3W, btnH, COL_ATTN_HI);
        DrawTextSmooth(state->mainFont, "[<>] SWITCH VISUALIZER MODE [TAB]", comboX + 25, btn3Y + 8, 12, COL_TEXT);
        if (mouse_on_btn3 && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            state->request_mode_toggle = true;
        }

        // Dropdown Items Menu overlay (draws on top of everything)
        if (state->model_dropdown_open) {
            int itemY = comboY + comboH + 2;
            int total_menu_h = state->num_available_models * comboH;
            DrawRectangle(comboX, itemY, comboW, total_menu_h, (Color){ 24, 24, 24, 250 });
            DrawRectangleLines(comboX, itemY, comboW, total_menu_h, COL_POS);

            for (int m = 0; m < state->num_available_models; m++) {
                int curr_item_y = itemY + (m * comboH);
                bool mouse_on_item = (mousePos.x >= comboX && mousePos.x <= comboX + comboW && mousePos.y >= curr_item_y && mousePos.y < curr_item_y + comboH);
                
                if (mouse_on_item) {
                    DrawRectangle(comboX + 1, curr_item_y, comboW - 2, comboH, (Color){ 50, 70, 50, 255 });
                }

                const char* m_name = state->available_models[m];
                const char* slash = strrchr(m_name, '/');
                if (slash) m_name = slash + 1;

                Color c = (m == state->current_model_index) ? COL_POS : (mouse_on_item ? COL_ACTIVATE : COL_TEXT_DIM);
                DrawTextSmooth(state->mainFont, TextFormat("%s %-25.25s", (m == state->current_model_index) ? "*" : " ", m_name), comboX + 10, curr_item_y + 8, 12, c);

                if (mouse_on_item && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    state->pending_model_index = m;
                    state->request_model_change = true;
                    state->model_dropdown_open = false;
                }
            }
        }

    } else {
        DrawTextSmooth(state->mainFont, "--- Classifier Architecture ---", uiX + 20, 265, 15, COL_TEXT_DIM);
        DrawTextSmooth(state->mainFont, "Structure: 6 -> 16 -> 12 -> 4", uiX + 20, 288, 13, COL_TEXT);
        DrawTextSmooth(state->mainFont, "Loss: Cross-Entropy Softmax", uiX + 20, 308, 13, COL_TEXT_DIM);
        DrawTextSmooth(state->mainFont, "Task: Intent & Safety Routing", uiX + 20, 328, 13, COL_POS);

        int comboX = uiX + 20;
        int btn3Y = 492;
        int btn3W = 310;
        int btnH = 32;
        Vector2 mousePos = GetMousePosition();
        bool mouse_on_btn3 = (mousePos.x >= comboX && mousePos.x <= comboX + btn3W && mousePos.y >= btn3Y && mousePos.y <= btn3Y + btnH);
        DrawRectangle(comboX, btn3Y, btn3W, btnH, mouse_on_btn3 ? (Color){ 50, 40, 70, 255 } : (Color){ 30, 25, 45, 255 });
        DrawRectangleLines(comboX, btn3Y, btn3W, btnH, COL_ATTN_HI);
        DrawTextSmooth(state->mainFont, "[<>] SWITCH VISUALIZER MODE [TAB]", comboX + 25, btn3Y + 8, 12, COL_TEXT);
        if (mouse_on_btn3 && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            state->request_mode_toggle = true;
        }
    }

    EndDrawing();
}
