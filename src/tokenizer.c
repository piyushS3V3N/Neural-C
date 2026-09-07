#include "tokenizer.h"

Tokenizer* create_tokenizer(int vocab_size) {
    Tokenizer* t = (Tokenizer*)malloc(sizeof(Tokenizer));
    if (!t) return NULL;
    
    t->vocab_size = vocab_size;
    t->max_token_length = 0;
    t->bos_id = 1;
    t->eos_id = 2;
    t->pad_id = 0;
    
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    
    for (int i = 0; i < vocab_size; i++) {
        t->vocab[i] = NULL;
        t->vocab_scores[i] = 0.0f;
    }
    
    // Initialize byte pieces table for raw bytes <0x00> to <0xFF>
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i][0] = (unsigned char)i;
        t->byte_pieces[i][1] = '\0';
    }
    
    return t;
}

void free_tokenizer(Tokenizer* t) {
    if (!t) return;
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) {
            if (t->vocab[i]) free(t->vocab[i]);
        }
        free(t->vocab);
    }
    if (t->vocab_scores) free(t->vocab_scores);
    free(t);
}

bool set_tokenizer_entry(Tokenizer* t, int id, const char* str, float score) {
    if (!t || id < 0 || id >= t->vocab_size) return false;
    
    if (t->vocab[id]) free(t->vocab[id]);
    t->vocab[id] = strdup(str);
    t->vocab_scores[id] = score;
    
    int len = (int)strlen(str);
    if (len > t->max_token_length) {
        t->max_token_length = len;
    }
    return true;
}

void set_tokenizer_special_tokens(Tokenizer* t, int bos_id, int eos_id, int pad_id) {
    if (!t) return;
    t->bos_id = bos_id;
    t->eos_id = eos_id;
    t->pad_id = pad_id;
}

static int find_vocab_token(Tokenizer* t, const char* str) {
    for (int i = 0; i < t->vocab_size; i++) {
        if (t->vocab[i] && strcmp(t->vocab[i], str) == 0) {
            return i;
        }
    }
    return -1;
}

int encode(Tokenizer* t, const char* text, bool add_bos, bool add_eos, int* tokens, int max_tokens) {
    if (!t || !text || !tokens || max_tokens <= 0) return 0;
    
    int n_tokens = 0;
    
    if (add_bos && n_tokens < max_tokens) {
        tokens[n_tokens++] = t->bos_id;
    }
    
    if (text[0] == '\0') {
        if (add_eos && n_tokens < max_tokens) {
            tokens[n_tokens++] = t->eos_id;
        }
        return n_tokens;
    }

    int str_len = (int)strlen(text);
    char buf[1024];
    
    // Initial tokenization: split into individual byte/character tokens
    int num_chars = 0;
    int *char_tokens = (int*)malloc(str_len * sizeof(int));
    
    for (int i = 0; i < str_len; i++) {
        int id = -1;
        if (text[i] == ' ') {
            id = find_vocab_token(t, "\xc4\xa0"); // Tiktoken / GPT-2 / Qwen BPE space Ġ
            if (id == -1) id = find_vocab_token(t, "\xe2\x96\x81"); // SentencePiece   space
        }
        if (id == -1) {
            char single_char[2] = { text[i], '\0' };
            id = find_vocab_token(t, single_char);
        }
        if (id == -1) {
            snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned char)text[i]);
            id = find_vocab_token(t, buf);
        }
        if (id == -1) id = t->pad_id;
        char_tokens[num_chars++] = id;
    }

    // Iterative BPE merging over adjacent token spans (from 2 up to num_chars)
    while (num_chars > 1) {
        float best_score = -1e9f;
        int best_id = -1;
        int best_start = -1;
        int best_span_len = 0;

        for (int span_len = 2; span_len <= num_chars; span_len++) {
            for (int i = 0; i <= num_chars - span_len; i++) {
                buf[0] = '\0';
                bool valid = true;
                for (int k = 0; k < span_len; k++) {
                    int tok_id = char_tokens[i + k];
                    if (tok_id < 0 || !t->vocab[tok_id]) {
                        valid = false;
                        break;
                    }
                    strcat(buf, t->vocab[tok_id]);
                }
                if (!valid) continue;

                int id = find_vocab_token(t, buf);
                if (id != -1 && t->vocab_scores[id] > best_score) {
                    best_score = t->vocab_scores[id];
                    best_id = id;
                    best_start = i;
                    best_span_len = span_len;
                }
            }
        }

        if (best_start == -1) break; // No more merges possible

        // Apply merge: replace char_tokens[best_start ... best_start + best_span_len - 1] with best_id
        char_tokens[best_start] = best_id;
        int shift = best_span_len - 1;
        for (int i = best_start + 1; i < num_chars - shift; i++) {
            char_tokens[i] = char_tokens[i + shift];
        }
        num_chars -= shift;
    }

    // Append merged tokens to output array
    for (int i = 0; i < num_chars && n_tokens < max_tokens; i++) {
        tokens[n_tokens++] = char_tokens[i];
    }
    
    free(char_tokens);
    
    if (add_eos && n_tokens < max_tokens) {
        tokens[n_tokens++] = t->eos_id;
    }
    
    return n_tokens;
}

const char* decode(Tokenizer* t, int prev_token, int token) {
    if (!t || token < 0 || token >= t->vocab_size) return "";
    const char* str = t->vocab[token];
    if (!str) return "";
    
    // Handle raw byte representation formatting like <0x0A> -> \n
    if (str[0] == '<' && str[1] == '0' && str[2] == 'x' && strlen(str) == 6 && str[5] == '>') {
        unsigned int byte_val;
        if (sscanf(str, "<0x%02X>", &byte_val) == 1) {
            return (const char*)t->byte_pieces[byte_val & 0xFF];
        }
    }

    // Clean up BPE / SentencePiece space and newline symbols
    static char clean_buf[1024];
    size_t out_idx = 0;
    size_t len = strlen(str);
    for (size_t i = 0; i < len; ) {
        // SentencePiece   (0xE2 0x96 0x81) -> space
        if (i + 2 < len && (unsigned char)str[i] == 0xE2 && (unsigned char)str[i+1] == 0x96 && (unsigned char)str[i+2] == 0x81) {
            clean_buf[out_idx++] = ' ';
            i += 3;
        }
        // Tiktoken / GPT-2 BPE Ġ (0xC4 0xA0) -> space
        else if (i + 1 < len && (unsigned char)str[i] == 0xC4 && (unsigned char)str[i+1] == 0xA0) {
            clean_buf[out_idx++] = ' ';
            i += 2;
        }
        // Tiktoken / GPT-2 BPE Ċ (0xC4 0x8A) -> newline
        else if (i + 1 < len && (unsigned char)str[i] == 0xC4 && (unsigned char)str[i+1] == 0x8A) {
            clean_buf[out_idx++] = '\n';
            i += 2;
        }
        // Regular character
        else {
            clean_buf[out_idx++] = str[i++];
        }
        if (out_idx >= sizeof(clean_buf) - 1) break;
    }
    clean_buf[out_idx] = '\0';
    return clean_buf;
}
