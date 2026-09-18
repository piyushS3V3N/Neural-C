#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

typedef struct {
    char *str;
    int id;
    float score;
} TokenItem;

// Minimal open-addressing string->int map (tokenizer index internals)
typedef struct { char *key; int val; unsigned char used; } TMapEntry;
typedef struct { TMapEntry *tab; size_t cap; size_t n; } TMap;

typedef struct {
    char **vocab;
    float *vocab_scores;
    int vocab_size;
    int max_token_length;
    unsigned char byte_pieces[512][2]; // Quick lookup for raw single-byte tokens
    
    // Special token IDs
    int bos_id;
    int eos_id;
    int pad_id;

    // GGUF tokenizer metadata (drives correct encode path)
    char model[32];          // e.g. "gpt2", "llama", "t5"
    char pre[32];            // e.g. "qwen2", "llama3", "gpt-2"
    bool add_bos_token;      // tokenizer.ggml.add_bos_token (Qwen: false)
    bool add_eos_token;      // tokenizer.ggml.add_eos_token
    int merge_count;         // BPE merge lines declared by the file
    char chat_template[2048];

    // Merge-driven (GPT-2 style) encoder state. Empty unless merges loaded.
    char **merge_left;       // left piece per accepted merge line
    char **merge_right;      // right piece per accepted merge line
    int n_merges;            // accepted merges stored
    int merge_cap;
    int *special_ids;        // vocab ids usable as atomic splits (<|..|>, bos/eos/pad)
    int n_specials;
    bool index_dirty;        // vocab/special index needs rebuild
    TMap vocab_map;          // token string -> id
    TMap merge_map;          // "left\x1Fright" -> rank
} Tokenizer;

// Core Tokenizer API
Tokenizer* create_tokenizer(int vocab_size);
void free_tokenizer(Tokenizer* tokenizer);

// Vocabulary Population
bool set_tokenizer_entry(Tokenizer* tokenizer, int id, const char* str, float score);
void set_tokenizer_special_tokens(Tokenizer* tokenizer, int bos_id, int eos_id, int pad_id);

// Merge table (GPT-2 style BPE ranks). rank = order of accepted lines.
bool tokenizer_add_merge(Tokenizer* tokenizer, const char* left, const char* right);
void tokenizer_build_index(Tokenizer* tokenizer);
// Fast vocab lookup via index (builds index on first use). Returns id or -1.
int tokenizer_find_token(Tokenizer* tokenizer, const char* str);

// Encoding (Text -> Tokens)
int encode(Tokenizer* tokenizer, const char* text, bool add_bos, bool add_eos, int* tokens, int max_tokens);

// Decoding (Token -> Text)
const char* decode(Tokenizer* tokenizer, int prev_token, int token);

// Chat template renderer
bool render_chat_template(const Tokenizer* tokenizer, const char* prompt, char* out, size_t out_size);

#endif // TOKENIZER_H
