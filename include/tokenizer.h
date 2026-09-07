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
} Tokenizer;

// Core Tokenizer API
Tokenizer* create_tokenizer(int vocab_size);
void free_tokenizer(Tokenizer* tokenizer);

// Vocabulary Population
bool set_tokenizer_entry(Tokenizer* tokenizer, int id, const char* str, float score);
void set_tokenizer_special_tokens(Tokenizer* tokenizer, int bos_id, int eos_id, int pad_id);

// Encoding (Text -> Tokens)
int encode(Tokenizer* tokenizer, const char* text, bool add_bos, bool add_eos, int* tokens, int max_tokens);

// Decoding (Token -> Text)
const char* decode(Tokenizer* tokenizer, int prev_token, int token);

#endif // TOKENIZER_H
