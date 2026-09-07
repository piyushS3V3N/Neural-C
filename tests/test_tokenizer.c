#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "tokenizer.h"

int main() {
    printf("=========================================\n");
    printf("  Module 1 Test: Tokenizer & Vocabulary  \n");
    printf("=========================================\n");

    // 1. Create Tokenizer
    int vocab_size = 10;
    Tokenizer* t = create_tokenizer(vocab_size);
    assert(t != NULL);
    assert(t->vocab_size == 10);
    printf("[PASS] Tokenizer initialization\n");

    // 2. Populate Vocabulary
    set_tokenizer_entry(t, 0, "<pad>", 0.0f);
    set_tokenizer_entry(t, 1, "<bos>", 0.0f);
    set_tokenizer_entry(t, 2, "<eos>", 0.0f);
    set_tokenizer_entry(t, 3, "H", 1.0f);
    set_tokenizer_entry(t, 4, "e", 1.0f);
    set_tokenizer_entry(t, 5, "l", 1.0f);
    set_tokenizer_entry(t, 6, "o", 1.0f);
    set_tokenizer_entry(t, 7, "He", 5.0f);   // Merged token with high score
    set_tokenizer_entry(t, 8, "llo", 10.0f); // Merged token with higher score
    set_tokenizer_entry(t, 9, "Hello", 20.0f);// Full word token with highest score
    set_tokenizer_special_tokens(t, 1, 2, 0);

    printf("[PASS] Vocabulary population\n");

    // 3. Test Encoding with BPE Merges
    int tokens[32];
    int n_tokens = encode(t, "Hello", true, true, tokens, 32);
    
    printf("Encoded 'Hello' -> Token IDs: [ ");
    for (int i = 0; i < n_tokens; i++) {
        printf("%d ", tokens[i]);
    }
    printf("]\n");

    assert(n_tokens == 3);
    assert(tokens[0] == 1); // <bos>
    assert(tokens[1] == 9); // "Hello" (highest score merged token)
    assert(tokens[2] == 2); // <eos>
    printf("[PASS] BPE Encoding & Greedy Merge\n");

    // 4. Test Decoding
    char decoded_str[128] = {0};
    for (int i = 0; i < n_tokens; i++) {
        const char* piece = decode(t, i > 0 ? tokens[i-1] : -1, tokens[i]);
        strcat(decoded_str, piece);
    }
    printf("Decoded Tokens -> '%s'\n", decoded_str);
    assert(strcmp(decoded_str, "<bos>Hello<eos>") == 0);
    printf("[PASS] Decoding Token IDs back to String\n");

    // 5. Cleanup
    free_tokenizer(t);
    printf("[PASS] Memory Cleanup\n");
    printf("\n>>> MODULE 1 (TOKENIZER & VOCABULARY ENGINE) PASSED ALL TESTS SUCCESSFULLY! <<<\n");
    return 0;
}
