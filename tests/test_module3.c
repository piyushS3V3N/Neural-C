#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "gguf_parser.h"

void create_mock_gguf_file(const char* filepath) {
    FILE* f = fopen(filepath, "wb");
    assert(f != NULL);

    // 1. Write Header
    GGUFHeader h = {
        .magic = GGUF_MAGIC,
        .version = 3,
        .tensor_count = 1,
        .metadata_kv_count = 1
    };
    fwrite(&h, sizeof(GGUFHeader), 1, f);

    // 2. Write Metadata KV: "general.alignment" = uint32_t(32)
    uint64_t key_len = strlen("general.alignment");
    fwrite(&key_len, sizeof(uint64_t), 1, f);
    fwrite("general.alignment", 1, key_len, f);

    uint32_t val_type = GGUF_TYPE_UINT32;
    fwrite(&val_type, sizeof(uint32_t), 1, f);
    uint32_t align_val = 32;
    fwrite(&align_val, sizeof(uint32_t), 1, f);

    // 3. Write Tensor Info Header: "token_embd.weight" (2D, FP32, offset 0)
    uint64_t tensor_name_len = strlen("token_embd.weight");
    fwrite(&tensor_name_len, sizeof(uint64_t), 1, f);
    fwrite("token_embd.weight", 1, tensor_name_len, f);

    uint32_t n_dims = 2;
    fwrite(&n_dims, sizeof(uint32_t), 1, f);
    uint64_t shape[2] = {128, 64};
    fwrite(shape, sizeof(uint64_t), 2, f);
    uint32_t ggml_type = 0; // FP32
    fwrite(&ggml_type, sizeof(uint32_t), 1, f);
    uint64_t offset = 0;
    fwrite(&offset, sizeof(uint64_t), 1, f);

    // 4. Pad alignment to 32 bytes
    long pos = ftell(f);
    long aligned_pos = (pos + 31) & ~31;
    for (long i = pos; i < aligned_pos; i++) fputc(0, f);

    // 5. Write Tensor Binary Payload (128 x 64 float values)
    float dummy_weights[128 * 64];
    for (int i = 0; i < 128 * 64; i++) dummy_weights[i] = (float)i * 0.001f;
    fwrite(dummy_weights, sizeof(float), 128 * 64, f);

    fclose(f);
}

int main() {
    printf("=========================================\n");
    printf("  Module 3 Test: GGUF Parser & mmap      \n");
    printf("=========================================\n");

    const char* mock_filepath = "mock_model.gguf";
    create_mock_gguf_file(mock_filepath);
    printf("[PASS] Created Mock GGUF Binary File '%s'\n", mock_filepath);

    // Test Opening & Memory Mapping GGUF Binary
    GGUFContext* ctx = open_gguf_file(mock_filepath);
    assert(ctx != NULL);
    assert(ctx->header.magic == GGUF_MAGIC);
    assert(ctx->header.version == 3);
    assert(ctx->header.tensor_count == 1);
    assert(ctx->alignment == 32);
    printf("[PASS] GGUF Magic Header & Metadata KV Parser\n");

    // Test Tensor Info Mapping
    assert(strcmp(ctx->tensors[0].name, "token_embd.weight") == 0);
    assert(ctx->tensors[0].n_dims == 2);
    assert(ctx->tensors[0].shape[0] == 128);
    assert(ctx->tensors[0].shape[1] == 64);
    assert(ctx->tensors[0].data_ptr != NULL);
    
    // Verify mapped binary data pointer contents
    float* mapped_weights = (float*)ctx->tensors[0].data_ptr;
    assert(mapped_weights[0] == 0.0f);
    assert(mapped_weights[10] == 10 * 0.001f);
    printf("[PASS] Zero-Copy mmap Tensor Pointer Offset Mapping\n");

    // Cleanup
    close_gguf_file(ctx);
    unlink(mock_filepath);
    printf("[PASS] Memory Unmap & File Cleanup\n");

    printf("\n>>> MODULE 3 (GGUF MODEL PARSER & MMAP ENGINE) PASSED ALL TESTS SUCCESSFULLY! <<<\n");
    return 0;
}
