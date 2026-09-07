#ifndef GGUF_PARSER_H
#define GGUF_PARSER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "model.h"
#include "tensor.h"
#include "tokenizer.h"

#define GGUF_MAGIC 0x46554747 // "GGUF" in Little Endian ASCII

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12
} GGUFValueType;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
} GGUFHeader;

typedef struct {
    char name[64];
    uint32_t n_dims;
    uint64_t shape[4];
    uint32_t ggml_type;
    uint64_t offset;
    void* data_ptr;
} GGUFTensorInfo;

typedef struct {
    int fd;
    size_t file_size;
    uint8_t* mapped_data;
    
    GGUFHeader header;
    GGUFTensorInfo* tensors;
    uint64_t tensor_data_offset;
    uint32_t alignment;

    Config config;
} GGUFContext;

// GGUF Parser API
GGUFContext* open_gguf_file(const char* filepath);
void close_gguf_file(GGUFContext* ctx);

// Model & Tokenizer Loader Helpers
bool load_gguf_tokenizer(GGUFContext* ctx, Tokenizer* tokenizer);
bool load_gguf_weights(GGUFContext* ctx, TransformerWeights* weights);

#endif // GGUF_PARSER_H
