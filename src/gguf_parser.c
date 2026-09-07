#include "gguf_parser.h"

static uint64_t align_offset(uint64_t offset, uint32_t alignment) {
    if (alignment == 0) return offset;
    return (offset + alignment - 1) & ~(uint64_t)(alignment - 1);
}

static uint64_t read_gguf_string(const uint8_t* data, uint64_t offset, char* out_buf, size_t buf_size) {
    uint64_t len = *(uint64_t*)(data + offset);
    offset += sizeof(uint64_t);
    
    size_t copy_len = len < (buf_size - 1) ? len : (buf_size - 1);
    memcpy(out_buf, data + offset, copy_len);
    out_buf[copy_len] = '\0';
    
    return offset + len;
}

GGUFContext* open_gguf_file(const char* filepath) {
    if (!filepath) return NULL;

    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        printf("[GGUF] ERROR: Failed to open file '%s'\n", filepath);
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }
    size_t file_size = st.st_size;

    uint8_t* mapped = (uint8_t*)mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        printf("[GGUF] ERROR: Failed to mmap file of size %zu bytes\n", file_size);
        close(fd);
        return NULL;
    }

    GGUFContext* ctx = (GGUFContext*)calloc(1, sizeof(GGUFContext));
    ctx->fd = fd;
    ctx->file_size = file_size;
    ctx->mapped_data = mapped;
    ctx->alignment = 32; // Default GGUF alignment

    uint64_t offset = 0;
    memcpy(&ctx->header, mapped + offset, sizeof(GGUFHeader));
    offset += sizeof(GGUFHeader);

    if (ctx->header.magic != GGUF_MAGIC) {
        printf("[GGUF] ERROR: Invalid magic header 0x%08X (expected GGUF 0x46554747)\n", ctx->header.magic);
        close_gguf_file(ctx);
        return NULL;
    }

    printf("[GGUF] Loaded Header: Version %u, Tensors: %llu, Metadata KVs: %llu\n",
        ctx->header.version, (unsigned long long)ctx->header.tensor_count, (unsigned long long)ctx->header.metadata_kv_count);

    // Read Metadata Key-Value pairs
    char key_buf[256];
    for (uint64_t i = 0; i < ctx->header.metadata_kv_count; i++) {
        offset = read_gguf_string(mapped, offset, key_buf, sizeof(key_buf));
        uint32_t val_type = *(uint32_t*)(mapped + offset);
        offset += sizeof(uint32_t);

        if (val_type == GGUF_TYPE_UINT32) {
            uint32_t val = *(uint32_t*)(mapped + offset);
            offset += sizeof(uint32_t);
            if (strcmp(key_buf, "general.alignment") == 0) ctx->alignment = val;
            else if (strstr(key_buf, "embedding_length")) ctx->config.dim = val;
            else if (strstr(key_buf, "feed_forward_length")) ctx->config.hidden_dim = val;
            else if (strstr(key_buf, "block_count")) ctx->config.n_layers = val;
            else if (strstr(key_buf, "attention.head_count_kv")) ctx->config.n_kv_heads = val;
            else if (strstr(key_buf, "attention.head_count")) ctx->config.n_heads = val;
            else if (strstr(key_buf, "context_length")) ctx->config.seq_len = val;
            else if (strstr(key_buf, "vocab_size")) ctx->config.vocab_size = val;
        } else if (val_type == GGUF_TYPE_UINT64) {
            uint64_t val = *(uint64_t*)(mapped + offset);
            offset += sizeof(uint64_t);
            if (strstr(key_buf, "embedding_length")) ctx->config.dim = (int)val;
            else if (strstr(key_buf, "feed_forward_length")) ctx->config.hidden_dim = (int)val;
            else if (strstr(key_buf, "block_count")) ctx->config.n_layers = (int)val;
            else if (strstr(key_buf, "attention.head_count_kv")) ctx->config.n_kv_heads = (int)val;
            else if (strstr(key_buf, "attention.head_count")) ctx->config.n_heads = (int)val;
            else if (strstr(key_buf, "context_length")) ctx->config.seq_len = (int)val;
            else if (strstr(key_buf, "vocab_size")) ctx->config.vocab_size = (int)val;
        } else if (val_type == GGUF_TYPE_INT32) {
            int32_t val = *(int32_t*)(mapped + offset);
            offset += sizeof(int32_t);
            if (strstr(key_buf, "embedding_length")) ctx->config.dim = val;
            else if (strstr(key_buf, "feed_forward_length")) ctx->config.hidden_dim = val;
            else if (strstr(key_buf, "block_count")) ctx->config.n_layers = val;
            else if (strstr(key_buf, "attention.head_count_kv")) ctx->config.n_kv_heads = val;
            else if (strstr(key_buf, "attention.head_count")) ctx->config.n_heads = val;
            else if (strstr(key_buf, "context_length")) ctx->config.seq_len = val;
            else if (strstr(key_buf, "vocab_size")) ctx->config.vocab_size = val;
        } else if (val_type == GGUF_TYPE_FLOAT32) {
            float val = *(float*)(mapped + offset);
            offset += sizeof(float);
            if (strstr(key_buf, "layer_norm_rms_epsilon")) ctx->config.norm_eps = val;
            else if (strstr(key_buf, "rope.freq_base") || strstr(key_buf, "rope_freq_base")) ctx->config.rope_freq_base = val;
        } else if (val_type == 13) { // GGUF_TYPE_FLOAT64
            double val = *(double*)(mapped + offset);
            offset += sizeof(double);
            if (strstr(key_buf, "rope.freq_base") || strstr(key_buf, "rope_freq_base")) ctx->config.rope_freq_base = (float)val;
        } else if (val_type == GGUF_TYPE_BOOL) {
            offset += sizeof(uint8_t);
        } else if (val_type == GGUF_TYPE_STRING) {
            char str_val[256];
            offset = read_gguf_string(mapped, offset, str_val, sizeof(str_val));
        } else if (val_type == GGUF_TYPE_ARRAY) {
            uint32_t arr_type = *(uint32_t*)(mapped + offset);
            offset += sizeof(uint32_t);
            uint64_t arr_len = *(uint64_t*)(mapped + offset);
            offset += sizeof(uint64_t);

            if (strstr(key_buf, "tokenizer.ggml.tokens")) {
                ctx->config.vocab_size = (int)arr_len;
            }

            for (uint64_t k = 0; k < arr_len; k++) {
                if (arr_type == GGUF_TYPE_STRING) {
                    char dummy[256];
                    offset = read_gguf_string(mapped, offset, dummy, sizeof(dummy));
                } else if (arr_type == GGUF_TYPE_FLOAT32) offset += sizeof(float);
                else if (arr_type == GGUF_TYPE_INT32 || arr_type == GGUF_TYPE_UINT32) offset += sizeof(uint32_t);
                else if (arr_type == GGUF_TYPE_INT8 || arr_type == GGUF_TYPE_UINT8) offset += sizeof(uint8_t);
            }
        }
    }

    if (ctx->config.rope_freq_base <= 0.0f) ctx->config.rope_freq_base = 10000.0f;
    if (ctx->config.n_kv_heads == 0) ctx->config.n_kv_heads = ctx->config.n_heads;
    if (ctx->config.n_heads > 0 && ctx->config.dim > 0) {
        ctx->config.head_dim = ctx->config.dim / ctx->config.n_heads;
    }

    // Read Tensor Info Headers
    ctx->tensors = (GGUFTensorInfo*)calloc(ctx->header.tensor_count, sizeof(GGUFTensorInfo));
    for (uint64_t i = 0; i < ctx->header.tensor_count; i++) {
        GGUFTensorInfo* info = &ctx->tensors[i];
        offset = read_gguf_string(mapped, offset, info->name, sizeof(info->name));
        info->n_dims = *(uint32_t*)(mapped + offset);
        offset += sizeof(uint32_t);

        for (uint32_t d = 0; d < info->n_dims; d++) {
            info->shape[d] = *(uint64_t*)(mapped + offset);
            offset += sizeof(uint64_t);
        }
        info->ggml_type = *(uint32_t*)(mapped + offset);
        offset += sizeof(uint32_t);
        info->offset = *(uint64_t*)(mapped + offset);
        offset += sizeof(uint64_t);
    }

    ctx->tensor_data_offset = align_offset(offset, ctx->alignment);
    for (uint64_t i = 0; i < ctx->header.tensor_count; i++) {
        ctx->tensors[i].data_ptr = mapped + ctx->tensor_data_offset + ctx->tensors[i].offset;
    }

    printf("[GGUF] Tensor Data Offset: %llu bytes. Alignment: %u bytes\n",
        (unsigned long long)ctx->tensor_data_offset, ctx->alignment);

    return ctx;
}

void close_gguf_file(GGUFContext* ctx) {
    if (!ctx) return;
    if (ctx->mapped_data && ctx->mapped_data != MAP_FAILED) {
        munmap(ctx->mapped_data, ctx->file_size);
    }
    if (ctx->fd >= 0) close(ctx->fd);
    if (ctx->tensors) free(ctx->tensors);
    free(ctx);
}

bool load_gguf_tokenizer(GGUFContext* ctx, Tokenizer* tokenizer) {
    if (!ctx || !tokenizer) return false;
    
    uint8_t* mapped = (uint8_t*)ctx->mapped_data;
    uint64_t offset = sizeof(GGUFHeader);

    char key_buf[256];
    int loaded = 0;

    for (uint64_t i = 0; i < ctx->header.metadata_kv_count; i++) {
        offset = read_gguf_string(mapped, offset, key_buf, sizeof(key_buf));
        uint32_t val_type = *(uint32_t*)(mapped + offset);
        offset += sizeof(uint32_t);

        if (val_type == GGUF_TYPE_ARRAY) {
            uint32_t arr_type = *(uint32_t*)(mapped + offset);
            offset += sizeof(uint32_t);
            uint64_t arr_len = *(uint64_t*)(mapped + offset);
            offset += sizeof(uint64_t);

            if (strcmp(key_buf, "tokenizer.ggml.tokens") == 0) {
                for (uint64_t k = 0; k < arr_len && k < (uint64_t)tokenizer->vocab_size; k++) {
                    char str_val[256];
                    offset = read_gguf_string(mapped, offset, str_val, sizeof(str_val));
                    set_tokenizer_entry(tokenizer, (int)k, str_val, (float)-(int)k);
                    loaded++;
                }
                set_tokenizer_special_tokens(tokenizer, 1, 2, 0);
                printf("[GGUF] Successfully populated %d real vocabulary tokens from GGUF metadata!\n", loaded);
            } else if (strcmp(key_buf, "tokenizer.ggml.scores") == 0 && arr_type == GGUF_TYPE_FLOAT32) {
                for (uint64_t k = 0; k < arr_len && k < (uint64_t)tokenizer->vocab_size; k++) {
                    float score = *(float*)(mapped + offset);
                    offset += sizeof(float);
                    tokenizer->vocab_scores[k] = score;
                }
            } else {
                for (uint64_t k = 0; k < arr_len; k++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        char dummy[256];
                        offset = read_gguf_string(mapped, offset, dummy, sizeof(dummy));
                    } else if (arr_type == GGUF_TYPE_FLOAT32) offset += sizeof(float);
                    else if (arr_type == GGUF_TYPE_INT32 || arr_type == GGUF_TYPE_UINT32) offset += sizeof(uint32_t);
                    else if (arr_type == GGUF_TYPE_INT8 || arr_type == GGUF_TYPE_UINT8) offset += sizeof(uint8_t);
                }
            }
        } else if (val_type == GGUF_TYPE_UINT32 || val_type == GGUF_TYPE_INT32) {
            uint32_t val = *(uint32_t*)(mapped + offset);
            offset += sizeof(uint32_t);
            if (strstr(key_buf, "bos_token_id")) tokenizer->bos_id = (int)val;
            else if (strstr(key_buf, "eos_token_id")) tokenizer->eos_id = (int)val;
            else if (strstr(key_buf, "padding_token_id")) tokenizer->pad_id = (int)val;
        } else if (val_type == GGUF_TYPE_UINT64) offset += sizeof(uint64_t);
        else if (val_type == GGUF_TYPE_FLOAT32) offset += sizeof(float);
        else if (val_type == GGUF_TYPE_BOOL) offset += sizeof(uint8_t);
        else if (val_type == GGUF_TYPE_STRING) {
            char str_val[256];
            offset = read_gguf_string(mapped, offset, str_val, sizeof(str_val));
        }
    }
    return false;
}

static QuantType convert_ggml_type(uint32_t ggml_type) {
    switch (ggml_type) {
        case 0: return QUANT_FP32;
        case 1: return QUANT_FP16;
        case 2: return QUANT_Q4_0;
        case 8: return QUANT_Q8_0;
        default: return QUANT_Q4_0;
    }
}

static void assign_tensor_mapping(Tensor* target, GGUFTensorInfo* t) {
    if (target->data && !target->is_mmap) {
        free(target->data);
        target->data = NULL;
    }
    target->data = t->data_ptr;
    target->type = convert_ggml_type(t->ggml_type);
    target->is_mmap = true;
}

bool load_gguf_weights(GGUFContext* ctx, TransformerWeights* weights) {
    if (!ctx || !weights) return false;

    int loaded_count = 0;
    for (uint64_t i = 0; i < ctx->header.tensor_count; i++) {
        GGUFTensorInfo* t = &ctx->tensors[i];
        
        if (strcmp(t->name, "token_embd.weight") == 0) {
            assign_tensor_mapping(&weights->token_embedding_table, t);
            printf("[GGUF TENSOR] token_embd.weight -> ggml_type: %u, shape: [%llu, %llu]\n", t->ggml_type, (unsigned long long)t->shape[0], (unsigned long long)t->shape[1]);
            loaded_count++;
        } else if (strcmp(t->name, "output_norm.weight") == 0) {
            assign_tensor_mapping(&weights->rms_final_weight, t);
            loaded_count++;
        } else if (strcmp(t->name, "output.weight") == 0) {
            assign_tensor_mapping(&weights->w_cls, t);
            loaded_count++;
        } else {
            int layer_idx = -1;
            if (sscanf(t->name, "blk.%d.", &layer_idx) == 1 && layer_idx >= 0 && layer_idx < ctx->config.n_layers) {
                if (strstr(t->name, "attn_norm.weight")) {
                    assign_tensor_mapping(&weights->rms_att_weight[layer_idx], t);
                } else if (strstr(t->name, "attn_q.weight")) {
                    assign_tensor_mapping(&weights->wq[layer_idx], t);
                } else if (strstr(t->name, "attn_k.weight")) {
                    assign_tensor_mapping(&weights->wk[layer_idx], t);
                } else if (strstr(t->name, "attn_v.weight")) {
                    assign_tensor_mapping(&weights->wv[layer_idx], t);
                } else if (strstr(t->name, "attn_q.bias")) {
                    assign_tensor_mapping(&weights->bq[layer_idx], t);
                } else if (strstr(t->name, "attn_k.bias")) {
                    assign_tensor_mapping(&weights->bk[layer_idx], t);
                } else if (strstr(t->name, "attn_v.bias")) {
                    assign_tensor_mapping(&weights->bv[layer_idx], t);
                } else if (strstr(t->name, "attn_output.weight")) {
                    assign_tensor_mapping(&weights->wo[layer_idx], t);
                } else if (strstr(t->name, "ffn_norm.weight")) {
                    assign_tensor_mapping(&weights->rms_ffn_weight[layer_idx], t);
                } else if (strstr(t->name, "ffn_gate.weight")) {
                    assign_tensor_mapping(&weights->w_gate[layer_idx], t);
                } else if (strstr(t->name, "ffn_up.weight")) {
                    assign_tensor_mapping(&weights->w_up[layer_idx], t);
                } else if (strstr(t->name, "ffn_down.weight")) {
                    assign_tensor_mapping(&weights->w_down[layer_idx], t);
                }
                loaded_count++;
            }
        }
    }

    if (!weights->w_cls.is_mmap) {
        if (weights->w_cls.data && !weights->w_cls.is_mmap) {
            free(weights->w_cls.data);
            weights->w_cls.data = NULL;
        }
        weights->w_cls = weights->token_embedding_table; // Weight-tying fallback
    }

    printf("[GGUF] Zero-Copy mmap mapped %d tensor weights into LLM Engine!\n", loaded_count);
    return true;
}
