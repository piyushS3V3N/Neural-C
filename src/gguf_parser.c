#include "gguf_parser.h"

static uint64_t align_offset(uint64_t offset, uint32_t alignment) {
    if (alignment == 0) return offset;
    return (offset + alignment - 1) & ~(uint64_t)(alignment - 1);
}

static uint64_t read_gguf_string_safe(const uint8_t* data, size_t file_size, uint64_t offset, char* out_buf, size_t buf_size) {
    if (!data || offset + sizeof(uint64_t) > file_size) return 0;
    uint64_t len = *(const uint64_t*)(data + offset);
    offset += sizeof(uint64_t);
    
    if (offset + len > file_size) return 0;
    
    if (out_buf && buf_size > 0) {
        size_t copy_len = len < (buf_size - 1) ? len : (buf_size - 1);
        memcpy(out_buf, data + offset, copy_len);
        out_buf[copy_len] = '\0';
    }
    
    return offset + len;
}

static uint64_t skip_gguf_value(const uint8_t* mapped, size_t file_size, uint64_t offset, uint32_t val_type, const char* key_buf) {
    if (!mapped || offset >= file_size) return 0;

    switch (val_type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            if (offset + 1 > file_size) return 0;
            return offset + 1;

        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            if (offset + 2 > file_size) return 0;
            return offset + 2;

        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            if (offset + 4 > file_size) return 0;
            return offset + 4;

        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            if (offset + 8 > file_size) return 0;
            return offset + 8;

        case GGUF_TYPE_STRING: {
            uint64_t next_off = read_gguf_string_safe(mapped, file_size, offset, NULL, 0);
            return next_off;
        }

        case GGUF_TYPE_ARRAY: {
            if (offset + sizeof(uint32_t) + sizeof(uint64_t) > file_size) return 0;
            uint32_t arr_type = *(const uint32_t*)(mapped + offset);
            offset += sizeof(uint32_t);
            uint64_t arr_len = *(const uint64_t*)(mapped + offset);
            offset += sizeof(uint64_t);

            for (uint64_t k = 0; k < arr_len; k++) {
                if (offset >= file_size) return 0;
                uint64_t next_off = skip_gguf_value(mapped, file_size, offset, arr_type, key_buf);
                if (next_off == 0 || next_off <= offset) return 0;
                offset = next_off;
            }
            return offset;
        }

        default:
            printf("[GGUF] WARN: Unknown val_type %u for key '%s' at offset %llu\n",
                val_type, key_buf ? key_buf : "?", (unsigned long long)offset);
            return 0;
    }
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
    if (!ctx) {
        munmap(mapped, file_size);
        close(fd);
        return NULL;
    }

    ctx->fd = fd;
    ctx->file_size = file_size;
    ctx->mapped_data = mapped;
    ctx->alignment = 32; // Default GGUF alignment

    if (file_size < sizeof(GGUFHeader)) {
        printf("[GGUF] ERROR: File size (%zu) smaller than GGUF header\n", file_size);
        close_gguf_file(ctx);
        return NULL;
    }

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
        uint64_t key_end = read_gguf_string_safe(mapped, file_size, offset, key_buf, sizeof(key_buf));
        if (key_end == 0 || key_end + sizeof(uint32_t) > file_size) {
            printf("[GGUF] WARN: Truncated key at metadata entry %llu\n", (unsigned long long)i);
            break;
        }
        offset = key_end;
        uint32_t val_type = *(const uint32_t*)(mapped + offset);
        uint64_t val_start = offset + sizeof(uint32_t);

        if (val_type == GGUF_TYPE_UINT32) {
            if (val_start + 4 <= file_size) {
                uint32_t val = *(const uint32_t*)(mapped + val_start);
                if (strcmp(key_buf, "general.alignment") == 0) ctx->alignment = val;
                else if (strstr(key_buf, "embedding_length")) ctx->config.dim = (int)val;
                else if (strstr(key_buf, "feed_forward_length")) ctx->config.hidden_dim = (int)val;
                else if (strstr(key_buf, "block_count")) ctx->config.n_layers = (int)val;
                else if (strstr(key_buf, "attention.head_count_kv")) ctx->config.n_kv_heads = (int)val;
                else if (strstr(key_buf, "attention.head_count")) ctx->config.n_heads = (int)val;
                else if (strstr(key_buf, "context_length")) ctx->config.seq_len = (int)val;
                else if (strstr(key_buf, "vocab_size")) ctx->config.vocab_size = (int)val;
            }
        } else if (val_type == GGUF_TYPE_UINT64) {
            if (val_start + 8 <= file_size) {
                uint64_t val = *(const uint64_t*)(mapped + val_start);
                if (strstr(key_buf, "embedding_length")) ctx->config.dim = (int)val;
                else if (strstr(key_buf, "feed_forward_length")) ctx->config.hidden_dim = (int)val;
                else if (strstr(key_buf, "block_count")) ctx->config.n_layers = (int)val;
                else if (strstr(key_buf, "attention.head_count_kv")) ctx->config.n_kv_heads = (int)val;
                else if (strstr(key_buf, "attention.head_count")) ctx->config.n_heads = (int)val;
                else if (strstr(key_buf, "context_length")) ctx->config.seq_len = (int)val;
                else if (strstr(key_buf, "vocab_size")) ctx->config.vocab_size = (int)val;
            }
        } else if (val_type == GGUF_TYPE_INT32) {
            if (val_start + 4 <= file_size) {
                int32_t val = *(const int32_t*)(mapped + val_start);
                if (strstr(key_buf, "embedding_length")) ctx->config.dim = val;
                else if (strstr(key_buf, "feed_forward_length")) ctx->config.hidden_dim = val;
                else if (strstr(key_buf, "block_count")) ctx->config.n_layers = val;
                else if (strstr(key_buf, "attention.head_count_kv")) ctx->config.n_kv_heads = val;
                else if (strstr(key_buf, "attention.head_count")) ctx->config.n_heads = val;
                else if (strstr(key_buf, "context_length")) ctx->config.seq_len = val;
                else if (strstr(key_buf, "vocab_size")) ctx->config.vocab_size = val;
            }
        } else if (val_type == GGUF_TYPE_FLOAT32) {
            if (val_start + 4 <= file_size) {
                float val = *(const float*)(mapped + val_start);
                if (strstr(key_buf, "layer_norm_rms_epsilon")) ctx->config.norm_eps = val;
                else if (strstr(key_buf, "rope.freq_base") || strstr(key_buf, "rope_freq_base")) ctx->config.rope_freq_base = val;
            }
        } else if (val_type == GGUF_TYPE_FLOAT64) {
            if (val_start + 8 <= file_size) {
                double val = *(const double*)(mapped + val_start);
                if (strstr(key_buf, "rope.freq_base") || strstr(key_buf, "rope_freq_base")) ctx->config.rope_freq_base = (float)val;
            }
        } else if (val_type == GGUF_TYPE_ARRAY) {
            if (val_start + 12 <= file_size) {
                uint64_t arr_len = *(const uint64_t*)(mapped + val_start + 4);
                if (strstr(key_buf, "tokenizer.ggml.tokens")) {
                    ctx->config.vocab_size = (int)arr_len;
                }
            }
        }

        uint64_t next_off = skip_gguf_value(mapped, file_size, val_start, val_type, key_buf);
        if (next_off == 0 || next_off <= offset) {
            printf("[GGUF] WARN: Skipping key '%s' (type %u) failed at offset %llu\n",
                key_buf, val_type, (unsigned long long)val_start);
            break;
        }
        offset = next_off;
    }

    if (ctx->config.rope_freq_base <= 0.0f) ctx->config.rope_freq_base = 10000.0f;
    if (ctx->config.n_kv_heads == 0) ctx->config.n_kv_heads = ctx->config.n_heads;
    if (ctx->config.n_heads > 0 && ctx->config.dim > 0) {
        ctx->config.head_dim = ctx->config.dim / ctx->config.n_heads;
    }

    int max_ctx = 8192;
    const char* env_max_ctx = getenv("NEURAL_C_MAX_CTX");
    if (env_max_ctx && atoi(env_max_ctx) > 0) {
        max_ctx = atoi(env_max_ctx);
    }
    if (ctx->config.seq_len > max_ctx) {
        printf("[GGUF] Context length %d exceeds cap (%d); capping seq_len to %d\n", ctx->config.seq_len, max_ctx, max_ctx);
        ctx->config.seq_len = max_ctx;
    }

    // Read Tensor Info Headers
    ctx->tensors = (GGUFTensorInfo*)calloc(ctx->header.tensor_count, sizeof(GGUFTensorInfo));
    if (!ctx->tensors && ctx->header.tensor_count > 0) {
        printf("[GGUF] ERROR: Failed to allocate memory for %llu tensors\n", (unsigned long long)ctx->header.tensor_count);
        close_gguf_file(ctx);
        return NULL;
    }

    for (uint64_t i = 0; i < ctx->header.tensor_count; i++) {
        GGUFTensorInfo* info = &ctx->tensors[i];
        uint64_t name_end = read_gguf_string_safe(mapped, file_size, offset, info->name, sizeof(info->name));
        if (name_end == 0 || name_end + sizeof(uint32_t) > file_size) {
            printf("[GGUF] ERROR: Truncated tensor name header at index %llu\n", (unsigned long long)i);
            close_gguf_file(ctx);
            return NULL;
        }
        offset = name_end;

        info->n_dims = *(const uint32_t*)(mapped + offset);
        offset += sizeof(uint32_t);

        uint32_t read_dims = info->n_dims;
        if (read_dims > 8) {
            printf("[GGUF] ERROR: Invalid n_dims %u for tensor '%s'\n", read_dims, info->name);
            close_gguf_file(ctx);
            return NULL;
        }

        if (offset + (uint64_t)read_dims * sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t) > file_size) {
            printf("[GGUF] ERROR: Truncated tensor info header for tensor '%s'\n", info->name);
            close_gguf_file(ctx);
            return NULL;
        }

        for (uint32_t d = 0; d < read_dims; d++) {
            uint64_t s = *(const uint64_t*)(mapped + offset);
            offset += sizeof(uint64_t);
            if (d < 4) info->shape[d] = s;
        }
        for (uint32_t d = read_dims; d < 4; d++) info->shape[d] = 1;

        info->ggml_type = *(const uint32_t*)(mapped + offset);
        offset += sizeof(uint32_t);

        info->offset = *(const uint64_t*)(mapped + offset);
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
    if (!ctx || !tokenizer || !ctx->mapped_data) return false;
    
    const uint8_t* mapped = ctx->mapped_data;
    size_t file_size = ctx->file_size;
    uint64_t offset = sizeof(GGUFHeader);

    char key_buf[256];
    int loaded = 0;
    bool have_bos = false, have_eos = false, have_pad = false;

    for (uint64_t i = 0; i < ctx->header.metadata_kv_count; i++) {
        uint64_t key_end = read_gguf_string_safe(mapped, file_size, offset, key_buf, sizeof(key_buf));
        if (key_end == 0 || key_end + sizeof(uint32_t) > file_size) {
            printf("[GGUF] WARN: Truncated key at tokenizer entry %llu\n", (unsigned long long)i);
            break;
        }
        offset = key_end;
        uint32_t val_type = *(const uint32_t*)(mapped + offset);
        uint64_t val_start = offset + sizeof(uint32_t);

        if (val_type == GGUF_TYPE_ARRAY) {
            if (val_start + 12 <= file_size) {
                uint32_t arr_type = *(const uint32_t*)(mapped + val_start);
                uint64_t arr_len = *(const uint64_t*)(mapped + val_start + 4);
                uint64_t arr_off = val_start + 12;

                if (strcmp(key_buf, "tokenizer.ggml.tokens") == 0 && arr_type == GGUF_TYPE_STRING) {
                    for (uint64_t k = 0; k < arr_len && k < (uint64_t)tokenizer->vocab_size; k++) {
                        char str_val[256];
                        arr_off = read_gguf_string_safe(mapped, file_size, arr_off, str_val, sizeof(str_val));
                        if (arr_off == 0) break;
                        set_tokenizer_entry(tokenizer, (int)k, str_val, (float)-(int)k);
                        loaded++;
                    }
                    if (!have_bos && !have_eos && !have_pad)
                        set_tokenizer_special_tokens(tokenizer, 1, 2, 0);
                    printf("[GGUF] Successfully populated %d real vocabulary tokens from GGUF metadata!\n", loaded);
                } else if (strcmp(key_buf, "tokenizer.ggml.merges") == 0 && arr_type == GGUF_TYPE_STRING) {
                    char sample[3][128] = {{0}};
                    int accepted = 0;
                    for (uint64_t k = 0; k < arr_len; k++) {
                        char str_val[256];
                        arr_off = read_gguf_string_safe(mapped, file_size, arr_off, str_val, sizeof(str_val));
                        if (arr_off == 0) break;
                        if (k < 3) snprintf(sample[k], sizeof(sample[k]), "%s", str_val);
                        char *sp = strchr(str_val, ' ');
                        if (!sp || sp == str_val || !sp[1] || strchr(sp + 1, ' ')) continue;
                        *sp = '\0';
                        if (tokenizer_add_merge(tokenizer, str_val, sp + 1)) accepted++;
                    }
                    tokenizer->merge_count = (arr_len > INT32_MAX) ? INT32_MAX : (int)arr_len;
                    printf("[TOKENIZER] merges: declared=%llu accepted=%d (sample: '%s' | '%s' | '%s')\n",
                        (unsigned long long)arr_len, accepted, sample[0], sample[1], sample[2]);
                } else if (strcmp(key_buf, "tokenizer.ggml.scores") == 0 && arr_type == GGUF_TYPE_FLOAT32) {
                    for (uint64_t k = 0; k < arr_len && k < (uint64_t)tokenizer->vocab_size; k++) {
                        if (arr_off + 4 > file_size) break;
                        float score = *(const float*)(mapped + arr_off);
                        arr_off += sizeof(float);
                        tokenizer->vocab_scores[k] = score;
                    }
                }
            }
        } else if (val_type == GGUF_TYPE_UINT32 || val_type == GGUF_TYPE_INT32) {
            if (val_start + 4 <= file_size) {
                uint32_t val = *(const uint32_t*)(mapped + val_start);
                if (strstr(key_buf, "bos_token_id")) { tokenizer->bos_id = (int)val; have_bos = true; }
                else if (strstr(key_buf, "eos_token_id")) { tokenizer->eos_id = (int)val; have_eos = true; }
                else if (strstr(key_buf, "padding_token_id")) { tokenizer->pad_id = (int)val; have_pad = true; }
            }
        } else if (val_type == GGUF_TYPE_BOOL) {
            if (val_start + 1 <= file_size) {
                uint8_t bval = *(const uint8_t*)(mapped + val_start);
                if (strcmp(key_buf, "tokenizer.ggml.add_bos_token") == 0) tokenizer->add_bos_token = (bval != 0);
                else if (strcmp(key_buf, "tokenizer.ggml.add_eos_token") == 0) tokenizer->add_eos_token = (bval != 0);
            }
        } else if (val_type == GGUF_TYPE_STRING) {
            char str_val[2048];
            if (read_gguf_string_safe(mapped, file_size, val_start, str_val, sizeof(str_val)) != 0) {
                if (strcmp(key_buf, "tokenizer.ggml.model") == 0) snprintf(tokenizer->model, sizeof(tokenizer->model), "%s", str_val);
                else if (strcmp(key_buf, "tokenizer.ggml.pre") == 0) snprintf(tokenizer->pre, sizeof(tokenizer->pre), "%s", str_val);
                else if (strcmp(key_buf, "tokenizer.chat_template") == 0) snprintf(tokenizer->chat_template, sizeof(tokenizer->chat_template), "%s", str_val);
            }
        }

        uint64_t next_off = skip_gguf_value(mapped, file_size, val_start, val_type, key_buf);
        if (next_off == 0 || next_off <= offset) break;
        offset = next_off;
    }
    printf("[TOKENIZER] model='%s' pre='%s' add_bos=%d add_eos=%d merges=%d bos=%d eos=%d pad=%d\n",
        tokenizer->model, tokenizer->pre, tokenizer->add_bos_token, tokenizer->add_eos_token,
        tokenizer->merge_count, tokenizer->bos_id, tokenizer->eos_id, tokenizer->pad_id);
    tokenizer_build_index(tokenizer);
    printf("[TOKENIZER] index: %d specials (atomic split)\n", tokenizer->n_specials);
    if (tokenizer->chat_template[0])
        printf("[TOKENIZER] chat_template: '%.160s...'\n", tokenizer->chat_template);
    else
        printf("[TOKENIZER] chat_template: (absent)\n");
    return loaded > 0;
}

static QuantType convert_ggml_type(uint32_t ggml_type) {
    switch (ggml_type) {
        case 0: return QUANT_FP32;
        case 1: return QUANT_FP16;
        case 2: return QUANT_Q4_0;
        case 8: return QUANT_Q8_0;
        case 14: return QUANT_Q6_K;
        default: return QUANT_Q4_K;
    }
}

static void assign_tensor_mapping(Tensor* target, GGUFTensorInfo* t) {
    if (!target || !t) return;
    if (target->data && !target->is_mmap) {
        free(target->data);
        target->data = NULL;
    }
    target->data = t->data_ptr;
    target->type = convert_ggml_type(t->ggml_type);
    target->is_mmap = true;
    target->n_dims = (t->n_dims > 4) ? 4 : (int)t->n_dims;
    target->numel = 1;
    for (int d = 0; d < target->n_dims; d++) {
        target->shape[d] = (t->shape[d] > INT32_MAX) ? INT32_MAX : (int)t->shape[d];
        target->numel *= (size_t)target->shape[d];
    }
    for (int d = target->n_dims; d < 4; d++) target->shape[d] = 1;
    if (target->n_dims == 0) target->numel = 0;
}

bool load_gguf_weights(GGUFContext* ctx, TransformerWeights* weights) {
    if (!ctx || !weights) return false;

    long type_hist[32] = {0};
    for (uint64_t i = 0; i < ctx->header.tensor_count; i++) {
        uint32_t gt = ctx->tensors[i].ggml_type;
        if (gt < 32) type_hist[gt]++;
    }
    printf("[GGUF] ggml_type histogram:");
    for (int k = 0; k < 32; k++) if (type_hist[k]) printf(" %d:%ld", k, type_hist[k]);
    printf("  (engine supports 0=F32 1=F16 2=Q4_0 8=Q8_0 14=Q6_K; others run as ZEROS)\n");

    int loaded_count = 0, mapped_count = 0, unsupported_warned = 0;
    bool fatal_unsupported = false;
    for (uint64_t i = 0; i < ctx->header.tensor_count; i++) {
        GGUFTensorInfo* t = &ctx->tensors[i];
        bool supported = (t->ggml_type == 0 || t->ggml_type == 1 ||
                          t->ggml_type == 2 || t->ggml_type == 8 ||
                          t->ggml_type == 14);
        if (!supported) {
            printf("[GGUF ERROR] Tensor '%s' uses unsupported quantization format (ggml_type %u). Only FP32(0), FP16(1), Q4_0(2), Q8_0(8), Q6_K(14) are supported.\n",
                t->name, t->ggml_type);
            fatal_unsupported = true;
        }
    }
    if (fatal_unsupported) {
        printf("[GGUF ERROR] Aborting model load due to unsupported quantization format.\n");
        return false;
    }

    for (uint64_t i = 0; i < ctx->header.tensor_count; i++) {
        GGUFTensorInfo* t = &ctx->tensors[i];

        if (strcmp(t->name, "token_embd.weight") == 0) {
            assign_tensor_mapping(&weights->token_embedding_table, t);
            printf("[GGUF TENSOR] token_embd.weight -> ggml_type: %u, shape: [%llu, %llu]\n", t->ggml_type, (unsigned long long)t->shape[0], (unsigned long long)t->shape[1]);
            loaded_count++; mapped_count++;
        } else if (strcmp(t->name, "output_norm.weight") == 0) {
            assign_tensor_mapping(&weights->rms_final_weight, t);
            loaded_count++; mapped_count++;
        } else if (strcmp(t->name, "output.weight") == 0) {
            bool supported = (t->ggml_type == 0 || t->ggml_type == 1 || t->ggml_type == 2 || t->ggml_type == 8 || t->ggml_type == 14);
            if (supported) {
                assign_tensor_mapping(&weights->w_cls, t);
                mapped_count++;
            }
            loaded_count++;
        } else {
            int layer_idx = -1;
            if (sscanf(t->name, "blk.%d.", &layer_idx) == 1 && layer_idx >= 0 && layer_idx < ctx->config.n_layers) {
                bool hit = true;
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
                } else hit = false;
                loaded_count++;
                if (hit) mapped_count++;
                else if (unsupported_warned < 14) {
                    printf("[GGUF] WARN: unmapped layer tensor '%s' (ignored)\n", t->name);
                    unsupported_warned++;
                }
            }
        }
    }

    if (!weights->w_cls.is_mmap) {
        if (weights->w_cls.data && !weights->w_cls.is_mmap) {
            free(weights->w_cls.data);
            weights->w_cls.data = NULL;
        }
        weights->w_cls = weights->token_embedding_table;
    }

    printf("[GGUF] Zero-Copy mmap mapped %d tensor weights into LLM Engine!\n", loaded_count);
    printf("[GGUF] name-mapped %d/%llu tensors.\n", mapped_count,
        (unsigned long long)ctx->header.tensor_count);

    const char* kind[15] = {"rms_att","wq","wk","wv","wo","bq","bk","bv","rms_ffn","w_gate","w_up","w_down","embd","rms_final","w_cls"};
    int missing[15] = {0};
    for (int l = 0; l < ctx->config.n_layers; l++) {
        if (!weights->rms_att_weight[l].data) missing[0]++;
        if (!weights->wq[l].data) missing[1]++;
        if (!weights->wk[l].data) missing[2]++;
        if (!weights->wv[l].data) missing[3]++;
        if (!weights->wo[l].data) missing[4]++;
        if (weights->bq && !weights->bq[l].data) missing[5]++;
        if (weights->bk && !weights->bk[l].data) missing[6]++;
        if (weights->bv && !weights->bv[l].data) missing[7]++;
        if (!weights->rms_ffn_weight[l].data) missing[8]++;
        if (!weights->w_gate[l].data) missing[9]++;
        if (!weights->w_up[l].data) missing[10]++;
        if (!weights->w_down[l].data) missing[11]++;
    }
    if (!weights->token_embedding_table.data) missing[12]++;
    if (!weights->rms_final_weight.data) missing[13]++;
    if (!weights->w_cls.data) missing[14]++;
    bool any_missing = false;
    for (int k = 0; k < 15; k++) if (missing[k]) any_missing = true;
    if (any_missing) {
        printf("[GGUF] WARN: NULL tensors (run as ZEROS):");
        for (int k = 0; k < 15; k++) if (missing[k]) printf(" %s:%d", kind[k], missing[k]);
        printf("\n  (bias NULLs are architecture-dependent; weight NULLs are fatal)\n");
    } else {
        printf("[GGUF] NULL audit: all weight tensors present.\n");
    }
    return true;
}
