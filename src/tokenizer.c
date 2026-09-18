#include "tokenizer.h"

static unsigned long tmap_hash(const char *s) {
    unsigned long h = 5381;
    while (*s) h = ((h << 5) + h + (unsigned char)*s++);
    return h;
}
static void tmap_clear_free(TMap *m) {
    if (!m->tab) { m->cap = m->n = 0; return; }
    for (size_t i = 0; i < m->cap; i++) if (m->tab[i].used) free(m->tab[i].key);
    free(m->tab); m->tab = NULL; m->cap = m->n = 0;
}
static void tmap_grow(TMap *m) {
    size_t ncap = m->cap ? m->cap * 2 : 1024;
    TMapEntry *nt = (TMapEntry*)calloc(ncap, sizeof(TMapEntry));
    if (!nt) return;
    for (size_t i = 0; i < m->cap; i++) if (m->tab[i].used) {
        unsigned long h = tmap_hash(m->tab[i].key);
        size_t j = h % ncap;
        while (nt[j].used) j = (j + 1) % ncap;
        nt[j] = m->tab[i];
    }
    free(m->tab); m->tab = nt; m->cap = ncap;
}
static void tmap_put(TMap *m, const char *key, int val) {
    if (!key) return;
    if (m->n * 4 >= m->cap * 3) tmap_grow(m);
    if (!m->tab) return;
    unsigned long h = tmap_hash(key);
    size_t j = h % m->cap;
    while (m->tab[j].used) {
        if (strcmp(m->tab[j].key, key) == 0) { m->tab[j].val = val; return; }
        j = (j + 1) % m->cap;
    }
    m->tab[j].key = strdup(key);
    if (!m->tab[j].key) return;
    m->tab[j].val = val; m->tab[j].used = 1; m->n++;
}
static int tmap_get(TMap *m, const char *key, int dflt) {
    if (!m->tab || !key) return dflt;
    unsigned long h = tmap_hash(key);
    size_t j = h % m->cap;
    size_t start = j;
    while (m->tab[j].used) {
        if (strcmp(m->tab[j].key, key) == 0) return m->tab[j].val;
        j = (j + 1) % m->cap;
        if (j == start) break;
    }
    return dflt;
}

Tokenizer* create_tokenizer(int vocab_size) {
    Tokenizer* t = (Tokenizer*)malloc(sizeof(Tokenizer));
    if (!t) return NULL;
    
    t->vocab_size = vocab_size;
    t->max_token_length = 0;
    t->bos_id = 1;
    t->eos_id = 2;
    t->pad_id = 0;

    // Defaults = legacy behavior (mock tests, LLaMA-style files).
    // Overwritten from GGUF metadata when present (Qwen: add_bos=false).
    snprintf(t->model, sizeof(t->model), "unknown");
    snprintf(t->pre, sizeof(t->pre), "unknown");
    t->add_bos_token = true;
    t->add_eos_token = false;
    t->merge_count = 0;
    t->chat_template[0] = '\0';
    
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

    t->merge_left = NULL; t->merge_right = NULL;
    t->n_merges = 0; t->merge_cap = 0;
    t->special_ids = NULL; t->n_specials = 0;
    t->index_dirty = true;
    t->vocab_map.tab = NULL; t->vocab_map.cap = t->vocab_map.n = 0;
    t->merge_map.tab = NULL; t->merge_map.cap = t->merge_map.n = 0;
    
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
    if (t->merge_left) {
        for (int i = 0; i < t->n_merges; i++) { free(t->merge_left[i]); free(t->merge_right[i]); }
        free(t->merge_left); free(t->merge_right);
    }
    if (t->special_ids) free(t->special_ids);
    tmap_clear_free(&t->vocab_map);
    tmap_clear_free(&t->merge_map);
    free(t);
}

bool tokenizer_add_merge(Tokenizer* t, const char* left, const char* right) {
    if (!t || !left || !right || !left[0] || !right[0]) return false;
    if (t->n_merges >= t->merge_cap) {
        int ncap = t->merge_cap ? t->merge_cap * 2 : 4096;
        char **nl = (char**)realloc(t->merge_left, (size_t)ncap * sizeof(char*));
        char **nr = (char**)realloc(t->merge_right, (size_t)ncap * sizeof(char*));
        if (!nl || !nr) return false;
        t->merge_left = nl; t->merge_right = nr; t->merge_cap = ncap;
    }
    t->merge_left[t->n_merges] = strdup(left);
    t->merge_right[t->n_merges] = strdup(right);
    if (!t->merge_left[t->n_merges] || !t->merge_right[t->n_merges]) {
        free(t->merge_left[t->n_merges]); free(t->merge_right[t->n_merges]);
        return false;
    }
    // rank = order of accepted lines (matches HF BpeConverter)
    size_t kl = strlen(left), kr = strlen(right);
    char *key = (char*)malloc(kl + kr + 2);
    if (key) {
        memcpy(key, left, kl); key[kl] = '\x1F';
        memcpy(key + kl + 1, right, kr + 1);
        tmap_put(&t->merge_map, key, t->n_merges);
        free(key);
    }
    t->n_merges++;
    return true;
}

static bool is_special_text(const Tokenizer* t, int id) {
    const char *s = (id >= 0 && id < t->vocab_size) ? t->vocab[id] : NULL;
    if (!s || !s[0]) return false;
    if (id == t->bos_id || id == t->eos_id || id == t->pad_id) return true;
    size_t L = strlen(s);
    // Qwen style <|...|> (also covers <|endoftext|>, <|im_start|>, <|im_end|>)
    if (L >= 5 && s[0] == '<' && s[1] == '|' && s[L-2] == '|' && s[L-1] == '>') return true;
    return false;
}

static int cmp_special_len_desc(const void* a, const void* b);
static Tokenizer *g_sort_tok = NULL;
static int cmp_special_len_desc(const void* a, const void* b) {
    size_t la = strlen(g_sort_tok->vocab[*(const int*)a]);
    size_t lb = strlen(g_sort_tok->vocab[*(const int*)b]);
    return (la < lb) - (la > lb);
}

void tokenizer_build_index(Tokenizer* t) {
    if (!t) return;
    tmap_clear_free(&t->vocab_map);
    for (int i = 0; i < t->vocab_size; i++)
        if (t->vocab[i]) tmap_put(&t->vocab_map, t->vocab[i], i);
    free(t->special_ids); t->special_ids = NULL; t->n_specials = 0;
    int *tmp = (int*)malloc((size_t)t->vocab_size * sizeof(int));
    if (tmp) {
        for (int i = 0; i < t->vocab_size; i++)
            if (is_special_text(t, i)) tmp[t->n_specials++] = i;
        if (t->n_specials) {
            t->special_ids = (int*)malloc((size_t)t->n_specials * sizeof(int));
            if (t->special_ids) {
                memcpy(t->special_ids, tmp, (size_t)t->n_specials * sizeof(int));
                g_sort_tok = t;
                qsort(t->special_ids, (size_t)t->n_specials, sizeof(int), cmp_special_len_desc);
                g_sort_tok = NULL;
            } else t->n_specials = 0;
        }
        free(tmp);
    }
    t->index_dirty = false;
}

static void tokenizer_ensure_index(Tokenizer* t) {
    if (t && t->index_dirty) tokenizer_build_index(t);
}

int tokenizer_find_token(Tokenizer* t, const char* str) {
    if (!t || !str) return -1;
    tokenizer_ensure_index(t);
    return tmap_get(&t->vocab_map, str, -1);
}

bool set_tokenizer_entry(Tokenizer* t, int id, const char* str, float score) {
    if (!t || id < 0 || id >= t->vocab_size) return false;
    
    if (t->vocab[id]) free(t->vocab[id]);
    t->vocab[id] = strdup(str);
    t->vocab_scores[id] = score;
    t->index_dirty = true;
    
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
    t->index_dirty = true;
}

static int find_vocab_token(Tokenizer* t, const char* str) {
    for (int i = 0; i < t->vocab_size; i++) {
        if (t->vocab[i] && strcmp(t->vocab[i], str) == 0) {
            return i;
        }
    }
    return -1;
}

#include <limits.h>

// GPT-2 bytes_to_unicode codepoint for a raw byte
static unsigned b2u_cp(unsigned char b) {
    if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))
        return b;
    unsigned idx;
    if (b <= 0x20) idx = b;            // 0x00..0x20 -> 0..32
    else if (b == 0xAD) idx = 33;
    else idx = 34 + (b - 0x7F);        // 0x7F..0xA0 -> 34..67
    return 0x100 + idx;                // U+0100..U+0143
}
static int utf8_encode(unsigned cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2;
}
// Map raw bytes to GPT-2 mapped UTF-8. Returns malloc'd NUL-terminated buffer.
static char *bytes_to_mapped(const unsigned char *s, size_t n, size_t *out_len) {
    char *o = (char*)malloc(n * 2 + 1);
    if (!o) return NULL;
    size_t p = 0;
    for (size_t i = 0; i < n; i++) { char tmp[2]; int L = utf8_encode(b2u_cp(s[i]), tmp); o[p++] = tmp[0]; if (L > 1) o[p++] = tmp[1]; }
    o[p] = '\0'; if (out_len) *out_len = p;
    return o;
}
static int merge_rank_of(Tokenizer *t, const char *a, const char *b) {
    size_t la = strlen(a), lb = strlen(b);
    if (la + lb + 2 > 1024) return -1;
    char key[1024];
    memcpy(key, a, la); key[la] = '\x1F';
    memcpy(key + la + 1, b, lb + 1);
    return tmap_get(&t->merge_map, key, -1);
}
// Decode first codepoint of a UTF-8 string; *clen = its byte length
static unsigned utf8_decode_first(const char *s, int *clen) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { if (clen) *clen = 1; return c; }
    if (clen) *clen = 2;
    return ((unsigned)(c & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F);
}
// Reverse GPT-2 mapping for a single-char symbol -> raw byte, or -1
static int mapped_symbol_to_byte(const char *sym) {
    int clen = 0; unsigned cp = utf8_decode_first(sym, &clen);
    if (!sym[clen]) {
        if (cp < 0x100) return (int)cp;
        if (cp >= 0x100 && cp <= 0x143) {
            unsigned idx = cp - 0x100;
            if (idx <= 32) return (int)idx;
            if (idx == 33) return 0xAD;
            return 0x7F + (int)(idx - 34);
        }
    }
    return -1;
}
// BPE-merge one word's raw bytes, appending ids. Returns ids appended.
static int bpe_piece(Tokenizer *t, const unsigned char *bytes, size_t n, int *out, int max_out) {
    if (n == 0 || max_out <= 0) return 0;
    size_t mlen = 0;
    char *m = bytes_to_mapped(bytes, n, &mlen);
    if (!m) return 0;
    // Split mapped string into codepoint symbols
    char **sym = (char**)malloc((mlen + 1) * sizeof(char*));
    if (!sym) { free(m); return 0; }
    int nsym = 0;
    for (size_t i = 0; i < mlen; ) {
        int cl = 1; (void)utf8_decode_first(m + i, &cl);
        char *s = (char*)malloc((size_t)cl + 1);
        if (!s) break;
        memcpy(s, m + i, (size_t)cl); s[cl] = '\0';
        sym[nsym++] = s; i += (size_t)cl;
    }
    // Greedy lowest-rank merges
    while (nsym > 1) {
        int best = -1, best_rank = INT_MAX;
        for (int i = 0; i < nsym - 1; i++) {
            int r = merge_rank_of(t, sym[i], sym[i + 1]);
            if (r >= 0 && r < best_rank) { best_rank = r; best = i; }
        }
        if (best < 0) break;
        size_t nl = strlen(sym[best]) + strlen(sym[best + 1]) + 1;
        char *ns = (char*)malloc(nl);
        if (!ns) break;
        snprintf(ns, nl, "%s%s", sym[best], sym[best + 1]);
        free(sym[best]); free(sym[best + 1]);
        sym[best] = ns;
        memmove(sym + best + 1, sym + best + 2, (size_t)(nsym - best - 2) * sizeof(char*));
        nsym--;
    }
    int cnt = 0;
    for (int i = 0; i < nsym && cnt < max_out; i++) {
        int id = tmap_get(&t->vocab_map, sym[i], -1);
        if (id < 0) {
            // Byte fallback: single mapped char -> <0xXX> token, else pad
            int b = mapped_symbol_to_byte(sym[i]);
            char fb[16];
            if (b >= 0) snprintf(fb, sizeof(fb), "<0x%02X>", b);
            else snprintf(fb, sizeof(fb), "<0x00>");
            id = tmap_get(&t->vocab_map, fb, -1);
            if (id < 0) {
                // Last resort: raw byte text (covers ASCII-complete vocabs)
                if (b >= 0 && b < 128) { char rb[2] = {(char)b, 0}; id = tmap_get(&t->vocab_map, rb, -1); }
                if (id < 0) id = t->pad_id;
            }
        }
        out[cnt++] = id;
    }
    for (int i = 0; i < nsym; i++) free(sym[i]);
    free(sym); free(m);
    return cnt;
}
// Longest special-token match at p (specials sorted desc by length)
static int match_special_at(Tokenizer *t, const char *p, size_t remain, size_t *mlen) {
    for (int k = 0; k < t->n_specials; k++) {
        int id = t->special_ids[k];
        const char *s = t->vocab[id];
        size_t L = strlen(s);
        if (L <= remain && memcmp(p, s, L) == 0) { if (mlen) *mlen = L; return id; }
    }
    return -1;
}
static bool is_qwen_alpha(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
static bool is_qwen_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}
static bool is_qwen_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\f' || c == '\v';
}
static bool is_qwen_newline(unsigned char c) {
    return c == '\r' || c == '\n';
}

static size_t qwen2_next_chunk(const char *text, size_t w, size_t j) {
    if (w >= j) return 0;
    
    // 1. Contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (text[w] == '\'') {
        size_t rem = j - w;
        if (rem >= 2) {
            char c1 = text[w+1];
            if (c1 == 's' || c1 == 'S' || c1 == 't' || c1 == 'T' || c1 == 'm' || c1 == 'M' || c1 == 'd' || c1 == 'D')
                return 2;
        }
        if (rem >= 3) {
            char c1 = text[w+1], c2 = text[w+2];
            if ((c1 == 'r' || c1 == 'R') && (c2 == 'e' || c2 == 'E')) return 3;
            if ((c1 == 'v' || c1 == 'V') && (c2 == 'e' || c2 == 'E')) return 3;
            if ((c1 == 'l' || c1 == 'L') && (c2 == 'l' || c2 == 'L')) return 3;
        }
    }
    
    // 2. Digits: \p{N}{1,3}
    if (is_qwen_digit((unsigned char)text[w])) {
        size_t len = 0;
        while (w + len < j && len < 3 && is_qwen_digit((unsigned char)text[w + len])) {
            len++;
        }
        return len;
    }
    
    // 3. Letters: [^\r\n\p{L}\p{N}]?\p{L}+
    unsigned char c0 = (unsigned char)text[w];
    if (is_qwen_alpha(c0) || c0 >= 0x80) {
        size_t len = 0;
        while (w + len < j) {
            unsigned char c = (unsigned char)text[w + len];
            if (is_qwen_alpha(c) || c >= 0x80) len++;
            else break;
        }
        return len;
    } else if (!is_qwen_newline(c0) && !is_qwen_digit(c0)) {
        if (w + 1 < j) {
            unsigned char c1 = (unsigned char)text[w + 1];
            if (is_qwen_alpha(c1) || c1 >= 0x80) {
                size_t len = 1;
                while (w + len < j) {
                    unsigned char c = (unsigned char)text[w + len];
                    if (is_qwen_alpha(c) || c >= 0x80) len++;
                    else break;
                }
                return len;
            }
        }
    }
    
    // 4. Punctuation / symbols: ?[^\s\p{L}\p{N}]+[\r\n]*
    if (c0 == ' ') {
        if (w + 1 < j) {
            unsigned char c1 = (unsigned char)text[w + 1];
            if (!is_qwen_space(c1) && !is_qwen_newline(c1) && !is_qwen_alpha(c1) && !is_qwen_digit(c1) && c1 < 0x80) {
                size_t len = 1;
                while (w + len < j) {
                    unsigned char c = (unsigned char)text[w + len];
                    if (!is_qwen_space(c) && !is_qwen_newline(c) && !is_qwen_alpha(c) && !is_qwen_digit(c) && c1 < 0x80) len++;
                    else break;
                }
                while (w + len < j && is_qwen_newline((unsigned char)text[w + len])) len++;
                return len;
            }
        }
    } else if (!is_qwen_space(c0) && !is_qwen_newline(c0) && !is_qwen_alpha(c0) && !is_qwen_digit(c0) && c0 < 0x80) {
        size_t len = 0;
        while (w + len < j) {
            unsigned char c = (unsigned char)text[w + len];
            if (!is_qwen_space(c) && !is_qwen_newline(c) && !is_qwen_alpha(c) && !is_qwen_digit(c) && c < 0x80) len++;
            else break;
        }
        while (w + len < j && is_qwen_newline((unsigned char)text[w + len])) len++;
        return len;
    }
    
    // 5. Newlines / Whitespace runs: \s*[\r\n]+ or \s+
    if (is_qwen_newline(c0)) {
        size_t len = 0;
        while (w + len < j && is_qwen_newline((unsigned char)text[w + len])) len++;
        return len;
    }
    if (is_qwen_space(c0)) {
        size_t len = 0;
        while (w + len < j && is_qwen_space((unsigned char)text[w + len])) len++;
        if (w + len < j && is_qwen_newline((unsigned char)text[w + len])) {
            while (w + len < j && is_qwen_newline((unsigned char)text[w + len])) len++;
        }
        return len;
    }
    
    return 1;
}

static int encode_bpe(Tokenizer *t, const char *text, bool add_bos, bool add_eos, int *tokens, int max_tokens) {
    int n = 0;
    if (add_bos && n < max_tokens) tokens[n++] = t->bos_id;
    size_t L = strlen(text), i = 0;
    while (i < L && n < max_tokens) {
        size_t slen = 0;
        int sid = match_special_at(t, text + i, L - i, &slen);
        if (sid >= 0) { tokens[n++] = sid; i += slen; continue; }
        // Normal run until next special (or end)
        size_t j = i + 1;
        while (j < L && match_special_at(t, text + j, L - j, NULL) < 0) j++;
        // Qwen2 / GPT-2 regex pre-tokenization split inside [i, j)
        size_t w = i;
        while (w < j && n < max_tokens) {
            size_t chunk_len = qwen2_next_chunk(text, w, j);
            if (chunk_len == 0) chunk_len = 1;
            n += bpe_piece(t, (const unsigned char*)text + w, chunk_len, tokens + n, max_tokens - n);
            w += chunk_len;
        }
        i = j;
    }
    if (add_eos && n < max_tokens) tokens[n++] = t->eos_id;
    return n;
}

static int encode_legacy(Tokenizer* t, const char* text, bool add_bos, bool add_eos, int* tokens, int max_tokens) {
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

int encode(Tokenizer* t, const char* text, bool add_bos, bool add_eos, int* tokens, int max_tokens) {
    if (!t || !text || !tokens || max_tokens <= 0) return 0;
    tokenizer_ensure_index(t);
    if (t->n_merges > 0) return encode_bpe(t, text, add_bos, add_eos, tokens, max_tokens);
    return encode_legacy(t, text, add_bos, add_eos, tokens, max_tokens);
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

    // Clean up BPE / SentencePiece space, newline, and mapped byte symbols
    static char clean_buf[1024];
    size_t out_idx = 0;
    size_t len = strlen(str);
    for (size_t i = 0; i < len; ) {
        // SentencePiece   (0xE2 0x96 0x81) -> space
        if (i + 2 < len && (unsigned char)str[i] == 0xE2 && (unsigned char)str[i+1] == 0x96 && (unsigned char)str[i+2] == 0x81) {
            clean_buf[out_idx++] = ' ';
            i += 3;
        }
        else {
            int clen = 0;
            unsigned cp = utf8_decode_first(str + i, &clen);
            if (cp >= 0x100 && cp <= 0x143) {
                unsigned idx = cp - 0x100;
                unsigned char raw_b;
                if (idx <= 32) raw_b = (unsigned char)idx;
                else if (idx == 33) raw_b = 0xAD;
                else raw_b = (unsigned char)(0x7F + (idx - 34));
                clean_buf[out_idx++] = (char)raw_b;
                i += (size_t)(clen > 0 ? clen : 1);
            } else {
                clean_buf[out_idx++] = str[i++];
            }
        }
        if (out_idx >= sizeof(clean_buf) - 1) break;
    }
    clean_buf[out_idx] = '\0';
    return clean_buf;
}

bool render_chat_template(const Tokenizer* t, const char* prompt, char* out, size_t out_size) {
    if (!t || !prompt || !out || out_size == 0) return false;
    if (t->chat_template[0] != '\0') {
        if (strstr(t->chat_template, "<|im_start|>")) {
            snprintf(out, out_size,
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                prompt);
            return true;
        }
        if (strstr(t->chat_template, "[INST]")) {
            snprintf(out, out_size, "[INST] %s [/INST]", prompt);
            return true;
        }
        if (strstr(t->chat_template, "<|start_header_id|>")) {
            snprintf(out, out_size,
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                prompt);
            return true;
        }
    }
    snprintf(out, out_size,
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
        prompt);
    return true;
}
