#include "LLMClassifier.h"
#include <ctype.h>

LLMClassifier* create_llm_classifier(void) {
    LLMClassifier* clf = (LLMClassifier*)calloc(1, sizeof(LLMClassifier));
    if (!clf) return NULL;

    clf->input_size = 6;
    clf->hidden1_size = 16;
    clf->hidden2_size = 12;
    clf->output_size = 4;

    size_t w1_size = clf->input_size * clf->hidden1_size;
    size_t w2_size = clf->hidden1_size * clf->hidden2_size;
    size_t w3_size = clf->hidden2_size * clf->output_size;

    clf->w1 = (float*)malloc(w1_size * sizeof(float));
    clf->b1 = (float*)calloc(clf->hidden1_size, sizeof(float));
    clf->w2 = (float*)malloc(w2_size * sizeof(float));
    clf->b2 = (float*)calloc(clf->hidden2_size, sizeof(float));
    clf->w3 = (float*)malloc(w3_size * sizeof(float));
    clf->b3 = (float*)calloc(clf->output_size, sizeof(float));

    clf->h1_act = (float*)calloc(clf->hidden1_size, sizeof(float));
    clf->h2_act = (float*)calloc(clf->hidden2_size, sizeof(float));
    clf->out_act = (float*)calloc(clf->output_size, sizeof(float));

    // Xavier/He-like Random Initialization
    for (size_t i = 0; i < w1_size; i++) clf->w1[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.5f;
    for (size_t i = 0; i < w2_size; i++) clf->w2[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.5f;
    for (size_t i = 0; i < w3_size; i++) clf->w3[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.5f;

    return clf;
}

void free_llm_classifier(LLMClassifier* clf) {
    if (!clf) return;
    if (clf->w1) free(clf->w1);
    if (clf->b1) free(clf->b1);
    if (clf->w2) free(clf->w2);
    if (clf->b2) free(clf->b2);
    if (clf->w3) free(clf->w3);
    if (clf->b3) free(clf->b3);
    if (clf->h1_act) free(clf->h1_act);
    if (clf->h2_act) free(clf->h2_act);
    if (clf->out_act) free(clf->out_act);
    free(clf);
}

void extract_prompt_features(const char* prompt, float features[6]) {
    if (!prompt || strlen(prompt) == 0) {
        for (int i = 0; i < 6; i++) features[i] = 0.0f;
        return;
    }

    size_t len = strlen(prompt);
    char lower[512];
    size_t copy_len = len < 511 ? len : 511;
    for (size_t i = 0; i < copy_len; i++) {
        lower[i] = (char)tolower(prompt[i]);
    }
    lower[copy_len] = '\0';

    // Feature 0: Has Code Keywords
    bool code_kw = (strstr(lower, "c") != NULL || strstr(lower, "code") != NULL || strstr(lower, "function") != NULL ||
                    strstr(lower, "python") != NULL || strstr(lower, "script") != NULL || strstr(lower, "sort") != NULL ||
                    strstr(lower, "bug") != NULL || strstr(lower, "error") != NULL || strstr(lower, "binary") != NULL);

    // Feature 1: Has Punctuation
    bool punct = (strchr(prompt, '?') != NULL || strchr(prompt, '!') != NULL || strchr(prompt, '.') != NULL);

    // Feature 2: Length > 30 chars
    bool len30 = (len > 30);

    // Feature 3: Has Question Word
    bool question = (strstr(lower, "how") != NULL || strstr(lower, "what") != NULL || strstr(lower, "why") != NULL ||
                     strstr(lower, "who") != NULL || strstr(lower, "where") != NULL);

    // Feature 4: Has Creative Keyword
    bool creative = (strstr(lower, "poem") != NULL || strstr(lower, "story") != NULL || strstr(lower, "song") != NULL ||
                     strstr(lower, "creative") != NULL || strstr(lower, "compose") != NULL || strstr(lower, "essay") != NULL);

    // Feature 5: Has Safety Keyword
    bool safety = (strstr(lower, "hack") != NULL || strstr(lower, "bypass") != NULL || strstr(lower, "malware") != NULL ||
                   strstr(lower, "attack") != NULL || strstr(lower, "bank") != NULL || strstr(lower, "security") != NULL);

    features[0] = code_kw ? 1.0f : 0.0f;
    features[1] = punct ? 1.0f : 0.0f;
    features[2] = len30 ? 1.0f : 0.0f;
    features[3] = question ? 1.0f : 0.0f;
    features[4] = creative ? 1.0f : 0.0f;
    features[5] = safety ? 1.0f : 0.0f;
}

static float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

ClassifierResult classify_prompt(LLMClassifier* clf, const char* prompt) {
    ClassifierResult res = {0};
    if (!clf || !prompt) return res;

    float features[6];
    extract_prompt_features(prompt, features);

    // Layer 1: Forward
    for (size_t j = 0; j < clf->hidden1_size; j++) {
        float sum = clf->b1[j];
        for (size_t i = 0; i < clf->input_size; i++) {
            sum += features[i] * clf->w1[i * clf->hidden1_size + j];
        }
        clf->h1_act[j] = relu(sum);
    }

    // Layer 2: Forward
    for (size_t j = 0; j < clf->hidden2_size; j++) {
        float sum = clf->b2[j];
        for (size_t i = 0; i < clf->hidden1_size; i++) {
            sum += clf->h1_act[i] * clf->w2[i * clf->hidden2_size + j];
        }
        clf->h2_act[j] = relu(sum);
    }

    // Layer 3: Output Logits
    float max_logit = -1e9f;
    for (size_t j = 0; j < clf->output_size; j++) {
        float sum = clf->b3[j];
        for (size_t i = 0; i < clf->hidden2_size; i++) {
            sum += clf->h2_act[i] * clf->w3[i * clf->output_size + j];
        }
        clf->out_act[j] = sum;
        if (sum > max_logit) max_logit = sum;
    }

    // Softmax
    float sum_exp = 0.0f;
    for (size_t j = 0; j < clf->output_size; j++) {
        clf->out_act[j] = expf(clf->out_act[j] - max_logit);
        sum_exp += clf->out_act[j];
    }
    for (size_t j = 0; j < clf->output_size; j++) {
        clf->out_act[j] /= sum_exp;
    }

    res.code_prob = clf->out_act[0];
    res.creative_prob = clf->out_act[1];
    res.qa_prob = clf->out_act[2];
    res.unsafe_prob = clf->out_act[3];

    // Determine primary intent
    PromptIntent max_intent = INTENT_CODE;
    float max_p = res.code_prob;
    if (res.creative_prob > max_p) { max_p = res.creative_prob; max_intent = INTENT_CREATIVE; }
    if (res.qa_prob > max_p) { max_p = res.qa_prob; max_intent = INTENT_QA_GENERAL; }
    if (res.unsafe_prob > max_p) { max_p = res.unsafe_prob; max_intent = INTENT_UNSAFE; }

    // Direct feature override rule for safety keyword if neural prediction is tied
    if (features[5] > 0.5f) {
        max_intent = INTENT_UNSAFE;
    }

    res.primary_intent = max_intent;

    // Auto-tune Hyperparameters based on intent
    switch (res.primary_intent) {
        case INTENT_CODE:
            res.recommended_temp = 0.2f;
            res.recommended_top_p = 0.85f;
            res.is_safe = true;
            break;
        case INTENT_CREATIVE:
            res.recommended_temp = 0.9f;
            res.recommended_top_p = 0.95f;
            res.is_safe = true;
            break;
        case INTENT_QA_GENERAL:
            res.recommended_temp = 0.6f;
            res.recommended_top_p = 0.90f;
            res.is_safe = true;
            break;
        case INTENT_UNSAFE:
            res.recommended_temp = 0.0f;
            res.recommended_top_p = 0.0f;
            res.is_safe = false;
            break;
    }

    return res;
}

bool save_classifier_weights(LLMClassifier* clf, const char* filename) {
    if (!clf || !filename) return false;
    FILE* f = fopen(filename, "wb");
    if (!f) return false;

    fwrite(clf->w1, sizeof(float), clf->input_size * clf->hidden1_size, f);
    fwrite(clf->b1, sizeof(float), clf->hidden1_size, f);
    fwrite(clf->w2, sizeof(float), clf->hidden1_size * clf->hidden2_size, f);
    fwrite(clf->b2, sizeof(float), clf->hidden2_size, f);
    fwrite(clf->w3, sizeof(float), clf->hidden2_size * clf->output_size, f);
    fwrite(clf->b3, sizeof(float), clf->output_size, f);

    fclose(f);
    return true;
}

bool load_classifier_weights(LLMClassifier* clf, const char* filename) {
    if (!clf || !filename) return false;
    FILE* f = fopen(filename, "rb");
    if (!f) return false;

    fread(clf->w1, sizeof(float), clf->input_size * clf->hidden1_size, f);
    fread(clf->b1, sizeof(float), clf->hidden1_size, f);
    fread(clf->w2, sizeof(float), clf->hidden1_size * clf->hidden2_size, f);
    fread(clf->b2, sizeof(float), clf->hidden2_size, f);
    fread(clf->w3, sizeof(float), clf->hidden2_size * clf->output_size, f);
    fread(clf->b3, sizeof(float), clf->output_size, f);

    fclose(f);
    return true;
}

void train_classifier_step(LLMClassifier* clf, const float inputs[6], int target_class, float lr) {
    if (!clf || target_class < 0 || target_class >= 4) return;

    // Forward pass
    for (size_t j = 0; j < clf->hidden1_size; j++) {
        float sum = clf->b1[j];
        for (size_t i = 0; i < clf->input_size; i++) sum += inputs[i] * clf->w1[i * clf->hidden1_size + j];
        clf->h1_act[j] = relu(sum);
    }
    for (size_t j = 0; j < clf->hidden2_size; j++) {
        float sum = clf->b2[j];
        for (size_t i = 0; i < clf->hidden1_size; i++) sum += clf->h1_act[i] * clf->w2[i * clf->hidden2_size + j];
        clf->h2_act[j] = relu(sum);
    }
    float max_logit = -1e9f;
    for (size_t j = 0; j < clf->output_size; j++) {
        float sum = clf->b3[j];
        for (size_t i = 0; i < clf->hidden2_size; i++) sum += clf->h2_act[i] * clf->w3[i * clf->output_size + j];
        clf->out_act[j] = sum;
        if (sum > max_logit) max_logit = sum;
    }
    float sum_exp = 0.0f;
    for (size_t j = 0; j < clf->output_size; j++) {
        clf->out_act[j] = expf(clf->out_act[j] - max_logit);
        sum_exp += clf->out_act[j];
    }
    for (size_t j = 0; j < clf->output_size; j++) clf->out_act[j] /= sum_exp;

    // Backprop Softmax Cross-Entropy Loss
    float d_out[4];
    for (size_t j = 0; j < clf->output_size; j++) {
        d_out[j] = clf->out_act[j] - (j == (size_t)target_class ? 1.0f : 0.0f);
    }

    // Update Layer 3 weights & biases
    float d_h2[12] = {0};
    for (size_t j = 0; j < clf->output_size; j++) {
        clf->b3[j] -= lr * d_out[j];
        for (size_t i = 0; i < clf->hidden2_size; i++) {
            d_h2[i] += d_out[j] * clf->w3[i * clf->output_size + j];
            clf->w3[i * clf->output_size + j] -= lr * d_out[j] * clf->h2_act[i];
        }
    }

    // Backprop Layer 2 (ReLU derivative)
    float d_h1[16] = {0};
    for (size_t i = 0; i < clf->hidden2_size; i++) {
        if (clf->h2_act[i] <= 0.0f) d_h2[i] = 0.0f;
        clf->b2[i] -= lr * d_h2[i];
        for (size_t k = 0; k < clf->hidden1_size; k++) {
            d_h1[k] += d_h2[i] * clf->w2[k * clf->hidden2_size + i];
            clf->w2[k * clf->hidden2_size + i] -= lr * d_h2[i] * clf->h1_act[k];
        }
    }

    // Backprop Layer 1 (ReLU derivative)
    for (size_t k = 0; k < clf->hidden1_size; k++) {
        if (clf->h1_act[k] <= 0.0f) d_h1[k] = 0.0f;
        clf->b1[k] -= lr * d_h1[k];
        for (size_t m = 0; m < clf->input_size; m++) {
            clf->w1[m * clf->hidden1_size + k] -= lr * d_h1[k] * inputs[m];
        }
    }
}
