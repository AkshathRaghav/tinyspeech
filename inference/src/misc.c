#include "misc.h"

float compute_mean_abs(int32_t *w, int32_t len) {
    int sum = 0.0f;
    for (int32_t i = 0; i < len; i++) {
        sum += fabsf((int)w[i]);
    }
    return sum / len;
}

int8_t clamp(int8_t val, int8_t min_val, int8_t max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

void quantize_weights(int32_t *w, int8_t *u, int32_t len, float* magn) {
    if (!magn) { 
        float computed_magn = compute_mean_abs(w, len);
        magn = &computed_magn;
    } 
    float scale = 32 / (*magn); // 32 

    for (int32_t i = 0; i < len; i++) {
        u[i] = (int8_t) clamp(roundf(w[i] * scale), -127.0f, 127.0f);
    }
}

void sigmoid(Tensor *tensor) {
    for (int i = 0; i < tensor->size; i++) {
        tensor->f_data[i] = 1.0f / (1.0f + expf(-tensor->f_data[i]));
    }
}

void attention(Tensor *residual, Tensor *S, Tensor *scale) { 
    for (int i = 0; i < S->size; i++) {
        S->f_data[i] = residual->data[i] + (residual->f_data[i] * (*scale->f_data)[i] * S->f_data[i])
    }
}

float mean(int8_t *data, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

void relu(Tensor* input) {
    for (int i = 0; i < input->size; i++) {
        if (input->data[i] < 0) {
            input->data[i] = 0;
        }
    }
}