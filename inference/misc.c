#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#define INT_MAX 2147483647

typedef struct {
    int8_t *data;
    int8_t shape[4]; // [N, C, H, W]
} Tensor;


float compute_mean_abs(int32_t *w, size_t len) {
    int sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        sum += fabsf((int)w[i]);
    }
    return sum / len;
}

int8_t clamp(int8_t val, int8_t min_val, int8_t max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

void quantize_weights(int32_t *w, int8_t *u, size_t len) {
    float mag = compute_mean_abs(w, len);
    float scale = 32 / mag;

    for (size_t i = 0; i < len; i++) {
        u[i] = (int8_t) clamp(roundf(w[i] * scale), -127.0f, 127.0f);
    }
}

void sigmoid(Tensor *tensor) {
    for (int i = 0; i < tensor->size; i++) {
        tensor->data[i] = 1.0f / (1.0f + expf(-tensor->data[i]));
    }
}

float mean(int8_t *data, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}
