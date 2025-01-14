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

void quantize_weights(Tensor *w, Tensor *u, float* scale, uint8_t retain_float) {
    for (int32_t i = 0; i < u->size; i++) {
        if (retain_float == 1) { 
            u->f_data[i] = clamp(roundf(w->f_data[i] / (*scale)), -127.0f, 127.0f);
        } else { 
            u->data[i] = (int8_t) clamp(roundf(w->f_data[i] / (*scale)), -127.0f, 127.0f);
        }
    }
}

void dequantize_weights(Tensor *quantized_weights, Tensor *dequantized_weights, float scale) {
    for (int32_t i = 0; i < dequantize_weights.size; i++) {
        dequantized_weights->f_data[i] = quantized_weights->[i] * scale;
    }
}


void sigmoid(Tensor *tensor) {
    #ifdef QUANT_MODE_QAT_SQ
        Tensor output = f_create_tensor(tensor->shape, 4); 
    #endif 

    for (int i = 0; i < tensor->size; i++) {
        #ifdef QUANT_MODE_QAT_SQ
            tensor->f_data[i] = 1.0f / (1.0f + expf(-tensor->f_data[i]));
        #else 
            output->f_data[i] = 1.0f / (1.0f + expf(-tensor->data[i]));
        #endif
    }

    #ifdef QUANT_MODE_QAT_SQ
        free_tensor(tensor);
        tensor = output; 
    #endif 
}

void attention(Tensor *residual, Tensor *S, Tensor *scale) { 
    for (int i = 0; i < S->size; i++) {
        #ifdef QUANT_MODE_QAT_SQ
            S->f_data[i] = residual->data[i] + (residual->f_data[i] * (*scale->f_data)[i] * S->f_data[i]);
        #else 
            S->f_data[i] = residual->f_data[i] + (residual->f_data[i] * (*scale->f_data)[i] * S->f_data[i]);
        #endif 
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
        #ifdef QUANT_MODE_QAT_SQ
            if (input->data[i] < 0) input->data[i] = 0;
        #else
            if (input->f_data[i] < 0) input->f_data[i] = 0;
        #endif
    }
}