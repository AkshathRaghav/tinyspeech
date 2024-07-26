#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "conv_layer.h"

// Define the dimensions as per your requirement
#define IN_CHANNELS 1
#define MID_CHANNELS 7
#define KERNEL_SIZE 1
#define INPUT_HEIGHT 12
#define INPUT_WIDTH 94
#define BATCH_SIZE 1

typedef struct {
    int8_t *data;
    int8_t shape[4]; // [N, C, H, W]
    float scale; 
} Tensor;

Tensor load_tensor(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    Tensor tensor;

    for (int i = 0; i < 4; ++i) {
        fread(&tensor.shape[i], sizeof(int8_t), 1, file);
    }

    fread(&tensor.scale, sizeof(float), 1, file);

    int num_elements = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3];
    tensor.data = (int8_t *)malloc(num_elements * sizeof(int8_t));

    fread(tensor.data, sizeof(int8_t), num_elements, file);
    fclose(file);

    fprintf(stdout, "Loaded tensor with shape [%d, %d, %d, %d] and scale %f\n", tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.scale);

    return tensor;
}

int get_index(int n, int c, int h, int w, int N, int C, int H, int W) {
    return n * (C * H * W) + c * (H * W) + h * W + w;
}


void relu(int8_t *input, int size) {
    for (int i = 0; i < size; i++) {
        if (input[i] < 0) {
            input[i] = 0;
        }
    }
}

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


Tensor conv2d(Tensor *input, Tensor *output) {
    int input_size = input->shape[0] * input->shape[1] * input->shape[2] * input->shape[3];

    output->shape[0] = BATCH_SIZE;
    output->shape[1] = MID_CHANNELS;
    output->shape[2] = INPUT_HEIGHT;
    output->shape[3] = INPUT_WIDTH;
    int output_size = output->shape[0] * output->shape[1] * output->shape[2] * output->shape[3];
    output->data = (int8_t *)malloc(output_size * sizeof(int8_t));
    int32_t *int32_output = (int32_t *)malloc(output_size * sizeof(int32_t));

    fprintf(stdout, "Input size: %d, Output size: %d\n", input_size, output_size);

    for (int n = 0; n < BATCH_SIZE; n++) {
        for (int oc = 0; oc < MID_CHANNELS; oc++) {
            for (int h = 0; h < INPUT_HEIGHT; h++) {
                for (int w = 0; w < INPUT_WIDTH; w++) {
                    int out_index = get_index(n, oc, h, w, BATCH_SIZE, MID_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH);
                    int32_output[out_index] = 0;
                    for (int ic = 0; ic < IN_CHANNELS; ic++) {
                        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                                int ih = h + kh;
                                int iw = w + kw;

                                int in_index = get_index(n, ic, ih, iw, BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH);
                                int weight_index = get_index(oc, ic, kh, kw, MID_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE);

                                int8_t input_value = (int8_t) input->data[in_index];
                                int8_t weight_value = (int8_t) conv_weights[weight_index];
                            
                                if (input_value != 0 && weight_value != 0) {
                                    int32_output[out_index] += (int32_t) (input_value * weight_value);
                                } else { 
                                    int32_output[out_index] += 0;
                                }

                            }
                        }
                    }
                }
            }
        }
    }
    quantize_weights(int32_output, output->data, output_size);
    relu(output->data, output_size);
}

float mean(int8_t *data, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

int main() {
    Tensor input_tensor = load_tensor("input_tensor.bin");
    Tensor expected_output_tensor = load_tensor("output_tensor.bin");
    Tensor output_tensor;

    conv2d(&input_tensor, &output_tensor);   

    int output_size = output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2] * output_tensor.shape[3];
    float expected = mean(expected_output_tensor.data, output_size);
    float actual = mean(output_tensor.data, output_size);
    
    fprintf(stdout, "Expected mean: %f, Actual mean: %f\n", expected, actual);
    // Don't try to equate them, C implements the quantized version 
    
    free(input_tensor.data);
    free(expected_output_tensor.data);
    free(output_tensor.data);
    return 1; 
}
