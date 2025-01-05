#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define INPUT_FEATURES 17
#define OUTPUT_FEATURES 10

typedef struct {
    float *data;
    int8_t shape[2]; // [N, F]
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

void linear(Tensor *input, Tensor *output, int8_t *weights) {
    int batch_size = input->shape[0];
    int input_features = INPUT_FEATURES;
    int output_features = OUTPUT_FEATURES;

    output->data = (float *)malloc(batch_size * output_features * sizeof(float));
    output->shape[0] = batch_size;
    output->shape[1] = output_features;

    int32_t int32_output = (int32_t *)malloc(batch_size * output_features * sizeof(int32_t));

    for (int n = 0; n < batch_size; n++) {
        for (int o = 0; o < output_features; o++) {
            int sum = 0; 
            for (int i = 0; i < input_features; i++) {
                sum += input->data[n * input_features + i] * weights[o * input_features + i];
            }
            int32_output[n * output_features + o] = sum;
        }
    }

    quantize_weights(int32_output, output->data, batch_size * output_features);
}

float mean(float *arr, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / size;
}

int main() { 

    Tensor input = load_tensor("input_tensor.bin");
    Tensor output;

    linear(&input, &output, fc_weights);

    Tensor expected_output = load_tensor("output_tensor.bin");

    fprintf(stdout, "Expected mean: %f, Actual mean: %f\n", mean(expected_output.data, expected_output.shape[0] * expected_output.shape[1]), mean(output.data, output.shape[0] * output.shape[1]));

    free(input); 
    free(output); 
    free(expected_output); 

    return 0; 
}