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

    int num_elements = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3];
    tensor.data = (int8_t *)malloc(num_elements * sizeof(int8_t));

    fread(tensor.data, sizeof(int8_t), num_elements, file);
    fclose(file);

    fprintf(stdout, "Loaded tensor with shape [%d, %d, %d, %d]\n", tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]);

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

void max_pool2d(Tensor *input, Tensor *output, int8_t kernel_size, int8_t stride) {
    int8_t batch_size = input->shape[0];
    int8_t channels = input->shape[1];
    int8_t input_height = input->shape[2];
    int8_t input_width = input->shape[3];

    int8_t output_height = (input_height - kernel_size) / stride + 1;
    int8_t output_width = (input_width - kernel_size) / stride + 1;

    output->data = (int8_t *)malloc(batch_size * channels * output_height * output_width * sizeof(int8_t));
    output->shape[0] = batch_size;
    output->shape[1] = channels;
    output->shape[2] = output_height;
    output->shape[3] = output_width;

    for (int8_t n = 0; n < batch_size; n++) {
        for (int8_t c = 0; c < channels; c++) {
            for (int8_t h = 0; h < output_height; h++) {
                for (int8_t w = 0; w < output_width; w++) {
                    int8_t max_val = -128;
                    for (int8_t kh = 0; kh < kernel_size; kh++) {
                        for (int8_t kw = 0; kw < kernel_size; kw++) {
                            int input_h = h * stride + kh;
                            int input_w = w * stride + kw;
                            int index = n * (channels * input_height * input_width) + c * (input_height * input_width) + input_h * input_width + input_w;
                            if (input->data[index] > max_val) {
                                max_val = input->data[index];
                            }
                        }
                    }
                    int output_index = n * (channels * output_height * output_width) + c * (output_height * output_width) + h * output_width + w;
                    output->data[output_index] = max_val;
                }
            }
        }
    }
}

void softmax(Tensor *input, Tensor *output) {
    int8_t batch_size = input->shape[0];
    int8_t num_classes = input->shape[1];

    output->data = (int8_t *)malloc(batch_size * num_classes * sizeof(int8_t));
    int32_t *int32_output = (int32_t *)malloc(batch_size * num_classes * sizeof(int32_t));
    output->shape[0] = batch_size;
    output->shape[1] = num_classes;

    for (int8_t n = 0; n < batch_size; n++) {
        int8_t max_val = -128;
        for (int8_t c = 0; c < num_classes; c++) {
            int index = n * num_classes + c;
            if (input->data[index] > max_val) {
                max_val = input->data[index];
            }
        }

        int32_t sum_exp = 0;
        for (int8_t c = 0; c < num_classes; c++) {
            int index = n * num_classes + c;
            sum_exp += (int32_t) expf(input->data[index] - max_val);
        }

        for (int8_t c = 0; c < num_classes; c++) {
            int index = n * num_classes + c;
            int32_output[index] = (int32_t) expf(input->data[index] - max_val) / sum_exp;
        }
    }

    quantize_weights(int32_output, output->data, batch_size * num_classes);
}

void adaptive_avg_pool2d(Tensor *input, Tensor *output) {
    int8_t batch_size = input->shape[0];
    int8_t channels = input->shape[1];
    int8_t height = input->shape[2];
    int8_t width = input->shape[3];

    output->data = (int8_t *)malloc(batch_size * channels * sizeof(int8_t));
    int32_t *int32_output = (int32_t *)malloc(batch_size * channels * sizeof(int32_t));
    output->shape[0] = batch_size;
    output->shape[1] = channels;
    output->shape[2] = 1;
    output->shape[3] = 1;

    for (int8_t n = 0; n < batch_size; n++) {
        for (int8_t c = 0; c < channels; c++) {
            int sum = 0;
            for (int8_t h = 0; h < height; h++) {
                for (int8_t w = 0; w < width; w++) {
                    int index = n * (channels * height * width) + c * (height * width) + h * width + w;
                    sum += input->data[index];
                }
            }
            int out_index = n * channels + c;
            int32_output[out_index] = sum / (height * width);
        }
    }

    quantize_weights(int32_output, output->data, batch_size * channels);
}

void upsample_nearest(Tensor *input, Tensor *output, int scale_factor) {
    int8_t batch_size = input->shape[0];
    int8_t channels = input->shape[1];
    int8_t input_height = input->shape[2];
    int8_t input_width = input->shape[3];

    int output_height = input_height * scale_factor;
    int output_width = input_width * scale_factor;

    output->data = (int8_t *)malloc(batch_size * channels * output_height * output_width * sizeof(int8_t));
    output->shape[0] = batch_size;
    output->shape[1] = channels;
    output->shape[2] = output_height;
    output->shape[3] = output_width;

    for (int n = 0; n < batch_size; n++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < output_height; h++) {
                for (int w = 0; w < output_width; w++) {
                    int input_h = h / scale_factor;
                    int input_w = w / scale_factor;
                    int input_index = n * (channels * input_height * input_width) + c * (input_height * input_width) + input_h * input_width + input_w;
                    int output_index = n * (channels * output_height * output_width) + c * (output_height * output_width) + h * output_width + w;
                    output->data[output_index] = input->data[input_index];
                }
            }
        }
    }
}

float mean(int8_t *data, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

int main() {
    Tensor output;
    Tensor input = load_tensor("input_tensor.bin");

    max_pool2d(&input, &output, 2, 2);
    float max_pool_mean = mean(output.data, output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3]);

    upsample_nearest(&input, &output, 2);
    float upsample_mean = mean(output.data, output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3]);

    adaptive_avg_pool2d(&input, &output);
    float avg_pool_mean = mean(output.data, output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3]);

    input = load_tensor("input_tensor_softmax.bin");

    softmax(&input, &output);
    float softmax_mean = mean(output.data, output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3]);

    fprintf(stdout, "Max pool mean: %f, Upsample mean: %f, Avg pool mean: %f, Softmax mean: %f\n", max_pool_mean, upsample_mean, avg_pool_mean, softmax_mean);

    return 0;
}