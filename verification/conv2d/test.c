#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "conv_layer.h"

void dequantize(int8_t *quantized, float scale, int8_t zp, float *output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (quantized[i] - zp) * scale;
    }
}

void conv2d(float *input, float *output, float *kernel, float *bias,
            int input_height, int input_width, int input_depth,
            int kernel_height, int kernel_width, int output_depth,
            int stride, int padding) {
    
    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;

    for (int od = 0; od < output_depth; od++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                float sum = 0.0;
                for (int id = 0; id < input_depth; id++) {
                    for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;
                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                sum += input[id * input_height * input_width + ih * input_width + iw] *
                                       kernel[od * input_depth * kernel_height * kernel_width +
                                              id * kernel_height * kernel_width +
                                              kh * kernel_width + kw];
                            }
                        }
                    }
                }
                output[od * output_height * output_width + oh * output_width + ow] = sum + bias[od];
            }
        }
    }
}

void conv2d_quantized(int8_t *input_q, int8_t *kernel_q, int8_t *bias_q, int8_t *output_q,
                      int input_height, int input_width, int input_depth,
                      int kernel_height, int kernel_width, int output_depth,
                      int stride, int padding, float input_scale, float kernel_scale,
                      float bias_scale, float output_scale, int8_t input_zp, int8_t kernel_zp,
                      int8_t bias_zp, int8_t output_zp) {
    
    int input_size = input_height * input_width * input_depth;
    int kernel_size = kernel_height * kernel_width * input_depth * output_depth;
    int bias_size = output_depth;
    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;
    int output_size = output_height * output_width * output_depth;

    float *input = (float *)malloc(input_size * sizeof(float));
    float *kernel = (float *)malloc(kernel_size * sizeof(float));
    float *bias = (float *)malloc(bias_size * sizeof(float));
    float *output = (float *)malloc(output_size * sizeof(float));

    dequantize(input_q, input_scale, input_zp, input, input_size);
    dequantize(kernel_q, kernel_scale, kernel_zp, kernel, kernel_size);
    dequantize(bias_q, bias_scale, bias_zp, bias, bias_size);

    conv2d(input, output, kernel, bias,
           input_height, input_width, input_depth,
           kernel_height, kernel_width, output_depth,
           stride, padding);

    for (int i = 0; i < output_size; i++) {
        output_q[i] = round(output[i] / output_scale) + output_zp;
    }

    free(input);
    free(kernel);
    free(bias);
    free(output);
}

int main() {
    // Load the quantized weights 
    const uint8_t *quantized_weights = conv_weights;
    const uint8_t *quantized_bias = conv_bias;

    int input_height = 12;
    int input_width = 94;
    int input_depth = 1;

    int kernel_height = 1;
    int kernel_width = 1;
    int output_depth = 7;

    int stride = 1;
    int padding = 0;
    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;

    float input_scale = 0.1f;
    float kernel_scale = 0.1f;
    float bias_scale = 0.1f;
    float output_scale = 0.1f;
    int8_t input_zp = 128;
    int8_t kernel_zp = 128;
    int8_t bias_zp = 128;
    int8_t output_zp = 128;

    // Allocate memory for the input and output tensors
    int8_t *input_q = (int8_t *)malloc(64 * input_height * input_width * sizeof(int8_t));
    int8_t *output_q = (int8_t *)malloc(64 * output_height * output_width * output_depth * sizeof(int8_t));

    for (int i = 0; i < 64 * input_height * input_width; i++) {
        input_q[i] = rand() % 256;
    }

    for (int i = 0; i < 64; i++) {
        conv2d_quantized(input_q + i * input_height * input_width, quantized_weights, quantized_bias, output_q + i * output_height * output_width * output_depth,
                         input_height, input_width, input_depth, kernel_height, kernel_width, output_depth, stride, padding,
                         input_scale, kernel_scale, bias_scale, output_scale, input_zp, kernel_zp, bias_zp, output_zp);
    }

    free(input_q);
    free(output_q);

    return 0;
}
