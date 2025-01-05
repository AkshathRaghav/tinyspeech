#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "../main.c"

int get_index(int n, int c, int h, int w, int N, int C, int H, int W) {
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

void relu(Tensor* input) {
    for (int i = 0; i < input->size; i++) {
        if (input->data[i] < 0) {
            input->data[i] = 0;
        }
    }
}

Tensor conv2d(Tensor *input, Tensor *weights, Tensor *bias, int stride, int padding) {
    int batch_size = input->shape[0];
    int in_channels = input->shape[1];
    int in_height = input->shape[2];
    int in_width = input->shape[3];

    int out_channels = weights->shape[0];
    int kernel_height = weights->shape[2];
    int kernel_width = weights->shape[3];

    int out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

    int8_t shape[4] = {batch_size, out_channels, out_height, out_width};
    Tensor output = create_tensor(shape, 4);

    for (int n = 0; n < batch_size; n++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int h = 0; h < out_height; h++) {
                for (int w = 0; w < out_width; w++) {
                    int32_t sum = 0;
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                            for (int kw = 0; kw < kernel_width; kw++) {
                                int ih = h * stride + kh - padding;
                                int iw = w * stride + kw - padding;

                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int in_index = n * (in_channels * in_height * in_width) +
                                                   ic * (in_height * in_width) +
                                                   ih * in_width + iw;

                                    int weight_index = oc * (in_channels * kernel_height * kernel_width) +
                                                       ic * (kernel_height * kernel_width) +
                                                       kh * kernel_width + kw;

                                    sum += input->data[in_index] * weights->data[weight_index];
                                }
                            }
                        }
                    }
                    sum += bias->data[oc];
                    int out_index = n * (out_channels * out_height * out_width) +
                                    oc * (out_height * out_width) +
                                    h * out_width + w;

                    output.data[out_index] = (int8_t)(sum/100);
                }
            }
        }
    }
    return output;
}   



int main() { 
    Tensor input = load_tensor("input_tensor.bin", 4);
    Tensor weights = load_tensor("weights.bin", 4);
    Tensor bias = load_tensor("bias.bin", 1);
    Tensor expected_output = load_tensor("output_tensor.bin", 4);

    Tensor output = conv2d(&input, &weights, &bias, 1, 1);

    confirm_equal(&output, &expected_output);

    free_tensor(&input);
    free_tensor(&output);
    free_tensor(&expected_output);

    return 0;
}