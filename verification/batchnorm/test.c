#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    float *data;
    int shape[4]; // [N, C, H, W]
} Tensor;

void batchnorm2d(Tensor *input, Tensor *output, float *mean, float *variance, float *gamma, float *beta) {
    int batch_size = input->shape[0];
    int channels = input->shape[1];
    int height = input->shape[2];
    int width = input->shape[3];

    output->data = (float *)malloc(batch_size * channels * height * width * sizeof(float));
    output->shape[0] = batch_size;
    output->shape[1] = channels;
    output->shape[2] = height;
    output->shape[3] = width;

    for (int n = 0; n < batch_size; n++) {
        for (int c = 0; c < channels; c++) {
            float mu = mean[c];
            float var = variance[c];
            float g = gamma[c];
            float b = beta[c];
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int index = n * (channels * height * width) + c * (height * width) + h * width + w;
                    float normalized = (input->data[index] - mu) / sqrtf(var);
                    output->data[index] = g * normalized + b;
                }
            }
        }
    }
}
