#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "../main.c"

int get_index(int n, int c, int h, int w, int N, int C, int H, int W) {
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

void batchnorm2d(Tensor* input, Tensor* mean, Tensor* variance, Tensor* gamma, Tensor* beta) {
    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];

    int8_t shape[4] = {N, C, H, W};
    Tensor output = create_tensor(shape, 4);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float var_sqrt = sqrt(variance->f_data[c] + 0.0001);
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = get_index(n, c, h, w, N, C, H, W);
                    output.data[idx] = (int8_t) ((gamma->f_data[c] * ((input->data[idx] - mean->f_data[c]) / var_sqrt) + beta->f_data[c])/10);
                }
            }
        }
    }
}

int main() {
    Tensor input = load_tensor("input_tensor.bin", 4);
    Tensor mean = f_load_tensor("mean.bin", 1);
    // f_print_tensor(&mean);
    Tensor variance = f_load_tensor("variance.bin", 1);
    // f_print_tensor(&variance);
    Tensor gamma = f_load_tensor("gamma.bin", 1);
    // f_print_tensor(&gamma);
    Tensor beta = f_load_tensor("beta.bin", 1);
    // f_print_tensor(&beta);
    Tensor expected_output = load_tensor("output_tensor.bin", 4);

    batchnorm2d(&input, &mean, &variance, &gamma, &beta);

    confirm_equal(&input, &expected_output);

    free_tensor(&input);
    free_tensor(&mean);
    free_tensor(&variance);
    free_tensor(&gamma);
    free_tensor(&beta);
    free_tensor(&expected_output);

    return 0;
}

// gcc test.c -o test -lm -Wall -Iverification && 	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out1.txt ./test 