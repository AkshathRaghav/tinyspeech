#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "../main.c"


void adaptive_avg_pool2d(Tensor *input, Tensor *output) {
    int8_t batch_size = input->shape[0];
    int8_t channels = input->shape[1];
    int8_t height = input->shape[2];
    int8_t width = input->shape[3];

    int8_t shape[4] = {batch_size, channels, 1, 1};
    Tensor output = create_tensor(shape, 4);

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
            output.data[out_index] = (int8_t*) (sum / (height * width));
        }
    }
}

int main() {
    Tensor input = load_tensor("input_tensor.bin", 4);
    print_tensor(&input);
    Tensor expected_output = load_tensor("output_tensor.bin", 4);
    print_tensor(&expected_output);
    Tensor output = adaptive_avg_pool2d(&input);
    print_tensor(&output);

    confirm_equal(&output, &expected_output);

    free_tensor(&input);
    free_tensor(&output);
    free_tensor(&expected_output);

    return 0;
}
