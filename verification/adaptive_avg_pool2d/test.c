#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "../main.c"


Tensor adaptive_avg_pool2d(Tensor *input) {
    int batch_size = input->shape[0];
    int channels = input->shape[1];
    int height = input->shape[2];
    int width = input->shape[3];

    int8_t shape[4] = {batch_size, channels, 1, 1};
    Tensor output = f_create_tensor(shape, 4);

    for (int n = 0; n < batch_size; n++) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;  
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int index = n * (channels * height * width) + 
                                c * (height * width) + 
                                h * width + 
                                w;
                    sum += input->f_data[index];
                }
            }
            int out_index = n * channels + c; // Adjust for the output shape
            output.f_data[out_index] = sum / (height * width);
        }
    }

    return output;
}


int main() {
    Tensor input = f_load_tensor("./input_tensor.bin", 4);
    Tensor expected_output = f_load_tensor("./output_tensor.bin", 4);
    Tensor output = adaptive_avg_pool2d(&input);

    confirm_equal(&output, &expected_output);

    free_tensor(&input);
    free_tensor(&output);
    free_tensor(&expected_output);

    return 0;
}

