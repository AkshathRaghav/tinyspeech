#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "../main.c"

// Using uint8_t for indexing, since Linear layer is used solely once at the end of the model. 
// 17x10 is the length of the tensor < 255.
Tensor fc_layer(Tensor *input, Tensor *weights) {
    int8_t batch_size = input->shape[0];
    int8_t input_features = input->shape[1];
    int8_t output_features = weights->shape[0];

    int8_t shape[2] = {batch_size, output_features};
    Tensor output = f_create_tensor(shape, 2);

    for (uint8_t n = 0; n < batch_size; n++) {
        for (uint8_t o = 0; o < output_features; o++) {
            float sum = 0.0f;
            for (int i = 0; i < input_features; i++) {
                sum += input->f_data[n * input_features + i] * weights->f_data[o * input_features + i];
            }
            output.f_data[n * output_features + o] = sum;
        }
    }
    return output; 
}

int main() { 
    Tensor input = f_load_tensor("input_tensor.bin", 2);
    Tensor weights = f_load_tensor("weights.bin", 2);
    Tensor expected_output = f_load_tensor("output_tensor.bin", 2);
    Tensor output = fc_layer(&input, &weights);

    confirm_equal(&output, &expected_output);

    free_tensor(&input);
    free_tensor(&output);
    free_tensor(&expected_output);

    return 0;
}