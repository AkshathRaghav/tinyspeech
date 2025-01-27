#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "../main.c"

// We get the resulting Tensor in the format of a "float". 
Tensor softmax(Tensor *input) {
    int batch_size = input->shape[0];
    int num_classes = input->shape[1];

    u_int8_t shape[2] = {batch_size, num_classes};
    Tensor output = f_create_tensor(shape, 2);

    for (int n = 0; n < batch_size; n++) {
        // Find max value for numerical stability
        float max_val = -FLT_MAX;
        for (int c = 0; c < num_classes; c++) {
            int index = n * num_classes + c;
            if (input->data[index] > max_val) {
                max_val = input->data[index];
            }
        }

        // Compute exponentials and their sum
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            int index = n * num_classes + c;
            output.f_data[index] = expf(input->data[index] - max_val);
            sum_exp += output.f_data[index];
        }

        // Normalize to get probabilities
        for (int c = 0; c < num_classes; c++) {
            int index = n * num_classes + c;
            output.f_data[index] /= sum_exp;
        }
    }

    return output; 
}

int main() {
    Tensor input = load_tensor("input_tensor.bin", 2);
    // print_tensor(&input);
    Tensor expected_output = f_load_tensor("output_tensor.bin", 2);
    // f_print_tensor(&expected_output);
    Tensor output = softmax(&input);
    // f_print_tensor(&output);

    confirm_equal(&output, &expected_output);

    free_tensor(&input);
    free_tensor(&output);
    free_tensor(&expected_output);

    return 0;
}

	