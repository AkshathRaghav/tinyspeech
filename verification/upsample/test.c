#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "../main.c"

Tensor upsample_nearest(Tensor* input, int in_size, int8_t scale_factor) {

    u_int8_t shape[4] = {input->shape[0], input->shape[1], input->shape[2] * scale_factor, input->shape[3] * scale_factor};
    Tensor output = create_tensor(shape, 4); 

    if (!output.data) {
        perror("Memory allocation failed for output.data");
        exit(EXIT_FAILURE);
    }

    for (int b = 0; b < output.shape[0]; b++) { // Batch 
        for (int c = 0; c < output.shape[1]; c++) { // Channel 
            for (int h = 0; h < output.shape[2]; h++) { // Height 
                int nearest_h = h / scale_factor;
                for (int w = 0; w < output.shape[3]; w++) { // Width 
                    int nearest_w = w / scale_factor;
                    int input_index = b * (input->shape[1] * input->shape[2] * input->shape[3]) +
                                      c * (input->shape[2] * input->shape[3]) +
                                      nearest_h * input->shape[3] +
                                      nearest_w;

                    int output_index = b * (output.shape[1] * output.shape[2] * output.shape[3]) +
                                       c * (output.shape[2] * output.shape[3]) +
                                       h * output.shape[3] +
                                       w;

                    output.data[output_index] = input->data[input_index];
                }
            }
        }
    }
    fprintf(stdout, "Loaded tensor with shape [%d, %d, %d, %d]\n", output.shape[0], output.shape[1], output.shape[2], output.shape[3]);
    return output;
}


int main() {
    Tensor input = load_tensor("input_tensor.bin", 4);
    Tensor expected_output = load_tensor("output_tensor.bin", 4);
    Tensor output; 

    output = upsample_nearest(&input, input.size, 2);

    confirm_equal(&output, &expected_output);

    free(input.data); 
    free(output.data); 
    free(expected_output.data); 

    return 0;
}

