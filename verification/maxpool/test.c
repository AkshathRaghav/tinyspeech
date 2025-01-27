#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "../main.c"

Tensor maxpool2d(Tensor* input, int kernel_size, int stride) {
    u_int8_t shape[4] =  {input->shape[0], input->shape[1], ((input->shape[2] - kernel_size) / stride + 1), ((input->shape[3] - kernel_size) / stride + 1)};
    Tensor output = create_tensor(shape, 4);

    for (int b = 0; b < output.shape[0]; b++) { // Batch 
        for (int c = 0; c < output.shape[1]; c++) { // Channel 
            for (int oh = 0; oh < output.shape[2]; oh++) { // Output height
                for (int ow = 0; ow < output.shape[3]; ow++) { // Output width
                    int max_value = INT8_MIN;

                    for (int kh = 0; kh < kernel_size; kh++) { // Kernel height
                        for (int kw = 0; kw < kernel_size; kw++) { // Kernel width
                            int ih = oh * stride + kh; 
                            int iw = ow * stride + kw; 

                            int input_index = b * (input->shape[1] * input->shape[2] * input->shape[3]) +
                                              c * (input->shape[2] * input->shape[3]) +
                                              ih * input->shape[3] +
                                              iw;

                            if (input->data[input_index] > max_value) {
                                max_value = input->data[input_index];
                            }
                        }
                    }

                    int output_index = b * (output.shape[1] * output.shape[2] * output.shape[3]) +
                                       c * (output.shape[2] * output.shape[3]) +
                                       oh * output.shape[3] +
                                       ow;

                    output.data[output_index] = max_value;
                }
            }
        }
    }

    return output;
}


int main() {
    Tensor input = load_tensor("input_tensor.bin", 4);
    Tensor expected_output = load_tensor("output_tensor.bin", 4);
    Tensor output = maxpool2d(&input, 2, 2); 

    confirm_equal(&output, &expected_output);

    free_tensor(&input);
    free_tensor(&output);
    free_tensor(&expected_output);

    return 0;
}
