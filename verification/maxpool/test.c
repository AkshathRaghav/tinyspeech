#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

float mean(int8_t *arr, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / size;
}

typedef struct {
    int size;
    int8_t *data;
    int8_t shape[4]; // [Batch, Channel, Height, Width]
} Tensor;

Tensor load_tensor(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    Tensor tensor;

    for (int i = 0; i < 4; ++i) {
        fread(&tensor.shape[i], sizeof(int8_t), 1, file);
    }

    tensor.size = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3];
    tensor.data = (int8_t*)malloc(tensor.size * sizeof(int8_t));
    if (!tensor.data) {
        perror("Memory allocation failed for tensor.data");
        exit(EXIT_FAILURE);
    }

    fread(tensor.data, sizeof(int8_t), tensor.size, file);
    fclose(file);

    fprintf(stdout, "Loaded tensor with shape [%d, %d, %d, %d]\n", tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]);

    return tensor;
}

Tensor maxpool2d(Tensor* input, int kernel_size, int stride) {
    Tensor output;

    output.shape[0] = input->shape[0]; // Batch size
    output.shape[1] = input->shape[1]; // Channels
    output.shape[2] = (input->shape[2] - kernel_size) / stride + 1; // Height
    output.shape[3] = (input->shape[3] - kernel_size) / stride + 1; // Width

    output.size = output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3];
    output.data = (int8_t*)malloc(output.size * sizeof(int8_t));
    if (!output.data) {
        perror("Memory allocation failed for output.data");
        exit(EXIT_FAILURE);
    }

    for (int b = 0; b < output.shape[0]; b++) { // Batch dimension
        for (int c = 0; c < output.shape[1]; c++) { // Channel dimension
            for (int oh = 0; oh < output.shape[2]; oh++) { // Output height
                for (int ow = 0; ow < output.shape[3]; ow++) { // Output width
                    int max_value = INT8_MIN;

                    for (int kh = 0; kh < kernel_size; kh++) { // Kernel height
                        for (int kw = 0; kw < kernel_size; kw++) { // Kernel width
                            int ih = oh * stride + kh; // Input height index
                            int iw = ow * stride + kw; // Input width index

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

int confirm_equal(Tensor* output, Tensor* expected_output) {
    if (output->size != expected_output->size) return 0;
    for (int i = 0; i < output->size; i++) {
        if (output->data[i] != expected_output->data[i]) return 0;
    }
    return 1;
}

int main() {
    Tensor input = load_tensor("input_tensor.bin");
    Tensor expected_output = load_tensor("output_tensor.bin");

    Tensor output = maxpool2d(&input, 2, 2); 

    if (confirm_equal(&output, &expected_output)) {
        printf("Success.\n");
    } else {
        printf("Failed.\n");
    }

    free(input.data);
    free(output.data);
    free(expected_output.data);

    return 0;
}
