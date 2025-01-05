#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "./main.h"

float mean(int8_t *arr, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / size;
}

void confirm_equal(Tensor* output, Tensor* expected_output) { 
    // printf("%d, %d\n", output->size, expected_output->size);
    int flag = 1; 
    if (output->size != expected_output->size) flag = 0; 
    for (int i = 0; i < output->size; i++) { 
        // printf("%d, %d\n", output->data[i], expected_output->data[i]);
        if (output->data != NULL) { 
            if (output->data[i] != expected_output->data[i]) { 
                flag = 0; 
                break; 
            }
        } else if (output->f_data != NULL) { // assumes both are float
            if (fabs(output->f_data[i] - expected_output->f_data[i]) > 0.1) { 
                flag = 0; 
                break; 
            }
        } else { 
            perror("tensor.data nor f_data initialized! Cannot run confirm_equal."); 
            exit(EXIT_FAILURE);
        }
    }
    if (flag) printf("Success.\n");
    else printf("Failed.\n");
}

Tensor f_create_tensor(int8_t* shape, int8_t dims) {
    Tensor tensor;

    tensor.dims = dims; 
    tensor.data = NULL;
    tensor.shape = (int8_t *)malloc(dims * sizeof(int8_t));
    if (!tensor.shape) {
        perror("Memory allocation failed for tensor.shape");
        exit(EXIT_FAILURE);
    }
    memcpy(tensor.shape, shape, dims * sizeof(int8_t));

    if (dims == 2) { 
        tensor.size = tensor.shape[0] * tensor.shape[1];
    } else if (dims == 4) { 
        tensor.size = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3];
    } else { 
        perror("Incorrect `dims` size. Expected 2 or 4");
        exit(EXIT_FAILURE);
    }

    tensor.f_data = (float *)malloc(tensor.size * sizeof(int8_t));
    if (!tensor.f_data) {
        perror("Memory allocation failed for tensor.data");
        exit(EXIT_FAILURE);
    }

    return tensor;
}


Tensor create_tensor(int8_t* shape, int8_t dims) {
    Tensor tensor;

    tensor.dims = dims; 
    tensor.f_data = NULL;
    tensor.shape = (int8_t *)malloc(dims * sizeof(int8_t));
    if (!tensor.shape) {
        perror("Memory allocation failed for tensor.shape");
        exit(EXIT_FAILURE);
    }
    memcpy(tensor.shape, shape, dims * sizeof(int8_t));

    if (dims == 2) { 
        tensor.size = tensor.shape[0] * tensor.shape[1];
    } else if (dims == 4) { 
        tensor.size = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3];
    } else { 
        perror("Incorrect `dims` size. Expected 2 or 4");
        exit(EXIT_FAILURE);
    }

    tensor.data = (int8_t *)malloc(tensor.size * sizeof(int8_t));
    if (!tensor.data) {
        perror("Memory allocation failed for tensor.data");
        exit(EXIT_FAILURE);
    }
    return tensor;
}

Tensor f_load_tensor(const char* filename, int8_t dims) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    Tensor tensor;

    tensor.dims = dims; 
    tensor.data = NULL;

    if (dims == 2) { 
        tensor.shape = (int8_t *)malloc(2 * sizeof(int8_t));
        fread(tensor.shape, sizeof(int8_t), 2, file);
        tensor.size = tensor.shape[0] * tensor.shape[1];
    } else if (dims == 4) { 
        tensor.shape = (int8_t *)malloc(4 * sizeof(int8_t));
        fread(tensor.shape, sizeof(int8_t), 4, file);
        tensor.size = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3];
    } else { 
        perror("Incorrect `dims` size. Expected 2 or 4");
        exit(EXIT_FAILURE);
    }

    tensor.f_data = (float *)malloc(tensor.size * sizeof(float));
    if (!tensor.f_data) {
        perror("Memory allocation failed for tensor.data");
        exit(EXIT_FAILURE);
    }
    fread(tensor.f_data, sizeof(float), tensor.size, file);

    fclose(file);

    if (dims == 2) { 
        fprintf(stdout, "Loaded tensor with shape [%d, %d]\n", tensor.shape[0], tensor.shape[1]);
    } else  { 
        fprintf(stdout, "Loaded tensor with shape [%d, %d, %d, %d]\n", tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]);
    }

    return tensor;
}


Tensor load_tensor(const char* filename, int8_t dims) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    Tensor tensor;

    tensor.dims = dims; 
    tensor.f_data = NULL;

    if (dims == 2) { 
        tensor.shape = (int8_t *)malloc(2 * sizeof(int8_t));
        fread(tensor.shape, sizeof(int8_t), 2, file);
        tensor.size = tensor.shape[0] * tensor.shape[1];
    } else if (dims == 4) { 
        tensor.shape = (int8_t *)malloc(4 * sizeof(int8_t));
        fread(tensor.shape, sizeof(int8_t), 4, file);
        tensor.size = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3];
    } else { 
        perror("Incorrect `dims` size. Expected 2 or 4");
        exit(EXIT_FAILURE);
    }

    tensor.data = (int8_t *)malloc(tensor.size * sizeof(int8_t));
    if (!tensor.data) {
        perror("Memory allocation failed for tensor.data");
        exit(EXIT_FAILURE);
    }
    fread(tensor.data, sizeof(int8_t), tensor.size, file);

    fclose(file);

    if (dims == 2) { 
        fprintf(stdout, "Loaded tensor with shape [%d, %d]\n", tensor.shape[0], tensor.shape[1]);
    } else  { 
        fprintf(stdout, "Loaded tensor with shape [%d, %d, %d, %d]\n", tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]);
    }

    return tensor;
}

void save_tensor(const char* filename, Tensor* tensor) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    fwrite(&(tensor->size), sizeof(int), 1, file);
    fwrite(&(tensor->dims), sizeof(int), 1, file);

    if (tensor->dims == 2) { 
        fwrite(tensor->shape, sizeof(int8_t), 2, file);
    } else { 
        fwrite(tensor->shape, sizeof(int8_t), 4, file);
    }

    fwrite(tensor->data, sizeof(float), tensor->size, file);

    fclose(file);

    if (tensor->dims == 2) { 
        fprintf(stdout, "Saved tensor with shape [%d, %d]\n", tensor->shape[0], tensor->shape[1]);
    } else  { 
        fprintf(stdout, "Saved tensor with shape [%d, %d, %d, %d]\n", tensor->shape[0], tensor->shape[1], tensor->shape[2], tensor->shape[3]);
    }
}

void free_tensor(Tensor* tensor) {
    if (!tensor->data) { 
        if (!tensor->f_data) { 
            perror("tensor.data nor f_data initialized!"); 
            exit(EXIT_FAILURE); 
        } else { 
            free(tensor->f_data);
        }
    } else { 
        free(tensor->data);
    }
}


void f_print_tensor_recursive(float *data, int8_t *shape, int dims, int current_dim, int indent_level) {
    // Print the opening bracket for the current dimension
    for (int i = 0; i < indent_level; i++) {
        printf(" ");
    }
    printf("[");

    if (current_dim == dims - 1) {
        // Base case: Print the values in the innermost dimension
        for (int i = 0; i < shape[current_dim]; i++) {
            printf("%.2f", data[i]);
            if (i < shape[current_dim] - 1) {
                printf(", ");
            }
        }
    } else {
        // Recursive case: Print subarrays for the next dimension
        int stride = 1;
        for (int i = current_dim + 1; i < dims; i++) {
            stride *= shape[i];
        }

        for (int i = 0; i < shape[current_dim]; i++) {
            if (i > 0) {
                printf(",\n");
            }
            f_print_tensor_recursive(data + i * stride, shape, dims, current_dim + 1, indent_level + 4);
        }
    }

    // Print the closing bracket for the current dimension
    printf("]");
    if (current_dim == 0) {
        printf("\n");
    }
}

void print_tensor_recursive(int8_t *data, int8_t *shape, int dims, int current_dim, int indent_level) {
    // Print the opening bracket for the current dimension
    for (int i = 0; i < indent_level; i++) {
        printf(" ");
    }
    printf("[");

    if (current_dim == dims - 1) {
        // Base case: Print the values in the innermost dimension
        for (int i = 0; i < shape[current_dim]; i++) {
            printf("%d", data[i]);
            if (i < shape[current_dim] - 1) {
                printf(", ");
            }
        }
    } else {
        // Recursive case: Print subarrays for the next dimension
        int stride = 1;
        for (int i = current_dim + 1; i < dims; i++) {
            stride *= shape[i];
        }

        for (int i = 0; i < shape[current_dim]; i++) {
            if (i > 0) {
                printf(",\n");
            }
            print_tensor_recursive((int8_t*) (data + i * stride), shape, dims, current_dim + 1, indent_level + 4);
        }
    }

    // Print the closing bracket for the current dimension
    printf("]");
    if (current_dim == 0) {
        printf("\n");
    }
}

void f_print_tensor(Tensor *tensor) {
    if (tensor->dims == 0 || tensor->shape == NULL || tensor->f_data == NULL) {
        printf("[]\n");
        return;
    }

    f_print_tensor_recursive(tensor->f_data, tensor->shape, tensor->dims, 0, 0);
}

void print_tensor(Tensor *tensor) {
    if (tensor->dims == 0 || tensor->shape == NULL || tensor->data == NULL) {
        printf("[]\n");
        return;
    }

    print_tensor_recursive(tensor->data, tensor->shape, tensor->dims, 0, 0);
}