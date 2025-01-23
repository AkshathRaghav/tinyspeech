#include "tensor.h"

void print_tensor_upto_n(Tensor* input, int len) { 
    for (int i = 0; i < len; i++) { 
        printf("%d, ", input->data[i]);
    }
    printf("\n");
}

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
    if (output->size != expected_output->size) { 
        flag = 0; 
        perror("Shape mismatch in tensors.");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < output->size; i++) { 
        if (output->data != NULL) { 
            // printf("%d, %d\n", output->data[i], expected_output->data[i]);

            // Verifying conv2d and batchnorm was really painful. 
            // I'm setting the error bar somewhat high, since I'm scaling the inputs and weights by factors of 10 (to ensure it doesnt get truncated when quantizing)
            // Then, I'm scaling the outputs back. The error is not to be dismissed, but was a test to see if the math was right, ignoring the type problem.

            if (fabs(output->data[i] - expected_output->data[i]) > 10) { 
                flag = 0; 
                break; 
            }
        } else if (output->f_data != NULL) { // assumes both are float
            // printf("%.2f, %.2f\n", output->f_data[i], expected_output->f_data[i]);
            if (fabs(output->f_data[i] - expected_output->f_data[i]) > 0.5) { 
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

Tensor f_create_tensor(u_int8_t* shape, int8_t dims) {
    Tensor tensor;

    tensor.dims = dims; 
    tensor.data = NULL;
    tensor.shape = (u_int8_t *)malloc(dims * sizeof(int8_t));
    if (!tensor.shape) {
        perror("Memory allocation failed for tensor.shape");
        exit(EXIT_FAILURE);
    }
    memcpy(tensor.shape, shape, dims * sizeof(int8_t));

    if (dims == 2) { 
        tensor.size = tensor.shape[0] * tensor.shape[1];
    } else if (dims == 4) { 
        tensor.size = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3];
    } else if (dims == 1) { 
        tensor.size = *tensor.shape;
    } else { 
        perror("Incorrect `dims` size. Expected 2 or 4");
        exit(EXIT_FAILURE);
    }

    tensor.f_data = (float *)malloc(tensor.size * sizeof(float));
    if (!tensor.f_data) {
        perror("Memory allocation failed for tensor.data");
        exit(EXIT_FAILURE);
    }

    return tensor;
}


Tensor create_tensor(u_int8_t* shape, int8_t dims) {
    Tensor tensor;

    tensor.dims = dims; 
    tensor.f_data = NULL;
    tensor.shape = (u_int8_t *)malloc(dims * sizeof(int8_t));
    if (!tensor.shape) {
        perror("Memory allocation failed for tensor.shape");
        exit(EXIT_FAILURE);
    }
    memcpy(tensor.shape, shape, dims * sizeof(int8_t));

    if (dims == 2) { 
        tensor.size = tensor.shape[0] * tensor.shape[1];
    } else if (dims == 4) { 
        tensor.size = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3];
    } else if (dims == 1) { 
        tensor.size = *tensor.shape;
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
        tensor.shape = (u_int8_t *)malloc(2 * sizeof(int8_t));
        fread(tensor.shape, sizeof(int8_t), 2, file);
        tensor.size = tensor.shape[0] * tensor.shape[1];
    } else if (dims == 4) { 
        tensor.shape = (u_int8_t *)malloc(4 * sizeof(int8_t));
        fread(tensor.shape, sizeof(int8_t), 4, file);
        tensor.size = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3];
    } else if (dims == 1) { 
        tensor.shape = (u_int8_t *)malloc(1 * sizeof(int8_t));
        fread(tensor.shape, sizeof(int8_t), 1, file);
        tensor.size = *tensor.shape;
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
    } else if (dims == 4) { 
        fprintf(stdout, "Loaded tensor with shape [%d, %d, %d, %d]\n", tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]);
    } else { 
        fprintf(stdout, "Loaded tensor with shape [%d]\n", *tensor.shape);
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
        tensor.shape = (u_int8_t *)malloc(2 * sizeof(int8_t));
        fread(tensor.shape, sizeof(int8_t), 2, file);
        tensor.size = tensor.shape[0] * tensor.shape[1];
    } else if (dims == 4) { 
        tensor.shape = (u_int8_t *)malloc(4 * sizeof(int8_t));
        fread(tensor.shape, sizeof(int8_t), 4, file);
        tensor.size = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3];
    } else if (dims == 1) { 
        tensor.shape = (u_int8_t *)malloc(1 * sizeof(int8_t));
        fread(tensor.shape, sizeof(int8_t), 1, file);
        tensor.size = *tensor.shape;
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
    } else if (dims == 4) { 
        fprintf(stdout, "Loaded tensor with shape [%d, %d, %d, %d]\n", tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]);
    } else { 
        fprintf(stdout, "Loaded tensor with shape [%d]\n", *tensor.shape);
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
    if (tensor->shape) free(tensor->shape);
    tensor->shape = tensor->data = NULL;
    tensor->f_data = NULL;
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