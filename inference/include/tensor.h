#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int dims;       // Number of Dimensions
    int size;      // Number of elements
    int8_t *shape; // 2D or 4D 
    int8_t *data;  // Flattened data
    float *f_data;  // Flattened data for float
} Tensor;

Tensor f_create_tensor(int8_t* shape, int8_t dims);
Tensor create_tensor(int8_t* shape, int8_t dims);
Tensor f_load_tensor(const char* filename, int8_t dims);
Tensor load_tensor(const char* filename, int8_t dims);
void free_tensor(Tensor* tensor);

// TODO: Remove these functions and their use in ./inference 
void save_tensor(const char* filename, Tensor* tensor);
void confirm_equal(Tensor* output, Tensor* expected_output);
void f_print_tensor(Tensor *tensor);
void print_tensor(Tensor *tensor);
void print_tensor_recursive(int8_t *data, int8_t *shape, int dims, int current_dim, int indent_level);
void f_print_tensor_recursive(float *data, int8_t *shape, int dims, int current_dim, int indent_level);

#endif