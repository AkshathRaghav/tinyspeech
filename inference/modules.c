/**
 * @brief Applies Batch Normalization to an array of 32-bit integers.
 * 
 * @param input Pointer to the input array of 32-bit integers.
 * @param output Pointer to the output array of 32-bit integers.
 * @param mean The mean value for the batch normalization.
 * @param variance The variance value for the batch normalization.
 * @param gamma The scale parameter.
 * @param beta The shift parameter.
 * @param n_input The number of elements in the input array.
 */
void BatchNorm(int32_t *input, int32_t *output, int32_t mean, int32_t variance, int32_t gamma, int32_t beta, uint32_t n_input) {
    int32_t inv_std = 1 << 14;  // Example fixed-point value, adjust as necessary
    int32_t tmp;

    // Pre-compute the fixed-point scaling factor (1/sqrt(variance + epsilon))
    // Assuming epsilon is a small value like 1e-5 represented in fixed-point
    variance += (1 << 5);  // Adding epsilon
    while (variance > 0) {
        variance >>= 1;
        inv_std >>= 1;
    }

    for (uint32_t i = 0; i < n_input; i++) {
        // Normalize
        tmp = (input[i] - mean) * inv_std;

        // Scale and shift
        tmp = (tmp * gamma) >> 14;  // Adjust the shift based on the fixed-point format
        output[i] = tmp + beta;
    }
}

/**
 * @brief Applies Max Pooling to a 2D input array.
 * 
 * @param input Pointer to the input 2D array of 32-bit integers.
 * @param output Pointer to the output 2D array of 32-bit integers.
 * @param height The height of the input array.
 * @param width The width of the input array.
 * @param pool_height The height of the pooling window.
 * @param pool_width The width of the pooling window.
 */
void MaxPool(int8_t *input, int8_t *output, uint32_t height, uint32_t width) {
    uint32_t out_height = height / 2;
    uint32_t out_width = width / 2;

    for (uint32_t h = 0; h < out_height; h++) {
        for (uint32_t w = 0; w < out_width; w++) {
            int32_t max_val = input[2*h*width + 2*w];
            max_val = (input[2*h*width + 2*w + 1] > max_val) ? input[2*h*width + 2*w + 1] : max_val;
            max_val = (input[(2*h+1)*width + 2*w] > max_val) ? input[(2*h+1)*width + 2*w] : max_val;
            max_val = (input[(2*h+1)*width + 2*w + 1] > max_val) ? input[(2*h+1)*width + 2*w + 1] : max_val;
            output[h*out_width + w] = max_val;
        }
    }
}

/**
 * @brief Applies Upsampling to a 2D input array using nearest neighbor interpolation.
 * 
 * @param input Pointer to the input 2D array of 32-bit integers.
 * @param output Pointer to the output 2D array of 32-bit integers.
 * @param height The height of the input array.
 * @param width The width of the input array.
 * @param scale The scale factor for upsampling.
 */
void Upsample(int32_t *input, int32_t *output, uint32_t height, uint32_t width, uint32_t scale) {
    uint32_t out_height = height * scale;
    uint32_t out_width = width * scale;

    for (uint32_t i = 0; i < out_height; i++) {
        for (uint32_t j = 0; j < out_width; j++) {
            uint32_t src_y = i / scale;
            uint32_t src_x = j / scale;
            output[i * out_width + j] = input[src_y * width + src_x];
        }
    }
}
