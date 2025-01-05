Tensor upsample_nearest(Tensor* input, int in_size, int8_t scale_factor) {
    Tensor output; 

    output.shape[0] = input->shape[0]; // Batch size
    output.shape[1] = input->shape[1]; // Channels
    output.shape[2] = input->shape[2] * scale_factor; // Height
    output.shape[3] = input->shape[3] * scale_factor; // Width

    output.size = output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3];

    output.data = (int8_t*)malloc(output.size * sizeof(int8_t));
    if (!output.data) {
        perror("Memory allocation failed for output.data");
        exit(EXIT_FAILURE);
    }

    for (int b = 0; b < output.shape[0]; b++) { // Batch dimension
        for (int c = 0; c < output.shape[1]; c++) { // Channel dimension
            for (int h = 0; h < output.shape[2]; h++) { // Height dimension
                int nearest_h = h / scale_factor; // Map output height to input height
                for (int w = 0; w < output.shape[3]; w++) { // Width dimension
                    int nearest_w = w / scale_factor; // Map output width to input width
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
