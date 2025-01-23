#include "modules.h"
#include "misc.h"
#include "weights.h"

Tensor batchnorm2d(Tensor* input, Tensor* gamma, Tensor* beta, Tensor* scale, Tensor* mean, Tensor* variance) {
    int8_t C = input->shape[1];
    int8_t H = input->shape[2];
    int8_t W = input->shape[3];

    Tensor output = f_create_tensor(input->shape, 4);

    for (int8_t n = 0; n < input->shape[0]; n++) {
        for (int8_t c = 0; c < C; c++) {
            float var_sqrt = (variance->f_data[c] != 0) ? sqrtf(variance->f_data[c] + 0.0001f) : 0.0001f;
            for (int8_t h = 0; h < H; h++) {
                for (int8_t w = 0; w < W; w++) {
                    int32_t idx = n * (C * H * W) + c * (H * W) + h * W + w;
                    float scaled = gamma->f_data[c] * (input->f_data[idx] - mean->f_data[c]) / var_sqrt + beta->f_data[c];
                    output.f_data[idx] = roundf(scaled);
                }
            }
        }
    }

    #ifdef QUANT_MODE_QAT_SQ
        Tensor quant_output = create_tensor(output.shape, 4);
        quantize_weights(&output, &quant_output, &(scale->f_data), CONVERT_INT8);

        free_tensor(output);
        output = quant_output;
    #endif 

    free_tensor(input);
    return output;
}

Tensor adaptive_avg_pool2d(Tensor *input) {
    int batch_size = input->shape[0];
    int channels = input->shape[1];
    int height = input->shape[2];
    int width = input->shape[3];

    int8_t shape[4] = {batch_size, channels, 1, 1};
    Tensor output = f_create_tensor(shape, 4);

    for (int n = 0; n < batch_size; n++) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;  
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int index = n * (channels * height * width) + 
                                c * (height * width) + 
                                h * width + 
                                w;
                    #ifdef QUANT_MODE_QAT_SQ
                        sum += input->data[index];
                    #else
                        sum += input->f_data[index];
                    #endif
                }
            }
            int out_index = n * channels + c; // Adjust for the output shape
            output.f_data[out_index] = sum / (height * width);
        }
    }

    return output;
}

Tensor conv2d(Tensor *input, Tensor *weights, Tensor *bias, Tensor *scale, u_int8_t stride, u_int8_t padding) {
    // avoids indexing overhead ig, direct access
    u_int8_t batch_size = input->shape[0];
    u_int8_t in_channels = input->shape[1];
    u_int8_t in_height = input->shape[2];
    u_int8_t in_width = input->shape[3];

    u_int8_t out_channels = weights->shape[0];
    u_int8_t kernel_height = weights->shape[2];
    u_int8_t kernel_width = weights->shape[3];

    u_int8_t out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    u_int8_t out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

    u_int8_t output_shape[4] = {batch_size, out_channels, out_height, out_width}; // make new 
    Tensor float_intermediate = f_create_tensor(output_shape, 4);

    #ifdef QUANT_MODE_DQ 
        quantize_weights(input, input, &(scale->f_data), CONVERT_FLOAT); // quant it again        
    #endif

    for (u_int8_t n = 0; n < batch_size; n++) {
        for (u_int8_t oc = 0; oc < out_channels; oc++) {
            for (u_int8_t h = 0; h < out_height; h++) {
                for (u_int8_t w = 0; w < out_width; w++) {
                    float sum = 0;
                    for (u_int8_t ic = 0; ic < in_channels; ic++) {
                        for (u_int8_t kh = 0; kh < kernel_height; kh++) {
                            for (u_int8_t kw = 0; kw < kernel_width; kw++) {
                                uint32_t ih = h * stride + kh - padding;
                                uint32_t iw = w * stride + kw - padding;

                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    uint32_t in_index = n * (in_channels * in_height * in_width) +
                                                   ic * (in_height * in_width) +
                                                   ih * in_width + iw;

                                    uint32_t weight_index = oc * (in_channels * kernel_height * kernel_width) +
                                                       ic * (kernel_height * kernel_width) +
                                                       kh * kernel_width + kw;

                                    #ifdef QUANT_MODE_DQ
                                        sum += input->f_data[in_index] * weights->data[weight_index];
                                    #elif QUANT_MODE_QAT_SQ
                                        sum += input->data[in_index] * weights->data[weight_index];
                                    #endif 
                                    
                                }
                            }
                        }
                    }
                    sum += bias->data[oc];
                    int32_t out_index = n * (out_channels * out_height * out_width) +
                                    oc * (out_height * out_width) +
                                    h * out_width + w;

                    float_intermediate.f_data[out_index] = sum;
                }
            }
        }
    }

    free_tensor(&input);
    
    #ifdef QUANT_MODE_DQ
        dequantize_weights(&float_intermediate, &float_intermediate, &(scale->f_data));
    #elif QUANT_MODE_QAT_SQ
        quantize_weights(&float_intermediate, &float_intermediate, &(scale->f_data), CONVERT_INT8);
    #endif 

    return float_intermediate;
}   

Tensor fc_layer(Tensor *input, Tensor *weights) {
    int8_t batch_size = input->shape[0];
    int8_t input_features = input->shape[1];
    int8_t output_features = weights->shape[0]; // Tracking the num_classes using weights itself

    int8_t shape[2] = {batch_size, output_features};
    Tensor output = f_create_tensor(shape, 2);

    for (u_int8_t n = 0; n < batch_size; n++) {
        for (u_int8_t o = 0; o < output_features; o++) {
            float sum = 0.0f;
            for (int i = 0; i < input_features; i++) {
                sum += input->f_data[n * input_features + i] * weights->f_data[o * input_features + i];
            }
            output.f_data[n * output_features + o] = sum;
        }
    }

    return output; 
}


Tensor maxpool2d(Tensor* input, int kernel_size, int stride) {
    int8_t shape[4] =  {input->shape[0], input->shape[1], ((input->shape[2] - kernel_size) / stride + 1), ((input->shape[3] - kernel_size) / stride + 1)};

    #ifdef QUANT_MODE_QAT_SQ
        Tensor output = create_tensor(shape, 4);
    #else   
        Tensor output = f_create_tensor(shape, 4);
    #endif
    

    for (u_int8_t b = 0; b < output.shape[0]; b++) { // Batch 
        for (u_int8_t c = 0; c < output.shape[1]; c++) { // Channel 
            for (u_int8_t oh = 0; oh < output.shape[2]; oh++) { // Output height
                for (u_int8_t ow = 0; ow < output.shape[3]; ow++) { // Output width
                    
                    #ifdef QUANT_MODE_QAT_SQ
                        int8_t max_value = INT8_MIN;
                    #else 
                        float max_value = FLOAT_MIN; 
                    #endif

                    for (u_int8_t kh = 0; kh < kernel_size; kh++) { // Kernel height
                        for (u_int8_t kw = 0; kw < kernel_size; kw++) { // Kernel width
                            int ih = oh * stride + kh; 
                            int iw = ow * stride + kw; 

                            int input_index = b * (input->shape[1] * input->shape[2] * input->shape[3]) +
                                              c * (input->shape[2] * input->shape[3]) +
                                              ih * input->shape[3] +
                                              iw;


                    #ifdef QUANT_MODE_QAT_SQ
                            if (input->data[input_index] > max_value) {
                                max_value = input->data[input_index];
                            }
                    #else 
                            if (input->f_data[input_index] > max_value) {
                                max_value = input->f_data[input_index];
                            }
                    #endif

                        }
                    }

                    int output_index = b * (output.shape[1] * output.shape[2] * output.shape[3]) +
                                       c * (output.shape[2] * output.shape[3]) +
                                       oh * output.shape[3] +
                                       ow;

                    #ifdef QUANT_MODE_QAT_SQ
                        output.data[output_index] = max_value;
                    #else
                        output.f_data[output_index] = max_value; 
                    #endif

                    
                }
            }
        }
    }

    return output;
}

Tensor softmax(Tensor *input) {
    int batch_size = input->shape[0];
    int num_classes = input->shape[1];

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
            input->f_data[index] = expf(input->f_data[index] - max_val);
            sum_exp += input->f_data[index];
        }

        // Normalize to get probabilities
        for (int c = 0; c < num_classes; c++) {
            int index = n * num_classes + c;
            input->f_data[index] /= sum_exp;
        }
    }
}

Tensor upsample_nearest(Tensor* input, int8_t scale_factor) {

    int8_t shape[4] = {input->shape[0], input->shape[1], input->shape[2] * scale_factor, input->shape[3] * scale_factor};

    #ifdef QUANT_MODE_QAT_SQ
        Tensor output = create_tensor(shape, 4);
    #else   
        Tensor output = f_create_tensor(shape, 4);
    #endif
    
    for (u_int8_t b = 0; b < output.shape[0]; b++) { // Batch 
        for (u_int8_t c = 0; c < output.shape[1]; c++) { // Channel 
            for (u_int8_t h = 0; h < output.shape[2]; h++) { // Height 
                uint32_t nearest_h = h / scale_factor;
                for (u_int8_t w = 0; w < output.shape[3]; w++) { // Width 
                    uint32_t nearest_w = w / scale_factor;
                    uint32_t input_index = b * (input->shape[1] * input->shape[2] * input->shape[3]) +
                                      c * (input->shape[2] * input->shape[3]) +
                                      nearest_h * input->shape[3] +
                                      nearest_w;

                    uint32_t output_index = b * (output.shape[1] * output.shape[2] * output.shape[3]) +
                                       c * (output.shape[2] * output.shape[3]) +
                                       h * output.shape[3] +
                                       w;

                #ifdef QUANT_MODE_QAT_SQ
                    output.data[output_index] = input->data[input_index];
                #else   
                    output.f_data[output_index] = input->f_data[input_index];
                #endif
    
                }
            }
        }
    }

    free_tensor(input);

    return output;
}


void AttentionCondenser(Tensor* input, int8_t in_channels, int8_t mid_channels, int8_t out_channels, u_int8_t* layer_id) { 

    Tensor Q = maxpool2d(input, 2, 2);
    Tensor K = conv2d(&Q, model_weights[(*layer_id)++].address, model_weights[(*layer_id)++].address, model_weights[(*layer_id)++].address, 1, 1); 
    *layer_id++; // ignoring calibrated weight-scale
    K = conv2d(&K, model_weights[(*layer_id)++].address, model_weights[(*layer_id)++].address, model_weights[(*layer_id)++].address, 1, 1); // K's data is de-allocated, ok to overwrite. 
    *layer_id++; 
    K = upsample_nearest(&K, 2); // K = A here
    K = conv2d(&K, model_weights[(*layer_id)++].address, model_weights[(*layer_id)++].address, model_weights[(*layer_id)++].address, 1, 1); // K = S here 
    *layer_id++;
    K = sigmoid(&K); 
    attention(&input, &K, model_weights[(*layer_id)++].address); // Overwriting to save space.
    return K; // S = V_prime

    // Note than normal SQ requires the output to go in int8 format. 
    // I've maintained it in float here, such that when the output goes into the batchnorm2d layer, it can maintain it's precision. Within the BatchNorm2d, we will then quant it back.
    // Technically, **this is still SQ**, because sigmoid and attention are mainly arithmetic operations, and batchnorm2d is the next "layer" with weights.  
    
}

Tensor Attn_BN_Block(Tensor* input, int8_t in_channels, int8_t mid_channels_0, int8_t out_channels_0, int8_t mid_channels_1, int8_t out_channels_1, u_int8_t* layer_id)  { 

    Tensor x_ = AttentionCondenser(input, in_channels, mid_channels_0, out_channels_0, layer_id, layer_id); free_tensor(input);
    x_ = batchnorm2d(x_, model_weights[(*layer_id)++].address, model_weights[(*layer_id)++].address, model_weights[(*layer_id)++].address, model_weights[(*layer_id) + 1], model_weights[(*layer_id) + 2]); 
    *layer_id += 3; 
    Tensor x = AttentionCondenser(x_, in_channels, mid_channels_1, out_channels_1, layer_id); free_tensor(x_);
    x = batchnorm2d(x, model_weights[(*layer_id)++].address, model_weights[(*layer_id)++].address, model_weights[(*layer_id)++].address, model_weights[(*layer_id)++].address); 
    *layer_id += 3; 

    // BatchNorm layers overwrite the input in-place. 
    // Note the ordering of the input to batch norms while referencing from model_weights. 

    return x; 

}

