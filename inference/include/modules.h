#ifndef MODULES_H
#define MODULES_H

#include "tensor.h"

Tensor upsample_nearest(Tensor* input, int in_size, int8_t scale_factor) ;
Tensor softmax(Tensor *input) ;
Tensor maxpool2d(Tensor* input, int kernel_size, int stride) ;
Tensor fc_layer(Tensor *input, Tensor *weights) ;
Tensor conv2d(Tensor *input, Tensor *weights, Tensor *bias, Tensor *scale, u_int8_t stride, u_int8_t padding);
Tensor batchnorm2d(Tensor* input, Tensor* gamma, Tensor* beta, Tensor* scale, Tensor* mean, Tensor* variance);
Tensor adaptive_avg_pool2d(Tensor *input);

// Declare these blocks an SRAM based function for speedup.
Tensor AttentionCondenser(Tensor* input, int8_t in_channels, int8_t mid_channels, int8_t out_channels, u_int8_t* layer_id) __attribute__((section(".srodata"))) __attribute__((used));;
Tensor Attn_BN_Block(Tensor* input, int8_t in_channels, int8_t mid_channels_0, int8_t out_channels_0, int8_t mid_channels_1, int8_t out_channels_1, u_int8_t* layer_id) __attribute__((section(".srodata"))) __attribute__((used));; 

#endif