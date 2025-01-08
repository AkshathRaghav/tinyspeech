#ifndef MODULES_H
#define MODULES_H

#include "tensor.h"

Tensor upsample_nearest(Tensor* input, int in_size, int8_t scale_factor) ;
Tensor softmax(Tensor *input) ;
Tensor maxpool2d(Tensor* input, int kernel_size, int stride) ;
Tensor fc_layer(Tensor *input, Tensor *weights) ;
Tensor conv2d(Tensor *input, Tensor *weights, Tensor *bias, int stride, int padding) ;
Tensor batchnorm2d(Tensor* input, Tensor* mean, Tensor* variance, Tensor* gamma, Tensor* beta) ;

// Declare these blocks an SRAM based function for speedup.
Tensor AttentionCondenser(Tensor* input, int8_t in_channels, int8_t mid_channels, int8_t out_channels) __attribute__((section(".srodata"))) __attribute__((used));;
Tensor Attn_BN_Block(Tensor* input, int8_t in_channels, int8_t mid_channels_0, int8_t out_channels_0, int8_t mid_channels_1, int8_t out_channels_1) __attribute__((section(".srodata"))) __attribute__((used));; 

#endif