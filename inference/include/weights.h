// Automatically generated header file
// Date: 2025-01-13 02:24:23.168861
// Quantized model exported from QModel.pth
// Quantization Type: 8bit
// Quantization Mode: QAT

#include <stdint.h>
#include "tensor.h"

#ifndef TINYSPEECH_WEIGHTS_H
#define TINYSPEECH_WEIGHTS_H

// Constants:
#define QUANT_MODE_QAT_SQ
#define CONVERT_FLOAT 1
#define CONVERT_INT8 0


static const int8_t shape_0[] = { 7, 1, 3, 3 };
static const int8_t data_1[] = { 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0xc0, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0xa0, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0xa0, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x20, 0x7f, 0x40, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80 };
static const Tensor CONV1_WEIGHT = {
    .dims = 4,
    .size = 63,
    .shape = shape_0,
    .data = data_1,
    .f_data = NULL
};
        
static const int8_t shape_2[] = { 7 };
static const int8_t data_3[] = { 0x80, 0x60, 0xc0, 0x7f, 0x80, 0x80, 0x80 };
static const Tensor CONV1_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_2,
    .data = data_3,
    .f_data = NULL
};
        
static const int8_t shape_4[] = { 1 };
static const float data_5[] = { 0x426c0000 };
static const Tensor CONV1_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_4,
    .data = NULL,
    .f_data = data_5
};
        
static const int8_t shape_6[] = { 1 };
static const float data_7[] = { 0x42fe0000 };
static const Tensor CONV1_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_6,
    .data = NULL,
    .f_data = data_7
};
        
static const int8_t shape_8[] = { 1 };
static const int8_t data_9[] = { 0x00 };
static const Tensor BLOCK1_LAYER1_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_8,
    .data = data_9,
    .f_data = NULL
};
        
static const int8_t shape_10[] = { 14, 7, 1, 1 };
static const int8_t data_11[] = { 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x7f, 0xa0, 0x7f, 0x80, 0x80, 0x7f, 0x60, 0xa0, 0x7f, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x60, 0xc0, 0x80, 0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x40, 0x7f, 0x80, 0x80, 0x80, 0x40, 0x80, 0x80, 0x80, 0x00, 0x80, 0xe0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0xc0, 0x7f, 0x7f, 0x80, 0x7f, 0xa0, 0x80, 0x80, 0x80, 0x60, 0xa0, 0x7f, 0x80, 0x7f, 0x80, 0xc0, 0x60, 0x7f, 0x7f, 0x7f, 0x80, 0x60, 0x80, 0xe0, 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0x60, 0x40, 0x80, 0x7f, 0xc0, 0x80, 0x7f };
static const Tensor BLOCK1_LAYER1_GROUP_CONV_WEIGHT = {
    .dims = 4,
    .size = 98,
    .shape = shape_10,
    .data = data_11,
    .f_data = NULL
};
        
static const int8_t shape_12[] = { 14 };
static const int8_t data_13[] = { 0x60, 0x80, 0x80, 0xe0, 0x7f, 0xa0, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x60, 0x7f };
static const Tensor BLOCK1_LAYER1_GROUP_CONV_BIAS = {
    .dims = 1,
    .size = 14,
    .shape = shape_12,
    .data = data_13,
    .f_data = NULL
};
        
static const int8_t shape_14[] = { 1 };
static const float data_15[] = { 0x428e0000 };
static const Tensor BLOCK1_LAYER1_GROUP_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_14,
    .data = NULL,
    .f_data = data_15
};
        
static const int8_t shape_16[] = { 1 };
static const float data_17[] = { 0x41d80000 };
static const Tensor BLOCK1_LAYER1_GROUP_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_16,
    .data = NULL,
    .f_data = data_17
};
        
static const int8_t shape_18[] = { 3, 14, 1, 1 };
static const int8_t data_19[] = { 0x40, 0xa0, 0x7f, 0x7f, 0x60, 0xc0, 0xc0, 0x20, 0xc0, 0xa0, 0x7f, 0x7f, 0xe0, 0xc0, 0x00, 0xe0, 0x40, 0x40, 0x80, 0xa0, 0x7f, 0x60, 0x00, 0x80, 0x80, 0x80, 0xa0, 0x60, 0x7f, 0x7f, 0x80, 0x80, 0xe0, 0x7f, 0x20, 0xc0, 0x7f, 0xa0, 0x80, 0x7f, 0x80, 0xc0 };
static const Tensor BLOCK1_LAYER1_POINTWISE_CONV_WEIGHT = {
    .dims = 4,
    .size = 42,
    .shape = shape_18,
    .data = data_19,
    .f_data = NULL
};
        
static const int8_t shape_20[] = { 3 };
static const int8_t data_21[] = { 0x80, 0x7f, 0x7f };
static const Tensor BLOCK1_LAYER1_POINTWISE_CONV_BIAS = {
    .dims = 1,
    .size = 3,
    .shape = shape_20,
    .data = data_21,
    .f_data = NULL
};
        
static const int8_t shape_22[] = { 1 };
static const float data_23[] = { 0x7fffffff };
static const Tensor BLOCK1_LAYER1_POINTWISE_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_22,
    .data = NULL,
    .f_data = data_23
};
        
static const int8_t shape_24[] = { 1 };
static const float data_25[] = { 0x41600000 };
static const Tensor BLOCK1_LAYER1_POINTWISE_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_24,
    .data = NULL,
    .f_data = data_25
};
        
static const int8_t shape_26[] = { 7, 3, 1, 1 };
static const int8_t data_27[] = { 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x00, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0xa0, 0x80, 0x80, 0x7f };
static const Tensor BLOCK1_LAYER1_EXPAND_CONV_WEIGHT = {
    .dims = 4,
    .size = 21,
    .shape = shape_26,
    .data = data_27,
    .f_data = NULL
};
        
static const int8_t shape_28[] = { 7 };
static const int8_t data_29[] = { 0x80, 0x7f, 0x80, 0x80, 0xa0, 0x80, 0x80 };
static const Tensor BLOCK1_LAYER1_EXPAND_CONV_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_28,
    .data = data_29,
    .f_data = NULL
};
        
static const int8_t shape_30[] = { 1 };
static const float data_31[] = { 0x7fffffff };
static const Tensor BLOCK1_LAYER1_EXPAND_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_30,
    .data = NULL,
    .f_data = data_31
};
        
static const int8_t shape_32[] = { 1 };
static const float data_33[] = { 0x42280000 };
static const Tensor BLOCK1_LAYER1_EXPAND_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_32,
    .data = NULL,
    .f_data = data_33
};
        
static const int8_t shape_34[] = { 7 };
static const int8_t data_35[] = { 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f };
static const Tensor BLOCK1_LAYER2_WEIGHT = {
    .dims = 1,
    .size = 7,
    .shape = shape_34,
    .data = data_35,
    .f_data = NULL
};
        
static const int8_t shape_36[] = { 7 };
static const int8_t data_37[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK1_LAYER2_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_36,
    .data = data_37,
    .f_data = NULL
};
        
static const int8_t shape_38[] = { 1 };
static const float data_39[] = { 0x7fffffff };
static const Tensor BLOCK1_LAYER2_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_38,
    .data = NULL,
    .f_data = data_39
};
        
static const int8_t shape_40[] = { 1 };
static const float data_41[] = { 0x42fe0000 };
static const Tensor BLOCK1_LAYER2_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_40,
    .data = NULL,
    .f_data = data_41
};
        
static const int8_t shape_42[] = { 7 };
static const int8_t data_43[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK1_LAYER2_RUNNING_MEAN = {
    .dims = 1,
    .size = 7,
    .shape = shape_42,
    .data = data_43,
    .f_data = NULL
};
        
static const int8_t shape_44[] = { 7 };
static const int8_t data_45[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK1_LAYER2_RUNNING_VAR = {
    .dims = 1,
    .size = 7,
    .shape = shape_44,
    .data = data_45,
    .f_data = NULL
};
        
static const int8_t shape_46[] = { 1 };
static const int8_t data_47[] = { 0x00 };
static const Tensor BLOCK1_LAYER3_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_46,
    .data = data_47,
    .f_data = NULL
};
        
static const int8_t shape_48[] = { 6, 7, 1, 1 };
static const int8_t data_49[] = { 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0xc0, 0x00, 0xa0, 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0xa0, 0x80, 0x60, 0x7f, 0x80, 0xa0, 0x80, 0xe0, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0xc0, 0xe0, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80 };
static const Tensor BLOCK1_LAYER3_GROUP_CONV_WEIGHT = {
    .dims = 4,
    .size = 42,
    .shape = shape_48,
    .data = data_49,
    .f_data = NULL
};
        
static const int8_t shape_50[] = { 6 };
static const int8_t data_51[] = { 0xe0, 0x80, 0x80, 0x00, 0xa0, 0x80 };
static const Tensor BLOCK1_LAYER3_GROUP_CONV_BIAS = {
    .dims = 1,
    .size = 6,
    .shape = shape_50,
    .data = data_51,
    .f_data = NULL
};
        
static const int8_t shape_52[] = { 1 };
static const float data_53[] = { 0x7fffffff };
static const Tensor BLOCK1_LAYER3_GROUP_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_52,
    .data = NULL,
    .f_data = data_53
};
        
static const int8_t shape_54[] = { 1 };
static const float data_55[] = { 0x41d80000 };
static const Tensor BLOCK1_LAYER3_GROUP_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_54,
    .data = NULL,
    .f_data = data_55
};
        
static const int8_t shape_56[] = { 7, 6, 1, 1 };
static const int8_t data_57[] = { 0x80, 0x80, 0x7f, 0x80, 0x60, 0x80, 0x20, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x60, 0x7f, 0x7f, 0xc0, 0x60, 0x80, 0xa0, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0xa0, 0x7f, 0x40, 0x80, 0x80, 0x7f, 0x80, 0xc0, 0x7f, 0x80, 0xe0, 0xe0, 0xa0, 0x80, 0x80, 0x80 };
static const Tensor BLOCK1_LAYER3_POINTWISE_CONV_WEIGHT = {
    .dims = 4,
    .size = 42,
    .shape = shape_56,
    .data = data_57,
    .f_data = NULL
};
        
static const int8_t shape_58[] = { 7 };
static const int8_t data_59[] = { 0x80, 0x7f, 0x80, 0x80, 0x60, 0x7f, 0x7f };
static const Tensor BLOCK1_LAYER3_POINTWISE_CONV_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_58,
    .data = data_59,
    .f_data = NULL
};
        
static const int8_t shape_60[] = { 1 };
static const float data_61[] = { 0x7fffffff };
static const Tensor BLOCK1_LAYER3_POINTWISE_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_60,
    .data = NULL,
    .f_data = data_61
};
        
static const int8_t shape_62[] = { 1 };
static const float data_63[] = { 0x41e00000 };
static const Tensor BLOCK1_LAYER3_POINTWISE_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_62,
    .data = NULL,
    .f_data = data_63
};
        
static const int8_t shape_64[] = { 7, 7, 1, 1 };
static const int8_t data_65[] = { 0x7f, 0x7f, 0x80, 0x20, 0x7f, 0xe0, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x20, 0x7f, 0x60, 0x40, 0x80, 0x80, 0x00, 0x7f, 0x7f, 0x80, 0xc0, 0x80, 0x80, 0x40, 0x7f, 0x7f, 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0x00, 0x7f, 0x80, 0x80, 0x7f, 0xa0, 0xc0, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x20, 0x7f, 0x80, 0xa0 };
static const Tensor BLOCK1_LAYER3_EXPAND_CONV_WEIGHT = {
    .dims = 4,
    .size = 49,
    .shape = shape_64,
    .data = data_65,
    .f_data = NULL
};
        
static const int8_t shape_66[] = { 7 };
static const int8_t data_67[] = { 0x7f, 0x20, 0x20, 0xa0, 0x80, 0x80, 0x80 };
static const Tensor BLOCK1_LAYER3_EXPAND_CONV_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_66,
    .data = data_67,
    .f_data = NULL
};
        
static const int8_t shape_68[] = { 1 };
static const float data_69[] = { 0x7fffffff };
static const Tensor BLOCK1_LAYER3_EXPAND_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_68,
    .data = NULL,
    .f_data = data_69
};
        
static const int8_t shape_70[] = { 1 };
static const float data_71[] = { 0x41c80000 };
static const Tensor BLOCK1_LAYER3_EXPAND_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_70,
    .data = NULL,
    .f_data = data_71
};
        
static const int8_t shape_72[] = { 7 };
static const int8_t data_73[] = { 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f };
static const Tensor BLOCK1_LAYER4_WEIGHT = {
    .dims = 1,
    .size = 7,
    .shape = shape_72,
    .data = data_73,
    .f_data = NULL
};
        
static const int8_t shape_74[] = { 7 };
static const int8_t data_75[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK1_LAYER4_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_74,
    .data = data_75,
    .f_data = NULL
};
        
static const int8_t shape_76[] = { 1 };
static const float data_77[] = { 0x7fffffff };
static const Tensor BLOCK1_LAYER4_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_76,
    .data = NULL,
    .f_data = data_77
};
        
static const int8_t shape_78[] = { 1 };
static const float data_79[] = { 0x42fe0000 };
static const Tensor BLOCK1_LAYER4_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_78,
    .data = NULL,
    .f_data = data_79
};
        
static const int8_t shape_80[] = { 7 };
static const int8_t data_81[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK1_LAYER4_RUNNING_MEAN = {
    .dims = 1,
    .size = 7,
    .shape = shape_80,
    .data = data_81,
    .f_data = NULL
};
        
static const int8_t shape_82[] = { 7 };
static const int8_t data_83[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK1_LAYER4_RUNNING_VAR = {
    .dims = 1,
    .size = 7,
    .shape = shape_82,
    .data = data_83,
    .f_data = NULL
};
        
static const int8_t shape_84[] = { 1 };
static const int8_t data_85[] = { 0x00 };
static const Tensor BLOCK2_LAYER1_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_84,
    .data = data_85,
    .f_data = NULL
};
        
static const int8_t shape_86[] = { 14, 7, 1, 1 };
static const int8_t data_87[] = { 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x80, 0xe0, 0x80, 0x80, 0x7f, 0x7f, 0x40, 0xc0, 0x80, 0x80, 0x60, 0x7f, 0xc0, 0x80, 0x7f, 0x80, 0x7f, 0x80, 0x00, 0xc0, 0x20, 0x80, 0x7f, 0x00, 0x7f, 0x7f, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0xe0, 0x80, 0x80, 0x40, 0x60, 0x60, 0xa0, 0x7f, 0x40, 0x80, 0xe0, 0x60, 0x7f, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0xc0, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x60, 0x7f, 0x80, 0x80, 0x40, 0x80, 0x80, 0x7f, 0xc0, 0x7f, 0x7f, 0x7f, 0x7f, 0x60, 0x00, 0xe0, 0x00, 0x7f, 0x7f, 0xa0, 0x80, 0xa0, 0x80, 0x40, 0x7f, 0x7f, 0x7f };
static const Tensor BLOCK2_LAYER1_GROUP_CONV_WEIGHT = {
    .dims = 4,
    .size = 98,
    .shape = shape_86,
    .data = data_87,
    .f_data = NULL
};
        
static const int8_t shape_88[] = { 14 };
static const int8_t data_89[] = { 0x80, 0x80, 0xa0, 0xc0, 0x20, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x7f, 0x00 };
static const Tensor BLOCK2_LAYER1_GROUP_CONV_BIAS = {
    .dims = 1,
    .size = 14,
    .shape = shape_88,
    .data = data_89,
    .f_data = NULL
};
        
static const int8_t shape_90[] = { 1 };
static const float data_91[] = { 0x428e0000 };
static const Tensor BLOCK2_LAYER1_GROUP_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_90,
    .data = NULL,
    .f_data = data_91
};
        
static const int8_t shape_92[] = { 1 };
static const float data_93[] = { 0x41b80000 };
static const Tensor BLOCK2_LAYER1_GROUP_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_92,
    .data = NULL,
    .f_data = data_93
};
        
static const int8_t shape_94[] = { 3, 14, 1, 1 };
static const int8_t data_95[] = { 0xe0, 0x7f, 0xe0, 0x7f, 0x7f, 0x80, 0x60, 0x40, 0x40, 0x20, 0x40, 0x80, 0xe0, 0x7f, 0x7f, 0x00, 0x80, 0x00, 0x80, 0x7f, 0x80, 0xe0, 0xc0, 0x40, 0x7f, 0x20, 0x7f, 0x7f, 0x7f, 0xa0, 0xc0, 0x7f, 0xc0, 0x7f, 0xc0, 0x80, 0x60, 0x20, 0x7f, 0x7f, 0x7f, 0x7f };
static const Tensor BLOCK2_LAYER1_POINTWISE_CONV_WEIGHT = {
    .dims = 4,
    .size = 42,
    .shape = shape_94,
    .data = data_95,
    .f_data = NULL
};
        
static const int8_t shape_96[] = { 3 };
static const int8_t data_97[] = { 0x00, 0x80, 0x80 };
static const Tensor BLOCK2_LAYER1_POINTWISE_CONV_BIAS = {
    .dims = 1,
    .size = 3,
    .shape = shape_96,
    .data = data_97,
    .f_data = NULL
};
        
static const int8_t shape_98[] = { 1 };
static const float data_99[] = { 0x7fffffff };
static const Tensor BLOCK2_LAYER1_POINTWISE_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_98,
    .data = NULL,
    .f_data = data_99
};
        
static const int8_t shape_100[] = { 1 };
static const float data_101[] = { 0x41800000 };
static const Tensor BLOCK2_LAYER1_POINTWISE_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_100,
    .data = NULL,
    .f_data = data_101
};
        
static const int8_t shape_102[] = { 7, 3, 1, 1 };
static const int8_t data_103[] = { 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x40, 0x7f, 0x60, 0x7f };
static const Tensor BLOCK2_LAYER1_EXPAND_CONV_WEIGHT = {
    .dims = 4,
    .size = 21,
    .shape = shape_102,
    .data = data_103,
    .f_data = NULL
};
        
static const int8_t shape_104[] = { 7 };
static const int8_t data_105[] = { 0x80, 0x00, 0x20, 0x7f, 0x7f, 0x7f, 0x80 };
static const Tensor BLOCK2_LAYER1_EXPAND_CONV_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_104,
    .data = data_105,
    .f_data = NULL
};
        
static const int8_t shape_106[] = { 1 };
static const float data_107[] = { 0x7fffffff };
static const Tensor BLOCK2_LAYER1_EXPAND_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_106,
    .data = NULL,
    .f_data = data_107
};
        
static const int8_t shape_108[] = { 1 };
static const float data_109[] = { 0x422c0000 };
static const Tensor BLOCK2_LAYER1_EXPAND_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_108,
    .data = NULL,
    .f_data = data_109
};
        
static const int8_t shape_110[] = { 7 };
static const int8_t data_111[] = { 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f };
static const Tensor BLOCK2_LAYER2_WEIGHT = {
    .dims = 1,
    .size = 7,
    .shape = shape_110,
    .data = data_111,
    .f_data = NULL
};
        
static const int8_t shape_112[] = { 7 };
static const int8_t data_113[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK2_LAYER2_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_112,
    .data = data_113,
    .f_data = NULL
};
        
static const int8_t shape_114[] = { 1 };
static const float data_115[] = { 0x7fffffff };
static const Tensor BLOCK2_LAYER2_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_114,
    .data = NULL,
    .f_data = data_115
};
        
static const int8_t shape_116[] = { 1 };
static const float data_117[] = { 0x42fe0000 };
static const Tensor BLOCK2_LAYER2_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_116,
    .data = NULL,
    .f_data = data_117
};
        
static const int8_t shape_118[] = { 7 };
static const int8_t data_119[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK2_LAYER2_RUNNING_MEAN = {
    .dims = 1,
    .size = 7,
    .shape = shape_118,
    .data = data_119,
    .f_data = NULL
};
        
static const int8_t shape_120[] = { 7 };
static const int8_t data_121[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK2_LAYER2_RUNNING_VAR = {
    .dims = 1,
    .size = 7,
    .shape = shape_120,
    .data = data_121,
    .f_data = NULL
};
        
static const int8_t shape_122[] = { 1 };
static const int8_t data_123[] = { 0x80 };
static const Tensor BLOCK2_LAYER3_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_122,
    .data = data_123,
    .f_data = NULL
};
        
static const int8_t shape_124[] = { 6, 7, 1, 1 };
static const int8_t data_125[] = { 0x00, 0x80, 0x80, 0xe0, 0x80, 0x7f, 0x20, 0x80, 0x7f, 0x80, 0x80, 0xc0, 0x80, 0x80, 0xc0, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x7f, 0x7f, 0x00, 0x40, 0x80, 0x80, 0x7f, 0xa0, 0xe0, 0x80, 0x80, 0x00, 0x80, 0xe0, 0x7f, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x7f, 0xa0 };
static const Tensor BLOCK2_LAYER3_GROUP_CONV_WEIGHT = {
    .dims = 4,
    .size = 42,
    .shape = shape_124,
    .data = data_125,
    .f_data = NULL
};
        
static const int8_t shape_126[] = { 6 };
static const int8_t data_127[] = { 0x80, 0x80, 0x80, 0x80, 0xa0, 0x40 };
static const Tensor BLOCK2_LAYER3_GROUP_CONV_BIAS = {
    .dims = 1,
    .size = 6,
    .shape = shape_126,
    .data = data_127,
    .f_data = NULL
};
        
static const int8_t shape_128[] = { 1 };
static const float data_129[] = { 0x7fffffff };
static const Tensor BLOCK2_LAYER3_GROUP_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_128,
    .data = NULL,
    .f_data = data_129
};
        
static const int8_t shape_130[] = { 1 };
static const float data_131[] = { 0x41c00000 };
static const Tensor BLOCK2_LAYER3_GROUP_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_130,
    .data = NULL,
    .f_data = data_131
};
        
static const int8_t shape_132[] = { 7, 6, 1, 1 };
static const int8_t data_133[] = { 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x7f, 0xa0, 0xe0, 0x80, 0x7f, 0x80, 0x7f, 0x20, 0x60, 0x7f, 0xc0, 0x80, 0xc0, 0x7f, 0x80, 0x60, 0x80, 0x7f, 0x7f, 0x20, 0x7f, 0x00, 0x7f, 0x80, 0xc0, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0xa0, 0xa0, 0x80, 0xc0, 0x7f, 0x40, 0x80 };
static const Tensor BLOCK2_LAYER3_POINTWISE_CONV_WEIGHT = {
    .dims = 4,
    .size = 42,
    .shape = shape_132,
    .data = data_133,
    .f_data = NULL
};
        
static const int8_t shape_134[] = { 7 };
static const int8_t data_135[] = { 0x7f, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80 };
static const Tensor BLOCK2_LAYER3_POINTWISE_CONV_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_134,
    .data = data_135,
    .f_data = NULL
};
        
static const int8_t shape_136[] = { 1 };
static const float data_137[] = { 0x7fffffff };
static const Tensor BLOCK2_LAYER3_POINTWISE_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_136,
    .data = NULL,
    .f_data = data_137
};
        
static const int8_t shape_138[] = { 1 };
static const float data_139[] = { 0x41d00000 };
static const Tensor BLOCK2_LAYER3_POINTWISE_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_138,
    .data = NULL,
    .f_data = data_139
};
        
static const int8_t shape_140[] = { 7, 7, 1, 1 };
static const int8_t data_141[] = { 0x7f, 0x7f, 0x40, 0x7f, 0x40, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x60, 0x80, 0x7f, 0x7f, 0x80, 0x60, 0xe0, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0xa0, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x7f };
static const Tensor BLOCK2_LAYER3_EXPAND_CONV_WEIGHT = {
    .dims = 4,
    .size = 49,
    .shape = shape_140,
    .data = data_141,
    .f_data = NULL
};
        
static const int8_t shape_142[] = { 7 };
static const int8_t data_143[] = { 0x7f, 0x7f, 0x40, 0x7f, 0x7f, 0x80, 0x80 };
static const Tensor BLOCK2_LAYER3_EXPAND_CONV_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_142,
    .data = data_143,
    .f_data = NULL
};
        
static const int8_t shape_144[] = { 1 };
static const float data_145[] = { 0x7fffffff };
static const Tensor BLOCK2_LAYER3_EXPAND_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_144,
    .data = NULL,
    .f_data = data_145
};
        
static const int8_t shape_146[] = { 1 };
static const float data_147[] = { 0x41e80000 };
static const Tensor BLOCK2_LAYER3_EXPAND_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_146,
    .data = NULL,
    .f_data = data_147
};
        
static const int8_t shape_148[] = { 7 };
static const int8_t data_149[] = { 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f };
static const Tensor BLOCK2_LAYER4_WEIGHT = {
    .dims = 1,
    .size = 7,
    .shape = shape_148,
    .data = data_149,
    .f_data = NULL
};
        
static const int8_t shape_150[] = { 7 };
static const int8_t data_151[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK2_LAYER4_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_150,
    .data = data_151,
    .f_data = NULL
};
        
static const int8_t shape_152[] = { 1 };
static const float data_153[] = { 0x7fffffff };
static const Tensor BLOCK2_LAYER4_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_152,
    .data = NULL,
    .f_data = data_153
};
        
static const int8_t shape_154[] = { 1 };
static const float data_155[] = { 0x42fe0000 };
static const Tensor BLOCK2_LAYER4_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_154,
    .data = NULL,
    .f_data = data_155
};
        
static const int8_t shape_156[] = { 7 };
static const int8_t data_157[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK2_LAYER4_RUNNING_MEAN = {
    .dims = 1,
    .size = 7,
    .shape = shape_156,
    .data = data_157,
    .f_data = NULL
};
        
static const int8_t shape_158[] = { 7 };
static const int8_t data_159[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK2_LAYER4_RUNNING_VAR = {
    .dims = 1,
    .size = 7,
    .shape = shape_158,
    .data = data_159,
    .f_data = NULL
};
        
static const int8_t shape_160[] = { 1 };
static const int8_t data_161[] = { 0x00 };
static const Tensor BLOCK3_LAYER1_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_160,
    .data = data_161,
    .f_data = NULL
};
        
static const int8_t shape_162[] = { 14, 7, 1, 1 };
static const int8_t data_163[] = { 0x7f, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x00, 0x7f, 0x80, 0x00, 0x7f, 0x80, 0x7f, 0x80, 0xa0, 0xa0, 0x80, 0x80, 0x7f, 0x60, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0xc0, 0xc0, 0x80, 0x20, 0x20, 0xc0, 0x40, 0x80, 0x60, 0x7f, 0x80, 0x80, 0x7f, 0x60, 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0x80, 0x7f, 0x80, 0x80, 0xc0, 0x80, 0x7f, 0xc0, 0x7f, 0x80, 0x7f, 0x80, 0x80, 0x40, 0x7f, 0x7f, 0x80, 0x80, 0x40, 0x60, 0x7f, 0x7f, 0xa0, 0x7f, 0x7f, 0xa0, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0xa0, 0x20, 0x7f, 0x80, 0x7f, 0x80, 0x7f, 0xe0, 0x80, 0x80, 0x80, 0x7f, 0x80, 0xe0, 0x60, 0x80, 0x60, 0x80, 0x80, 0x80 };
static const Tensor BLOCK3_LAYER1_GROUP_CONV_WEIGHT = {
    .dims = 4,
    .size = 98,
    .shape = shape_162,
    .data = data_163,
    .f_data = NULL
};
        
static const int8_t shape_164[] = { 14 };
static const int8_t data_165[] = { 0x80, 0x80, 0x40, 0xc0, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0x7f };
static const Tensor BLOCK3_LAYER1_GROUP_CONV_BIAS = {
    .dims = 1,
    .size = 14,
    .shape = shape_164,
    .data = data_165,
    .f_data = NULL
};
        
static const int8_t shape_166[] = { 1 };
static const float data_167[] = { 0x428e0000 };
static const Tensor BLOCK3_LAYER1_GROUP_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_166,
    .data = NULL,
    .f_data = data_167
};
        
static const int8_t shape_168[] = { 1 };
static const float data_169[] = { 0x41c80000 };
static const Tensor BLOCK3_LAYER1_GROUP_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_168,
    .data = NULL,
    .f_data = data_169
};
        
static const int8_t shape_170[] = { 2, 14, 1, 1 };
static const int8_t data_171[] = { 0x7f, 0x7f, 0x40, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x7f, 0x60, 0x20, 0x80, 0xe0, 0x7f, 0x40, 0x20, 0xc0, 0xa0, 0x7f, 0x20, 0x7f, 0xc0, 0x7f, 0x7f, 0x20, 0x7f, 0x7f, 0x7f };
static const Tensor BLOCK3_LAYER1_POINTWISE_CONV_WEIGHT = {
    .dims = 4,
    .size = 28,
    .shape = shape_170,
    .data = data_171,
    .f_data = NULL
};
        
static const int8_t shape_172[] = { 2 };
static const int8_t data_173[] = { 0x7f, 0xc0 };
static const Tensor BLOCK3_LAYER1_POINTWISE_CONV_BIAS = {
    .dims = 1,
    .size = 2,
    .shape = shape_172,
    .data = data_173,
    .f_data = NULL
};
        
static const int8_t shape_174[] = { 1 };
static const float data_175[] = { 0x7fffffff };
static const Tensor BLOCK3_LAYER1_POINTWISE_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_174,
    .data = NULL,
    .f_data = data_175
};
        
static const int8_t shape_176[] = { 1 };
static const float data_177[] = { 0x41880000 };
static const Tensor BLOCK3_LAYER1_POINTWISE_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_176,
    .data = NULL,
    .f_data = data_177
};
        
static const int8_t shape_178[] = { 7, 2, 1, 1 };
static const int8_t data_179[] = { 0x80, 0x7f, 0x00, 0x80, 0x80, 0x20, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f };
static const Tensor BLOCK3_LAYER1_EXPAND_CONV_WEIGHT = {
    .dims = 4,
    .size = 14,
    .shape = shape_178,
    .data = data_179,
    .f_data = NULL
};
        
static const int8_t shape_180[] = { 7 };
static const int8_t data_181[] = { 0x7f, 0x00, 0x7f, 0x7f, 0x80, 0x40, 0xc0 };
static const Tensor BLOCK3_LAYER1_EXPAND_CONV_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_180,
    .data = data_181,
    .f_data = NULL
};
        
static const int8_t shape_182[] = { 1 };
static const float data_183[] = { 0x7fffffff };
static const Tensor BLOCK3_LAYER1_EXPAND_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_182,
    .data = NULL,
    .f_data = data_183
};
        
static const int8_t shape_184[] = { 1 };
static const float data_185[] = { 0x420c0000 };
static const Tensor BLOCK3_LAYER1_EXPAND_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_184,
    .data = NULL,
    .f_data = data_185
};
        
static const int8_t shape_186[] = { 7 };
static const int8_t data_187[] = { 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f };
static const Tensor BLOCK3_LAYER2_WEIGHT = {
    .dims = 1,
    .size = 7,
    .shape = shape_186,
    .data = data_187,
    .f_data = NULL
};
        
static const int8_t shape_188[] = { 7 };
static const int8_t data_189[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK3_LAYER2_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_188,
    .data = data_189,
    .f_data = NULL
};
        
static const int8_t shape_190[] = { 1 };
static const float data_191[] = { 0x7fffffff };
static const Tensor BLOCK3_LAYER2_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_190,
    .data = NULL,
    .f_data = data_191
};
        
static const int8_t shape_192[] = { 1 };
static const float data_193[] = { 0x42fe0000 };
static const Tensor BLOCK3_LAYER2_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_192,
    .data = NULL,
    .f_data = data_193
};
        
static const int8_t shape_194[] = { 7 };
static const int8_t data_195[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK3_LAYER2_RUNNING_MEAN = {
    .dims = 1,
    .size = 7,
    .shape = shape_194,
    .data = data_195,
    .f_data = NULL
};
        
static const int8_t shape_196[] = { 7 };
static const int8_t data_197[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK3_LAYER2_RUNNING_VAR = {
    .dims = 1,
    .size = 7,
    .shape = shape_196,
    .data = data_197,
    .f_data = NULL
};
        
static const int8_t shape_198[] = { 1 };
static const int8_t data_199[] = { 0x00 };
static const Tensor BLOCK3_LAYER3_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_198,
    .data = data_199,
    .f_data = NULL
};
        
static const int8_t shape_200[] = { 4, 7, 1, 1 };
static const int8_t data_201[] = { 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x7f, 0x80, 0xc0, 0x7f, 0xc0, 0xc0, 0x80, 0x60, 0x80, 0x00, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0xe0, 0x7f, 0x80, 0x80, 0x80, 0xa0, 0x80, 0x80 };
static const Tensor BLOCK3_LAYER3_GROUP_CONV_WEIGHT = {
    .dims = 4,
    .size = 28,
    .shape = shape_200,
    .data = data_201,
    .f_data = NULL
};
        
static const int8_t shape_202[] = { 4 };
static const int8_t data_203[] = { 0x7f, 0x40, 0x80, 0x80 };
static const Tensor BLOCK3_LAYER3_GROUP_CONV_BIAS = {
    .dims = 1,
    .size = 4,
    .shape = shape_202,
    .data = data_203,
    .f_data = NULL
};
        
static const int8_t shape_204[] = { 1 };
static const float data_205[] = { 0x7fffffff };
static const Tensor BLOCK3_LAYER3_GROUP_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_204,
    .data = NULL,
    .f_data = data_205
};
        
static const int8_t shape_206[] = { 1 };
static const float data_207[] = { 0x41c00000 };
static const Tensor BLOCK3_LAYER3_GROUP_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_206,
    .data = NULL,
    .f_data = data_207
};
        
static const int8_t shape_208[] = { 7, 4, 1, 1 };
static const int8_t data_209[] = { 0x80, 0x80, 0x80, 0x80, 0x60, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x40, 0x80, 0x00, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0xa0, 0x60 };
static const Tensor BLOCK3_LAYER3_POINTWISE_CONV_WEIGHT = {
    .dims = 4,
    .size = 28,
    .shape = shape_208,
    .data = data_209,
    .f_data = NULL
};
        
static const int8_t shape_210[] = { 7 };
static const int8_t data_211[] = { 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0xc0 };
static const Tensor BLOCK3_LAYER3_POINTWISE_CONV_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_210,
    .data = data_211,
    .f_data = NULL
};
        
static const int8_t shape_212[] = { 1 };
static const float data_213[] = { 0x7fffffff };
static const Tensor BLOCK3_LAYER3_POINTWISE_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_212,
    .data = NULL,
    .f_data = data_213
};
        
static const int8_t shape_214[] = { 1 };
static const float data_215[] = { 0x42040000 };
static const Tensor BLOCK3_LAYER3_POINTWISE_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_214,
    .data = NULL,
    .f_data = data_215
};
        
static const int8_t shape_216[] = { 7, 7, 1, 1 };
static const int8_t data_217[] = { 0x80, 0x7f, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x80, 0xe0, 0x80, 0x40, 0x7f, 0xc0, 0xe0, 0x7f, 0xc0, 0x80, 0x7f, 0x80, 0xe0, 0x20, 0xe0, 0x80, 0x7f, 0x80, 0x80, 0x60, 0x80, 0x80, 0x80, 0x80, 0x80, 0xa0, 0x60, 0x80, 0x80, 0x00, 0x40, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x7f };
static const Tensor BLOCK3_LAYER3_EXPAND_CONV_WEIGHT = {
    .dims = 4,
    .size = 49,
    .shape = shape_216,
    .data = data_217,
    .f_data = NULL
};
        
static const int8_t shape_218[] = { 7 };
static const int8_t data_219[] = { 0xc0, 0x7f, 0x80, 0x40, 0x80, 0x7f, 0x80 };
static const Tensor BLOCK3_LAYER3_EXPAND_CONV_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_218,
    .data = data_219,
    .f_data = NULL
};
        
static const int8_t shape_220[] = { 1 };
static const float data_221[] = { 0x7fffffff };
static const Tensor BLOCK3_LAYER3_EXPAND_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_220,
    .data = NULL,
    .f_data = data_221
};
        
static const int8_t shape_222[] = { 1 };
static const float data_223[] = { 0x41b80000 };
static const Tensor BLOCK3_LAYER3_EXPAND_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_222,
    .data = NULL,
    .f_data = data_223
};
        
static const int8_t shape_224[] = { 7 };
static const int8_t data_225[] = { 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f };
static const Tensor BLOCK3_LAYER4_WEIGHT = {
    .dims = 1,
    .size = 7,
    .shape = shape_224,
    .data = data_225,
    .f_data = NULL
};
        
static const int8_t shape_226[] = { 7 };
static const int8_t data_227[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK3_LAYER4_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_226,
    .data = data_227,
    .f_data = NULL
};
        
static const int8_t shape_228[] = { 1 };
static const float data_229[] = { 0x7fffffff };
static const Tensor BLOCK3_LAYER4_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_228,
    .data = NULL,
    .f_data = data_229
};
        
static const int8_t shape_230[] = { 1 };
static const float data_231[] = { 0x42fe0000 };
static const Tensor BLOCK3_LAYER4_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_230,
    .data = NULL,
    .f_data = data_231
};
        
static const int8_t shape_232[] = { 7 };
static const int8_t data_233[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK3_LAYER4_RUNNING_MEAN = {
    .dims = 1,
    .size = 7,
    .shape = shape_232,
    .data = data_233,
    .f_data = NULL
};
        
static const int8_t shape_234[] = { 7 };
static const int8_t data_235[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK3_LAYER4_RUNNING_VAR = {
    .dims = 1,
    .size = 7,
    .shape = shape_234,
    .data = data_235,
    .f_data = NULL
};
        
static const int8_t shape_236[] = { 1 };
static const int8_t data_237[] = { 0x00 };
static const Tensor BLOCK4_LAYER1_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_236,
    .data = data_237,
    .f_data = NULL
};
        
static const int8_t shape_238[] = { 14, 7, 1, 1 };
static const int8_t data_239[] = { 0xa0, 0x80, 0x00, 0x7f, 0x7f, 0x80, 0x00, 0xe0, 0x80, 0x7f, 0x00, 0x60, 0x80, 0x80, 0x40, 0x7f, 0x60, 0x80, 0x7f, 0xc0, 0x80, 0x80, 0x7f, 0x00, 0xc0, 0x80, 0xa0, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x00, 0x7f, 0x80, 0x7f, 0x80, 0x00, 0x80, 0xa0, 0x80, 0x80, 0x80, 0x80, 0x20, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x20, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0x80, 0x7f, 0x80, 0x7f, 0x40, 0x60, 0xc0, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0xa0, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x00, 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x7f };
static const Tensor BLOCK4_LAYER1_GROUP_CONV_WEIGHT = {
    .dims = 4,
    .size = 98,
    .shape = shape_238,
    .data = data_239,
    .f_data = NULL
};
        
static const int8_t shape_240[] = { 14 };
static const int8_t data_241[] = { 0x80, 0x7f, 0xc0, 0x80, 0x7f, 0x80, 0xe0, 0x7f, 0x80, 0x80, 0x40, 0xc0, 0x60, 0x80 };
static const Tensor BLOCK4_LAYER1_GROUP_CONV_BIAS = {
    .dims = 1,
    .size = 14,
    .shape = shape_240,
    .data = data_241,
    .f_data = NULL
};
        
static const int8_t shape_242[] = { 1 };
static const float data_243[] = { 0x428e0000 };
static const Tensor BLOCK4_LAYER1_GROUP_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_242,
    .data = NULL,
    .f_data = data_243
};
        
static const int8_t shape_244[] = { 1 };
static const float data_245[] = { 0x41c80000 };
static const Tensor BLOCK4_LAYER1_GROUP_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_244,
    .data = NULL,
    .f_data = data_245
};
        
static const int8_t shape_246[] = { 11, 14, 1, 1 };
static const int8_t data_247[] = { 0x00, 0x80, 0x80, 0x80, 0x7f, 0x40, 0x80, 0x7f, 0x60, 0x40, 0x80, 0x20, 0x7f, 0x7f, 0x80, 0x7f, 0x20, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0x80, 0x20, 0x80, 0xc0, 0x00, 0xe0, 0xe0, 0x20, 0x7f, 0x7f, 0x7f, 0x60, 0x7f, 0x00, 0x20, 0x40, 0x80, 0x80, 0x20, 0xa0, 0x00, 0xc0, 0x20, 0x7f, 0xa0, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0xa0, 0xe0, 0xc0, 0x80, 0xa0, 0xa0, 0x7f, 0xa0, 0x80, 0x7f, 0x00, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x00, 0x7f, 0xa0, 0x80, 0x40, 0x60, 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0x40, 0x80, 0x7f, 0x20, 0x40, 0x60, 0x7f, 0x80, 0xc0, 0x40, 0x20, 0x7f, 0x80, 0x00, 0x40, 0xc0, 0x7f, 0x20, 0x80, 0xe0, 0x80, 0x20, 0xe0, 0x80, 0xc0, 0xe0, 0xa0, 0x7f, 0xc0, 0x7f, 0x40, 0x80, 0x40, 0x7f, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x7f, 0xe0, 0x80, 0x20, 0x60, 0xa0, 0x60, 0x80, 0x7f, 0x7f, 0x7f, 0x20, 0x60, 0x80, 0xc0, 0xe0, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x60, 0x60, 0x7f, 0x7f, 0x7f, 0xa0, 0x7f, 0x7f, 0x80, 0x80 };
static const Tensor BLOCK4_LAYER1_POINTWISE_CONV_WEIGHT = {
    .dims = 4,
    .size = 154,
    .shape = shape_246,
    .data = data_247,
    .f_data = NULL
};
        
static const int8_t shape_248[] = { 11 };
static const int8_t data_249[] = { 0x40, 0x7f, 0x80, 0x20, 0x00, 0x40, 0x80, 0x7f, 0x7f, 0x80, 0x00 };
static const Tensor BLOCK4_LAYER1_POINTWISE_CONV_BIAS = {
    .dims = 1,
    .size = 11,
    .shape = shape_248,
    .data = data_249,
    .f_data = NULL
};
        
static const int8_t shape_250[] = { 1 };
static const float data_251[] = { 0x7fffffff };
static const Tensor BLOCK4_LAYER1_POINTWISE_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_250,
    .data = NULL,
    .f_data = data_251
};
        
static const int8_t shape_252[] = { 1 };
static const float data_253[] = { 0x41880000 };
static const Tensor BLOCK4_LAYER1_POINTWISE_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_252,
    .data = NULL,
    .f_data = data_253
};
        
static const int8_t shape_254[] = { 7, 11, 1, 1 };
static const int8_t data_255[] = { 0x00, 0x60, 0x7f, 0x60, 0xc0, 0x7f, 0x80, 0xa0, 0x20, 0x80, 0x20, 0x80, 0x20, 0x20, 0x20, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x7f, 0x80, 0x20, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x00, 0x7f, 0x80, 0xc0, 0x80, 0x20, 0x60, 0x20, 0x80, 0x80, 0x60, 0x80, 0xc0, 0x00, 0x80, 0x7f, 0x20, 0x7f, 0xe0, 0x80, 0x7f, 0x20, 0xa0, 0x80, 0x40, 0x80, 0x60, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x20, 0x7f, 0xa0, 0x80, 0x60, 0x60, 0x7f, 0x00, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0x80, 0x7f, 0x00 };
static const Tensor BLOCK4_LAYER1_EXPAND_CONV_WEIGHT = {
    .dims = 4,
    .size = 77,
    .shape = shape_254,
    .data = data_255,
    .f_data = NULL
};
        
static const int8_t shape_256[] = { 7 };
static const int8_t data_257[] = { 0x80, 0x60, 0xc0, 0x80, 0xa0, 0x20, 0x7f };
static const Tensor BLOCK4_LAYER1_EXPAND_CONV_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_256,
    .data = data_257,
    .f_data = NULL
};
        
static const int8_t shape_258[] = { 1 };
static const float data_259[] = { 0x7fffffff };
static const Tensor BLOCK4_LAYER1_EXPAND_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_258,
    .data = NULL,
    .f_data = data_259
};
        
static const int8_t shape_260[] = { 1 };
static const float data_261[] = { 0x41980000 };
static const Tensor BLOCK4_LAYER1_EXPAND_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_260,
    .data = NULL,
    .f_data = data_261
};
        
static const int8_t shape_262[] = { 7 };
static const int8_t data_263[] = { 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f };
static const Tensor BLOCK4_LAYER2_WEIGHT = {
    .dims = 1,
    .size = 7,
    .shape = shape_262,
    .data = data_263,
    .f_data = NULL
};
        
static const int8_t shape_264[] = { 7 };
static const int8_t data_265[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK4_LAYER2_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_264,
    .data = data_265,
    .f_data = NULL
};
        
static const int8_t shape_266[] = { 1 };
static const float data_267[] = { 0x7fffffff };
static const Tensor BLOCK4_LAYER2_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_266,
    .data = NULL,
    .f_data = data_267
};
        
static const int8_t shape_268[] = { 1 };
static const float data_269[] = { 0x42fe0000 };
static const Tensor BLOCK4_LAYER2_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_268,
    .data = NULL,
    .f_data = data_269
};
        
static const int8_t shape_270[] = { 7 };
static const int8_t data_271[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK4_LAYER2_RUNNING_MEAN = {
    .dims = 1,
    .size = 7,
    .shape = shape_270,
    .data = data_271,
    .f_data = NULL
};
        
static const int8_t shape_272[] = { 7 };
static const int8_t data_273[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK4_LAYER2_RUNNING_VAR = {
    .dims = 1,
    .size = 7,
    .shape = shape_272,
    .data = data_273,
    .f_data = NULL
};
        
static const int8_t shape_274[] = { 1 };
static const int8_t data_275[] = { 0x80 };
static const Tensor BLOCK4_LAYER3_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_274,
    .data = data_275,
    .f_data = NULL
};
        
static const int8_t shape_276[] = { 22, 7, 1, 1 };
static const int8_t data_277[] = { 0x80, 0x7f, 0x40, 0x00, 0x40, 0x80, 0x80, 0x80, 0x80, 0x80, 0x20, 0x80, 0x80, 0xc0, 0xe0, 0x80, 0x20, 0xc0, 0x60, 0x20, 0x80, 0x80, 0x00, 0x80, 0x20, 0xe0, 0x7f, 0x60, 0x7f, 0x00, 0x80, 0x7f, 0x80, 0xc0, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x60, 0x7f, 0x80, 0x40, 0x7f, 0x7f, 0x80, 0x7f, 0x80, 0x00, 0xa0, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x00, 0x80, 0x7f, 0x7f, 0x40, 0x80, 0x60, 0x7f, 0x7f, 0x80, 0x80, 0x7f, 0x20, 0x80, 0x80, 0x80, 0x7f, 0xc0, 0x7f, 0x7f, 0x40, 0x60, 0xe0, 0x7f, 0x80, 0x7f, 0xc0, 0x7f, 0x00, 0x80, 0xe0, 0x7f, 0x80, 0x80, 0xa0, 0x80, 0x7f, 0x7f, 0xe0, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x60, 0x7f, 0x7f, 0x7f, 0xe0, 0x40, 0x80, 0x7f, 0x80, 0x7f, 0x40, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x40, 0x7f, 0x7f, 0x7f, 0x7f, 0xc0, 0x7f, 0x60, 0x7f, 0x7f, 0x60, 0x80, 0xe0, 0x60, 0x40, 0xa0, 0x7f, 0x7f, 0x80, 0xe0, 0xe0, 0x80, 0x40, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x60, 0x60, 0x7f };
static const Tensor BLOCK4_LAYER3_GROUP_CONV_WEIGHT = {
    .dims = 4,
    .size = 154,
    .shape = shape_276,
    .data = data_277,
    .f_data = NULL
};
        
static const int8_t shape_278[] = { 22 };
static const int8_t data_279[] = { 0x40, 0x80, 0x80, 0xa0, 0x00, 0x7f, 0x7f, 0x7f, 0x40, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0xe0, 0x80, 0x00, 0x7f, 0x00 };
static const Tensor BLOCK4_LAYER3_GROUP_CONV_BIAS = {
    .dims = 1,
    .size = 22,
    .shape = shape_278,
    .data = data_279,
    .f_data = NULL
};
        
static const int8_t shape_280[] = { 1 };
static const float data_281[] = { 0x7fffffff };
static const Tensor BLOCK4_LAYER3_GROUP_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_280,
    .data = NULL,
    .f_data = data_281
};
        
static const int8_t shape_282[] = { 1 };
static const float data_283[] = { 0x41b00000 };
static const Tensor BLOCK4_LAYER3_GROUP_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_282,
    .data = NULL,
    .f_data = data_283
};
        
static const int8_t shape_284[] = { 7, 22, 1, 1 };
static const int8_t data_285[] = { 0x00, 0x20, 0x60, 0xc0, 0x80, 0x40, 0xe0, 0xa0, 0x60, 0x00, 0xe0, 0xa0, 0xe0, 0x80, 0x40, 0x00, 0x7f, 0x00, 0x20, 0x40, 0x7f, 0xa0, 0x7f, 0x40, 0x80, 0x40, 0x80, 0xc0, 0xa0, 0x80, 0x7f, 0x80, 0x60, 0x80, 0xe0, 0x00, 0x7f, 0xc0, 0x40, 0x20, 0x7f, 0x00, 0x60, 0x80, 0x80, 0x7f, 0x80, 0xe0, 0xe0, 0x7f, 0xe0, 0x80, 0xc0, 0x7f, 0x60, 0x7f, 0xe0, 0x60, 0x80, 0x60, 0x60, 0x7f, 0x80, 0xc0, 0x40, 0x80, 0x80, 0x80, 0x00, 0x7f, 0x40, 0x00, 0x40, 0x80, 0x20, 0xc0, 0xa0, 0x7f, 0x80, 0x20, 0x80, 0x00, 0xa0, 0x40, 0x80, 0x7f, 0x80, 0x7f, 0x80, 0x7f, 0xa0, 0x00, 0xe0, 0x7f, 0x40, 0x80, 0x20, 0x80, 0x7f, 0x20, 0xc0, 0xa0, 0x40, 0x80, 0x60, 0x7f, 0x7f, 0x40, 0x80, 0x7f, 0xc0, 0xc0, 0xe0, 0x7f, 0x7f, 0x20, 0x60, 0x80, 0x00, 0x80, 0x40, 0x80, 0xc0, 0x7f, 0x7f, 0x80, 0x80, 0x20, 0x60, 0xe0, 0xc0, 0x80, 0x60, 0x7f, 0xa0, 0x60, 0x60, 0x7f, 0xe0, 0x00, 0x7f, 0x7f, 0x7f, 0xc0, 0x60, 0x7f, 0x80, 0x20, 0xc0, 0x7f, 0xa0, 0x00, 0x7f, 0x80 };
static const Tensor BLOCK4_LAYER3_POINTWISE_CONV_WEIGHT = {
    .dims = 4,
    .size = 154,
    .shape = shape_284,
    .data = data_285,
    .f_data = NULL
};
        
static const int8_t shape_286[] = { 7 };
static const int8_t data_287[] = { 0x80, 0xa0, 0x7f, 0x7f, 0x80, 0x20, 0x80 };
static const Tensor BLOCK4_LAYER3_POINTWISE_CONV_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_286,
    .data = data_287,
    .f_data = NULL
};
        
static const int8_t shape_288[] = { 1 };
static const float data_289[] = { 0x7fffffff };
static const Tensor BLOCK4_LAYER3_POINTWISE_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_288,
    .data = NULL,
    .f_data = data_289
};
        
static const int8_t shape_290[] = { 1 };
static const float data_291[] = { 0x41500000 };
static const Tensor BLOCK4_LAYER3_POINTWISE_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_290,
    .data = NULL,
    .f_data = data_291
};
        
static const int8_t shape_292[] = { 7, 7, 1, 1 };
static const int8_t data_293[] = { 0x80, 0x7f, 0x7f, 0xa0, 0x40, 0x40, 0x60, 0x80, 0x80, 0x7f, 0x40, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x60, 0x7f, 0x00, 0x60, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0xe0, 0x7f, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0xc0, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x20, 0x7f, 0x20, 0xe0, 0x80, 0x80 };
static const Tensor BLOCK4_LAYER3_EXPAND_CONV_WEIGHT = {
    .dims = 4,
    .size = 49,
    .shape = shape_292,
    .data = data_293,
    .f_data = NULL
};
        
static const int8_t shape_294[] = { 7 };
static const int8_t data_295[] = { 0x80, 0x7f, 0xc0, 0x80, 0x20, 0x20, 0xe0 };
static const Tensor BLOCK4_LAYER3_EXPAND_CONV_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_294,
    .data = data_295,
    .f_data = NULL
};
        
static const int8_t shape_296[] = { 1 };
static const float data_297[] = { 0x7fffffff };
static const Tensor BLOCK4_LAYER3_EXPAND_CONV_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_296,
    .data = NULL,
    .f_data = data_297
};
        
static const int8_t shape_298[] = { 1 };
static const float data_299[] = { 0x41c80000 };
static const Tensor BLOCK4_LAYER3_EXPAND_CONV_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_298,
    .data = NULL,
    .f_data = data_299
};
        
static const int8_t shape_300[] = { 7 };
static const int8_t data_301[] = { 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f };
static const Tensor BLOCK4_LAYER4_WEIGHT = {
    .dims = 1,
    .size = 7,
    .shape = shape_300,
    .data = data_301,
    .f_data = NULL
};
        
static const int8_t shape_302[] = { 7 };
static const int8_t data_303[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK4_LAYER4_BIAS = {
    .dims = 1,
    .size = 7,
    .shape = shape_302,
    .data = data_303,
    .f_data = NULL
};
        
static const int8_t shape_304[] = { 1 };
static const float data_305[] = { 0x7fffffff };
static const Tensor BLOCK4_LAYER4_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_304,
    .data = NULL,
    .f_data = data_305
};
        
static const int8_t shape_306[] = { 1 };
static const float data_307[] = { 0x42fe0000 };
static const Tensor BLOCK4_LAYER4_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_306,
    .data = NULL,
    .f_data = data_307
};
        
static const int8_t shape_308[] = { 7 };
static const int8_t data_309[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK4_LAYER4_RUNNING_MEAN = {
    .dims = 1,
    .size = 7,
    .shape = shape_308,
    .data = data_309,
    .f_data = NULL
};
        
static const int8_t shape_310[] = { 7 };
static const int8_t data_311[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const Tensor BLOCK4_LAYER4_RUNNING_VAR = {
    .dims = 1,
    .size = 7,
    .shape = shape_310,
    .data = data_311,
    .f_data = NULL
};
        
static const int8_t shape_312[] = { 17, 7, 3, 3 };
static const int8_t data_313[] = { 0x00, 0xe0, 0x00, 0x80, 0x80, 0x80, 0x80, 0xa0, 0x80, 0xc0, 0x00, 0xe0, 0xe0, 0xa0, 0xc0, 0x80, 0xc0, 0xc0, 0xc0, 0xa0, 0x80, 0x80, 0x80, 0x80, 0xe0, 0x80, 0x80, 0xa0, 0x80, 0xa0, 0x80, 0x80, 0x80, 0x20, 0x80, 0xe0, 0x80, 0x80, 0xa0, 0x80, 0xc0, 0x80, 0x80, 0x00, 0xc0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0x00, 0xa0, 0x80, 0xe0, 0x20, 0x80, 0x80, 0xa0, 0x40, 0xe0, 0x20, 0x80, 0x40, 0xa0, 0x00, 0x40, 0x20, 0x00, 0x80, 0x80, 0x80, 0xc0, 0x80, 0xa0, 0x20, 0x40, 0x80, 0x00, 0xc0, 0x80, 0x00, 0x00, 0x00, 0x00, 0xc0, 0xe0, 0x80, 0xe0, 0x80, 0xe0, 0xe0, 0x20, 0xe0, 0xe0, 0x00, 0xe0, 0xe0, 0xa0, 0xc0, 0xa0, 0x00, 0xa0, 0x20, 0x40, 0x80, 0xe0, 0xc0, 0x80, 0xe0, 0xa0, 0xe0, 0x80, 0x00, 0x80, 0x80, 0x00, 0x40, 0x00, 0x80, 0xa0, 0x80, 0xc0, 0x80, 0xc0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0xa0, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x60, 0xc0, 0x80, 0x80, 0x60, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x80, 0xe0, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0xa0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x80, 0xc0, 0x80, 0x80, 0x80, 0x80, 0xa0, 0x00, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x00, 0x80, 0x80, 0x00, 0x20, 0xc0, 0x00, 0x20, 0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xc0, 0x80, 0xc0, 0xc0, 0x20, 0xc0, 0xc0, 0xc0, 0x80, 0x20, 0xc0, 0x00, 0x80, 0xc0, 0x00, 0xe0, 0xa0, 0x00, 0xa0, 0x80, 0x80, 0x80, 0x80, 0xe0, 0xa0, 0xa0, 0xe0, 0xe0, 0xc0, 0x40, 0xc0, 0x20, 0x00, 0x80, 0xc0, 0x00, 0xa0, 0xe0, 0x00, 0xa0, 0xa0, 0xc0, 0x80, 0xa0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x20, 0x80, 0x80, 0x7f, 0xa0, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0xc0, 0x80, 0x40, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x20, 0xe0, 0x80, 0x80, 0x60, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0xe0, 0x7f, 0xe0, 0xc0, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x7f, 0x40, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x40, 0xa0, 0x20, 0x7f, 0x7f, 0xe0, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0x80, 0x60, 0x80, 0x80, 0x00, 0x80, 0x80, 0x80, 0x80, 0xa0, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0xa0, 0xc0, 0x80, 0x80, 0x80, 0x80, 0xa0, 0x20, 0x60, 0x7f, 0x7f, 0x80, 0x60, 0x00, 0xa0, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x40, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0xa0, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0xc0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xc0, 0x80, 0x80, 0x00, 0xc0, 0x80, 0x20, 0xa0, 0x80, 0x80, 0x80, 0x80, 0x60, 0x80, 0x80, 0x80, 0x60, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x40, 0x80, 0x80, 0x80, 0xe0, 0x7f, 0xc0, 0x7f, 0x80, 0x80, 0x80, 0xc0, 0x20, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x20, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x60, 0x20, 0x80, 0x80, 0xc0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0xc0, 0x40, 0x7f, 0x7f, 0x40, 0x40, 0x7f, 0x7f, 0x7f, 0x00, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x20, 0xa0, 0x00, 0xe0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x60, 0x7f, 0x80, 0x80, 0xa0, 0xc0, 0xe0, 0x80, 0xc0, 0xe0, 0x00, 0xa0, 0xa0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xa0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x20, 0x00, 0xc0, 0x7f, 0x20, 0x80, 0xa0, 0x80, 0xe0, 0x80, 0x80, 0x40, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x20, 0x00, 0xa0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0xa0, 0xc0, 0x20, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0xc0, 0xa0, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x40, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xc0, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0xc0, 0x80, 0x00, 0x7f, 0xe0, 0xa0, 0x80, 0x80, 0x80, 0xe0, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0xe0, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0xc0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x60, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xc0, 0x40, 0x40, 0x7f, 0x20, 0x40, 0x20, 0x7f, 0x60, 0x7f, 0x80, 0x00, 0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x00, 0x80, 0xa0, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x40, 0x80, 0x00, 0x80, 0x80, 0xa0, 0x7f, 0x00, 0x60, 0xa0, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x00, 0x20, 0xc0, 0x00, 0x80, 0x80, 0x80, 0x60, 0x20, 0x80, 0x80, 0x60, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0xc0, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x80, 0xa0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0xa0, 0x7f, 0xc0, 0xa0, 0xc0, 0x80, 0x80, 0x80, 0xc0, 0x60, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x60, 0x7f, 0xa0, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0xe0, 0x20, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x20, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xe0, 0xe0, 0x7f, 0xa0, 0x20, 0xc0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xa0, 0xa0, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xa0, 0x80, 0x80, 0x80, 0x80, 0x7f, 0xe0, 0x80, 0x80, 0x80, 0xa0, 0x80, 0x80, 0x80, 0x80, 0xa0, 0x80, 0x80, 0xe0, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0xa0, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x60, 0x80, 0x60, 0x60, 0xa0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0x60, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x80, 0xa0, 0x80, 0x80, 0x7f, 0x7f, 0x80, 0x7f, 0x7f, 0x80 };
static const Tensor CONV2_WEIGHT = {
    .dims = 4,
    .size = 1071,
    .shape = shape_312,
    .data = data_313,
    .f_data = NULL
};
        
static const int8_t shape_314[] = { 17 };
static const int8_t data_315[] = { 0xa0, 0xa0, 0x00, 0xe0, 0xa0, 0x7f, 0x20, 0xa0, 0x40, 0x40, 0xc0, 0x40, 0xa0, 0x20, 0x7f, 0x60, 0x40 };
static const Tensor CONV2_BIAS = {
    .dims = 1,
    .size = 17,
    .shape = shape_314,
    .data = data_315,
    .f_data = NULL
};
        
static const int8_t shape_316[] = { 1 };
static const float data_317[] = { 0x42540000 };
static const Tensor CONV2_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_316,
    .data = NULL,
    .f_data = data_317
};
        
static const int8_t shape_318[] = { 1 };
static const float data_319[] = { 0x42240000 };
static const Tensor CONV2_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_318,
    .data = NULL,
    .f_data = data_319
};
        
static const int8_t shape_320[] = { 6, 17 };
static const int8_t data_321[] = { 0x60, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x7f, 0x80, 0x7f, 0x80, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x7f, 0x60, 0x80, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x40, 0x7f, 0x80, 0x7f, 0xe0, 0x7f, 0x7f, 0x80, 0x80, 0xe0, 0xe0, 0x7f, 0x7f, 0x80, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0x80, 0x7f, 0x80, 0x7f, 0x80, 0x80, 0x7f, 0xc0, 0x80, 0x7f, 0x7f, 0x80, 0x20, 0x80, 0x7f, 0x80, 0x7f, 0x7f, 0x80, 0x7f, 0x80, 0x7f, 0x80, 0x7f };
static const Tensor FC_WEIGHT = {
    .dims = 2,
    .size = 102,
    .shape = shape_320,
    .data = data_321,
    .f_data = NULL
};
        
static const int8_t shape_322[] = { 1 };
static const float data_323[] = { 0x42a80000 };
static const Tensor FC_ACTIVATION_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_322,
    .data = NULL,
    .f_data = data_323
};
        
static const int8_t shape_324[] = { 1 };
static const float data_325[] = { 0x42ca0000 };
static const Tensor FC_WEIGHT_SCALE = {
    .dims = 1,
    .size = 1,
    .shape = shape_324,
    .data = NULL,
    .f_data = data_325
};
        
typedef struct {
    const uint8_t id;
    int *address;
} VariableMap;

VariableMap model_weights[] = {
    { 0, &CONV1_WEIGHT },
    { 1, &CONV1_BIAS },
    { 2, &CONV1_ACTIVATION_SCALE },
    { 3, &CONV1_WEIGHT_SCALE },
    { 4, &BLOCK1_LAYER1_SCALE },
    { 5, &BLOCK1_LAYER1_GROUP_CONV_WEIGHT },
    { 6, &BLOCK1_LAYER1_GROUP_CONV_BIAS },
    { 7, &BLOCK1_LAYER1_GROUP_CONV_ACTIVATION_SCALE },
    { 8, &BLOCK1_LAYER1_GROUP_CONV_WEIGHT_SCALE },
    { 9, &BLOCK1_LAYER1_POINTWISE_CONV_WEIGHT },
    { 10, &BLOCK1_LAYER1_POINTWISE_CONV_BIAS },
    { 11, &BLOCK1_LAYER1_POINTWISE_CONV_ACTIVATION_SCALE },
    { 12, &BLOCK1_LAYER1_POINTWISE_CONV_WEIGHT_SCALE },
    { 13, &BLOCK1_LAYER1_EXPAND_CONV_WEIGHT },
    { 14, &BLOCK1_LAYER1_EXPAND_CONV_BIAS },
    { 15, &BLOCK1_LAYER1_EXPAND_CONV_ACTIVATION_SCALE },
    { 16, &BLOCK1_LAYER1_EXPAND_CONV_WEIGHT_SCALE },
    { 17, &BLOCK1_LAYER2_WEIGHT },
    { 18, &BLOCK1_LAYER2_BIAS },
    { 19, &BLOCK1_LAYER2_ACTIVATION_SCALE },
    { 20, &BLOCK1_LAYER2_WEIGHT_SCALE },
    { 21, &BLOCK1_LAYER2_RUNNING_MEAN },
    { 22, &BLOCK1_LAYER2_RUNNING_VAR },
    { 23, &BLOCK1_LAYER3_SCALE },
    { 24, &BLOCK1_LAYER3_GROUP_CONV_WEIGHT },
    { 25, &BLOCK1_LAYER3_GROUP_CONV_BIAS },
    { 26, &BLOCK1_LAYER3_GROUP_CONV_ACTIVATION_SCALE },
    { 27, &BLOCK1_LAYER3_GROUP_CONV_WEIGHT_SCALE },
    { 28, &BLOCK1_LAYER3_POINTWISE_CONV_WEIGHT },
    { 29, &BLOCK1_LAYER3_POINTWISE_CONV_BIAS },
    { 30, &BLOCK1_LAYER3_POINTWISE_CONV_ACTIVATION_SCALE },
    { 31, &BLOCK1_LAYER3_POINTWISE_CONV_WEIGHT_SCALE },
    { 32, &BLOCK1_LAYER3_EXPAND_CONV_WEIGHT },
    { 33, &BLOCK1_LAYER3_EXPAND_CONV_BIAS },
    { 34, &BLOCK1_LAYER3_EXPAND_CONV_ACTIVATION_SCALE },
    { 35, &BLOCK1_LAYER3_EXPAND_CONV_WEIGHT_SCALE },
    { 36, &BLOCK1_LAYER4_WEIGHT },
    { 37, &BLOCK1_LAYER4_BIAS },
    { 38, &BLOCK1_LAYER4_ACTIVATION_SCALE },
    { 39, &BLOCK1_LAYER4_WEIGHT_SCALE },
    { 40, &BLOCK1_LAYER4_RUNNING_MEAN },
    { 41, &BLOCK1_LAYER4_RUNNING_VAR },
    { 42, &BLOCK2_LAYER1_SCALE },
    { 43, &BLOCK2_LAYER1_GROUP_CONV_WEIGHT },
    { 44, &BLOCK2_LAYER1_GROUP_CONV_BIAS },
    { 45, &BLOCK2_LAYER1_GROUP_CONV_ACTIVATION_SCALE },
    { 46, &BLOCK2_LAYER1_GROUP_CONV_WEIGHT_SCALE },
    { 47, &BLOCK2_LAYER1_POINTWISE_CONV_WEIGHT },
    { 48, &BLOCK2_LAYER1_POINTWISE_CONV_BIAS },
    { 49, &BLOCK2_LAYER1_POINTWISE_CONV_ACTIVATION_SCALE },
    { 50, &BLOCK2_LAYER1_POINTWISE_CONV_WEIGHT_SCALE },
    { 51, &BLOCK2_LAYER1_EXPAND_CONV_WEIGHT },
    { 52, &BLOCK2_LAYER1_EXPAND_CONV_BIAS },
    { 53, &BLOCK2_LAYER1_EXPAND_CONV_ACTIVATION_SCALE },
    { 54, &BLOCK2_LAYER1_EXPAND_CONV_WEIGHT_SCALE },
    { 55, &BLOCK2_LAYER2_WEIGHT },
    { 56, &BLOCK2_LAYER2_BIAS },
    { 57, &BLOCK2_LAYER2_ACTIVATION_SCALE },
    { 58, &BLOCK2_LAYER2_WEIGHT_SCALE },
    { 59, &BLOCK2_LAYER2_RUNNING_MEAN },
    { 60, &BLOCK2_LAYER2_RUNNING_VAR },
    { 61, &BLOCK2_LAYER3_SCALE },
    { 62, &BLOCK2_LAYER3_GROUP_CONV_WEIGHT },
    { 63, &BLOCK2_LAYER3_GROUP_CONV_BIAS },
    { 64, &BLOCK2_LAYER3_GROUP_CONV_ACTIVATION_SCALE },
    { 65, &BLOCK2_LAYER3_GROUP_CONV_WEIGHT_SCALE },
    { 66, &BLOCK2_LAYER3_POINTWISE_CONV_WEIGHT },
    { 67, &BLOCK2_LAYER3_POINTWISE_CONV_BIAS },
    { 68, &BLOCK2_LAYER3_POINTWISE_CONV_ACTIVATION_SCALE },
    { 69, &BLOCK2_LAYER3_POINTWISE_CONV_WEIGHT_SCALE },
    { 70, &BLOCK2_LAYER3_EXPAND_CONV_WEIGHT },
    { 71, &BLOCK2_LAYER3_EXPAND_CONV_BIAS },
    { 72, &BLOCK2_LAYER3_EXPAND_CONV_ACTIVATION_SCALE },
    { 73, &BLOCK2_LAYER3_EXPAND_CONV_WEIGHT_SCALE },
    { 74, &BLOCK2_LAYER4_WEIGHT },
    { 75, &BLOCK2_LAYER4_BIAS },
    { 76, &BLOCK2_LAYER4_ACTIVATION_SCALE },
    { 77, &BLOCK2_LAYER4_WEIGHT_SCALE },
    { 78, &BLOCK2_LAYER4_RUNNING_MEAN },
    { 79, &BLOCK2_LAYER4_RUNNING_VAR },
    { 80, &BLOCK3_LAYER1_SCALE },
    { 81, &BLOCK3_LAYER1_GROUP_CONV_WEIGHT },
    { 82, &BLOCK3_LAYER1_GROUP_CONV_BIAS },
    { 83, &BLOCK3_LAYER1_GROUP_CONV_ACTIVATION_SCALE },
    { 84, &BLOCK3_LAYER1_GROUP_CONV_WEIGHT_SCALE },
    { 85, &BLOCK3_LAYER1_POINTWISE_CONV_WEIGHT },
    { 86, &BLOCK3_LAYER1_POINTWISE_CONV_BIAS },
    { 87, &BLOCK3_LAYER1_POINTWISE_CONV_ACTIVATION_SCALE },
    { 88, &BLOCK3_LAYER1_POINTWISE_CONV_WEIGHT_SCALE },
    { 89, &BLOCK3_LAYER1_EXPAND_CONV_WEIGHT },
    { 90, &BLOCK3_LAYER1_EXPAND_CONV_BIAS },
    { 91, &BLOCK3_LAYER1_EXPAND_CONV_ACTIVATION_SCALE },
    { 92, &BLOCK3_LAYER1_EXPAND_CONV_WEIGHT_SCALE },
    { 93, &BLOCK3_LAYER2_WEIGHT },
    { 94, &BLOCK3_LAYER2_BIAS },
    { 95, &BLOCK3_LAYER2_ACTIVATION_SCALE },
    { 96, &BLOCK3_LAYER2_WEIGHT_SCALE },
    { 97, &BLOCK3_LAYER2_RUNNING_MEAN },
    { 98, &BLOCK3_LAYER2_RUNNING_VAR },
    { 99, &BLOCK3_LAYER3_SCALE },
    { 100, &BLOCK3_LAYER3_GROUP_CONV_WEIGHT },
    { 101, &BLOCK3_LAYER3_GROUP_CONV_BIAS },
    { 102, &BLOCK3_LAYER3_GROUP_CONV_ACTIVATION_SCALE },
    { 103, &BLOCK3_LAYER3_GROUP_CONV_WEIGHT_SCALE },
    { 104, &BLOCK3_LAYER3_POINTWISE_CONV_WEIGHT },
    { 105, &BLOCK3_LAYER3_POINTWISE_CONV_BIAS },
    { 106, &BLOCK3_LAYER3_POINTWISE_CONV_ACTIVATION_SCALE },
    { 107, &BLOCK3_LAYER3_POINTWISE_CONV_WEIGHT_SCALE },
    { 108, &BLOCK3_LAYER3_EXPAND_CONV_WEIGHT },
    { 109, &BLOCK3_LAYER3_EXPAND_CONV_BIAS },
    { 110, &BLOCK3_LAYER3_EXPAND_CONV_ACTIVATION_SCALE },
    { 111, &BLOCK3_LAYER3_EXPAND_CONV_WEIGHT_SCALE },
    { 112, &BLOCK3_LAYER4_WEIGHT },
    { 113, &BLOCK3_LAYER4_BIAS },
    { 114, &BLOCK3_LAYER4_ACTIVATION_SCALE },
    { 115, &BLOCK3_LAYER4_WEIGHT_SCALE },
    { 116, &BLOCK3_LAYER4_RUNNING_MEAN },
    { 117, &BLOCK3_LAYER4_RUNNING_VAR },
    { 118, &BLOCK4_LAYER1_SCALE },
    { 119, &BLOCK4_LAYER1_GROUP_CONV_WEIGHT },
    { 120, &BLOCK4_LAYER1_GROUP_CONV_BIAS },
    { 121, &BLOCK4_LAYER1_GROUP_CONV_ACTIVATION_SCALE },
    { 122, &BLOCK4_LAYER1_GROUP_CONV_WEIGHT_SCALE },
    { 123, &BLOCK4_LAYER1_POINTWISE_CONV_WEIGHT },
    { 124, &BLOCK4_LAYER1_POINTWISE_CONV_BIAS },
    { 125, &BLOCK4_LAYER1_POINTWISE_CONV_ACTIVATION_SCALE },
    { 126, &BLOCK4_LAYER1_POINTWISE_CONV_WEIGHT_SCALE },
    { 127, &BLOCK4_LAYER1_EXPAND_CONV_WEIGHT },
    { 128, &BLOCK4_LAYER1_EXPAND_CONV_BIAS },
    { 129, &BLOCK4_LAYER1_EXPAND_CONV_ACTIVATION_SCALE },
    { 130, &BLOCK4_LAYER1_EXPAND_CONV_WEIGHT_SCALE },
    { 131, &BLOCK4_LAYER2_WEIGHT },
    { 132, &BLOCK4_LAYER2_BIAS },
    { 133, &BLOCK4_LAYER2_ACTIVATION_SCALE },
    { 134, &BLOCK4_LAYER2_WEIGHT_SCALE },
    { 135, &BLOCK4_LAYER2_RUNNING_MEAN },
    { 136, &BLOCK4_LAYER2_RUNNING_VAR },
    { 137, &BLOCK4_LAYER3_SCALE },
    { 138, &BLOCK4_LAYER3_GROUP_CONV_WEIGHT },
    { 139, &BLOCK4_LAYER3_GROUP_CONV_BIAS },
    { 140, &BLOCK4_LAYER3_GROUP_CONV_ACTIVATION_SCALE },
    { 141, &BLOCK4_LAYER3_GROUP_CONV_WEIGHT_SCALE },
    { 142, &BLOCK4_LAYER3_POINTWISE_CONV_WEIGHT },
    { 143, &BLOCK4_LAYER3_POINTWISE_CONV_BIAS },
    { 144, &BLOCK4_LAYER3_POINTWISE_CONV_ACTIVATION_SCALE },
    { 145, &BLOCK4_LAYER3_POINTWISE_CONV_WEIGHT_SCALE },
    { 146, &BLOCK4_LAYER3_EXPAND_CONV_WEIGHT },
    { 147, &BLOCK4_LAYER3_EXPAND_CONV_BIAS },
    { 148, &BLOCK4_LAYER3_EXPAND_CONV_ACTIVATION_SCALE },
    { 149, &BLOCK4_LAYER3_EXPAND_CONV_WEIGHT_SCALE },
    { 150, &BLOCK4_LAYER4_WEIGHT },
    { 151, &BLOCK4_LAYER4_BIAS },
    { 152, &BLOCK4_LAYER4_ACTIVATION_SCALE },
    { 153, &BLOCK4_LAYER4_WEIGHT_SCALE },
    { 154, &BLOCK4_LAYER4_RUNNING_MEAN },
    { 155, &BLOCK4_LAYER4_RUNNING_VAR },
    { 156, &CONV2_WEIGHT },
    { 157, &CONV2_BIAS },
    { 158, &CONV2_ACTIVATION_SCALE },
    { 159, &CONV2_WEIGHT_SCALE },
    { 160, &FC_WEIGHT },
    { 161, &FC_ACTIVATION_SCALE },
    { 162, &FC_WEIGHT_SCALE },
    {NULL, NULL}
};
#endif
