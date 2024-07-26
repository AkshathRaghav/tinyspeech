#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <stdint.h>

int8_t conv_weights[7] = {-23.0, 35.0, -40.0, 42.0, 11.0, 42.0, -31.0};

float conv_weights_scale = 46.96187973022461;

char* bpw = "8bit";
#endif // CONV_LAYER_H
