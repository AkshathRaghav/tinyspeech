#ifndef TINYSPEECH_H
#define TINYSPEECH_H

#include "tensor.h"
#include "weights.h"
#include "modules.h"

Tensor TinySpeechZ(Tensor* input, u_int8_t num_classes);

#endif