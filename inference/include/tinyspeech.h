#ifndef TINYSPEECH_H
#define TINYSPEECH_H

#include "tensor.h"
#include "weights.h"
#include "modules.h"

Tensor TinySpeechZ(Tensor* input, uint8_t num_classes);

#endif