import torch
import torch.nn as nn
import numpy as np

def quantize_tensor(tensor, scale, zero_point):
    return ((tensor / scale).round() + zero_point).clamp(0, 255).to(torch.uint8)

def dequantize_tensor(tensor, scale, zero_point):
    return (tensor.to(torch.float32) - zero_point) * scale

def export_to_hfile(weights, bias, filename):
    with open(filename, 'w') as f:
        f.write("#ifndef CONV_LAYER_H\n")
        f.write("#define CONV_LAYER_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"const uint8_t conv_weights[{weights.size}] = {{")
        f.write(", ".join(map(str, weights.flatten())))
        f.write("};\n\n")

        f.write(f"const uint8_t conv_bias[{bias.size}] = {{")
        f.write(", ".join(map(str, bias.flatten())))
        f.write("};\n\n")

        f.write("#endif // CONV_LAYER_H\n")

if __name__ == "__main__":
    weight_scale = 0.1
    weight_zero_point = 128

    in_channels = 1
    mid_channels = 7
    kernel_size = 1
    conv_layer = nn.Conv2d(in_channels, mid_channels, kernel_size)

    input_tensor = torch.randn(64, 1, 12, 94)

    output_tensor = conv_layer(input_tensor)

    quantized_weights = quantize_tensor(conv_layer.weight.data, weight_scale, weight_zero_point)
    quantized_bias = quantize_tensor(conv_layer.bias.data, weight_scale, weight_zero_point)

    np.savez("quantized_conv_layer.npz", weights=quantized_weights.numpy(), bias=quantized_bias.numpy())

    export_to_hfile(quantized_weights.numpy(), quantized_bias.numpy(), "conv_layer.h")

\