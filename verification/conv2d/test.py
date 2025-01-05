import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.nn.functional as F

def save_tensor(tensor, filename):
    shape = tensor.shape
    with open(filename, 'wb') as f:
        for dim in shape:
            f.write(np.array(dim, dtype=np.int8).tobytes())
        f.write(tensor.flatten().astype(np.int8).tobytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script') 
    parser.add_argument('--scale', type=float, default=0.1, help='Scaling for quantization')
    parser.add_argument('--bpw', type=str, default="8bit", help='Bits per weight')
    args = parser.parse_args()

    input_tensor = torch.randint(-128, 128, (1, 3, 5, 5), dtype=torch.float32)
    conv_layer = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)

    # Naive scaling to mimic quant
    conv_layer.weight.data = conv_layer.weight.data * 100
    conv_layer.bias.data = conv_layer.bias.data * 10

    with torch.no_grad(): 
        output_tensor = conv_layer(input_tensor)

    # Naive de-scaling to mimic quant
    output_tensor /= 100

    save_tensor(input_tensor.detach().numpy(), "input_tensor.bin")
    save_tensor(conv_layer.weight.detach().numpy(), "weights.bin")
    save_tensor(conv_layer.bias.detach().numpy(), "bias.bin")
    save_tensor(output_tensor.detach().numpy(), "output_tensor.bin")

    print("Input Tensor:")
    print(input_tensor)

    print("\nWeights:")
    print(conv_layer.weight)

    print("\nBias:")
    print(conv_layer.bias)

    print("\nOutput Tensor (after Conv2D):")
    print(output_tensor)



