import numpy as np
import torch
import torch.nn as nn
import argparse

def save_tensor(tensor, filename):
    shape = tensor.shape
    with open(filename, 'wb') as f:
        for dim in shape:
            f.write(np.array(dim, dtype=np.int8).tobytes())
        f.write(tensor.flatten().astype(np.int8).tobytes())

def save_tensor_f(tensor, filename):
    shape = tensor.shape
    with open(filename, 'wb') as f:
        for dim in shape:
            f.write(np.array(dim, dtype=np.int8).tobytes())
        f.write(tensor.flatten().tobytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script') 
    parser.add_argument('--scale', type=float, default=0.25, help='Scaling for quantization')
    parser.add_argument('--bpw', type=str, default="8bit", help='Bits per weight')
    args = parser.parse_args()

    input_tensor = torch.tensor([
        [2, 1, 4],
        [1, 2, 3]
    ], dtype=torch.float)

    with torch.no_grad(): 
        softmax_layer = nn.Softmax(dim=1)
        output_tensor = softmax_layer(input_tensor)

    save_tensor(input_tensor.detach().numpy(), "./input_tensor.bin")
    save_tensor_f(output_tensor.detach().numpy(), "./output_tensor.bin")
