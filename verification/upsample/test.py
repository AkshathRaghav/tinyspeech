import torch 
import numpy as np
import torch.nn as nn
import argparse

def save_tensor_custom(tensor, filename):
    shape = tensor.shape
    with open(filename, 'wb') as f:
        for dim in shape:
            f.write(np.array(dim, dtype=np.int8).tobytes())
        f.write(tensor.flatten().astype(np.int8).tobytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script') 
    parser.add_argument('--scale', type=float, default=0.25, help='Scaling for quantization')
    parser.add_argument('--bpw', type=str, default="8bit", help='Bits per weight')
    args = parser.parse_args()

    upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')

    with torch.no_grad(): 
        input_tensor = torch.tensor([[1., 2., 3.], 
                        [4., 5., 6.], 
                        ]) 
        input_tensor = input_tensor.view(1,1,2,3) 
        output_tensor = upsample_layer(input_tensor)


    save_tensor_custom(input_tensor.detach().numpy(), "./input_tensor.bin")
    save_tensor_custom(output_tensor.detach().numpy(), "./output_tensor.bin")

