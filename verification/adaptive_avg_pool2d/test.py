import numpy as np
import torch
import torch.nn as nn

def save_tensor(tensor, filename):
    shape = tensor.shape
    with open(filename, "wb") as f:
        f.write(np.array(shape, dtype=np.int8).tobytes())
        f.write(tensor.flatten().astype(np.int8).tobytes())

if __name__ == "__main__":
    # Define a sample input tensor
    input_tensor = torch.tensor([
        [[2, 1, 0],
         [1, 2, 3]],
        [[5, 2, 8],
         [1, 2, 3]]
    ], dtype=torch.int)

    # Define the adaptive average pooling layer
    adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)

    # Apply adaptive average pooling
    output_tensor = adaptive_avg_pool(input_tensor)

    # Save input and output tensors
    save_tensor(input_tensor, "input_tensor.bin")
    save_tensor(output_tensor, "output_tensor.bin")

    print("Input Tensor:")
    print(input_tensor)

    print("\nOutput Tensor (after AdaptiveAvgPool2d):")
    print(output_tensor)
