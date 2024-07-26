import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def save_tensor_custom(tensor, filename):
    shape = tensor.shape
    with open(filename, 'wb') as f:
        if len(shape) == 2: shape = (*shape, 0, 0)
        for dim in shape:
            f.write(np.array(dim, dtype=np.int8).tobytes())
        f.write(tensor.tobytes())

if __name__ == "__main__":
    batch_size = 10
    input_tensor = torch.randn(batch_size, 1, 12, 94)
    save_tensor_custom(input_tensor.numpy(), "input_tensor.bin")

    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    output_tensor = max_pool(input_tensor)

    save_tensor_custom(
        output_tensor.detach().numpy(), 
        "output_tensor_max_pool.bin")

    upsample = nn.Upsample(scale_factor=2, mode='nearest')
    output_tensor = upsample(output_tensor)

    save_tensor_custom(
        output_tensor.detach().numpy(), 
        "output_tensor_upsample.bin")
    
    global_pool = nn.AdaptiveAvgPool2d(1)
    output_tensor = global_pool(input_tensor)

    save_tensor_custom(
        output_tensor.detach().numpy(), 
        "output_tensor_global_pool.bin")
    
    softmax = nn.Softmax(dim=1)
    input_tensor = torch.randn(batch_size, 10)
    output_tensor = softmax(input_tensor)

    save_tensor_custom(input_tensor.numpy(), "input_tensor_softmax.bin")    
    save_tensor_custom(
        output_tensor.detach().numpy(), 
        "output_tensor_softmax.bin")
