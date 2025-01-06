import torch
import torch.nn as nn
import numpy as np

def save_tensor(tensor, filename):
    shape = tensor.shape
    with open(filename, 'wb') as f:
        for dim in shape:
            f.write(np.array(dim, dtype=np.int8).tobytes())
        f.write(tensor.flatten().astype(np.int8).tobytes())

def f_save_tensor(tensor, filename):
    shape = tensor.shape
    with open(filename, 'wb') as f:
        for dim in shape:
            f.write(np.array(dim, dtype=np.int8).tobytes())
        f.write(tensor.flatten().astype(np.float32).tobytes())

if __name__ == "__main__":
    input_tensor = torch.randint(-15, 15, (1, 3, 5, 5), dtype=torch.float32)
    batchnorm_layer = nn.BatchNorm2d(num_features=3)

    f_save_tensor(batchnorm_layer.running_mean.detach().numpy(), "./mean.bin")
    f_save_tensor(batchnorm_layer.running_var.detach().numpy(), "./variance.bin")
    f_save_tensor(batchnorm_layer.weight.detach().numpy(), "./gamma.bin")
    f_save_tensor(batchnorm_layer.bias.detach().numpy(), "./beta.bin")
    
    # Piece of shit layer uses running statistics WHICH DO GET UPDATED within no_grad=True.
    with torch.no_grad(): 
        output_tensor = batchnorm_layer(input_tensor)

    save_tensor(input_tensor.detach().numpy(), "./input_tensor.bin")
    save_tensor(output_tensor.detach().numpy(), "./output_tensor.bin")

    # print("Input Tensor:")
    # print(input_tensor)

    # print("\nMean:")
    # print(batchnorm_layer.running_mean.shape)
    # print(batchnorm_layer.running_mean)

    # print("\nVariance:")
    # print(batchnorm_layer.running_var.shape)
    # print(batchnorm_layer.running_var)

    # print("\nGamma (Scale):")
    # print(batchnorm_layer.weight.shape)
    # print(batchnorm_layer.weight)

    # print("\nBeta (Shift):")
    # print(batchnorm_layer.bias.shape)
    # print(batchnorm_layer.bias)

    # print("\nOutput Tensor (after BatchNorm2d):")
    # print(output_tensor.shape)
    # print(output_tensor)
