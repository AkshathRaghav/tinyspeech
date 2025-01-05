import numpy as np
import torch
import torch.nn as nn

def save_tensor(tensor, filename):
    shape = tensor.shape
    with open(filename, "wb") as f:
        f.write(np.array(shape, dtype=np.int8).tobytes())
        f.write(tensor.flatten().astype(np.float32).tobytes())

if __name__ == "__main__":
    input_tensor = torch.randint(-128, 128, (1, 17, 32, 32), dtype=torch.float)
    adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)
    output_tensor = adaptive_avg_pool(input_tensor)

    save_tensor(input_tensor.detach().numpy(), "./input_tensor.bin")
    save_tensor(output_tensor.detach().numpy(), "./output_tensor.bin")