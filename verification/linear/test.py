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
        f.write(tensor.tobytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verification Parser') 
    parser.add_argument('--scale', type=float, default=0.25, help='Scaling for quantization')
    parser.add_argument('--bpw', type=str, default="8bit", help='Bits per weight')
    args = parser.parse_args()

    in_features = 17
    out_features = 10

    input_tensor = torch.randint(-128, 128, (1, 17), dtype=torch.float)
    fc_layer = nn.Linear(17, 10)
    output_tensor = fc_layer(input_tensor)

    save_tensor(input_tensor.detach().numpy(), "input_tensor.bin")
    save_tensor(fc_layer.weight.data.detach().numpy(), "weights.bin")
    save_tensor(output_tensor.detach().numpy(), "output_tensor.bin")

    # fc_layer = BitLinear(in_features, out_features, QuantType=args.bpw, quantscale=args.scale)
    # quant = BitQuant(QuantType=args.bpw, quantscale=args.scale)

    # quant_weight, quant_weight_scale, _ = quant.weight_quant(fc_layer.weight.data)
    # export_to_hfile(args.bpw, quant_weight.numpy(), quant_weight_scale.numpy(), "fc_layer.h")

    # with torch.no_grad(): 
    #     input_tensor = torch.randn(1, 17) # post GAP 
    #     output_tensor = fc_layer(input_tensor)

    # input_tensor, input_scale = quant.activation_quant(input_tensor)
    # output_tensor, output_scale = quant.activation_quant(output_tensor)