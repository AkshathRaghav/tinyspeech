from training.modules import * 

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime


# ---- 
# w/ Quant
# ----

class QTinySpeechZ(nn.Module): 
    def __init__(self, num_classes, QuantType='8bit', WScale='PerTensor', NormType='RMS', quantscale=0.25, test=0): 
        super(QTinySpeechZ, self).__init__()
        self.conv1 = BitConv2d(1, 7, kernel_size=3, stride=1, padding=1, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)

        self.block1 = QAttn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=3, mid_channels_1=6, out_channels_1=7, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale, test=test)
        self.block2 = QAttn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=3, mid_channels_1=6, out_channels_1=7, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale, test=test)
        self.block3 = QAttn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=2, mid_channels_1=4, out_channels_1=7, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale, test=test)
        self.block4 = QAttn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=11, mid_channels_1=22, out_channels_1=7, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale, test=test)
        # self.block5 = QAttn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=14, mid_channels_1=28, out_channels_1=7, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale, test=test)
        # self.block6 = QAttn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=10, mid_channels_1=20, out_channels_1=7, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale, test=test)

        self.conv2 = BitConv2d(in_channels=7, out_channels=17, kernel_size=3, stride=1, padding=1, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = BitLinear(17, num_classes, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            self.cuda()

    def __str__(self): 
        return "QTinySpeechZ"

    def forward(self, x):
        x = x.to(self.device)   
        x = F.relu(self.conv1(x))  
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        x = F.relu(self.conv2(x))  
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc(x)  
        return x

# ---- 
# w/o Quant
# ----

class TinySpeechZ(nn.Module): 
    def __init__(self, num_classes, test=0): 
        super(TinySpeechZ, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, stride=1, padding=1)

        self.block1 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=3, mid_channels_1=6, out_channels_1=7, test=test)
        self.block2 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=3, mid_channels_1=6, out_channels_1=7, test=test)
        self.block3 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=2, mid_channels_1=4, out_channels_1=7, test=test)
        self.block4 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=11, mid_channels_1=22, out_channels_1=7, test=test)
        self.block5 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=14, mid_channels_1=28, out_channels_1=7, test=test)
        self.block6 = Attn_BN_Block(in_channels=7, mid_channels_0=14, out_channels_0=10, mid_channels_1=20, out_channels_1=7, test=test)

        self.conv2 = nn.Conv2d(in_channels=7, out_channels=17, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(17, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.test = test 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            self.cuda()

    def __str__(self): 
        return "TinySpeechZ"

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))  
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.relu(self.conv2(x))  
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc(x)  
            
        return x


class TinySpeechZ1(TinySpeechZ): 
    def __init__(self, num_classes, test=0): 
        super().__init__(num_classes=num_classes)

        if torch.cuda.is_available():
            self.cuda()

    def __str__(self): 
        return "TinySpeechZ-1"
    
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))  
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.relu(self.conv2(x))  
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc(x)  
            
        return x

class TinySpeechZ2(TinySpeechZ): 
    def __init__(self, num_classes, test=0): 
        super().__init__(num_classes=num_classes)

        if torch.cuda.is_available():
            self.cuda()

    def __str__(self): 
        return "TinySpeechZ-2"
    
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))  
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.relu(self.conv2(x))  
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc(x)  
            
        return x

class TinySpeechZ3(TinySpeechZ): 
    def __init__(self, num_classes, test=0): 
        super().__init__(num_classes=num_classes)

    def __str__(self): 
        return "TinySpeechZ-3"

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))  
        x = self.block1(x)
        x = self.block2(x)
        x = F.relu(self.conv2(x))  
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc(x)  
            
        return x

if __name__ == "__main__": 

    import numpy as np 

    def generate_header_file(model, filename_, runname): 
        if not model.__class__.__name__ in ["QTinySpeechZ"]: 
            raise ValueError("Model not in supported quants.")

        filename = []
        final_struct = {} 
        bit_quant = BitQuant(QuantType="8bit", WScale="PerTensor", quantscale=0.25)
        id_ = [0]


        def product(shape): 
            prod = 1
            for i in shape: 
                prod *= i
            return prod 

        def encode(weights): 
            return [f"0x{w & 0xFF:02x}" for w in weights]

        def generate_c_code_int(tensor_, tensor_name):
            tensor = tensor_
            shape_str = ", ".join(encode(tensor.shape))

            val, scale, _ = bit_quant.weight_quant(tensor)
            
            tensor = val.detach().cpu().numpy()
            if np.isnan(tensor).any() or np.isinf(tensor).any():
                tensor = np.nan_to_num(tensor, nan=0.0, posinf=127, neginf=-128)
            data_str = ", ".join(encode(tensor.flatten().astype(np.int8)))

            c_code = f"""
    static const int8_t arr_{id_[0]} = {{ {shape_str} }};
    static const int8_t arr_{id_[0] + 1} = {{ {data_str} }};
    static const Tensor {tensor_name} = {{
        .dims = {len(tensor.shape)},
        .size = {product(tensor.shape)},
        .shape = &arr_{id_[0]},
        .data = &arr_{id_[0]+1},
        .f_data = NULL
    }};
        """
            id_[0] += 2
            return c_code


        def generate_c_code_float(tensor_, tensor_name):
            tensor = tensor_.detach().cpu().numpy()
            shape_str = ", ".join(encode(tensor.shape))
            data_str = ", ".join(f"{x:.6f}" for x in tensor)  

            c_code = f"""
    static const int8_t arr_{id_[0]} = {{ {shape_str} }};
    static const float arr_{id_[0] + 1} = {{ {data_str} }};
    static const Tensor {tensor_name} = {{
        .dims = {len(tensor.shape)},
        .size = {product(tensor.shape)},
        .shape = &arr_{id_[0]},
        .data = NULL,
        .f_data = &arr_{id_[0]+1}
    }};
        """
            id_[0] += 2
            return c_code

        def extract_model_structure(filename, model, parent_name=""):
            for name, module in model.named_children():
                full_name = f"{parent_name}_{name}".upper() if parent_name else name.upper()
                temp = None 
                for param_name, param in module.named_parameters(recurse=False):
                    if (param_name == "scale"): 
                        temp = True 
                        filename += [f"\n{generate_c_code_int(param, f'{full_name}_SCALE')}"]
                    else:
                        final_struct[f"{full_name}_{param_name.upper()}"] = param.shape
                        filename += [f"\n{generate_c_code_int(param, f'{full_name}_{param_name.upper()}')}"]
                for buffer_name, buffer in module.named_buffers(recurse=False):
                    if (buffer.shape): 
                        final_struct[f"{full_name}_{buffer_name.upper()}"] = buffer.shape
                        filename += [f"\n{generate_c_code_int(buffer, f'{full_name}_{buffer_name.upper()}')}"]

                if len(list(module.children())) > 0:
                    extract_model_structure(filename, module, parent_name=full_name)

                if temp: 
                    final_struct[f"{full_name}_SCALE"] = param.shape

        extract_model_structure(filename, model)
        filename += [""" 
    typedef struct {
        const uint8_t id;
        int *address;
    } VariableMap;

    VariableMap model_weights[] = {
    """]

        for i, names in enumerate(final_struct.keys()): 
            filename += ['    {', str(i), ', &', names, '},\n'] 

        filename += ["    {NULL, NULL}\n};"]


        with open(filename_, "w") as f: 
            f.write(f'// Automatically generated header file\n')
            f.write(f'// Date: {datetime.now()}\n')
            f.write(f'// Quantized model exported from {runname}.pth\n\n')

            f.write('#include <stdint.h>\n#include "tensor.h"\n\n')

            f.write('#ifndef TINYSPEECH_WEIGHTS_H\n')
            f.write('#define TINYSPEECH_WEIGHTS_H')

            if model.__class__.__name__ == "QTinySpeechZ": 
                f.write("""

    // Block 1
    #define B1_IN 7
    #define B1_MC_0 14
    #define B1_OC_0 3
    #define B1_MC_1 6
    #define B1_OC_1 7

    // Block 2
    #define B2_IN 7
    #define B2_MC_0 14
    #define B2_OC_0 3
    #define B2_MC_1 6
    #define B2_OC_1 7

    // Block 3
    #define B3_IN 7
    #define B3_MC_0 14
    #define B3_OC_0 2
    #define B3_MC_1 4
    #define B3_OC_1 7

    // Block 4
    #define B4_IN 7
    #define B4_MC_0 14
    #define B4_OC_0 11
    #define B4_MC_1 22
    #define B4_OC_1 7

    // Block 5
    #define B5_IN 7
    #define B5_MC_0 14
    #define B5_OC_0 14
    #define B5_MC_1 28
    #define B5_OC_1 7

    // Block 6
    #define B6_IN 7
    #define B6_MC_0 14
    #define B6_OC_0 10
    #define B6_MC_1 20
    #define B6_OC_1 7
                """)


            f.write("".join(filename))
            f.write('\n\n#endif\n')

    model = QTinySpeechZ(num_classes=6, QuantType=f"8bit")
    model = QModel(model)
    model.load_checkpoint('/depot/euge/data/araviki/vsdsquadronmini/models/QTinySpeechZ_quant.pth')
    # state_dict = torch.load('/depot/euge/data/araviki/vsdsquadronmini/models/QTinySpeechZ')
    # model.load_state_dict(state_dict)
    # generate_header_file(model, "/depot/euge/data/araviki/vsdsquadronmini/inference/include/weights.h", '/depot/euge/data/araviki/vsdsquadronmini/models/QTinySpeechZ')

        
    # torch.Size([1, 1, 32, 32])
    # torch.Size([1, 7, 32, 32])
    # torch.Size([1, 7, 32, 32])
    # torch.Size([1, 7, 32, 32])
    # torch.Size([1, 7, 32, 32])
    # torch.Size([1, 7, 32, 32])
    # torch.Size([1, 17, 32, 32])
    # torch.Size([1, 17, 1, 1])
    # torch.Size([1, 17])
    # torch.Size([1, 10])
