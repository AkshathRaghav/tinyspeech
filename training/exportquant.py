# Taken from https://github.com/cpldcpu/BitNetMCU/blob/main/exportquant.py

import numpy as np 
from datetime import datetime
from modules import BitQuant
import struct

def generate_model_header_file(model, output_file, quant_mode, quant_type):
    if model.__class__.__name__ not in ["QModel"]:
        raise ValueError("Unsupported model type. Supported types: QModel")

    header_lines = []  
    model_structure = {}  
    quantizer = BitQuant(QuantType=quant_type, WScale="PerTensor", quantscale=0.25)
    tensor_id = [0]  

    def calculate_product(shape):
        product = 1
        for dim in shape:
            product *= dim
        return product

    def encode_int(weights):
        return [f"0x{value & 0xFF:02x}" for value in weights]

    def encode_float(value):
        packed = struct.pack('>f', value) 
        unsigned_int = struct.unpack('>I', packed)[0]  
        return f"0x{unsigned_int:08x}" 

    def generate_int_tensor_code(tensor, tensor_name):
        tensor_np = tensor.detach().cpu().numpy()
        shape_str = ", ".join(map(str, tensor_np.shape))

        quantized_values, _, _ = quantizer.weight_quant(tensor)
        quantized_tensor = quantized_values.detach().cpu().numpy()

        if np.isnan(quantized_tensor).any() or np.isinf(quantized_tensor).any():
            quantized_tensor = np.nan_to_num(quantized_tensor, nan=0.0, posinf=127, neginf=-128)

        data_str = ", ".join(encode_int(quantized_tensor.flatten().astype(np.int8)))

        c_code = f"""
static const int8_t shape_{tensor_id[0]}[] = {{ {shape_str} }};
static const int8_t data_{tensor_id[0] + 1}[] = {{ {data_str} }};
static const Tensor {tensor_name} = {{
    .dims = {len(tensor_np.shape)},
    .size = {calculate_product(tensor_np.shape)},
    .shape = shape_{tensor_id[0]},
    .data = data_{tensor_id[0] + 1},
    .f_data = NULL
}};
        """
        tensor_id[0] += 2
        return c_code

    def generate_float_tensor_code(tensor, tensor_name):
        tensor_np = tensor.detach().cpu().numpy()
        shape_str = ", ".join(map(str, tensor_np.shape))
        data_str = ", ".join(encode_float(value) for value in tensor_np.flatten())
        dims = len(tensor_np.shape)
        if (dims == 0): 
            shape_str = str(1)
            dims = 1


        c_code = f"""
static const int8_t shape_{tensor_id[0]}[] = {{ {shape_str} }};
static const float data_{tensor_id[0] + 1}[] = {{ {data_str} }};
static const Tensor {tensor_name} = {{
    .dims = {dims},
    .size = {calculate_product(tensor_np.shape)},
    .shape = shape_{tensor_id[0]},
    .data = NULL,
    .f_data = data_{tensor_id[0] + 1}
}};
        """
        tensor_id[0] += 2
        return c_code

    def extract_model_structure(header_lines, model, parent_name=""):
        for module_name, module in model.named_children():
            full_name = f"{parent_name}_{module_name}".upper() if parent_name else module_name.upper()

            for param_name, param in module.named_parameters(recurse=False):
                tensor_name = f"{full_name}_{param_name.upper()}"
                model_structure[tensor_name] = param.shape
                if "calibrated" in param_name: 
                    header_lines.append(generate_float_tensor_code(param, tensor_name))
                else: 
                    header_lines.append(generate_int_tensor_code(param, tensor_name))

            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer.shape:
                    tensor_name = f"{full_name}_{buffer_name.upper()}"
                    model_structure[tensor_name] = buffer.shape
                    if "calibrated" in buffer_name: 
                        header_lines.append(generate_float_tensor_code(buffer, tensor_name))
                    else: 
                        header_lines.append(generate_int_tensor_code(buffer, tensor_name))

            if len(list(module.children())) > 0:
                extract_model_structure(header_lines, module, parent_name=full_name)

    extract_model_structure(header_lines, model)
    
    header_lines.append("""
// Block 1
#define B1_IN       7    // Input channels
#define B1_MC_0     14   // Mid channels 0
#define B1_OC_0     3    // Out channels 0
#define B1_MC_1     6    // Mid channels 1
#define B1_OC_1     7    // Out channels 1

// Block 2
#define B2_IN       7    // Input channels
#define B2_MC_0     14   // Mid channels 0
#define B2_OC_0     3    // Out channels 0
#define B2_MC_1     6    // Mid channels 1
#define B2_OC_1     7    // Out channels 1

// Block 3
#define B3_IN       7    // Input channels
#define B3_MC_0     14   // Mid channels 0
#define B3_OC_0     2    // Out channels 0
#define B3_MC_1     4    // Mid channels 1
#define B3_OC_1     7    // Out channels 1

// Block 4
#define B4_IN       7    // Input channels
#define B4_MC_0     14   // Mid channels 0
#define B4_OC_0     11   // Out channels 0
#define B4_MC_1     22   // Mid channels 1
#define B4_OC_1     7    // Out channels 1

// Block 5
#define B5_IN       7    // Input channels
#define B5_MC_0     14   // Mid channels 0
#define B5_OC_0     14   // Out channels 0
#define B5_MC_1     28   // Mid channels 1
#define B5_OC_1     7    // Out channels 1

// Block 6
#define B6_IN       7    // Input channels
#define B6_MC_0     14   // Mid channels 0
#define B6_OC_0     10   // Out channels 0
#define B6_MC_1     20   // Mid channels 1
#define B6_OC_1     7    // Out channels 1
""")

    header_lines.append("""
typedef struct {
    const u_int8_t id;
    int *address;
} VariableMap;

VariableMap model_weights[] = {
""")

    for idx, tensor_name in enumerate(model_structure.keys()):
        header_lines.append(f"    {{ {idx}, &{tensor_name} }},\n")

    header_lines.append("    {NULL, NULL}\n};")

    with open(output_file, "w") as file:
        file.write("// Automatically generated header file\n")
        file.write(f"// Date: {datetime.now()}\n")
        file.write(f"// Quantized model exported from {model.__class__.__name__}.pth\n")
        file.write(f"// Quantization Type: {quant_type}\n")
        file.write(f"// Quantization Mode: {quant_mode}\n\n")

        file.write("#include <stdint.h>\n#include \"tensor.h\"\n\n")

        file.write("#ifndef TINYSPEECH_WEIGHTS_H\n")
        file.write("#define TINYSPEECH_WEIGHTS_H\n\n")

        file.write("// Constants:\n")
        if quant_mode in ["QAT", "SQ"]:  file.write(f"#define QUANT_MODE_QAT_SQ\n")
        else: file.write(f"#define QUANT_MODE_{quant_mode}\n")
        file.write(f"#define CONVERT_FLOAT 1\n")
        file.write(f"#define CONVERT_INT8 0\n\n")

        file.write("".join(header_lines).replace("MODEL_", "").replace("CALIBRATED_", ""))

        file.write("\n#endif\n")

if __name__ == "__main__": 
    model = QTinySpeechZ(num_classes=6, QuantType=f"8bit", WScale="PerTensor", quantscale=0.25)
    model = QModel(model)
    model.load_checkpoint('/depot/euge/data/araviki/vsdsquadronmini/models/QTinySpeechZ_quant.pth')
    generate_model_header_file(model, "/depot/euge/data/araviki/vsdsquadronmini/inference/include/weights.h", "QAT", "8bit")