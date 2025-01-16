import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def log_tensor(name, tensor): 
    tensor = tensor.detach().cpu().numpy()
    print(f"Layer Name: {name}, Output Tensor Shape: {tensor.shape}")

    if " " not in name.strip(): 
        with open(f"./logs/{name}.bin", 'wb') as f:
            for dim in tensor.shape:
                f.write(np.array(dim, dtype=np.int8).tobytes())
            f.write(tensor.flatten().astype(float).tobytes())


# ---- 
# w/ Quant
# ----

class QModel(nn.Module):
    def __init__(self, model, quant_type='4bitsym', w_scale='PerTensor', quant_scale=0.25):
        super(QModel, self).__init__()
        self.model = model
        self.bit_quant = BitQuant(QuantType=quant_type, WScale=w_scale, quantscale=quant_scale)
        self.original_weights = {}
        self.scales = {}

    def __str__(self): 
        return str(self.model)

    def quantize(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name or 'bias' in name or 'scale' in name or 'calibrated' in name:
                q_param, scale, _ = self.bit_quant.weight_quant(param.data)
                self.original_weights[name] = param.data.clone()
                param.data = q_param
                self.scales[name] = scale

    def forward(self, x):
        # Unquantize weights for inference
        for name, scale in self.scales.items():
            param = self.get_submodule(name)
            param.data = self.bit_quant.dequantize_tensor(param.data, scale, 0)

        # Forward pass
        output = self.model(x)

        # Requantize weights after inference
        for name, original_weight in self.original_weights.items():
            param = self.get_submodule(name)
            param.data = original_weight
        
        return output

    def get_submodule(self, target):
        names = target.split('.')
        mod = self.model
        for name in names[:-1]:
            mod = getattr(mod, name)
        return getattr(mod, names[-1])

    def numel(self):
        total_params = 0
        param_details = {}
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            param_details[name] = num_params
            total_params += num_params
        return total_params, param_details

    def save_checkpoint(self, filepath):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'original_weights': self.original_weights,
            'scales': self.scales,
            'quant_type': self.bit_quant.QuantType,
            'w_scale': self.bit_quant.WScale,
            'quant_scale': self.bit_quant.quantscale
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        print(checkpoint['model_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.original_weights = checkpoint['original_weights']
        self.scales = checkpoint['scales']
        self.bit_quant.QuantType = checkpoint['quant_type']
        self.bit_quant.WScale = checkpoint['w_scale']
        self.bit_quant.quantscale = checkpoint['quant_scale']

class BitQuant:
    def __init__(self, QuantType='4bitsym', WScale='PerTensor', quantscale=0.25):
        self.QuantType = QuantType
        self.WScale = WScale
        self.quantscale = quantscale
        self.activations_mean = None

        self.calibrated_activation_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calibrated_activation_scale.requires_grad = False
        self.calibrated_weight_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calibrated_weight_scale.requires_grad = False

    def activation_quant(self, x):
        mag = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        if not self.activations_mean: 
            self.activations_mean = x.abs().mean().clamp_(min=1e-5)
        else: 
            self.activations_mean += x.abs().mean().clamp_(min=1e-5)
            self.activations_mean /= 2
        mag = mag * self.calibrated_activation_scale

        if self.QuantType == '2bitsym':
            scale = 1.0 / mag
            y = (x * scale).round().clamp_(-2, 1)
        elif self.QuantType == '4bitsym':
            scale = 8.0 / mag
            y = (x * scale).round().clamp_(-8, 7)
        elif self.QuantType == '8bit':
            scale = 127.0 / mag
            y = (x * scale).round().clamp_(-128, 127)
        else:
            raise AssertionError(f'Invalid QuantType: {self.QuantType}. Expected one of: "2bitsym", "4bitsym", "8bit"')

        return y, scale

    def update_clipping_scalar(self, activations_mean, w):

        self.calibrated_activation_scale = torch.nn.Parameter(activations_mean.clone().detach() / self.quantscale)
        self.calibrated_activation_scale.requires_grad = False  

        if self.WScale == 'PerOutput':
            s = w.abs().max(dim=-1, keepdim=True)[0].clamp_(min=1e-5) / self.quantscale
        else:
            s = w.abs().mean().clamp_(min=1e-5) / self.quantscale

        self.calibrated_weight_scale = torch.nn.Parameter(s)
        self.calibrated_weight_scale.requires_grad = False  

    def weight_quant(self, w):
        if self.QuantType == '2bitsym':
            scale = 1.0 / self.calibrated_weight_scale
            u = ((w * scale - 0.5).round().clamp_(-2, 1) + 0.5)
            bpw = 2
        elif self.QuantType == '4bitsym':
            scale = self.quantscale * 8.0 / self.calibrated_weight_scale
            u = ((w * scale - 0.5).round().clamp_(-8, 7) + 0.5)
            bpw = 4
        elif self.QuantType == '8bit':
            scale = 128.0 * self.quantscale / self.calibrated_weight_scale
            u = (w * scale).round().clamp_(-128, 127)
            bpw = 8
        else:
            raise AssertionError(f"Invalid QuantType: {self.QuantType}. Expected one of: '2bitsym', '4bitsym', '8bit'")

        return u, scale, bpw

    def ste(self, x_q, x_scale, x): 
        """ 
        Applying STE for approximating gradients during backprop to avoid clipping  
        """
        return x + (x_q / x_scale - x).detach() 

    def norm(self, x):
        """Using only RMS"""
        y = torch.sqrt(torch.mean(x**2, dim=(-2,-1), keepdim=True))
        return x / y

    def dequantize_tensor(self, quantized_tensor, scale):
        if self.QuantType == '2bitsym':
            dequantized_tensor = (quantized_tensor + 0.5) / scale
        elif self.QuantType == '4bitsym':
            dequantized_tensor = (quantized_tensor + 0.5) / scale
        elif self.QuantType == '8bit':
            dequantized_tensor = quantized_tensor / scale
        else:
            raise AssertionError(f"Invalid QuantType: {self.QuantType}. Expected one of: '2bitsym', '4bitsym', '8bit'")
        return dequantized_tensor

class BitParameter(nn.Parameter, BitQuant):
    def __init__(self, val, QuantType='4bitsym', WScale='PerTensor', NormType='RMS', quantscale=0.25):
        nn.Parameter.__init__(val)
        BitQuant.__init__(self, QuantType, WScale, quantscale)
        self.NormType = NormType

    def forward(self, x):
        x_norm = self.norm(x)
        param_norm = self.norm(self.data) 
        
        x_int, x_scale = self.activation_quant(x_norm)
        x_quant = self.ste(x_int, x_scale, x_norm)
        
        param_int, param_scale, _ = self.weight_quant(param_norm)
        param_quant = self.ste(param_int, param_scale, param_norm)

        y = x_quant * param_quant
    
        return y

class BitBatchNorm2d(nn.BatchNorm2d, BitQuant):
    def __init__(self, num_features, QuantType='4bitsym', WScale='PerTensor', NormType='RMS', quantscale=0.25):
        nn.BatchNorm2d.__init__(self, num_features)
        BitQuant.__init__(self, QuantType, WScale, quantscale)
        self.NormType = NormType

    def forward(self, x):
        x_norm = self.norm(x)
        w = self.weight
        b = self.bias
        
        x_int, x_scale = self.activation_quant(x_norm)
        x_quant = self.ste(x_int, x_scale, x_norm)
        
        w_int, w_scale, _ = self.weight_quant(w)
        w_quant = self.ste(w_int, w_scale, w)
        
        b_int, b_scale, _ = self.weight_quant(b)
        b_quant = self.ste(b_int, b_scale, b)
        
        y = F.batch_norm(x_quant, self.running_mean, self.running_var, w_quant, b_quant, self.training, self.momentum, self.eps)
    
        return y

class BitLinear(nn.Linear, BitQuant):
    def __init__(self, in_features, out_features, bias=False, QuantType='Binary', WScale='PerTensor', NormType='RMS', quantscale=0.25):
        nn.Linear.__init__(self, in_features, out_features, bias=False)
        BitQuant.__init__(self, QuantType, WScale, quantscale)

        self.NormType = NormType

    def forward(self, x):
        w = self.weight # a weight tensor with shape [d, k]
        x_norm = self.norm(x)

        x_int, x_scale = self.activation_quant(x_norm)
        x_quant = self.ste(x_int, x_scale, x_norm)

        w_int, w_scale, _ = self.weight_quant(w)
        w_quant = self.ste(w_int, w_scale, w)

        y = F.linear(x_quant, w_quant)
        return y

class BitConv2d(nn.Conv2d, BitQuant):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1,  QuantType='4bitsym', WScale='PerTensor', NormType='RMS', quantscale=0.25):
        nn.Conv2d.__init__(self,in_channels, out_channels, kernel_size=kernel_size, stride=(stride,), padding=(padding,), groups=groups, bias=True)
        BitQuant.__init__(self, QuantType, WScale, quantscale)

        self.NormType = NormType
        self.groups = groups
        self.stride = (stride,)
        self.padding = (padding,)

    def forward(self, x):
        w = self.weight 
        x_norm = self.norm(x)

        x_int, x_scale = self.activation_quant(x_norm)
        x_quant = self.ste(x_int, x_scale, x_norm)  

        w_int, w_scale, _ = self.weight_quant(w)
        w_quant = self.ste(w_int, w_scale, w)

        # Interesting: If you switch baise=None to bias=self.bias, the training stagnates in the second epoch itself. Why is this? Is self.bias pointing to something else?     
        y = F.conv2d(x_quant, w_quant, groups=self.groups, stride=self.stride, padding=self.padding, bias=None)
        return y
  
class QAttentionCondenser(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, QuantType='4bitsym', WScale='PerTensor', NormType='RMS', quantscale=0.25):
        super(QAttentionCondenser, self).__init__()
        self.condense = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_conv = BitConv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=1, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.pointwise_conv = BitConv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.scale = BitParameter(torch.Tensor(1))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  
        self.expand_conv = BitConv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)

    def forward(self, x):
        residual = x  
        Q = self.condense(x)  
        K = F.relu(self.group_conv(Q)) 
        K = F.relu(self.pointwise_conv(K))
        A = self.upsample(K)
        A = self.expand_conv(A)   
        S = torch.sigmoid(A)  
        V_prime = residual * S * self.scale  
        V_prime += residual  
        return V_prime

class QAttn_BN_Block(nn.Module): 
    def __init__(self, in_channels, mid_channels_0, out_channels_0, mid_channels_1, out_channels_1, QuantType='4bitsym', WScale='PerTensor', NormType='RMS', quantscale=0.25, test=0): 
        super(QAttn_BN_Block, self).__init__()
        self.layer1 = QAttentionCondenser(in_channels, mid_channels_0, out_channels_0, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.layer2 = BitBatchNorm2d(in_channels, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.layer3 = QAttentionCondenser(in_channels, mid_channels_1, out_channels_1, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.layer4 = BitBatchNorm2d(in_channels, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.test = test 

    def forward(self, x): 
        x_ = self.layer1(x)
        x_ = self.layer2(x_)
        x_ = self.layer3(x_)
        x_ = self.layer4(x_)
        x_ += x
        return x 

# ---- 
# w/o Quant
# ----

class AttentionCondenser(nn.Module):
    def __init__(self, name, in_channels, mid_channels, out_channels, test=0):
        super(AttentionCondenser, self).__init__()
        self.condense = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=1)  
        self.pointwise_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.scale = nn.Parameter(torch.Tensor(1))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  
        self.expand_conv = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.test = test
        self.name = name

    def forward(self, x):
        residual = x  
        Q = self.condense(x)  
        if self.test: log_tensor(f"{self.name}_MaxPool2d", x_)
        K = F.relu(self.group_conv(Q)) 
        if self.test: log_tensor(f"{self.name}_Conv1", x_)
        K = F.relu(self.pointwise_conv(K))
        if self.test: log_tensor(f"{self.name}_Conv2", x_)
        A = self.upsample(K)
        if self.test: log_tensor(f"{self.name}_Upsample", x_)
        A = self.expand_conv(A)   
        if self.test: log_tensor(f"{self.name}_Conv3", x_)
        S = torch.sigmoid(A)  
        if self.test: log_tensor(f"{self.name}_Sigmoid", x_)
        V_prime = residual * S * self.scale  
        V_prime += residual  
        return V_prime

class Attn_BN_Block(nn.Module): 
    def __init__(self, name, in_channels, mid_channels_0, out_channels_0, mid_channels_1, out_channels_1, test=0): 
        super(Attn_BN_Block, self).__init__()
        self.name = name
        self.layer1 = AttentionCondenser(f"{self.name}_AttnCond1", in_channels, mid_channels_0, out_channels_0)
        self.layer2 = nn.BatchNorm2d(num_features=in_channels)
        self.layer3 = AttentionCondenser(f"{self.name}_AttnCond2", in_channels, mid_channels_1, out_channels_1)
        self.layer4 = nn.BatchNorm2d(num_features=in_channels)
        self.test = test 
    
    def forward(self, x): 
        x_ = self.layer1(x)
        if self.test: log_tensor(f"{self.name}_AttnCond1", x_)
        x_ = self.layer2(x_)
        if self.test: log_tensor(f"{self.name}_BatchNorm2d", x_)
        x_ = self.layer3(x_)
        if self.test: log_tensor(f"{self.name}_AttnCond2", x_)
        x_ = self.layer4(x_)
        if self.test: log_tensor(f"{self.name}_BatchNorm2d", x_)
        x_ += x
        return x 
