import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F

# ---- 
# w/ Quant
# ----

class BitQuant:
    def __init__(self, QuantType='4bitsym', WScale='PerTensor', quantscale=0.25):
        self.QuantType = QuantType
        self.WScale = WScale
        self.quantscale = quantscale

    def activation_quant(self, x):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-128, 127)
        return y, scale

    def weight_quant(self, w):
        if self.WScale == 'PerOutput':
            mag = w.abs().max(dim=-1, keepdim=True)[0].clamp_(min=1e-5)
        elif self.WScale == 'PerTensor':
            mag = w.abs().mean().clamp_(min=1e-5)
        else:
            raise AssertionError(f"Invalid WScale: {self.WScale}. Expected one of: 'PerTensor', 'PerOutput'")

        if self.QuantType == '2bitsym':
            scale = 1.0 / mag
            u = ((w * scale - 0.5).round().clamp_(-2, 1) + 0.5)
            bpw = 2
        elif self.QuantType == '4bitsym':
            scale = self.quantscale * 8.0 / mag
            u = ((w * scale - 0.5).round().clamp_(-8, 7) + 0.5)
            bpw = 4
        elif self.QuantType == '8bit':
            scale = 128.0 * self.quantscale / mag
            u = (w * scale).round().clamp_(-128, 127)
            bpw = 8
        else:
            raise AssertionError(f"Invalid QuantType: {self.QuantType}. Expected one of: '2bitsym', '4bitsym', '8bit'")

        return u, scale, bpw

    def norm(self, x):
        """Using only RMS"""
        y = torch.sqrt(torch.mean(x**2, dim=(-2,-1), keepdim=True))
        return x / y

class BitParameter(nn.Module, BitQuant):
    def __init__(self, shape, QuantType='4bitsym', WScale='PerTensor', NormType='RMS', quantscale=0.25):
        super(BitParameter, self).__init__()
        self.param = nn.Parameter(torch.randn(shape))
        BitQuant.__init__(self, QuantType, WScale, quantscale)
        self.NormType = NormType

    def forward(self, x):
        x_norm = self.norm(x)
        param_norm = self.norm(self.param)
        
        if self.QuantType == 'None':
            y = x_norm * param_norm
        else:
            x_int, x_scale = self.activation_quant(x_norm)
            x_quant = x_norm + (x_int / x_scale - x_norm).detach()
            
            param_int, param_scale, _ = self.weight_quant(param_norm)
            param_quant = param_norm + (param_int / param_scale - param_norm).detach()
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
        
        if self.QuantType == 'None':
            y = F.batch_norm(x_norm, self.running_mean, self.running_var, w, b, self.training, self.momentum, self.eps)
        else:
            x_int, x_scale = self.activation_quant(x_norm)
            x_quant = x_norm + (x_int / x_scale - x_norm).detach()
            
            w_int, w_scale, _ = self.weight_quant(w)
            w_quant = w + (w_int / w_scale - w).detach()
            
            b_int, b_scale, _ = self.weight_quant(b)
            b_quant = b + (b_int / b_scale - b).detach()
            
            y = F.batch_norm(x_quant, self.running_mean, self.running_var, w_quant, b_quant, self.training, self.momentum, self.eps)
        
        return y

class BitMaxPool2d(nn.MaxPool2d, BitQuant):
    def __init__(self, kernel_size, stride=None, padding=0, QuantType='4bitsym', WScale='PerTensor', NormType='RMS', quantscale=0.25):
        nn.MaxPool2d.__init__(self, kernel_size, stride, padding)
        BitQuant.__init__(self, QuantType, WScale, quantscale)
        self.NormType = NormType

    def forward(self, x):
        x_norm = self.norm(x)
        if self.QuantType == 'None':
            y = F.max_pool2d(x_norm, self.kernel_size, self.stride, self.padding)
        else:
            x_int, x_scale = self.activation_quant(x_norm)
            x_quant = x_norm + (x_int / x_scale - x_norm).detach()
            y = F.max_pool2d(x_quant, self.kernel_size, self.stride, self.padding)
        return y


class BitConvTranspose2d(nn.ConvTranspose2d, BitQuant):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, QuantType='4bitsym', WScale='PerTensor', NormType='RMS', quantscale=0.25):
        nn.ConvTranspose2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias=False)
        BitQuant.__init__(self, QuantType, WScale, quantscale)
        self.NormType = NormType

    def forward(self, x):
        x_norm = self.norm(x)
        w = self.weight
        if self.QuantType == 'None':
            y = F.conv_transpose2d(x_norm, w, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups)
        else:
            x_int, x_scale = self.activation_quant(x_norm)
            x_quant = x_norm + (x_int / x_scale - x_norm).detach()
            w_int, w_scale, _ = self.weight_quant(w)
            w_quant = w + (w_int / w_scale - w).detach()
            y = F.conv_transpose2d(x_quant, w_quant, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups)
        return y

class QAttentionCondenser(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, QuantType='4bitsym', WScale='PerTensor', NormType='RMS', quantscale=0.25):
        super(QAttentionCondenser, self).__init__()
        self.condense = BitMaxPool2d(kernel_size=2, stride=2, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.group_conv = BitConv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=1, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.pointwise_conv = BitConv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.expand = BitConvTranspose2d(out_channels, in_channels, kernel_size=2, stride=2, padding=0, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.scale = BitParameter((1, in_channels, 1, 1), QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
    
    def forward(self, x):
        residual = x
        Q = self.condense(x)
        K = F.relu(self.group_conv(Q))
        K = F.relu(self.pointwise_conv(K))
        A = self.expand(K)
        if A.size() != x.size():
            A = F.interpolate(A, size=x.size()[2:])
        S = torch.sigmoid(A)
        V_prime = residual * S * self.scale(x)  # Apply scale parameter
        V_prime += residual
        return V_prime

class QAttnCond_Block(nn.Module): 
    def __init__(self, in_channels, mid_channels_0, out_channels_0, mid_channels_1, out_channels_1, QuantType='4bitsym', WScale='PerTensor', NormType='RMS', quantscale=0.25, test=0): 
        super(QAttnCond_Block, self).__init__()
        self.layer1 = AttentionCondenser(in_channels, mid_channels_0, out_channels_0, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.layer2 = BitBatchNorm2d(out_channels_0, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.layer3 = AttentionCondenser(out_channels_0, mid_channels_1, out_channels_1, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
        self.layer4 = BitBatchNorm2d(out_channels_1, QuantType=QuantType, WScale=WScale, NormType=NormType, quantscale=quantscale)
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
    def __init__(self, in_channels, mid_channels, out_channels):
        super(AttentionCondenser, self).__init__()
        self.condense = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=1)  
        self.pointwise_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.scale = nn.Parameter(torch.Tensor(1))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  
        self.expand_conv = nn.Conv2d(out_channels, in_channels, kernel_size=1)

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


class Attn_BN_Block(nn.Module): 
    def __init__(self, in_channels, mid_channels_0, out_channels_0, mid_channels_1, out_channels_1, test=0): 
        super(Attn_BN_Block, self).__init__()
        self.layer1 = AttentionCondenser(in_channels, mid_channels_0, out_channels_0)
        self.layer2 = nn.BatchNorm2d(num_features=in_channels)
        self.layer3 = AttentionCondenser(in_channels, mid_channels_1, out_channels_1)
        self.layer4 = nn.BatchNorm2d(num_features=in_channels)
        self.test = test 
    
    def forward(self, x): 
        x_ = self.layer1(x)
        x_ = self.layer2(x_)
        x_ = self.layer3(x_)
        x_ = self.layer4(x_)

        x_ += x

        return x 

class TinySpeechZ(nn.Module): 
    def __init__(self, num_classes, test=0): 
        super(TinySpeech, self).__init__()
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
        self.test = test 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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