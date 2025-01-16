from modules import * 

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime
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

        self.block1 = Attn_BN_Block("AttnCondBlock1", in_channels=7, mid_channels_0=14, out_channels_0=3, mid_channels_1=6, out_channels_1=7, test=test)
        self.block2 = Attn_BN_Block("AttnCondBlock2", in_channels=7, mid_channels_0=14, out_channels_0=3, mid_channels_1=6, out_channels_1=7, test=test)
        self.block3 = Attn_BN_Block("AttnCondBlock3", in_channels=7, mid_channels_0=14, out_channels_0=2, mid_channels_1=4, out_channels_1=7, test=test)
        self.block4 = Attn_BN_Block("AttnCondBlock4", in_channels=7, mid_channels_0=14, out_channels_0=11, mid_channels_1=22, out_channels_1=7, test=test)
        self.block5 = Attn_BN_Block("AttnCondBlock5", in_channels=7, mid_channels_0=14, out_channels_0=14, mid_channels_1=28, out_channels_1=7, test=test)
        self.block6 = Attn_BN_Block("AttnCondBlock6", in_channels=7, mid_channels_0=14, out_channels_0=10, mid_channels_1=20, out_channels_1=7, test=test)

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
        if self.test: log_tensor("<Input to Model>", x)

        x = F.relu(self.conv1(x))  
        if self.test: log_tensor("Conv1", x)

        x = self.block1(x)
        if self.test: log_tensor("AttnCondBlock1", x)
        x = self.block2(x)
        if self.test: log_tensor("AttnCondBlock2", x)
        x = self.block3(x)
        if self.test: log_tensor("AttnCondBlock3", x)
        x = self.block4(x)
        if self.test: log_tensor("AttnCondBlock4", x)
        x = self.block5(x)
        if self.test: log_tensor("AttnCondBlock5", x)
        x = self.block6(x)
        if self.test: log_tensor("AttnCondBlock6", x)

        x = F.relu(self.conv2(x))  
        if self.test: log_tensor("Conv2", x)
        x = self.global_pool(x)  
        if self.test: log_tensor("Global_Pool", x)
        x = x.view(x.size(0), -1)  
        if self.test: log_tensor("<Tensor Shape change before FC Layer>", x)
        x = self.fc(x)  
        if self.test: log_tensor("<Output from Model>", x)

        return x


