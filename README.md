<div align="center">

# 🗣️🔥 QP-TinySpeech

*Low-Bit <ins>Quantized + Pruned</ins> TinySpeech-Z for low-power MCUs*

</div>

**Update: TinySpeechZ has been quantized with symmertrical 4bit, achieving 91% accuracy. Custom kernels are being written for the condenser layers**

# Overview 

This repository implements the attention-condenser-based TinySpeech architecture, aimed at offsetting the dependence (and subsequent computational cost) of  sparse convulution layers. We also provide a training and inference engine for the Z, M and X families achieving 91%+ accuracy. Moreover, this project contains drivers to run the model on the VSDSquadron-Mini board, carrying the CH32V003 MCU and equipped with only 2kb SRAM | 16kb Flash.

Paper: https://arxiv.org/abs/2008.04245
Official Code: N/A 
    
# Results 

QP-TinySpeech deployments allows for on-device command recognition for voice-assistants on low-power devices. 

# Components Utilized 

- [VSDSquadron Mini](https://www.vlsisystemdesign.com/vsdsquadronmini/)
- [ESP32 WROOM Moule](https://www.espressif.com/en/products/socs/esp32) 
- Miscelleneous (Wires, Battery, etc.)

# Experiments 

First, install required packages using `pip install -r requirements.txt`. 

### Training 

You can train the TinySpeech family of models using: 

```
python train.py --save_pth "models" --epochs 50 --batch_size 64 --lr 0.01 --momentum 0.9 --seed 42 --device "cuda"
```

**Preferred**: Use one of the `config_*.yaml` files provided in './experiments' like follows: 

```
python train.py --config tinyspeechz_google_speech.yaml
```

# Remarks

Run `python -m torch.utils.bottleneck train.py --config <your_config_yaml>` to evaluate efficiency. 

Attention condensers are designed to replace or reduce the need for traditional convolutional layers, which are typically resource-intensive. The idea is to leverage a self-contained self-attention mechanism that can effectively capture and model both local and cross-channel activation relationships within the input data.

We're dealing with only quantization-aware training for the TinySpeech-Z and TinySpeech-M variants for now, considering their smaller size. In Quantization Aware Training (QAT), the model weights should not be fully converted to 4 bits throughout the entire training process. Instead, the weights should remain in their higher precision format (typically fp32) during training, but simulated as lower precision (e.g., 4 bits) during the forward and backward passes. This approach allows for the benefits of quantization while still leveraging the precision of higher bit-widths for weight updates.

# Acknowledgements 

- Our quantization-aware training modules were adapted from [BitNetMCU: High Accuracy Low-Bit Quantized Neural Networks on a low-end Microcontroller](https://github.com/cpldcpu/BitNetMCU). This project itself, was inspired by their early work on simple 3-layer CNN inference on a low-end MCU. 

# Citation 

If you find our work useful, please cite us. 

```
@software{araviki-2024-qp_tinyspeech, 
    title="QP-TinySpeech: Extremely Low-Bit Quantized + Pruned TinySpeech-Z for low-power MCUs", 
    author="Ravikiran, Akshath Raghav"
    year={2024}
}
```

Original Paper: 
```
@misc{wong-etal-2020-tinyspeech, 
    title="TinySpeech: Attention Condensers for Deep Speech Recognition Neural Networks on Edge Devices", 
    author="Wong, Alexander and 
            Famouri Mahmoud and 
            Pavlova Maya and 
            Surana Siddharth", 
    year={2020},
    eprint={2008.04245},
    archivePrefix={arXiv},
}
```




