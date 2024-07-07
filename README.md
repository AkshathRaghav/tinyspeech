<div align="center">

# 🗣️🔥 QP-TinySpeech

🚀 Extremely Low-Bit Quantized + Pruned TinySpeech-Z for low-power MCUs 🚀

</div>

# Overview 

This repository contains (Iterative Pruning & QA-Training) scripts implementing the attention condenser block based architecture, aimed at offsetting the computational cost of self-attention networks. Moreover, it contains inference scripts designed to run on the VSDSquadron-Mini board, carrying the CH32V003 micro-controller and equipped with only 2kb of RAM and 16kb of Flash.

This project requires a ESP32 for interfacing with the audio sensor, and is responsible for sampling them into MFCC embeddings. These vectors will then be passed into the board for inference. 

> Please reach out to araviki`at`purdue`dot`edu for any questions 

# Results 

# Components Utilized 

- [VSDSquadron Mini](https://www.vlsisystemdesign.com/vsdsquadronmini/)
- [ESP32 WROOM Moule](https://www.espressif.com/en/products/socs/esp32) 
- Miscelleneous (Wires, Battery, etc.)

# Remarks

The smallest existing TinySpeech model requires about 20 kbits of memory. Pruning aims to streamline this further, reducing it to under 16 kbits to better suit the memory constraints of the VSDSquadron Mini. Training the model, with quantized weights, allows the model to learn how to perform well that precision. 




