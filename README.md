<div align="center">

# 🗣️🔥 QP-TinySpeech

*Extremely Low-Bit <ins>Quantized + Pruned</ins> TinySpeech-Z for low-power MCUs*

</div>

# Overview 

This repository contains Iterative Pruning & QA-Training scripts implementing the attention-condenser-based TinySpeech architecture, aimed at offsetting the computational cost of using sparse convulution modules. Moreover, it contains inference scripts designed to run on the VSDSquadron-Mini board, carrying the CH32V003 microcontroller and equipped with only 2kb of RAM and 16kb of Flash.

QP-TinySpeech deployments allow for on-device command recognition for voice-assistants on low-power devices. 

> Please reach out to araviki`at`purdue`dot`edu for any questions 

# Results 

# Components Utilized 

- [VSDSquadron Mini](https://www.vlsisystemdesign.com/vsdsquadronmini/)
- [ESP32 WROOM Moule](https://www.espressif.com/en/products/socs/esp32) 
- Miscelleneous (Wires, Battery, etc.)

# Remarks

The smallest existing TinySpeech model requires about 20 kbits of memory. Pruning aims to streamline this further, reducing it to under 16 kbits to better suit the memory constraints of the VSDSquadron Mini. Training the model, with quantized weights, allows the model to learn how to perform well that precision. 




