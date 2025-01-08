## Notes: 

> np.int8 consumes 8 bits (1 byte) per element. An input tensor of [1, 12, 94] is 1128 bytes. Weights of 2.7k parameters is 2.7 bytes.  

1. We use Quant-aware training during the Pythonic training phase. Meaning, on every pass through a layer, we take the incoming float values, quantize it and then pass it through the actual PyTorch layers. 
> Simulated quantization is applied during the forward pass, introducing quantization noise to weights and activations. The model learns to adapt to these effects. Gradients computed in float32. 

2. During inference time, we keep all the weights quantized in int8. They are directly loading when inferencing. 

3. Towards the end of attention condensers (not the blocks), we do a sigmoid, and get the results out in 'float'. Then we do the V' calculations, and quantize it again before passing it along. 

4. Idea for loading and storing weights: 
    1. We'll pre-make Tensor objects and store them in h files. 
    2. We'll name them based on the BLOCK(#)_CONDENSER(#)_CONV2D(#)_WEIGHT 
    3. Only thing we need to pass into the blocks and condensers ids.  