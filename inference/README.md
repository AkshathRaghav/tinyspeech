# Explaining the inference process: 

We've implemented the following schemes post-training for the inference engine: **Dynamic Quantization**, **Static Quantization**, and **Quantization-Aware Training (QAT)** during **inference**.

QAT and Static Quantization are identical for the inference stage, but different in training. You could probably notice this in the `./train.py` file. 

### **Comparison Table**

| **Aspect**                    | **Dynamic Quantization**                         | **Static Quantization**                           | **Quantization-Aware Training**                     |
|--------------------------------|------------------------------------------------|--------------------------------------------------|---------------------------------------------------|
| **Quantized During Computation** | Weights (int8), Activations (dynamically int8)  | Weights (int8), Activations (precomputed int8)    | Fake quantized weights & activations (float32)     |
| **Dequantized During Computation** | Activations/output back to float32             | None (stays in int8)                              | None during training (all float32)                 |
| **Stored Weights**             | Quantized `int8` weights                        | Quantized `int8` weights                         | Quantized `int8` weights after conversion          |
| **Stored Activations**         | None                                           | Precomputed scale & zero-point                   | Precomputed scale & zero-point after conversion    |


---

### **Dynamic Quantization**

```python
def forward(self, x):
    # x is the input activation
    # Type: float (input is in float32 at the start)

    residual = x  
    # Type: float (no quantization yet)

    Q = self.condense(x)
    # Type: float (MaxPool2d works directly on float)

    K = F.relu(self.group_conv(Q))
    # Type (Q -> float intermediate -> K): float
    # Action: 
    # 1. group_conv weights are quantized to int8.
    # 2. Multiply int8 weights with float inputs, producing float outputs.

    K = F.relu(self.pointwise_conv(K))
    # Type (K -> float intermediate -> K): float
    # Action:
    # 1. pointwise_conv weights are quantized to int8.
    # 2. Multiply int8 weights with float intermediate, output is float.

    A = self.upsample(K)
    # Type (K -> A): float
    # Action: Nearest-neighbor upsampling works on float.

    A = self.expand_conv(A)
    # Type (A -> float intermediate -> A): float
    # Action:
    # 1. expand_conv weights are quantized to int8.
    # 2. Multiply int8 weights with float inputs, output is float.

    S = torch.sigmoid(A)
    # Type (A -> S): float
    # Action: Sigmoid operates directly on float.

    V_prime = residual * S * self.scale
    # Type: float
    # Action: Element-wise multiplication and addition in float.

    return V_prime
    # Output Type: float
```

---

### **Static Quantization / Quantization-Aware Training**

```python
def forward(self, x):
    # x is the input activation
    # Type: int8 (already quantized at the start)

    residual = x  
    # Type: int8 (kept in quantized form)

    Q = self.condense(x)
    # Type: int8
    # Action: MaxPool2d works directly on int8.

    K = F.relu(self.group_conv(Q))
    # Type (Q -> float intermediate -> K): int8
    # Action:
    # Input: int8 
    # Intermediate: float (single val) 
    # Output: int8, diff shape 

    K = F.relu(self.pointwise_conv(K))
    # Type (K -> float intermediate -> K): int8
    # Action: Same as above.

    A = self.upsample(K)
    # Type (K -> A): int8
    # Action: Nearest-neighbor upsampling works on int8.

    A = self.expand_conv(A)
    # Type (A -> float intermediate -> A): int8
    # Action: Same as above.

    S = torch.sigmoid(A)
    # Type (A -> S): float
    # Action: Dequantize A to float for sigmoid operation.

    V_prime = residual * S * self.scale
    # Type: float
    # Action: Dequantize residual and perform element-wise multiplication and addition.

    return V_prime
    # Output Type: float
```

