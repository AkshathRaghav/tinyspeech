#include "./tensor.h"
#include "../drivers/driver.h"

Tensor TinySpeechZ(Tensor* input) { 
    uint8_t layer_id = 0; 

    Tensor x = relu(conv2d(&input, model_weights[layer_id++], model_weights[layer_id++], model_weights[layer_id++], 3, 1)); layer_id++; 

    x = Attn_BN_Block(&x, B1_IN, B1_MC_0, B1_OC_0, B1_MC_1, B1_OC_1, &layer_id); 
    x = Attn_BN_Block(&x, B2_IN, B2_MC_0, B2_OC_0, B2_MC_1, B2_OC_1, &layer_id); 
    x = Attn_BN_Block(&x, B3_IN, B3_MC_0, B3_OC_0, B3_MC_1, B3_OC_1, &layer_id); 
    x = Attn_BN_Block(&x, B4_IN, B4_MC_0, B4_OC_0, B4_MC_1, B4_OC_1, &layer_id); 

    Tensor x = relu(conv2d(&x, model_weights[layer_id++], model_weights[layer_id++], model_weights[layer_id++], 3, 1)); layer_id++; 
    Tensor pooled = adaptive_avg_pool2d(&x); free_tensor(&x);
    x = fc_layer(&pooled, Tensor weights); free_tensor(pooled);
    softmax(&x);
    // TODO: Logic check the shape here 

    return x; 
}

// Attn_BN_Block(&x, B5_IN, B5_MC_0, B5_OC_0, B5_MC_1, B5_OC_1, &layer_id); 
// Attn_BN_Block(&x, B6_IN, B6_MC_0, B6_OC_0, B6_MC_1, B6_OC_1, &layer_id); 

int main() { 

    #ifdef TEST_RUN
        fprintf("Loading in TEST mode.\n");
        uint8_t shape[4] = { 1, 1, 12, 94 };

        #ifdef QUANT_MODE_QAT_SQ
            Tensor input = f_load_tensor("../logs/ModelInput.bin", 4);
        #else QUANT_MODE_DQ
            Tensor input = load_tensor("../logs/ModelInput.bin", 4);
        #endif

        Tensor output = TinySpeechZ(input);
        f_print_tensor(output);
    #endif

    return 0;
}