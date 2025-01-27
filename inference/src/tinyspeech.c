#include "./tensor.h"
#include "./weights.h"
#include "./modules.h"
#include "./misc.h"

// #include "../drivers/driver.h"

Tensor TinySpeechZ(Tensor* input) { 
    u_int8_t layer_id = 0; 

    Tensor x = conv2d(input, model_weights[layer_id + 1].address, model_weights[layer_id + 2].address, model_weights[layer_id + 3].address, 3, 1); layer_id += 4; 
    relu(&x);

    x = Attn_BN_Block(&x, B1_IN, B1_MC_0, B1_OC_0, B1_MC_1, B1_OC_1, &layer_id); 
    x = Attn_BN_Block(&x, B2_IN, B2_MC_0, B2_OC_0, B2_MC_1, B2_OC_1, &layer_id); 
    x = Attn_BN_Block(&x, B3_IN, B3_MC_0, B3_OC_0, B3_MC_1, B3_OC_1, &layer_id); 
    x = Attn_BN_Block(&x, B4_IN, B4_MC_0, B4_OC_0, B4_MC_1, B4_OC_1, &layer_id); 

    x = conv2d(input, model_weights[layer_id + 1].address, model_weights[layer_id + 2].address, model_weights[layer_id + 3].address, 3, 1); layer_id += 4; 
    relu(&x); 

    Tensor pooled = adaptive_avg_pool2d(&x); free_tensor(&x);
    x = fc_layer(&pooled, model_weights[layer_id++].address); free_tensor(&pooled);
    softmax(&x);

    return x; 
}

// Attn_BN_Block(&x, B5_IN, B5_MC_0, B5_OC_0, B5_MC_1, B5_OC_1, &layer_id); 
// Attn_BN_Block(&x, B6_IN, B6_MC_0, B6_OC_0, B6_MC_1, B6_OC_1, &layer_id); 

int main() { 

    #ifdef TEST_RUN
        fprintf(stdout, "Loading in TEST mode.\n");

        #ifdef QUANT_MODE_QAT_SQ
            Tensor input = f_load_tensor("../logs/ModelInput.bin", 4);
        #else 
            Tensor input = load_tensor("../logs/ModelInput.bin", 4);
        #endif

        Tensor output = TinySpeechZ(&input);
        f_print_tensor(&output);
    #endif

    return 0;
}