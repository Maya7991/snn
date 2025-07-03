#include "../test_cases.h"
#include "test_data/kws_sample_input.h"
#include "test_data/kws_test_data.h"

#define IN_CHANNELS 1
#define TIMESTEPS 35
#define IMG_H 101
#define IMG_W 64

#define FILTER1 16
#define FILTER2 32
#define FILTER3 64
#define KERNEL_SIZE1 3
#define KERNEL_SIZE2 5
#define KERNEL_SIZE3 3

#define PADDING 1
#define STRIDE 2

#define FLATTEN_OUT (FILTER3*3*2)
#define FC1_NEURONS 512
#define CLASSES 11


// #define CONV_OUT (IMG_H - KERNEL_SIZE + 1)  // didnt consider padding or stride or difefrenet H and W

/**
 * This is the KWS network
*/


void kws(){

    printf("\n----- AIfES SNN KWS Model Test -----\n");

    float decay = 1.0f;
    float v_threshold = 1.0f;

    uint16_t input_shape[] = {1, TIMESTEPS, IN_CHANNELS, IMG_H, IMG_W};     //  N x T x C x H x W
    // float input_data[1*TIMESTEPS*IN_CHANNELS*IMG_H*IMG_W];              // comment this when using real input data
    aitensor_t input_tensor = AITENSOR_5D_F32(input_shape, sample_input);

    float output_data[1*TIMESTEPS*CLASSES];                                // shape: [1, 35, 11] => (NTN)
    uint16_t output_shape[] = {1, TIMESTEPS, CLASSES};
    aitensor_t output_tensor = AITENSOR_3D_F32(output_shape, output_data);

    // float bias_data[FILTER1] = {0.0f, 0.0f};
    // float lif1_vmem[FILTERS*(CONV_OUT/2)*(CONV_OUT/2)]; // 4D -> NCHW        
    // float fc1_bias[NEURONS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                                      
    // float fc1_vmem[TIMESTEPS*NEURONS];    

    // float conv1_bias[FILTER1];  // import from PyTorch
    // float conv1_weight[IN_CHANNELS*FILTER1*KERNEL_SIZE1*KERNEL_SIZE1];
    float lif1_vmem[1*FILTER1*50*31];

    // float conv2_bias[FILTER2];  // import from PyTorch
    // float conv2_weight[FILTER1*FILTER2*KERNEL_SIZE2*KERNEL_SIZE2];
    float lif2_vmem[1*FILTER2*12*7];

    // float conv3_bias[FILTER3];  // import from PyTorch
    // float conv3_weight[FILTER2*FILTER3*KERNEL_SIZE3*KERNEL_SIZE3];
    float lif3_vmem[1*FILTER3*3*2];

    // float fc1_weight[FLATTEN_OUT*FC1_NEURONS];
    // float fc1_bias[FC1_NEURONS];
    float fc1_vmem[1*TIMESTEPS*FC1_NEURONS];

    // float fc2_weight[FC1_NEURONS*CLASSES];
    // float fc2_bias[CLASSES];
    float fc2_vmem[1*TIMESTEPS*CLASSES];
// ----------------------------------------------------------------------------------------------------------------------------------

    aimodel_t model;

    // Layer definition
    uint16_t input_layer_shape[]                        = {1, TIMESTEPS, IN_CHANNELS, IMG_H, IMG_W}; // [batch-size, channels, height, width] <- channels first format
    ailayer_input_f32_t input_layer                     = AILAYER_INPUT_F32_M(/*input dimension=*/ 5, input_layer_shape);

    ailayer_snn_conv2d_f32_t snn_conv2d_layer_1         = AILAYER_SNN_CONV2D_F32_M(
                                                                /* filters =*/     FILTER1,
                                                                /* kernel_size =*/ HW(KERNEL_SIZE1, KERNEL_SIZE1),
                                                                /* stride =*/      HW(STRIDE, STRIDE),
                                                                /* dilation =*/    HW(1, 1),
                                                                /* padding =*/     HW(PADDING, PADDING),
                                                                /* weights =*/     conv1_weight,
                                                                /* bias =*/        conv1_bias
                                                            );                         
                                                                    
    ailayer_lif_f32_t lif1                              = AILAYER_LIF_F32_M(decay, v_threshold, lif1_vmem);
    // --------------------------------------------------------------------------------------------------------------------
    
    ailayer_snn_conv2d_f32_t snn_conv2d_layer_2         = AILAYER_SNN_CONV2D_F32_M(
                                                                /* filters =*/     FILTER2,
                                                                /* kernel_size =*/ HW(KERNEL_SIZE2, KERNEL_SIZE2),
                                                                /* stride =*/      HW(STRIDE, STRIDE),
                                                                /* dilation =*/    HW(1, 1),
                                                                /* padding =*/     HW(PADDING, PADDING),
                                                                /* weights =*/     conv2_weight,
                                                                /* bias =*/        conv2_bias
                                                            );                         
    ailayer_snn_maxpool2d_t snn_maxpool2d_layer_1       = AILAYER_SNN_MAXPOOL2D_F32_A(          // equivalent to PyTorch's nn.MaxPool2d(2,2)
                                                                /* pool_size =*/   HW(2, 2),
                                                                /* stride =*/      HW(2, 2),
                                                                /* padding =*/     HW(0, 0)
                                                            );                                                        
    ailayer_lif_f32_t lif2                             = AILAYER_LIF_F32_M(decay, v_threshold, lif2_vmem);
    // --------------------------------------------------------------------------------------------------------------------

    ailayer_snn_conv2d_f32_t snn_conv2d_layer_3         = AILAYER_SNN_CONV2D_F32_M(
                                                                /* filters =*/     FILTER3,
                                                                /* kernel_size =*/ HW(KERNEL_SIZE3, KERNEL_SIZE3),
                                                                /* stride =*/      HW(STRIDE, STRIDE),
                                                                /* dilation =*/    HW(1, 1),
                                                                /* padding =*/     HW(PADDING, PADDING),
                                                                /* weights =*/     conv3_weight,
                                                                /* bias =*/        conv3_bias
                                                            );                         
    ailayer_snn_maxpool2d_t snn_maxpool2d_layer_2       = AILAYER_SNN_MAXPOOL2D_F32_A(          // equivalent to PyTorch's nn.MaxPool2d(2,2)
                                                                /* pool_size =*/   HW(2, 2),
                                                                /* stride =*/      HW(2, 2),
                                                                /* padding =*/     HW(0, 0)
                                                            );                                                        
    ailayer_lif_f32_t lif3                             = AILAYER_LIF_F32_M(decay, v_threshold, lif3_vmem);

                                                                    
    // --------------------------------------------------------------------------------------------------------------------
    ailayer_snn_flatten_t flatten_layer                 = AILAYER_SNN_FLATTEN_F32_M();

    ailayer_spiking_dense_f32_t spiking_dense_layer_1   = AILAYER_SPIKING_DENSE_F32_M(   
                                                                    /*neurons=*/ FC1_NEURONS, 
                                                                    /* timesteps=*/ TIMESTEPS, 
                                                                    /*weights=*/ fc1_weight, 
                                                                    /*bias=*/ fc1_bias,
                                                                    /*v_mem=*/ fc1_vmem);
    ailayer_spiking_dense_f32_t spiking_dense_layer_2   = AILAYER_SPIKING_DENSE_F32_M(   
                                                                    /*neurons=*/ CLASSES, 
                                                                    /* timesteps=*/ TIMESTEPS, 
                                                                    /*weights=*/ fc2_weight, 
                                                                    /*bias=*/ fc2_bias,
                                                                    /*v_mem=*/ fc2_vmem);                                                                 

    void *inference_memory;
    ailayer_t *x;           // Layer pointer to perform the connection

    // The channels first related functions ("chw" or "cfirst") are used, because the input data is given as channels first format.
    model.input_layer = ailayer_input_f32_default(&input_layer);
        x = ailayer_snn_conv2d_f32_default(&snn_conv2d_layer_1, model.input_layer);
        x = ailayer_lif_f32_default(&lif1, x);
        x = ailayer_snn_conv2d_f32_default(&snn_conv2d_layer_2, x);
        x = ailayer_snn_maxpool2d_f32_default(&snn_maxpool2d_layer_1, x);
        x = ailayer_lif_f32_default(&lif2, x); //
        x = ailayer_snn_conv2d_f32_default(&snn_conv2d_layer_3, x);
        x = ailayer_snn_maxpool2d_f32_default(&snn_maxpool2d_layer_2, x);
        x = ailayer_lif_f32_default(&lif3, x);
        x = ailayer_snn_flatten_f32_default(&flatten_layer, x);
        x = ailayer_spiking_dense_f32_default(&spiking_dense_layer_1, x);
        x = ailayer_spiking_dense_f32_default(&spiking_dense_layer_2, x);       
        model.output_layer = x;

    // Finish the model creation by checking the connections and setting some parameters for further processing
    aialgo_compile_model(&model);

    printf("\n-------------- Neural network model structure ---------------\n");
    aialgo_print_model_structure(&model);
    printf("--------------------------------------------------------\n\n");

    // Allocate memory for intermediate results of the inference
    uint32_t inference_memory_size = aialgo_sizeof_inference_memory(&model);
    printf("The model needs %.4f MB of memory for inference.\n", inference_memory_size/1000000.0f);
    inference_memory = malloc(inference_memory_size);

    // Schedule the memory to the model
    aialgo_schedule_inference_memory(&model, inference_memory, inference_memory_size);

    // Run the neural network and write the results into the output_tensor
    aialgo_inference_model(&model, &input_tensor, &output_tensor);

    printf("\nPrediction shape: ");
    for(uint32_t i=0; i<(output_tensor.dim); i++){
        printf("%d,", output_tensor.shape[i]);
    }

    for(uint32_t t=0; t<TIMESTEPS; t++)
    {
        printf("\nT(%d) =>   ", t);
        for(uint32_t c=0; c<CLASSES; c++){
            // if(c!=0) printf("      ");
            printf("%.1f,  ", ((float*)output_tensor.data)[t*CLASSES + c]);
        }           
    }
    
    printf("\nFinished Inference\n\n");
    free(inference_memory);
    return;

    // ----------------------------------------------------------------------------------------------------------------------------------    
}