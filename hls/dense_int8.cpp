#include <iostream>
#include <cmath>
#include "weights.h" // Include greutatile generate de tine

// Dimensiuni fixe (bazate pe SmallCNN)
#define IMG_H 28
#define IMG_W 28
#define C1_CH 8
#define C2_CH 16
#define FC_IN 12544 // 16 * 28 * 28
#define FC_OUT 10

// Functie simpla de ReLU + Quantization clamp
// In hardware real, scale-ul ar fi tot int, dar aici folosim float pt simplitate
int8_t quant_relu(int32_t acc, float scale) {
    float val = acc * scale; 
    if (val < 0) return 0; // ReLU
    if (val > 127) return 127; // Saturare INT8
    return (int8_t)val;
}

// Functia Top-Level
void dense_int8(const float input_image[784], float output_scores[10]) {
    // Directive pentru interfata AXI (comunicare cu procesorul/memoria)
    #pragma HLS INTERFACE m_axi port=input_image depth=784
    #pragma HLS INTERFACE m_axi port=output_scores depth=10
    #pragma HLS INTERFACE s_axilite port=return

    // Buffere interne pentru feature maps
    // Folosim static pentru a nu aloca pe stack
    static int8_t layer1_out[C1_CH][IMG_H][IMG_W];
    static int8_t layer2_out[C2_CH][IMG_H][IMG_W];
    
    // 1. CONVOLUTIA 1 (1 ch -> 8 ch)
    // ------------------------------------------------
    for(int oc = 0; oc < C1_CH; oc++) {
        for(int y = 0; y < IMG_H; y++) {
            for(int x = 0; x < IMG_W; x++) {
                
                int32_t sum = conv1_b[oc]; // Initializam cu bias-ul (int32)

                // Kernel 3x3
                for(int ky = 0; ky < 3; ky++) {
                    for(int kx = 0; kx < 3; kx++) {
                        // Padding=1 implicit: coordonatele originale sunt (y-1, x-1)
                        int iy = y + ky - 1;
                        int ix = x + kx - 1;

                        int8_t pixel_val = 0;
                        // Verificam padding (boundary check)
                        if (iy >= 0 && iy < IMG_H && ix >= 0 && ix < IMG_W) {
                            // Convertim input float la int8 "on the fly" (x * 127)
                            // Sau citim direct daca input_image e deja int
                            pixel_val = (int8_t)(input_image[iy * IMG_W + ix] * 127.0f);
                        }
                        
                        // Weights sunt linearizate in weights.h: [oc][ic][ky][kx]
                        // ic=0 (doar 1 canal intrare)
                        int w_idx = oc * (1 * 3 * 3) + ky * 3 + kx;
                        sum += pixel_val * conv1_w[w_idx];
                    }
                }
                // Aplicam Scale + ReLU si salvam
                layer1_out[oc][y][x] = quant_relu(sum, scale_w1); 
            }
        }
    }

    // 2. CONVOLUTIA 2 (8 ch -> 16 ch)
    // ------------------------------------------------
    for(int oc = 0; oc < C2_CH; oc++) {
        for(int y = 0; y < IMG_H; y++) {
            for(int x = 0; x < IMG_W; x++) {
                
                int32_t sum = conv2_b[oc];

                for(int ic = 0; ic < C1_CH; ic++) {
                    for(int ky = 0; ky < 3; ky++) {
                        for(int kx = 0; kx < 3; kx++) {
                            int iy = y + ky - 1;
                            int ix = x + kx - 1;
                            
                            int8_t val_in = 0;
                            if (iy >= 0 && iy < IMG_H && ix >= 0 && ix < IMG_W) {
                                val_in = layer1_out[ic][iy][ix];
                            }

                            // Indexare weights: [oc][ic][ky][kx]
                            int w_idx = oc * (C1_CH * 9) + ic * 9 + ky * 3 + kx;
                            sum += val_in * conv2_w[w_idx];
                        }
                    }
                }
                layer2_out[oc][y][x] = quant_relu(sum, scale_w2);
            }
        }
    }

    // 3. FULLY CONNECTED (16*28*28 -> 10)
    // ------------------------------------------------
    // Atentie: Aici nu avem pooling, deci matricea e mare!
    
    for(int i = 0; i < FC_OUT; i++) {
        int32_t sum = fc_b[i];
        
        // Flattening implicit prin iterarea bufferului 3D
        int flat_idx = 0;
        for(int c = 0; c < C2_CH; c++) {
            for(int y = 0; y < IMG_H; y++) {
                for(int x = 0; x < IMG_W; x++) {
                    
                    int8_t val_in = layer2_out[c][y][x];
                    // fc_w este [10][12544]
                    int8_t w_val = fc_w[i * FC_IN + flat_idx];
                    
                    sum += val_in * w_val;
                    flat_idx++;
                }
            }
        }
        // Output final (logits) - lasam ca float pentru a citi usor in testbench
        output_scores[i] = (float)sum; 
    }
}