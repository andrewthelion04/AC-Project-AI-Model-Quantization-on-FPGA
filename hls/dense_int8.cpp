#include <ap_int.h>
#include "requantizer.h"

extern "C" {
void dense_int8(
    const ap_int<8>* x,          // [IN]
    const ap_int<8>* w,          // [OUT x IN] flattened
    const ap_int<32>* b,         // [OUT]
    ap_int<8>* y,                // [OUT]
    int IN, int OUT,
    ap_int<32> mul, int shift, ap_int<32> add
){
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem3

#pragma HLS INTERFACE s_axilite port=IN bundle=control
#pragma HLS INTERFACE s_axilite port=OUT bundle=control
#pragma HLS INTERFACE s_axilite port=mul bundle=control
#pragma HLS INTERFACE s_axilite port=shift bundle=control
#pragma HLS INTERFACE s_axilite port=add bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  for (int o = 0; o < OUT; o++){
#pragma HLS PIPELINE II=1
    ap_int<32> acc = b[o];
    for (int i = 0; i < IN; i++){
      ap_int<16> prod = (ap_int<16>)x[i] * (ap_int<16>)w[o*IN + i];
      acc += (ap_int<32>)prod;
    }
    y[o] = requant<8>(acc, mul, shift, add);
  }
}
}
