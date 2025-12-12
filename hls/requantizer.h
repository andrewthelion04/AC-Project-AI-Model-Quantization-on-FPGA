#pragma once
#include <ap_int.h>

// y = clip( round( (x * mul) >> shift ) + add )
template<int OUT_BITS>
ap_int<OUT_BITS> requant(ap_int<32> x, ap_int<32> mul, int shift, ap_int<32> add){
#pragma HLS INLINE
  ap_int<64> prod = (ap_int<64>)x * (ap_int<64>)mul;
  ap_int<64> rnd = (shift > 0) ? (ap_int<64>)(1LL << (shift-1)) : (ap_int<64>)0;
  ap_int<64> y64 = (prod + rnd) >> shift;
  y64 += add;

  const ap_int<64> qmax = (1LL << (OUT_BITS-1)) - 1;
  const ap_int<64> qmin = -(1LL << (OUT_BITS-1));
  if (y64 > qmax) y64 = qmax;
  if (y64 < qmin) y64 = qmin;
  return (ap_int<OUT_BITS>)y64;
}
