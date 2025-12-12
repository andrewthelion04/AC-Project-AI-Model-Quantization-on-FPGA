#pragma once
#include <cstdint>

template<int BITS>
struct QInt {
  int32_t v;
  static constexpr int32_t QMAX = (1 << (BITS-1)) - 1;
  static constexpr int32_t QMIN = -(1 << (BITS-1));

  static int32_t sat(int64_t x) {
    if (x > QMAX) return QMAX;
    if (x < QMIN) return QMIN;
    return (int32_t)x;
  }
};

// Gemmlowp-style: y ≈ (x * mul) >> shift with rounding.
inline int32_t requantize_gemmlowp_style(int32_t x, int32_t mul, int shift) {
  int64_t prod = (int64_t)x * (int64_t)mul;
  int64_t rnd = (shift > 0) ? (int64_t(1) << (shift-1)) : 0;
  int64_t y = (prod + rnd) >> shift;
  return (int32_t)y;
}
