#include "fixed_point.h"
#include <iostream>

int main(){
  std::cout << "sat8(999)=" << QInt<8>::sat(999) << "\n";
  std::cout << "sat8(-999)=" << QInt<8>::sat(-999) << "\n";

  int32_t x = 123456;
  int32_t mul = 1073741824; // ~2^30
  int32_t y = requantize_gemmlowp_style(x, mul, 30);
  std::cout << "requant=" << y << "\n";
  return 0;
}
