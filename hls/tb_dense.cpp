#include <iostream>
#include <vector>
#include <ap_int.h>

// prototype
extern "C" void dense_int8(
    const ap_int<8>* x,
    const ap_int<8>* w,
    const ap_int<32>* b,
    ap_int<8>* y,
    int IN, int OUT,
    ap_int<32> mul, int shift, ap_int<32> add
);

int main(){
  const int IN = 8;
  const int OUT = 4;
  std::vector<ap_int<8>> x(IN);
  std::vector<ap_int<8>> w(OUT*IN);
  std::vector<ap_int<32>> b(OUT);
  std::vector<ap_int<8>> y(OUT);

  for(int i=0;i<IN;i++) x[i] = i-4;
  for(int o=0;o<OUT;o++){
    b[o] = 0;
    for(int i=0;i<IN;i++) w[o*IN+i] = (i%3)-1;
  }

  ap_int<32> mul = 1<<10; // fake scale
  int shift = 10;
  ap_int<32> add = 0;

  dense_int8(x.data(), w.data(), b.data(), y.data(), IN, OUT, mul, shift, add);

  for(int o=0;o<OUT;o++){
    std::cout << "y["<<o<<"]=" << (int)y[o] << "\n";
  }
  return 0;
}
