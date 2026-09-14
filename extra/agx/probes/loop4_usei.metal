// target: a loop the compiler keeps. s = b0[t]; 4 times s += i; b1[t] = s
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, uint t [[thread_position_in_grid]]) {
  float s = b0[t];
  #pragma clang loop unroll(disable)
  for (int i = 0; i < 4; i++) s += (float)i;
  b1[t] = s; }
