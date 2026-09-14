// target: a store that executes on every iteration. b1[4t+i] = b0[t] + i
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, uint t [[thread_position_in_grid]]) {
  float x = b0[t];
  #pragma clang loop unroll(disable)
  for (int i = 0; i < 4; i++) b1[4*t + i] = x + (float)i; }
