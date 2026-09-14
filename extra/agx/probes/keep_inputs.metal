// target: fadd dst. b0 and b1 stay live after the add, so the sum cannot overwrite r0 or r1
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, device float* b2, device float* b3, device float* b4, uint t [[thread_position_in_grid]]) {
  float x = b0[t], y = b1[t]; b2[t] = x + y; b3[t] = x; b4[t] = y; }
