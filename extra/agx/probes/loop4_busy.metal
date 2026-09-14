// target: counter register when r2..r5 hold live values. s = b0[t]; x,y,z live across the loop
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, device float* b2, device float* b3, uint t [[thread_position_in_grid]]) {
  float s = b0[t], x = b1[t], y = b2[t], z = b3[t];
  #pragma clang loop unroll(disable)
  for (int i = 0; i < 4; i++) s += 1.0f;
  b1[t] = s; b2[t] = x + y; b3[t] = z; }
