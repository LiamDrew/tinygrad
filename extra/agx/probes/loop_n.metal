// target: loop count from memory (no unrolling possible). n = c[0]
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, device int* c, uint t [[thread_position_in_grid]]) {
  float s = b0[t]; int n = c[0];
  for (int i = 0; i < n; i++) s += 1.0f;
  b1[t] = s; }
