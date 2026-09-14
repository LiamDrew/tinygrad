// target: two results live at once: (b0+b1) and (b2+b3), then their sum
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, device float* b2, device float* b3, device float* b4, uint t [[thread_position_in_grid]]) {
  b4[t] = (b0[t] + b1[t]) + (b2[t] + b3[t]); }
