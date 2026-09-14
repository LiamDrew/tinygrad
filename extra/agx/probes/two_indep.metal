// target: two independent results live at once -> both need named destinations
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, device float* b2, device float* b3, uint t [[thread_position_in_grid]]) { b2[t] = b0[t] + b1[t]; b3[t] = b0[t] * b1[t]; }
