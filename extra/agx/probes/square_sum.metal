// target: one result used twice: s = b0+b1; b2 = s*s
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, device float* b2, uint t [[thread_position_in_grid]]) { float s = b0[t] + b1[t]; b2[t] = s * s; }
