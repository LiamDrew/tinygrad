// target: store byte 6 with three stores
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, device float* b2, device float* b3, uint t [[thread_position_in_grid]]) { float x = b0[t]; b1[t] = x; b2[t] = x; b3[t] = x; }
