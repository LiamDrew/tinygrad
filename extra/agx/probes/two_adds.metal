// target: fadd destination register. two dependent adds force the first result into a named register
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, device float* b2, device float* b3, uint t [[thread_position_in_grid]]) { b3[t] = (b0[t] + b1[t]) + b2[t]; }
