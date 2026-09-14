// target: fmul. same shape as sum3 with * instead of +
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, device float* b2, uint t [[thread_position_in_grid]]) { b2[t] = b0[t] * b1[t]; }
