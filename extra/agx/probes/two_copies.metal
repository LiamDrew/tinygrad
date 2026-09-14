// target: store src operand for loaded registers r0 and r1
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, device float* b2, device float* b3, uint t [[thread_position_in_grid]]) { b2[t] = b0[t]; b3[t] = b1[t]; }
