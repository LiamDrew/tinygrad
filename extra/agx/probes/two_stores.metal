// target: store byte 6 (0x20 vs 0x21). one loaded value stored to two buffers
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, device float* b2, uint t [[thread_position_in_grid]]) { float x = b0[t]; b1[t] = x; b2[t] = x; }
