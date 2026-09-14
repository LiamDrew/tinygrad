// target: immediate offset on a load. b1[t] = b0[t+1]
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, uint t [[thread_position_in_grid]]) { b1[t] = b0[t + 1]; }
