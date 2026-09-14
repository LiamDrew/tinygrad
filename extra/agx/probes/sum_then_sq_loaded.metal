// two ops, second reads a loaded value again. s = b0+1; b1 = s*b0
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, uint t [[thread_position_in_grid]]) { float s = b0[t] + 1.0f; b1[t] = s * b0[t]; }
