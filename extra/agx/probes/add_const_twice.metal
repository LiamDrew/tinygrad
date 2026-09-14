// target: (b0+1)+2 kept separate via volatile-ish dependency: use different immediates and a second buffer read between
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, uint t [[thread_position_in_grid]]) { float s = b0[t] + 1.0f; b1[t] = s + b1[t]; }
