// float compare / select: o[t] = (int)(a > b) (diff against ../int/ilt.metal and friends)
#include <metal_stdlib>
using namespace metal;
kernel void k(device int* o, device float* ba, device float* bb, uint t [[thread_position_in_grid]]) { float a = ba[t], b = bb[t]; o[t] = (int)(a > b); }
