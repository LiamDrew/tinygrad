// float compare / select: o[t] = a < b ? c : b (diff against ../int/ilt.metal and friends)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* ba, device float* bb, device float* bc, uint t [[thread_position_in_grid]]) { float a = ba[t], b = bb[t]; float c = bc[t]; o[t] = a < b ? c : b; }
