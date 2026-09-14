// 4-wide load at an immediate offset: o[t] = sum of b[t + 1]
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float4* b, uint t [[thread_position_in_grid]]) { float4 v = b[t + 1]; o[t] = v.x + v.y + v.z + v.w; }
