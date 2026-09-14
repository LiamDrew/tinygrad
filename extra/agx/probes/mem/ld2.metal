// 2-wide load: o[t] = sum of the float2 b[t]
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float2* b, uint t [[thread_position_in_grid]]) { float2 v = b[t]; o[t] = v.x + v.y; }
