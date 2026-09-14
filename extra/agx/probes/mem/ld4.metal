// 4-wide load: o[t] = sum of the float4 b[t] (diff against ld2.metal / ../copy.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float4* b, uint t [[thread_position_in_grid]]) { float4 v = b[t]; o[t] = v.x + v.y + v.z + v.w; }
