// 4-wide load, .x and .w used: o[t] = b[t].x + b[t].w (component mask?)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float4* b, uint t [[thread_position_in_grid]]) { float4 v = b[t]; o[t] = v.x + v.w; }
