// 4-wide load, only .x used: o[t] = b[t].x (does the compiler narrow it?)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float4* b, uint t [[thread_position_in_grid]]) { o[t] = b[t].x; }
