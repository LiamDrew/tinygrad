// 4-wide store: o[t] = float4(a, a, a, a) with a = b[t]
#include <metal_stdlib>
using namespace metal;
kernel void k(device float4* o, device float* b, uint t [[thread_position_in_grid]]) { float a = b[t]; o[t] = float4(a, a + 1.0f, a + 2.0f, a + 3.0f); }
