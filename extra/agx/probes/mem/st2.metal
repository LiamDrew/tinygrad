// 2-wide store: o[t] = float2(a, a + 1)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float2* o, device float* b, uint t [[thread_position_in_grid]]) { float a = b[t]; o[t] = float2(a, a + 1.0f); }
