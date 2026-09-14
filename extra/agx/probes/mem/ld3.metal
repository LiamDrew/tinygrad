// 3-wide load (packed_float3): o[t] = x + y + z
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device packed_float3* b, uint t [[thread_position_in_grid]]) { float3 v = float3(b[t]); o[t] = v.x + v.y + v.z; }
