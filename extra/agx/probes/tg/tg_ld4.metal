// 4-wide threadgroup load: float4 array
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]]) {
  threadgroup float s[64]; s[t] = b[t]; threadgroup_barrier(mem_flags::mem_threadgroup); float4 v = *(threadgroup float4*)(s + 4 * (t & 15)); o[t] = v.x + v.y + v.z + v.w; }
