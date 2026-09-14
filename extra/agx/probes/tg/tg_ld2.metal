// 2-wide threadgroup load: float2 from a threadgroup array
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]]) {
  threadgroup float s[64]; s[t] = b[t]; threadgroup_barrier(mem_flags::mem_threadgroup); float2 v = *(threadgroup float2*)(s + 2 * (t & 31)); o[t] = v.x + v.y; }
