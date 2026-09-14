// 4-wide threadgroup store: float4 into a threadgroup array, then scalar read back
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]]) {
  threadgroup float s[256]; *(threadgroup float4*)(s + 4 * t) = float4(b[t], b[t] + 1.0f, b[t] + 2.0f, b[t] + 3.0f); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = s[(t * 4 + 5) & 255]; }
