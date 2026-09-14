// threadgroup memory, constant offset: s[t] = b[t]; barrier; o[t] = s[t + 8] (grid of 56)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]]) {
  threadgroup float s[64]; s[t] = b[t]; threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = s[t + 8]; }
