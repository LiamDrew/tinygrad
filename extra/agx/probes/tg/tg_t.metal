// threadgroup memory indexed by the 32-bit thread index (r1): s[t] = b[t]; barrier; o[t] = s[t]  (grid of at most 64 threads)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]]) {
  threadgroup float s[64]; s[t] = b[t]; threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = s[t]; }
