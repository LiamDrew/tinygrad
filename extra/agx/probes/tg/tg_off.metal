// threadgroup load at an immediate offset: o[t] = s[l + 4]
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]], uint l [[thread_position_in_threadgroup]]) {
  threadgroup float s[64]; s[l] = b[t]; threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = s[(l + 4) & 63]; }
