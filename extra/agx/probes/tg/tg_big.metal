// larger threadgroup array (metadata allocation size): s[1024]
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]], uint l [[thread_position_in_threadgroup]]) {
  threadgroup float s[1024]; s[l] = b[t]; threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = s[l ^ 1]; }
