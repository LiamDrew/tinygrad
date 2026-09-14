// threadgroup store then load without a barrier (isolates the barrier bytes)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]], uint l [[thread_position_in_threadgroup]]) {
  threadgroup float s[64]; s[l] = b[t]; o[t] = s[l ^ 1]; }
