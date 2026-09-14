// target: the naive matmul shape, one thread: for i, j: s = 0; for k: s += a[4i+k]*b[4k+j]; c[4i+j] = s
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* a, device float* b, device float* c, uint t [[thread_position_in_grid]]) {
  #pragma clang loop unroll(disable)
  for (int i = 0; i < 4; i++)
  #pragma clang loop unroll(disable)
  for (int j = 0; j < 4; j++) { float s = 0.0f;
    #pragma clang loop unroll(disable)
    for (int k = 0; k < 4; k++) s += a[4*i + k] * b[4*k + j];
    c[4*i + j] = s; } }
