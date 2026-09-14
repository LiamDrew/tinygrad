// multiply-accumulate: D = A * B + C, C = b[128..191]
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]]) {
  simdgroup_float8x8 a, bb, c, d; simdgroup_load(a, b, 8); simdgroup_load(bb, b + 64, 8); simdgroup_load(c, b + 128, 8); simdgroup_multiply_accumulate(d, a, bb, c); simdgroup_store(d, o, 8); }
