// one simdgroup multiply: D = A * B, A = b[0..63], B = b[64..127], D -> o[0..63]
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]]) {
  simdgroup_float8x8 a, bb, d; simdgroup_load(a, b, 8); simdgroup_load(bb, b + 64, 8); simdgroup_multiply(d, a, bb); simdgroup_store(d, o, 8); }
