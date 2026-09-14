// multiply through thread_elements: A, B from b, D's per-lane elements written to o[2t], o[2t+1] (no simdgroup_store)
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]]) {
  simdgroup_float8x8 a, bb, d; simdgroup_load(a, b, 8); simdgroup_load(bb, b + 64, 8); simdgroup_multiply(d, a, bb);
  o[2 * t] = d.thread_elements()[0]; o[2 * t + 1] = d.thread_elements()[1]; }
