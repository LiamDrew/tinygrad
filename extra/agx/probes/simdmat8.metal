// target: register budget with the matrix unit. 8 simdgroup_float8x8 fragments live at once, each multiply-accumulated with its neighbour
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void k(device float* b, device float* o, uint t [[thread_position_in_grid]]) {
  simdgroup_float8x8 m0; simdgroup_load(m0, b + 0 * 64, 8); simdgroup_float8x8 m1; simdgroup_load(m1, b + 1 * 64, 8); simdgroup_float8x8 m2; simdgroup_load(m2, b + 2 * 64, 8); simdgroup_float8x8 m3; simdgroup_load(m3, b + 3 * 64, 8); simdgroup_float8x8 m4; simdgroup_load(m4, b + 4 * 64, 8); simdgroup_float8x8 m5; simdgroup_load(m5, b + 5 * 64, 8); simdgroup_float8x8 m6; simdgroup_load(m6, b + 6 * 64, 8); simdgroup_float8x8 m7; simdgroup_load(m7, b + 7 * 64, 8);
  simdgroup_multiply_accumulate(m0, m0, m1, m0); simdgroup_multiply_accumulate(m1, m1, m2, m1); simdgroup_multiply_accumulate(m2, m2, m3, m2); simdgroup_multiply_accumulate(m3, m3, m4, m3); simdgroup_multiply_accumulate(m4, m4, m5, m4); simdgroup_multiply_accumulate(m5, m5, m6, m5); simdgroup_multiply_accumulate(m6, m6, m7, m6); simdgroup_multiply_accumulate(m7, m7, m0, m7);
  simdgroup_store(m7, o + 7 * 64, 8); simdgroup_store(m6, o + 6 * 64, 8); simdgroup_store(m5, o + 5 * 64, 8); simdgroup_store(m4, o + 4 * 64, 8); simdgroup_store(m3, o + 3 * 64, 8); simdgroup_store(m2, o + 2 * 64, 8); simdgroup_store(m1, o + 1 * 64, 8); simdgroup_store(m0, o + 0 * 64, 8); }
