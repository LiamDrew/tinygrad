// 16x16 product as the API allows it: four 8x8 output fragments, each two multiply-accumulates (how many cf instructions?)
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]]) {
  simdgroup_float8x8 a[2][2], bb[2][2], d[2][2];
  for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) { simdgroup_load(a[i][j], b + i * 128 + j * 8, 16); simdgroup_load(bb[i][j], b + 256 + i * 128 + j * 8, 16); d[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); }
  for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) for (int kk = 0; kk < 2; kk++) simdgroup_multiply_accumulate(d[i][j], a[i][kk], bb[kk][j], d[i][j]);
  for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) simdgroup_store(d[i][j], o + i * 128 + j * 8, 16); }
