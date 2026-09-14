// fragment layout: load the 8x8 matrix b[0..63] (row-major, stride 8) and write each lane's two elements: o[2t] = e0, o[2t+1] = e1 (32 threads)
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]]) {
  simdgroup_float8x8 m; simdgroup_load(m, b, 8); float2 e = float2(m.thread_elements()[0], m.thread_elements()[1]); o[2 * t] = e.x; o[2 * t + 1] = e.y; }
