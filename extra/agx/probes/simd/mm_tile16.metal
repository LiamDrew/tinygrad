// reference for the hand-written kernel: one SIMD group per 16x16 tile of C = A * B (N x N, row-major), K in steps of 8, four accumulators
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant uint N = 512;
kernel void k(device float* o, device float* a, device float* b, uint3 g [[threadgroup_position_in_grid]]) {
  simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), c01 = c00, c10 = c00, c11 = c00, fa0, fa1, fb0, fb1;
  const device float* ap = a + g.y * 16 * N; const device float* bp = b + g.x * 16;
  for (uint k = 0; k < N; k += 8) {
    simdgroup_load(fa0, ap + k, N); simdgroup_load(fa1, ap + 8 * N + k, N);
    simdgroup_load(fb0, bp + k * N, N); simdgroup_load(fb1, bp + k * N + 8, N);
    simdgroup_multiply_accumulate(c00, fa0, fb0, c00); simdgroup_multiply_accumulate(c01, fa0, fb1, c01);
    simdgroup_multiply_accumulate(c10, fa1, fb0, c10); simdgroup_multiply_accumulate(c11, fa1, fb1, c11);
  }
  device float* op = o + g.y * 16 * N + g.x * 16;
  simdgroup_store(c00, op, N); simdgroup_store(c01, op + 8, N); simdgroup_store(c10, op + 8 * N, N); simdgroup_store(c11, op + 8 * N + 8, N);
}
