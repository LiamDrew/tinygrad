// reference for extra/agx/mm_staged.py: 128 threads (4 SIMD groups) per 32x32 tile of C, A 32x16 and B 16x32 tiles staged through threadgroup memory
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant uint N = 512;
kernel void k(device float* o, device float* a, device float* b, uint3 g [[threadgroup_position_in_grid]], uint l [[thread_index_in_threadgroup]]) {
  threadgroup float sa[32 * 16], sb[16 * 32];
  uint si = l >> 6, sj = (l >> 5) & 1;
  simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), c01 = c00, c10 = c00, c11 = c00, fa0, fa1, fb0, fb1;
  const device float4* pa = (const device float4*)(a + (g.y * 32 + (l >> 2)) * N + 4 * (l & 3));
  const device float4* pb = (const device float4*)(b + (l >> 3) * N + g.x * 32 + 4 * (l & 7));
  for (uint k = 0; k < N; k += 16) {
    float4 va = pa[k / 4], vb = pb[(k / 16) * 16 * N / 4];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *(threadgroup float4*)(sa + 4 * l) = va; *(threadgroup float4*)(sb + 4 * l) = vb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ks = 0; ks < 16; ks += 8) {
      simdgroup_load(fa0, sa + (16 * si) * 16 + ks, 16); simdgroup_load(fa1, sa + (16 * si + 8) * 16 + ks, 16);
      simdgroup_load(fb0, sb + ks * 32 + 16 * sj, 32); simdgroup_load(fb1, sb + ks * 32 + 16 * sj + 8, 32);
      simdgroup_multiply_accumulate(c00, fa0, fb0, c00); simdgroup_multiply_accumulate(c01, fa0, fb1, c01);
      simdgroup_multiply_accumulate(c10, fa1, fb0, c10); simdgroup_multiply_accumulate(c11, fa1, fb1, c11);
    }
  }
  device float* op = o + (g.y * 32 + 16 * si) * N + g.x * 32 + 16 * sj;
  simdgroup_store(c00, op, N); simdgroup_store(c01, op + 8, N); simdgroup_store(c10, op + 8 * N, N); simdgroup_store(c11, op + 8 * N + 8, N);
}
