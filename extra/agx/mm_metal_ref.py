#!/usr/bin/env python3
"""The same algorithm as extra/agx/mm_direct.py, written in Metal and compiled by Apple's compiler, run through the AGX dispatch so the
comparison is like for like: 128 threads = 4 SIMD groups per threadgroup, each SIMD group a 32x32 tile of C in 16 accumulator pairs,
four A and four B fragments per K step of 8 loaded straight from device memory, 4x4 outer product, no threadgroup memory.
  DEV=AGX DEBUG=2 python3 extra/agx/mm_metal_ref.py [N]"""
import os, sys
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.runtime import ops_agx, ops_metal
from tinygrad import codegen
from tinygrad.runtime.support import hcq2
from tinygrad.engine import realize
from extra.agx.probe import gpu_binary, compile_metal
N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
Device['AGX'].wait_timeout_ms = 10000.0
# ROLLED=1 keeps the 4x4 loops as loops (Apple's compiler does not unroll them and the kernel is 4x slower); the default writes
# every fragment and accumulator as its own named variable, which is the shape tinygrad's Metal renderer emits.
ROLLED = os.getenv("ROLLED") == "1"
decl = "simdgroup_float8x8 " + ", ".join([f"c{i}{j}" for i in range(4) for j in range(4)] + [f"fa{i}" for i in range(4)] + [f"fb{j}" for j in range(4)]) + ";"
init = "\n  ".join(f"c{i}{j} = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);" for i in range(4) for j in range(4))
loads = "\n    ".join([f"simdgroup_load(fa{i}, ap + {8 * i} * N + k, N);" for i in range(4)] +
                      [f"simdgroup_load(fb{j}, bp + k * N + {8 * j}, N);" for j in range(4)])
mmas = "\n    ".join(f"simdgroup_multiply_accumulate(c{i}{j}, fa{i}, fb{j}, c{i}{j});" for i in range(4) for j in range(4))
stores = "\n  ".join(f"simdgroup_store(c{i}{j}, op + {8 * i} * N + {8 * j}, N);" for i in range(4) for j in range(4))
BODY_ROLLED = """  simdgroup_float8x8 c[4][4], fa[4], fb[4];
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) c[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const device float* ap = a + g.y * 32 * N;
  const device float* bp = b + g.x * 128 + sg * 32;
  for (uint k = 0; k < N; k += 8) {
    for (int i = 0; i < 4; i++) simdgroup_load(fa[i], ap + 8 * i * N + k, N);
    for (int j = 0; j < 4; j++) simdgroup_load(fb[j], bp + k * N + 8 * j, N);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(c[i][j], fa[i], fb[j], c[i][j]);
  }
  device float* op = o + g.y * 32 * N + g.x * 128 + sg * 32;
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) simdgroup_store(c[i][j], op + 8 * i * N + 8 * j, N);"""
BODY_FLAT = f"""  {decl}
  {init}
  const device float* ap = a + g.y * 32 * N;
  const device float* bp = b + g.x * 128 + sg * 32;
  for (uint k = 0; k < N; k += 8) {{
    {loads}
    {mmas}
  }}
  device float* op = o + g.y * 32 * N + g.x * 128 + sg * 32;
  {stores}"""
SRC = f"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant uint N = {N};
kernel void k(device float* o, device float* a, device float* b, uint3 g [[threadgroup_position_in_grid]], uint l [[thread_index_in_threadgroup]]) {{
  uint sg = l >> 5;
{BODY_ROLLED if ROLLED else BODY_FLAT}
}}
"""
if os.getenv("DUMP"): open(os.getenv("DUMP"), "w").write(SRC); sys.exit(0)
apple = gpu_binary(compile_metal(SRC))
for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
A, B = np.random.default_rng(0).random((N, N), dtype=np.float32), np.random.default_rng(1).random((N, N), dtype=np.float32)
with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: "metal reference"), \
     patch.object(ops_agx.AGXCompiler, "compile", lambda self, s: apple), \
     patch.object(ops_metal.MetalQueue, "dims", staticmethod(lambda prg: (N // 128, N // 32, 1, 128, 1, 1))):
  x, y = Tensor(A.ravel()).contiguous().realize(), Tensor(B.ravel()).contiguous().realize()
  for _ in range(int(os.getenv("RUNS", "3"))): out = (x + y).realize()
  got = np.array(out.tolist(), np.float32).reshape(N, N)
print("maxerr", np.abs(got - A @ B).max())
