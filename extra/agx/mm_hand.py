#!/usr/bin/env python3
"""hand-written N x N x N fp32 matmul on the simdgroup matrix unit (dsl.MMA): one SIMD group (32 lanes) per 16x16 tile of C, four accumulator
pairs, K in steps of 8 with two A and two B fragment loads per step. Runs through the AGX device with the renderer patched to return this text.
  DEV=AGX DEBUG=2 python3 extra/agx/mm_hand.py [N]      # 512: 264 us on M3, the same as Apple's compiler on probes/simd/mm_tile16.metal"""
import sys
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.runtime import ops_agx, ops_metal
from tinygrad import codegen
from tinygrad.runtime.support import hcq2
from tinygrad.engine import realize
N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
Device['AGX'].wait_timeout_ms = 10000.0
SRC = f"""
.buffers 3
getsr r2, lid.x
getsr r8, gid.y
getsr r10, gid.x
movimm r4, #0x00000003
movimm r12, #0x{8 * N:08x}
movimm r13, #0x00000008
ushr r3, r2, #1
iand r3, r3, r4
ushr r5, r2, #4
ishl r5, r5, #2
iadd r3, r3, r5          ; row(t) = ((t>>1)&3) + 4*(t>>4)
movimm r4, #0x00000001
iand r6, r2, r4
ishl r6, r6, #1
ushr r7, r2, #3
iand r7, r7, r4
ishl r7, r7, #2
iadd r6, r6, r7          ; col(t) = 2*(t&1) + 4*((t>>3)&1)
ishl r8, r8, #4
iadd r8, r8, r3
ishl r8, r8, #{N.bit_length() - 1}
iadd r8, r8, r6          ; A lane base: (16*gy + row) * N + col
ishl r9, r3, #{N.bit_length() - 1}
ishl r10, r10, #4
iadd r9, r9, r10
iadd r9, r9, r6          ; B lane base: row * N + 16*gx + col
iadd r11, r8, r10        ; C lane base: (16*gy + row) * N + 16*gx + col
movimm r32, #0.0
movimm r33, #0.0
movimm r34, #0.0
movimm r35, #0.0
movimm r36, #0.0
movimm r37, #0.0
movimm r38, #0.0
movimm r39, #0.0
loop r14, #{N // 8}
body
load2 r40, 1, r8
load2 r42, 1, r8, #{8 * N}
load2 r44, 2, r9
load2 r46, 2, r9, #8
mma r32, r40, r44, r32
mma r34, r40, r46, r34
mma r36, r42, r44, r36
mma r38, r42, r46, r38
iadd r8, r8, r13         ; A: next 8 columns
iadd r9, r9, r12         ; B: next 8 rows
endloop
store2 r32, 0, r11
store2 r34, 0, r11, #8
iadd r15, r11, r12
store2 r36, 0, r15
store2 r38, 0, r15, #8
"""
for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
A, B = np.random.default_rng(0).random((N, N), dtype=np.float32), np.random.default_rng(1).random((N, N), dtype=np.float32)
with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: SRC), patch.object(ops_metal.MetalQueue, "dims", staticmethod(lambda prg: (N // 16, N // 16, 1, 32, 1, 1))):
  x, y = Tensor(A.ravel()).contiguous().realize(), Tensor(B.ravel()).contiguous().realize()
  for _ in range(3): out = (x + y).realize()
  got = np.array(out.tolist(), np.float32).reshape(N, N)
print("maxerr", np.abs(got - A @ B).max(), " rows 0..1 cols 0..3:", got[:2, :4], (A @ B)[:2, :4])
