#!/usr/bin/env python3
"""hand-written N x N x N fp32 matmul, tiles staged through threadgroup memory: a threadgroup of 4 SIMD groups (128 threads) owns a 32x32 tile of C.
Per K step of 16: every thread loads one float4 of the 32x16 A tile and one of the 16x32 B tile (coalesced), stores them to threadgroup memory,
barrier, then each SIMD group reads its 16x16 sub-tile's fragments with 2-wide threadgroup loads and runs 8 mmas on four accumulators.
  DEV=AGX DEBUG=2 python3 extra/agx/mm_staged.py [N]"""
import sys
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.runtime import ops_agx, ops_metal
from tinygrad import codegen
from tinygrad.runtime.support import hcq2
from tinygrad.engine import realize
N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
L = N.bit_length() - 1
Device['AGX'].wait_timeout_ms = 10000.0
SRC = f"""
.buffers 3
.tgmem 4096
getsr r2, lid.x           ; l = 0..127
getsr r8, gid.y
getsr r9, gid.x
movimm r4, #0x0000001f
iand r3, r2, r4           ; t = lane 0..31
movimm r4, #0x00000003
ushr r5, r3, #1
iand r5, r5, r4
ushr r6, r3, #4
ishl r6, r6, #2
iadd r5, r5, r6           ; row(t)
movimm r4, #0x00000001
iand r6, r3, r4
ishl r6, r6, #1
ushr r7, r3, #3
iand r7, r7, r4
ishl r7, r7, #2
iadd r6, r6, r7           ; col(t)
ushr r7, r2, #6           ; si = l >> 6 (row block of the SIMD group)
ishl r7, r7, #4
iadd r11, r7, r5          ; 16*si + row
ishl r11, r11, #4
iadd r11, r11, r6         ; A fragment index in tg memory: (16*si + row) * 16 + col
ushr r7, r2, #5
iand r7, r7, r4           ; sj = (l >> 5) & 1
ishl r7, r7, #4           ; 16*sj
ishl r12, r5, #5
iadd r12, r12, r7
iadd r12, r12, r6
movimm r4, #0x00000200
iadd r12, r12, r4         ; B fragment index: 512 + row * 32 + 16*sj + col
ishl r10, r2, #2          ; 4*l: this thread's float4 slot in both tiles
ishl r8, r8, #5           ; 32*gy
ushr r4, r2, #2
iadd r4, r4, r8           ; 32*gy + (l >> 2)
ishl r4, r4, #{L}
movimm r13, #0x00000003
iand r13, r2, r13
ishl r13, r13, #2
iadd r8, r4, r13          ; A staging source: (32*gy + l/4) * N + 4*(l%4)   [+16k]
ishl r9, r9, #5           ; 32*gx
ushr r4, r2, #3
ishl r4, r4, #{L}
iadd r4, r4, r9
movimm r13, #0x00000007
iand r13, r2, r13
ishl r13, r13, #2
iadd r9, r4, r13          ; B staging source: (l/8) * N + 32*gx + 4*(l%8)   [+16k*N]
ushr r4, r2, #6
ishl r4, r4, #4
iadd r4, r4, r5
ishl r4, r4, #5
ushr r4, r4, #5           ; (16*si + row)  (kept in r4)
ishl r14, r2, #0
ushr r14, r14, #5
iand r14, r14, r13        ; scratch
ishl r6, r7, #0           ; 16*sj
iadd r6, r6, r5
ushr r6, r6, #0
ishl r13, r2, #0
ushr r13, r13, #6
ishl r13, r13, #4
iadd r13, r13, r5         ; 16*si + row
iadd r13, r13, r8
ishl r6, r7, #0
movimm r13, #0x00000010   ; 16
movimm r14, #0x{16 * N:08x}
{chr(10).join(f"movimm r{r}, #0.0" for r in range(32, 40))}
loop r15, #{N // 16}
body
load4 r16, 1, r8
load4 r20, 2, r9
barrier
tgstore4 r16, r10
tgstore4 r20, r10, #512
barrier
tgload2 r24, r11
tgload2 r26, r11, #128
tgload2 r28, r12
tgload2 r30, r12, #8
mma r32, r24, r28, r32
mma r34, r24, r30, r34
mma r36, r26, r28, r36
mma r38, r26, r30, r38
tgload2 r24, r11, #8
tgload2 r26, r11, #136
tgload2 r28, r12, #256
tgload2 r30, r12, #264
mma r32, r24, r28, r32
mma r34, r24, r30, r34
mma r36, r26, r28, r36
mma r38, r26, r30, r38
iadd r8, r8, r13
iadd r9, r9, r14
endloop
getsr r8, gid.y
getsr r9, gid.x
ishl r8, r8, #5
ushr r4, r2, #6
ishl r4, r4, #4
iadd r8, r8, r4
iadd r8, r8, r5           ; 32*gy + 16*si + row
ishl r8, r8, #{L}
ishl r9, r9, #5
iadd r8, r8, r9
ushr r4, r2, #5
movimm r13, #0x00000001
iand r4, r4, r13
ishl r4, r4, #4
iadd r8, r8, r4
movimm r4, #0x00000001
iand r6, r3, r4
ishl r6, r6, #1
ushr r7, r3, #3
iand r7, r7, r4
ishl r7, r7, #2
iadd r6, r6, r7           ; col(t) again
iadd r8, r8, r6           ; C lane base
store2 r32, 0, r8
store2 r34, 0, r8, #8
movimm r13, #0x{8 * N:08x}
iadd r9, r8, r13
store2 r36, 0, r9
store2 r38, 0, r9, #8
"""
for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
A, B = np.random.default_rng(0).random((N, N), dtype=np.float32), np.random.default_rng(1).random((N, N), dtype=np.float32)
with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: SRC), patch.object(ops_metal.MetalQueue, "dims", staticmethod(lambda prg: (N // 32, N // 32, 1, 128, 1, 1))):
  x, y = Tensor(A.ravel()).contiguous().realize(), Tensor(B.ravel()).contiguous().realize()
  for _ in range(int(__import__("os").getenv("RUNS", "3"))): out = (x + y).realize()
  got = np.array(out.tolist(), np.float32).reshape(N, N)
print("maxerr", np.abs(got - A @ B).max())
