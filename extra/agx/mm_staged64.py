#!/usr/bin/env python3
"""hand-written N x N x N fp32 matmul, 64x64 tile of C per threadgroup of 4 SIMD groups (each a 32x32 sub-tile: 16 accumulator pairs r32..r63),
A 64x16 and B 16x64 tiles staged through threadgroup memory (two float4 per thread per tile), K step 16.
  DEV=AGX DEBUG=2 python3 extra/agx/mm_staged64.py [N]"""
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
acc = lambda i, j: 32 + 2 * (4 * i + j)
# tg layout: A tile 64 rows x 16 cols at 0 (row stride 16); B tile 16 rows x 64 cols at 1024 (row stride 64)
# staging: thread l stores A float4s at 4l and 4l + 512 (rows l/4 and 32 + l/4, cols 4*(l%4)); B float4s at 1024 + 4l and 1024 + 4l + 512 (rows l/16, 8 + l/16, cols 4*(l%16))
def substep(ks):
  s = [f"tgload2 r{16 + 2*i}, r11, #{128 * i + 8 * ks}" for i in range(4)]        # A frags: rows 32*si + 8i, cols 8ks
  s += [f"tgload2 r{24 + 2*j}, r{12 if ks == 0 else 4}, #{8 * j}" for j in range(4)]   # B frags: rows 8ks (r4 = r12 + 512), cols 32*sj + 8j
  s += [f"mma r{acc(i, j)}, r{16 + 2*i}, r{24 + 2*j}, r{acc(i, j)}" for i in range(4) for j in range(4)]
  return "\n".join(s)
SRC = f"""
.buffers 3
.tgmem 8192
getsr r2, lid.x           ; l = 0..127
getsr r8, gid.y
getsr r9, gid.x
movimm r4, #0x0000001f
iand r3, r2, r4           ; lane t
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
ushr r7, r2, #6           ; si
ishl r7, r7, #5           ; 32*si
iadd r11, r7, r5
ishl r11, r11, #4
iadd r11, r11, r6         ; A fragment index: (32*si + row) * 16 + col
ushr r7, r2, #5
iand r7, r7, r4           ; sj
ishl r7, r7, #5           ; 32*sj
ishl r12, r5, #6
iadd r12, r12, r7
iadd r12, r12, r6
movimm r4, #0x00000400
iadd r12, r12, r4         ; B fragment index: 1024 + row * 64 + 32*sj + col
ishl r10, r2, #2          ; 4*l
ishl r8, r8, #6           ; 64*gy
ushr r4, r2, #2
iadd r4, r4, r8           ; 64*gy + l/4
ishl r4, r4, #{L}
movimm r13, #0x00000003
iand r13, r2, r13
ishl r13, r13, #2
iadd r8, r4, r13          ; A staging source: (64*gy + l/4) * N + 4*(l%4)   [+16k]; second float4 at +32*N
ishl r9, r9, #6           ; 64*gx
ushr r4, r2, #4
ishl r4, r4, #{L}
iadd r4, r4, r9
movimm r13, #0x0000000f
iand r13, r2, r13
ishl r13, r13, #2
iadd r9, r4, r13          ; B staging source: (l/16) * N + 64*gx + 4*(l%16)  [+16k*N]; second float4 at +8*N
movimm r13, #0x00000010
movimm r14, #0x{16 * N:08x}
movimm r4, #0x{32 * N:08x}
iadd r7, r8, r4           ; A staging source 2
movimm r4, #0x00000200
iadd r4, r12, r4          ; B fragment index for the second k sub-step (+8 rows)
movimm r6, #0x{8 * N:08x}
iadd r6, r9, r6           ; B staging source 2 (col(t) in r6 is dead: the C base recomputes it)
{chr(10).join(f"movimm r{r}, #0.0" for r in range(32, 64))}
movimm r31, #0x{N // 16:08x}
loop r15, r31
body
load4 r16, 1, r8
load4 r20, 1, r7
load4 r24, 2, r9
load4 r28, 2, r6
barrier
tgstore4 r16, r10
tgstore4 r20, r10, #512
tgstore4 r24, r10, #1024
tgstore4 r28, r10, #1536
barrier
{substep(0)}
{substep(1)}
iadd r8, r8, r13
iadd r7, r7, r13
iadd r9, r9, r14
iadd r6, r6, r14
endloop
getsr r8, gid.y
getsr r9, gid.x
ishl r8, r8, #6
ushr r4, r2, #6
ishl r4, r4, #5
iadd r8, r8, r4
iadd r8, r8, r5           ; 64*gy + 32*si + row
ishl r8, r8, #{L}
ishl r9, r9, #6
iadd r8, r8, r9
ushr r4, r2, #5
movimm r13, #0x00000001
iand r4, r4, r13
ishl r4, r4, #5
iadd r8, r8, r4           ; + 64*gx + 32*sj
movimm r4, #0x00000001
iand r6, r3, r4
ishl r6, r6, #1
ushr r7, r3, #3
iand r7, r7, r4
ishl r7, r7, #2
iadd r6, r6, r7           ; col(t)
iadd r8, r8, r6           ; + col
movimm r13, #0x{8 * N:08x}
{chr(10).join((f"store2 r{acc(i, j)}, 0, r8" + (f", #{8 * j}" if j else "")) for j in range(4) for i in [0])}
iadd r8, r8, r13
{chr(10).join((f"store2 r{acc(1, j)}, 0, r8" + (f", #{8 * j}" if j else "")) for j in range(4))}
iadd r8, r8, r13
{chr(10).join((f"store2 r{acc(2, j)}, 0, r8" + (f", #{8 * j}" if j else "")) for j in range(4))}
iadd r8, r8, r13
{chr(10).join((f"store2 r{acc(3, j)}, 0, r8" + (f", #{8 * j}" if j else "")) for j in range(4))}
"""
for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
A, B = np.random.default_rng(0).random((N, N), dtype=np.float32), np.random.default_rng(1).random((N, N), dtype=np.float32)
with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: SRC), patch.object(ops_metal.MetalQueue, "dims", staticmethod(lambda prg: (N // 64, N // 64, 1, 128, 1, 1))):
  x, y = Tensor(A.ravel()).contiguous().realize(), Tensor(B.ravel()).contiguous().realize()
  for _ in range(int(__import__("os").getenv("RUNS", "3"))): out = (x + y).realize()
  got = np.array(out.tolist(), np.float32).reshape(N, N)
print("maxerr", np.abs(got - A @ B).max())
