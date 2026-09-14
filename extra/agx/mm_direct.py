#!/usr/bin/env python3
"""hand-written N x N x N fp32 matmul in the shape Apple's own fast kernel uses (scratch mm_metal_1024.dis): NO threadgroup memory and no
barriers. 128 threads = 4 SIMD groups per threadgroup; each SIMD group owns a 32x32 tile of C in 16 accumulator pairs (r32..r63) and per K step
of 8 loads four A and four B 8x8 fragments straight from device memory with 2-wide loads, then does a 4x4 outer product (8 loads -> 16 mmas).
Reuse comes from the cache, not from staging.
  DEV=AGX DEBUG=2 python3 extra/agx/mm_direct.py [N]"""
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
AFRAG = [16, 18, 20, 22]
BFRAG = [24, 26, 28, 30]
import os
# SLOTS=1: put half the fragments on scoreboard slot 1 (dsl.SCOREBOARD) and order the mmas so the four that need only the slot-0
# fragments run while the slot-1 loads are still in flight. Each mma waits on whichever slots are still pending.
SLOTS = os.getenv("SLOTS") == "1"
sb = lambda i: ", s1" if SLOTS and i >= 2 else ""
pairs = ([(i, j) for i in (0, 1) for j in (0, 1)] + [(i, j) for i in range(4) for j in range(4) if i >= 2 or j >= 2]) if SLOTS else \
        [(i, j) for i in range(4) for j in range(4)]
body = "\n".join([f"load2 r{AFRAG[i]}, 1, r{5 + i}{sb(i)}" for i in range(4)] +
                 [f"load2 r{BFRAG[j]}, 2, r9" + (f", #{8 * j}" if j else "") + sb(j) for j in range(4)] +
                 [f"mma r{acc(i, j)}, r{AFRAG[i]}, r{BFRAG[j]}, r{acc(i, j)}" for i, j in pairs] +
                 [f"iadd r{5 + i}, r{5 + i}, r11" for i in range(4)] + ["iadd r9, r9, r10"])
stores = []
for i in range(4):
  if i: stores.append("iadd r13, r13, r10")
  stores += [f"store2 r{acc(i, j)}, 0, r13" + (f", #{8 * j}" if j else "") for j in range(4)]
SRC = f"""
.buffers 3
getsr r0, lid.x
movimm r15, #0x0000001f
iand r2, r0, r15          ; lane t = l & 31
ushr r13, r0, #5          ; sg = l >> 5 (which 32-column strip of the 128)
movimm r15, #0x00000003
ushr r3, r2, #1
iand r3, r3, r15
ushr r5, r2, #4
ishl r5, r5, #2
iadd r3, r3, r5           ; row(t) = ((t>>1)&3) + 4*(t>>4)
movimm r15, #0x00000001
iand r4, r2, r15
ishl r4, r4, #1
ushr r5, r2, #3
iand r5, r5, r15
ishl r5, r5, #2
iadd r4, r4, r5           ; col(t) = 2*(t&1) + 4*((t>>3)&1)
getsr r5, gid.x
ishl r5, r5, #7           ; 128*gx
ishl r13, r13, #5         ; 32*sg
iadd r5, r5, r13
iadd r9, r5, r4           ; colbase = 128*gx + 32*sg + col(t)
getsr r6, gid.y
ishl r6, r6, #5
iadd r6, r6, r3           ; rowbase = 32*gy + row(t)
ishl r5, r6, #{L}
iadd r5, r5, r4           ; A fragment 0 address = rowbase * N + col(t)
ishl r13, r6, #{L}
iadd r13, r13, r9         ; C address = rowbase * N + colbase
ishl r7, r3, #{L}
iadd r9, r7, r9           ; B fragment 0 address = row(t) * N + colbase
movimm r10, #0x{8 * N:08x}
movimm r11, #0x00000008
iadd r6, r5, r10          ; A fragments 1..3: rows +8, +16, +24
iadd r7, r6, r10
iadd r8, r7, r10
{chr(10).join(f"movimm r{r}, #0.0" for r in range(32, 64))}
movimm r12, #0x{N // 8:08x}
loop r14, r12
body
{body}
endloop
{chr(10).join(stores)}
"""
for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
A, B = np.random.default_rng(0).random((N, N), dtype=np.float32), np.random.default_rng(1).random((N, N), dtype=np.float32)
with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: SRC), \
     patch.object(ops_metal.MetalQueue, "dims", staticmethod(lambda prg: (N // 128, N // 32, 1, 128, 1, 1))):
  x, y = Tensor(A.ravel()).contiguous().realize(), Tensor(B.ravel()).contiguous().realize()
  for _ in range(int(__import__("os").getenv("RUNS", "3"))): out = (x + y).realize()
  got = np.array(out.tolist(), np.float32).reshape(N, N)
print("maxerr", np.abs(got - A @ B).max())
