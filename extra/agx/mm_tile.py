#!/usr/bin/env python3
"""Parameterised tiling experiment. Same structure as extra/agx/mm_direct.py (no threadgroup memory, fragments loaded straight from
device memory, MI x MJ outer product per K step of 8) but with the tile shape as knobs, so it can be swept:

  MI  A fragments per SIMD group  -> the SIMD group owns 8*MI rows of C      (1, 2 or 4)
  MJ  B fragments per SIMD group  -> and 8*MJ columns                        (1, 2 or 4)
  SG  SIMD groups per threadgroup -> threadgroup tile is 8*MI by 8*MJ*SG     (1, 2, 4 or 8)

Registers: addresses r0..r15, fragments r16.., accumulators after them; MI*MJ accumulator pairs. Powers of two only, so every
address is a shift. mm_direct is MI=4 MJ=4 SG=4.
  MI=4 MJ=4 SG=4 DEV=AGX DEBUG=2 python3 extra/agx/mm_tile.py [N]"""
import os, sys
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.runtime import ops_agx, ops_metal
from tinygrad import codegen
from tinygrad.runtime.support import hcq2
from tinygrad.engine import realize
N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
MI, MJ, SG = (int(os.getenv(k, d)) for k, d in (("MI", "4"), ("MJ", "4"), ("SG", "4")))
for v in (MI, MJ, SG): assert v & (v - 1) == 0, "powers of two only"
assert MI <= 4, "A fragment addresses live in r5..r8"
L = N.bit_length() - 1
TROWS, TCOLS = 8 * MI, 8 * MJ * SG                      # threadgroup tile
assert N % TROWS == 0 and N % TCOLS == 0, f"N={N} does not divide into a {TROWS}x{TCOLS} tile"
Device['AGX'].wait_timeout_ms = 10000.0
AF = [16 + 2 * i for i in range(MI)]
BF = [16 + 2 * MI + 2 * j for j in range(MJ)]
ACC0 = 16 + 2 * (MI + MJ)
acc = lambda i, j: ACC0 + 2 * (MJ * i + j)
assert acc(MI - 1, MJ - 1) + 1 <= 63, f"{MI}x{MJ} needs r{acc(MI-1,MJ-1)+1}, past r63"
body = ([f"load2 r{AF[i]}, 1, r{5 + i}" for i in range(MI)] +
        [f"load2 r{BF[j]}, 2, r9" + (f", #{8 * j}" if j else "") for j in range(MJ)] +
        [f"mma r{acc(i, j)}, r{AF[i]}, r{BF[j]}, r{acc(i, j)}" for i in range(MI) for j in range(MJ)] +
        [f"iadd r{5 + i}, r{5 + i}, r11" for i in range(MI)] + ["iadd r9, r9, r10"])
stores = []
for i in range(MI):
  if i: stores.append("iadd r13, r13, r10")
  stores += [f"store2 r{acc(i, j)}, 0, r13" + (f", #{8 * j}" if j else "") for j in range(MJ)]
SRC = f"""
.buffers 3
getsr r0, lid.x
movimm r15, #0x0000001f
iand r2, r0, r15          ; lane t
ushr r13, r0, #5          ; sg
movimm r15, #0x00000003
ushr r3, r2, #1
iand r3, r3, r15
ushr r5, r2, #4
ishl r5, r5, #2
iadd r3, r3, r5           ; row(t)
movimm r15, #0x00000001
iand r4, r2, r15
ishl r4, r4, #1
ushr r5, r2, #3
iand r5, r5, r15
ishl r5, r5, #2
iadd r4, r4, r5           ; col(t)
getsr r5, gid.x
ishl r5, r5, #{TCOLS.bit_length() - 1}
ishl r13, r13, #{(8 * MJ).bit_length() - 1}
iadd r5, r5, r13
iadd r9, r5, r4           ; colbase = TCOLS*gx + 8*MJ*sg + col(t)
getsr r6, gid.y
ishl r6, r6, #{TROWS.bit_length() - 1}
iadd r6, r6, r3           ; rowbase = TROWS*gy + row(t)
ishl r5, r6, #{L}
iadd r5, r5, r4           ; A fragment 0
ishl r13, r6, #{L}
iadd r13, r13, r9         ; C address
ishl r7, r3, #{L}
iadd r9, r7, r9           ; B fragment 0
movimm r10, #0x{8 * N:08x}
movimm r11, #0x00000008
{chr(10).join(f"iadd r{6 + i}, r{5 + i}, r10" for i in range(MI - 1))}
{chr(10).join(f"movimm r{r}, #0.0" for r in range(ACC0, ACC0 + 2 * MI * MJ))}
movimm r12, #0x{N // 8:08x}
loop r14, r12
body
{chr(10).join(body)}
endloop
{chr(10).join(stores)}
"""
for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
A, B = np.random.default_rng(0).random((N, N), dtype=np.float32), np.random.default_rng(1).random((N, N), dtype=np.float32)
with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: SRC), \
     patch.object(ops_metal.MetalQueue, "dims", staticmethod(lambda prg: (N // TCOLS, N // TROWS, 1, 32 * SG, 1, 1))):
  x, y = Tensor(A.ravel()).contiguous().realize(), Tensor(B.ravel()).contiguous().realize()
  for _ in range(int(os.getenv("RUNS", "3"))): out = (x + y).realize()
  got = np.array(out.tolist(), np.float32).reshape(N, N)
print(f"MI={MI} MJ={MJ} SG={SG} tile {TROWS}x{TCOLS} groups {(N // TCOLS) * (N // TROWS)}  maxerr", np.abs(got - A @ B).max())
