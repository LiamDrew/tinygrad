#!/usr/bin/env python3
"""mm_direct with the memory stall pipelined away, using the far register bank for accumulators.

The 16 accumulator pairs move to r64..r95 (hardware-verified usable; r112 and above hang), which frees r16..r47 for TWO sets of
fragment registers. That allows genuine double buffering: while the 16 mmas of one K step run, the next step's eight loads are
already in flight on the other scoreboard slot, writing the other register set, so the write-after-read hazard cannot bite.
  DEV=AGX DEBUG=2 python3 extra/agx/mm_pipelined.py [N]"""
import os, sys
os.environ.setdefault("AGX_NREGS", "96")          # cover r95; Apple's largest observed declaration is 96
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
acc = lambda i, j: 64 + 2 * (4 * i + j)           # 16 accumulator pairs in the far bank
SET = [{"A": [16, 18, 20, 22], "B": [24, 26, 28, 30]}, {"A": [32, 34, 36, 38], "B": [40, 42, 44, 46]}]
def loads(which, slot, off):                       # eight loads of one K step into register set `which`, on scoreboard slot `slot`
  s = SET[which]; tag = f", s{slot}"
  return ([f"load2 r{s['A'][i]}, 1, r{5 + i}" + (f", #{off}" if off else "") + tag for i in range(4)] +
          [f"load2 r{s['B'][j]}, 2, r{9 if off == 0 else 15}" + (f", #{8 * j}" if j else "") + tag for j in range(4)])
def mmas(which):
  s = SET[which]
  return [f"mma r{acc(i, j)}, r{s['A'][i]}, r{s['B'][j]}, r{acc(i, j)}" for i in range(4) for j in range(4)]
body = (loads(1, 1, 8) + mmas(0) +                                  # prefetch the odd K step, compute the even one
        [f"iadd r{5 + i}, r{5 + i}, r11" for i in range(4)] + ["iadd r9, r9, r10", "iadd r15, r15, r10"] +
        loads(0, 0, 0) + mmas(1))                                    # prefetch the next even step, compute the odd one
stores = []
for i in range(4):
  if i: stores.append("iadd r13, r13, r14")
  stores += [f"store2 r{acc(i, j)}, 0, r13" + (f", #{8 * j}" if j else "") for j in range(4)]
SRC = f"""
.buffers 3
getsr r0, lid.x
movimm r15, #0x0000001f
iand r2, r0, r15
ushr r13, r0, #5
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
ishl r5, r5, #7
ishl r13, r13, #5
iadd r5, r5, r13
iadd r9, r5, r4           ; colbase
getsr r6, gid.y
ishl r6, r6, #5
iadd r6, r6, r3           ; rowbase
ishl r5, r6, #{L}
iadd r5, r5, r4           ; A fragment 0
ishl r13, r6, #{L}
iadd r13, r13, r9         ; C address
ishl r7, r3, #{L}
iadd r9, r7, r9           ; B fragment 0
movimm r0, #0x{8 * N:08x}
iadd r6, r5, r0
iadd r7, r6, r0
iadd r8, r7, r0           ; A fragments 1..3
iadd r15, r9, r0          ; B address for the odd K sub-step
movimm r10, #0x{16 * N:08x}   ; B advance per iteration (two K steps)
movimm r11, #0x00000010       ; A advance per iteration
{chr(10).join(f"movimm r{r}, #0.0" for r in range(48, 64))}
{chr(10).join(f"ishl r{acc(i, j) + k}, r{48 + (4 * i + j) % 16}, #0" for i in range(4) for j in range(4) for k in (0, 1))}
{chr(10).join(f"load2 r{SET[0]['A'][i]}, 1, r{5 + i}, s0" for i in range(4))}
{chr(10).join(f"load2 r{SET[0]['B'][j]}, 2, r9" + (f", #{8 * j}" if j else "") + ", s0" for j in range(4))}
movimm r12, #0x{N // 16:08x}
loop r14, r12
body
{chr(10).join(body)}
endloop
movimm r14, #0x{8 * N:08x}
{chr(10).join(stores)}
"""
for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
A, B = np.random.default_rng(0).random((N, N), dtype=np.float32), np.random.default_rng(1).random((N, N), dtype=np.float32)
pad = np.zeros(64, np.float32)                     # the pipeline prefetches one K step past the end; keep those reads in bounds
with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: SRC), \
     patch.object(ops_metal.MetalQueue, "dims", staticmethod(lambda prg: (N // 128, N // 32, 1, 128, 1, 1))):
  x, y = Tensor(np.concatenate([A.ravel(), pad])).contiguous().realize(), Tensor(np.concatenate([B.ravel(), pad])).contiguous().realize()
  for _ in range(int(os.getenv("RUNS", "3"))): out = (x + y).realize()
  got = np.array(out.tolist(), np.float32)[:N * N].reshape(N, N)
print("maxerr", np.abs(got - A @ B).max())
