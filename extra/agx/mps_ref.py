#!/usr/bin/env python3
"""Apple's own tuned matmul, MPSMatrixMultiplication, driven straight through objc and timed with the command buffer's GPU
timestamps. This is the baseline that matters: not tinygrad's Metal path, but the vendor library.
  PYTHONPATH=. python3 extra/agx/mps_ref.py [N]"""
import ctypes, ctypes.util, sys, time
import numpy as np
ctypes.CDLL("/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders", mode=ctypes.RTLD_GLOBAL)
from tinygrad.runtime.autogen import metal
import tinygrad.runtime.support.objc as objc
P = ctypes.c_void_p
lib = objc.lib
def cls(n): return lib.objc_getClass(n.encode())
def msg(recv, sel, *args, restype=P, argtypes=()):
  f = ctypes.CFUNCTYPE(restype, P, P, *argtypes)(ctypes.cast(lib.objc_msgSend, P).value)
  return f(recv, objc.getsel(sel.encode()), *args)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
MPSDataTypeFloat32 = 0x10000000 | 32
dev = metal.MTLCreateSystemDefaultDevice().value
q = msg(dev, 'newCommandQueue')
nbytes = N * N * 4
A = (np.random.default_rng(0).random((N, N), dtype=np.float32))
B = (np.random.default_rng(1).random((N, N), dtype=np.float32))
def buf(arr=None):
  b = msg(dev, 'newBufferWithLength:options:', ctypes.c_ulong(nbytes), ctypes.c_ulong(0), argtypes=[ctypes.c_ulong, ctypes.c_ulong])
  if arr is not None: ctypes.memmove(msg(b, 'contents'), arr.ctypes.data, nbytes)
  return b
ba, bb, bc = buf(A), buf(B), buf()
def desc(rows, cols):
  return msg(cls('MPSMatrixDescriptor'), 'matrixDescriptorWithRows:columns:rowBytes:dataType:',
             ctypes.c_ulong(rows), ctypes.c_ulong(cols), ctypes.c_ulong(cols * 4), ctypes.c_uint(MPSDataTypeFloat32),
             argtypes=[ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_uint])
def mat(b, rows, cols):
  return msg(msg(cls('MPSMatrix'), 'alloc'), 'initWithBuffer:descriptor:', b, desc(rows, cols), argtypes=[P, P])
ma, mb, mc = mat(ba, N, N), mat(bb, N, N), mat(bc, N, N)
mm = msg(msg(cls('MPSMatrixMultiplication'), 'alloc'),
         'initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:',
         dev, ctypes.c_bool(False), ctypes.c_bool(False), ctypes.c_ulong(N), ctypes.c_ulong(N), ctypes.c_ulong(N),
         ctypes.c_double(1.0), ctypes.c_double(0.0),
         argtypes=[P, ctypes.c_bool, ctypes.c_bool, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_double, ctypes.c_double])
assert mm, "MPSMatrixMultiplication init failed"
REPS = int(__import__('os').getenv("REPS", "1"))   # multiplications per command buffer: REPS>1 keeps the GPU busy so the clock does not drop
ts = []
for i in range(12):
  cb = msg(q, 'commandBuffer')
  for _ in range(REPS):
    msg(mm, 'encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:', cb, ma, mb, mc, restype=None, argtypes=[P, P, P, P])
  msg(cb, 'commit', restype=None); msg(cb, 'waitUntilCompleted', restype=None)
  f = ctypes.CFUNCTYPE(ctypes.c_double, P, P)(ctypes.cast(lib.objc_msgSend, P).value)
  t0 = f(cb, objc.getsel(b'GPUStartTime')); t1 = f(cb, objc.getsel(b'GPUEndTime'))
  ts.append((t1 - t0) * 1e6 / REPS)
got = np.frombuffer(ctypes.string_at(msg(bc, 'contents'), nbytes), np.float32).reshape(N, N)
print(f"MPS N={N}: min {min(ts):8.1f} us  median {sorted(ts)[len(ts)//2]:8.1f} us  maxerr {np.abs(got - A @ B).max():.4g}")
