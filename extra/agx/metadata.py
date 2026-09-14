#!/usr/bin/env python3
"""metadata.py: Apple's __GPU_METADATA function table for a Metal kernel (register count f0 and friends), to calibrate asm.gen_metadata.
  python3 extra/agx/metadata.py extra/agx/probes/live24.metal"""
import sys, struct, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from tinygrad.runtime.ops_agx import split_archive
from extra.agx.probe import gpu_binary, compile_metal

def section(inner:bytes, name:bytes) -> bytes:
  o = 32
  for _ in range(struct.unpack('<I', inner[16:20])[0]):
    cmd, sz = struct.unpack('<II', inner[o:o + 8])
    if cmd == 0x19:
      for i in range(struct.unpack('<I', inner[o + 64:o + 68])[0]):
        sh = o + 72 + 80 * i
        if inner[sh + 16:sh + 32].rstrip(b"\0") == name:  # segment name field holds the __GPU_* name
          size, off = struct.unpack('<QI', inner[sh + 40:sh + 52]); return inner[off:off + size]
    o += sz
  raise ValueError(name)

def table(fb:bytes, pos:int) -> dict[int, int]: # flatbuffer table at pos -> {field slot: absolute offset of the field}
  vt = pos - struct.unpack_from('<i', fb, pos)[0]
  vtsize = struct.unpack_from('<H', fb, vt)[0]
  return {i: pos + off for i, off in enumerate(struct.unpack_from(f'<{(vtsize - 4) // 2}H', fb, vt + 4)) if off}
def ref(fb:bytes, at:int) -> int: return at + struct.unpack_from('<I', fb, at)[0]
def u32(fb:bytes, at:int) -> int: return struct.unpack_from('<I', fb, at)[0]

def fn_table(src:str) -> dict[int, int]:
  fb = section(split_archive(gpu_binary(compile_metal(src)))[0], b'__GPU_METADATA')
  root = table(fb, ref(fb, 0)); fn = table(fb, ref(fb, root[0]))
  return {slot: u32(fb, at) for slot, at in sorted(fn.items())}

if __name__ == "__main__":
  for f in sys.argv[1:]:
    t = fn_table(pathlib.Path(f).read_text())
    print(f"{f}: f0 (registers) = {t[0]}; raw fields {t}")
