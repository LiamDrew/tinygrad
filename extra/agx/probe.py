#!/usr/bin/env python3
"""probe.py: what does Apple's compiler emit for a small Metal kernel? Compile in-process, capture the GPU binary through a
binary archive, and list the machine code with everything our table already knows decoded and the rest highlighted.

  python3 extra/agx/probe.py extra/agx/probes/add_const.metal              # listing
  python3 extra/agx/probe.py --diff probes/fadd.metal probes/fmul.metal    # two listings side by side, differing bytes marked
  python3 extra/agx/probe.py --dump probes/fadd.metal out.bin              # raw __text
  python3 extra/agx/probe.py --hex probes/fadd.metal                       # bytes only, for grep and diff tools
"""
import sys, ctypes, tempfile, pathlib, argparse, difflib, hashlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from tinygrad.runtime.ops_metal import MetalCompiler, to_ns_str
from tinygrad.runtime.ops_agx import split_archive, text_section, _msg, _cls, P
from tinygrad.runtime.autogen import metal
import tinygrad.runtime.support.objc as objc
from tinygrad.renderer.agx import dsl

def compile_metal(src:str) -> bytes: return MetalCompiler().compile(src)

def gpu_binary(metallib:bytes) -> bytes:
  """metallib -> the fat binary archive Apple writes for its compiled pipeline (contains the applegpu slice)"""
  dev = metal.MTLCreateSystemDefaultDevice()
  lib = dev.newLibraryWithData_error(objc.dispatch_data_create(metallib, len(metallib), None, None), ctypes.byref(err:=metal.NSError()))
  assert err.value is None, "newLibraryWithData failed"
  name = _msg(_msg(lib.value, 'functionNames'), 'firstObject')
  desc = metal.MTLComputePipelineDescriptor.new()
  desc.setComputeFunction(metal.MTLFunction(_msg(lib.value, 'newFunctionWithName:', name, argtypes=[P])))
  e = P(0)
  archive = _msg(dev.value, 'newBinaryArchiveWithDescriptor:error:', _msg(_msg(_cls('MTLBinaryArchiveDescriptor'), 'alloc'), 'init'),
                 ctypes.byref(e), argtypes=[P, ctypes.POINTER(P)])
  assert archive and not e.value, "newBinaryArchive failed"
  ok = _msg(archive, 'addComputePipelineFunctionsWithDescriptor:error:', desc.value, ctypes.byref(e), restype=ctypes.c_bool, argtypes=[P, ctypes.POINTER(P)])
  assert ok, "addComputePipelineFunctions failed"
  with tempfile.TemporaryDirectory() as d:
    path = f"{d}/archive.bin"
    url = _msg(_cls('NSURL'), 'fileURLWithPath:', to_ns_str(path).value, argtypes=[P])
    ok = _msg(archive, 'serializeToURL:error:', url, ctypes.byref(e), restype=ctypes.c_bool, argtypes=[P, ctypes.POINTER(P)])
    assert ok, "serializeToURL failed"
    return pathlib.Path(path).read_bytes()

def text_of(src:str) -> bytes: return text_section(split_archive(gpu_binary(compile_metal(src)))[0])

RED, DIM, END = "\033[31m", "\033[2m", "\033[0m"
def listing(code:bytes, color=True) -> list[str]:
  out = []
  for off, bs, txt in dsl.disassemble(code):
    unknown = txt.startswith(".short")
    line = f"{off:04x}: {bs.hex(' '):<26} {txt}"
    out.append((RED if unknown else "") + line + (END if unknown else "") if color else line)
  return out

def show(path:str, color=True):
  src = pathlib.Path(path).read_text()
  code = text_of(src)
  print(f"{DIM}// {path}  ({len(code)} bytes, sha256 {hashlib.sha256(code).hexdigest()[:12]}){END}")
  print(DIM + src.strip() + END + "\n")
  print("\n".join(listing(code, color)))
  return code

def diff(a:str, b:str):
  ca, cb = text_of(pathlib.Path(a).read_text()), text_of(pathlib.Path(b).read_text())
  la, lb = listing(ca, color=False), listing(cb, color=False)
  w = max(len(l) for l in la[1:] + lb[1:]) + 2 # the 64-byte preamble line is wide; it never differs
  print(f"{a:<{w}}| {b}")
  la, lb = la[1:], lb[1:]
  for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, la, lb).get_opcodes():
    for k in range(max(i2 - i1, j2 - j1)):
      l, r = la[i1 + k] if i1 + k < i2 else "", lb[j1 + k] if j1 + k < j2 else ""
      if tag == "equal": print(f"{DIM}{l:<{w}}| {r}{END}")
      else: print(f"{RED}{l:<{w}}| {r}{END}")

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("files", nargs="+")
  ap.add_argument("--diff", action="store_true"); ap.add_argument("--dump"); ap.add_argument("--hex", action="store_true")
  args = ap.parse_args()
  if args.diff: diff(*args.files[:2])
  elif args.dump: pathlib.Path(args.dump).write_bytes(text_of(pathlib.Path(args.files[0]).read_text())); print(f"wrote {args.dump}")
  elif args.hex: print(text_of(pathlib.Path(args.files[0]).read_text()).hex(" "))
  else:
    for f in args.files: show(f); print()
