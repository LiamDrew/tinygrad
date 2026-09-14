# AGX: hand-assembled Apple GPU machine code on the HCQ2 Metal device. Metal only mints the pipeline state; the GPU runs our bytes.
from __future__ import annotations
import ctypes, struct, functools
from tinygrad.helpers import Target, DEBUG
from tinygrad.device import Compiler
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.runtime.autogen import metal
import tinygrad.runtime.support.objc as objc
from tinygrad.runtime.ops_metal import MetalDevice, MetalQueue, checked, to_ns_str
from tinygrad.runtime.support.hcq2 import encode_submit
from tinygrad.runtime.support.agx import asm
from tinygrad.renderer.agx import dsl

# step 1 stub: every kernel renders to the same three-buffer add, data0[t] = data1[t] + data2[t]
ADD_ASM = """
.buffers 3
load r0, 1
wait
load r1, 2
wait
fadd r2, r0, r1
store r2, 0
"""

class AGXCompiler(Compiler):
  def __init__(self): super().__init__("compile_agx")
  def compile(self, src:str) -> bytes: return asm.build(src)[0]
  def compile_cached(self, src:str) -> bytes:
    lib = super().compile_cached(src)
    if DEBUG >= 4: self.disassemble(lib) # hcq2's DEBUG>=5 printers cannot render a submit yet, so show the bytes here, cache hit or not
    return lib
  def disassemble(self, lib:bytes): print(dsl.disasm_str(text_section(split_archive(lib)[0])))

class AGXRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  global_max, local_max = None, None
  compiler = AGXCompiler()
  def render(self, uops:list[UOp]) -> str:
    params = [u for u in uops if u.op is Ops.PARAM]
    assert len(params) == 3, f"the stub add kernel needs exactly 3 buffers, got {len(params)}"
    return ADD_ASM

# *****************
# in-memory binary archive: Metal's archive loading is path-only, so we hook the lookup and insert our compute object under whatever key it asks

P, UL = ctypes.c_void_p, ctypes.c_ulong
_lib = objc.lib
for f, r, a in (('class_getInstanceMethod', P, [P, P]), ('method_setImplementation', P, [P, P]), ('method_getImplementation', P, [P]),
                ('class_getInstanceVariable', P, [P, ctypes.c_char_p]), ('ivar_getOffset', ctypes.c_long, [P])):
  getattr(_lib, f).restype, getattr(_lib, f).argtypes = r, a
def _cls(name:str): return _lib.objc_getClass(name.encode())
def _msg(receiver, sel:str, *args, restype=P, argtypes=()):
  return ctypes.CFUNCTYPE(restype, P, P, *argtypes)(ctypes.cast(_lib.objc_msgSend, P).value)(receiver, objc.getsel(sel.encode()), *args)

_keep:list = [] # ctypes objects that must outlive the Metal objects using them
def _dispatch_data(b:bytes):
  _keep.append(buf:=ctypes.create_string_buffer(b, len(b)))
  return objc.dispatch_data_create(buf, len(b), None, None)

_pending:dict[int, bytes] = {} # archive -> compute object awaiting insertion
_hooked = False
def _hook():
  global _hooked
  if _hooked: return
  iv = _lib.class_getInstanceVariable(_cls('MTLBinaryEntry'), b'_reflectionFlags')
  m = _lib.class_getInstanceMethod(_cls('_MTLBinaryArchive'), objc.getsel(b'getBinaryDataForKey:reflectionType:'))
  assert iv and m and _lib.class_getInstanceMethod(_cls('_MTLBinaryArchive'), objc.getsel(b'addBinaryEntryInternal:forKey:')), "Metal internals changed"
  flags_off = _lib.ivar_getOffset(iv)
  LOOKUP = ctypes.CFUNCTYPE(P, P, P, P, ctypes.c_char)
  orig = LOOKUP(_lib.method_getImplementation(m))
  def lookup(archive, sel, key, rtype):
    if (inner:=_pending.pop(archive, None)) is not None:
      entry = _msg(_msg(_cls('MTLBinaryEntry'), 'alloc'), 'initWithData:reflectionBlock:binaryPosition:',
                   _dispatch_data(inner), _dispatch_data(bytes(16)), UL(0xffffffffffffffff), argtypes=[P, P, UL])
      ctypes.c_int.from_address(entry + flags_off).value = 2
      _msg(archive, 'addBinaryEntryInternal:forKey:', entry, key, restype=None, argtypes=[P, P])
    return orig(archive, sel, key, rtype)
  _keep.append(hooked:=LOOKUP(lookup))
  _lib.method_setImplementation(m, ctypes.cast(hooked, P))
  _hooked = True

def split_archive(archive:bytes) -> tuple[bytes, bytes]:
  """fat archive -> (compute object = slice0's __compute section, metallib slice)"""
  nfat = struct.unpack('>I', archive[4:8])[0]
  slices = {s[0]: (s[2], s[3]) for s in (struct.unpack('>iiIII', archive[8 + 20 * i:28 + 20 * i]) for i in range(nfat))}
  (s0off, s0sz), (mloff, mlsz) = slices[0x1000013], slices[0x1000017]
  s0, o = archive[s0off:s0off + s0sz], 32
  for _ in range(struct.unpack('<I', s0[16:20])[0]):
    cmd, sz = struct.unpack('<II', s0[o:o + 8])
    if cmd == 0x19 and s0[o + 8:o + 14] == b'__TEXT':
      for i in range(struct.unpack('<I', s0[o + 64:o + 68])[0]):
        sh = o + 72 + 80 * i
        if s0[sh:sh + 9] == b'__compute':
          size, off = struct.unpack('<QI', s0[sh + 40:sh + 52])
          return s0[off:off + size], archive[mloff:mloff + mlsz]
    o += sz
  raise ValueError('no __compute section in slice0')

def text_section(inner:bytes) -> bytes:
  """__text of the compute object (an object Mach-O): the machine code"""
  o = 32
  for _ in range(struct.unpack('<I', inner[16:20])[0]):
    cmd, sz = struct.unpack('<II', inner[o:o + 8])
    if cmd == 0x19:
      for i in range(struct.unpack('<I', inner[o + 64:o + 68])[0]):
        sh = o + 72 + 80 * i
        if inner[sh:sh + 6] == b'__text':
          size, off = struct.unpack('<QI', inner[sh + 40:sh + 52])
          return inner[off:off + size]
    o += sz
  raise ValueError('no __text section')

# *****************
# device

class AGXDevice(MetalDevice):
  pm_encode = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_agx_compute", name="submit"), lambda ctx, submit: encode_submit(MetalQueue(ctx, submit))),
  ])

  def __init__(self, device:str=""):
    super().__init__(device)
    self.renderers = [AGXRenderer]
    _hook()

  @functools.cache
  def pipeline(self, lib:bytes, name:str) -> metal.MTLComputePipelineState:
    inner, ml = split_archive(lib)
    dev = self.sysdevice.value
    archive = checked(lambda d, e: _msg(dev, 'newBinaryArchiveWithDescriptor:error:', d, e, argtypes=[P, P]),
                      _msg(_msg(_cls('MTLBinaryArchiveDescriptor'), 'alloc'), 'init'))
    _pending[archive] = inner
    library = checked(self.sysdevice.newLibraryWithData_error, objc.dispatch_data_create(ml, len(ml), None, None))
    fn = library.newFunctionWithName(to_ns_str("k"))
    descriptor = metal.MTLComputePipelineDescriptor.new()
    descriptor.setComputeFunction(fn)
    descriptor.setSupportIndirectCommandBuffers(True)
    _msg(descriptor.value, 'setBinaryArchives:', _msg(_cls('NSArray'), 'arrayWithObject:', archive, argtypes=[P]), restype=None, argtypes=[P])
    state = checked(self.sysdevice.newComputePipelineStateWithDescriptor_options_reflection_error, descriptor,
                    metal.MTLPipelineOptionFailOnBinaryArchiveMiss, None)
    assert archive not in _pending, "Metal never consulted the binary archive"
    return state
