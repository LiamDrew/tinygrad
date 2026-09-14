# AGX: hand-assembled Apple GPU machine code on the HCQ2 Metal device. Metal only mints the pipeline state; the GPU runs our bytes.
from __future__ import annotations
import ctypes, struct, functools
from tinygrad.helpers import Target
from tinygrad.device import Compiler
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import metal
import tinygrad.runtime.support.objc as objc
from tinygrad.runtime.ops_metal import MetalDevice, MetalQueue, checked, to_ns_str
from tinygrad.runtime.support.hcq2 import encode_submit
from tinygrad.runtime.support.agx import asm
from tinygrad.renderer.agx import dsl

# *****************
# renderer: linear UOps -> our assembly (tinygrad/runtime/support/agx/asm.py syntax). Naive kernels only: NOOPT=1, no SPECIAL, one thread.
# registers: r1 is the thread index; floats, constants and loop counters take r2..r14 (4-bit destination fields); int address math takes r16+.

def cval(u:UOp): # constant value behind CASTs, or None
  while u.op is Ops.CAST: u = u.src[0]
  return u.arg if u.op is Ops.CONST else None
def base(u:UOp) -> UOp: # look through AFTER to the buffer
  while u.op is Ops.AFTER: u = u.src[0]
  return u

class AGXCompiler(Compiler):
  def __init__(self): super().__init__(None) # no disk cache: assembling is instant and a stale binary hides ISA changes
  def compile(self, src:str) -> bytes: return asm.build(src)[0]
  def disassemble(self, lib:bytes): print(dsl.disasm_str(text_section(split_archive(lib)[0])))

class AGXRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  global_max, local_max = None, None
  compiler = AGXCompiler()

  def render(self, uops:list[UOp]) -> str:
    lines, reg, lo, hi = [], {}, 0, 14                         # reg: UOp -> register; lo/hi pools
    def new_lo():
      nonlocal lo; lo += 2; assert lo <= 14, "out of low registers"; return lo
    def new_hi():
      nonlocal hi; hi += 2; assert hi <= 62, "out of high registers"; return hi
    uses:dict[UOp, list[UOp]] = {}
    for u in uops:
      for src in u.src: uses.setdefault(src, []).append(u)
    params = [u for u in uops if u.op is Ops.PARAM]
    assert [p.arg.slot for p in params] == list(range(len(params))), "buffers must be the call args 0..n-1"
    lines.append(f".buffers {len(params)}")
    slot = {p: p.arg.slot for p in params}
    done:set[UOp] = set()

    def intop(u:UOp) -> int: # int ALU op -> register holding its value, via addr (shift-add). emits code.
      if u in reg: return reg[u]
      a, b = u.src[0], u.src[1]
      if u.op is Ops.MUL and cval(a) is not None: a, b = b, a
      if u.op is Ops.ADD and cval(a) is not None: a, b = b, a
      ra = intop(a) if cval(a) is None else None
      assert ra is not None, f"int op on two constants should have folded: {u}"
      d = new_hi()
      if u.op is Ops.MUL:
        n = cval(b); assert n is not None and n > 0 and n & (n - 1) == 0 and n <= 16, f"only power-of-two strides up to 16 for now, got {n}"
        lines.append(f"addr r{d}, r{ra}, {n.bit_length() - 1}, #0")
      elif u.op is Ops.ADD:
        if cval(b) is not None: lines.append(f"addr r{d}, r{ra}, 0, #{cval(b)}")
        else: lines.append(f"addr r{d}, r{ra}, 0, r{intop(b)}")
      else: raise NotImplementedError(f"int {u.op}")
      reg[u] = d; done.add(u); return d

    def idxreg(idx:UOp) -> int: # register holding an element index (constants go through movimm)
      if (c:=cval(idx)) is not None:
        d = new_lo(); lines.append(f"movimm r{d}, #0x{c:08x}"); return d
      return intop(idx)

    def fsrc(u:UOp) -> str: # float operand: register or E4M3 immediate
      return f"#{cval(u)}" if cval(u) is not None else f"r{reg[u]}"

    i = 0
    while i < len(uops):
      u = uops[i]; i += 1
      if u in done or u.op in {Ops.PARAM, Ops.CONST, Ops.CAST, Ops.AFTER, Ops.SINK, Ops.NOOP, Ops.INDEX}: continue
      if u.op is Ops.BUFFER: reg[u] = new_lo()                 # a register-resident accumulator
      elif u.op is Ops.RANGE:
        n = cval(u.src[0]); assert n is not None, "symbolic loop bounds are not supported yet"
        c = new_lo()
        lines.append(f"loop r{c}, #{n}")
        # the counter reads 0..n-1 only in the pre section; copy it there so inner loops (where it already reads 1..n) see the 0-based value
        v = new_hi(); reg[u] = v; lines.append(f"addr r{v}, r{c}, 0, #0")
        # pre section: every int op inside this loop that depends only on constants, outer values and this counter
        depth, j = 0, i
        while j < len(uops) and not (uops[j].op is Ops.END and uops[j].src[1] is u):
          v = uops[j]; j += 1
          if v.dtype == dtypes.int and v.op in {Ops.ADD, Ops.MUL} and all(cval(s) is not None or s in reg or s is u for s in v.src): intop(v)
        lines.append("body")
      elif u.op is Ops.END: lines.append("endloop")
      elif u.op is Ops.LOAD:
        ix = u.src[0]; buf = base(ix.src[0])
        if buf.op is Ops.BUFFER: reg[u] = reg[buf]              # accumulator read: alias
        else:
          d = new_lo(); reg[u] = d
          lines.append(f"load r{d}, {slot[buf]}, r{idxreg(ix.src[1])}"); lines.append("wait")
      elif u.op is Ops.STORE:
        ix, v = u.src[0], u.src[1]; buf = base(ix.src[0])
        if buf.op is Ops.BUFFER:                               # accumulator write
          acc = reg[buf]
          if cval(v) is not None: lines.append(f"movimm r{acc}, #{float(cval(v))}")
          else: assert reg[v] == acc, "value must have been computed into the accumulator"
        else: lines.append(f"store r{reg[v]}, {slot[buf]}, r{idxreg(ix.src[1])}")
      elif u.op in {Ops.ADD, Ops.MUL} and u.dtype == dtypes.float:
        us = uses.get(u, [])
        if u.op is Ops.MUL and len(us) == 1 and us[0].op is Ops.ADD and us[0].dtype == dtypes.float: continue   # fused by the ADD below
        # destination: the accumulator when this value's only use is an accumulator store
        dst = reg[base(us[0].src[0].src[0])] if len(us) == 1 and us[0].op is Ops.STORE and base(us[0].src[0].src[0]).op is Ops.BUFFER else new_lo()
        reg[u] = dst
        a, b = u.src
        if u.op is Ops.ADD and b.op is Ops.MUL and b.dtype == dtypes.float and len(uses.get(b, [])) == 1 and b not in reg: a, b = b, a
        if u.op is Ops.ADD and a.op is Ops.MUL and a.dtype == dtypes.float and len(uses.get(a, [])) == 1 and a not in reg:
          done.add(a); lines.append(f"fma r{dst}, {fsrc(a.src[0])}, r{reg[a.src[1]]}, {fsrc(b)}")   # a*b + c fused
        else: lines.append(f"{'fadd' if u.op is Ops.ADD else 'fmul'} r{dst}, r{reg[a]}, {fsrc(b)}")
      elif u.op in {Ops.ADD, Ops.MUL} and u.dtype == dtypes.int: intop(u)
      else: raise NotImplementedError(f"AGX renderer: {u.op} {u.dtype}")
    return "\n".join(lines) + "\n"

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
