# AGX: hand-assembled Apple GPU machine code on the HCQ2 Metal device. Metal only mints the pipeline state; the GPU runs our bytes.
from __future__ import annotations
import ctypes, struct, functools, bisect
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
# registers (dsl.REGISTER_FILE): r1 is the thread index; every value takes the lowest free register in r0, r2..r63 and gives it back after
# its last read. Loop counters must fit the 4-bit loop_init field (r0..r15) and an addr's shifted operand its 5-bit field (r0..r31): the
# lowest-free policy keeps both small; the DSL asserts if not. r64..r95 are unused until sources above r63 are hardware-verified.

def cval(u:UOp): # constant value behind CASTs, or None
  while u.op is Ops.CAST: u = u.src[0]
  return u.arg if u.op is Ops.CONST else None
def strip(u:UOp) -> UOp: # the value-carrying UOp behind CAST/AFTER/NOOP (all no-ops for us: one dtype width, no ordering to keep)
  while u.op in {Ops.CAST, Ops.AFTER, Ops.NOOP}: u = u.src[0]
  return u
def is_e4m3(v) -> bool:
  try: dsl.e4m3(float(v)); return True
  except ValueError: return False

class AGXCompiler(Compiler):
  def __init__(self): super().__init__(None) # no disk cache: assembling is instant and a stale binary hides ISA changes
  def compile(self, src:str) -> bytes: return asm.build(src)[0]
  def disassemble(self, lib:bytes): print(dsl.disasm_str(text_section(split_archive(lib)[0])))

class AGXRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  global_max, local_max = None, None
  compiler = AGXCompiler()

  def render(self, uops:list[UOp]) -> str:
    lines:list[str] = []
    pos = {u: i for i, u in enumerate(uops)}
    uses:dict[UOp, list[UOp]] = {}
    for u in uops:
      for src in u.src: uses.setdefault(src, []).append(u)
    params = [u for u in uops if u.op is Ops.PARAM]
    assert [p.arg.slot for p in params] == list(range(len(params))), "buffers must be the call args 0..n-1"
    lines.append(f".buffers {len(params)}")
    slot = {p: p.arg.slot for p in params}
    # fma: a float MUL whose only use is a float ADD folds into it (its operands are then read at the ADD)
    fused:dict[UOp, UOp] = {}
    for u in uops:
      if u.op is Ops.ADD and u.dtype == dtypes.float:
        for s in u.src:
          if s.op is Ops.MUL and s.dtype == dtypes.float and uses.get(s) == [u]: fused[s] = u; break   # one product per add

    # liveness: the last position each value is read, looking through INDEX/CAST/AFTER. a value read inside a loop that did not define it
    # is live until that loop's END (it is read again next iteration).
    TRANSPARENT = {Ops.INDEX, Ops.CAST, Ops.AFTER, Ops.NOOP}
    def roots(u:UOp) -> list[UOp]: return [r for s in u.src for r in roots(s)] if u.op in TRANSPARENT else [u]
    end_of = {e.src[1]: e for e in uops if e.op is Ops.END}
    last:dict[UOp, int] = {}
    open_loops:list[UOp] = []
    for i, u in enumerate(uops):
      if u.op is Ops.END: open_loops.pop()
      at = pos[fused[u]] if u in fused else i
      for s in u.src:
        for r in roots(s):
          j = at
          for l in open_loops:
            if pos[r] < pos[l]: j = max(j, pos[end_of[l]])
          last[r] = max(last.get(r, -1), j)
      if u.op is Ops.RANGE: open_loops.append(u); last[u] = pos[end_of[u]]   # the counter lives for the whole loop
    dying:dict[int, list[UOp]] = {}
    for u in uops: dying.setdefault(last.get(u, pos[u]), []).append(u)

    # the register file
    NREGS, COUNTER_MAX = 64, 15
    free = sorted(set(range(NREGS)) - {1})
    owned:dict[UOp, list[int]] = {}    # registers a value gives back when it dies (aliases of an accumulator own nothing)
    reg:dict[UOp, int] = {}
    def alloc(u:UOp, limit:int=NREGS - 1) -> int:
      for r in free:
        if r <= limit: free.remove(r); owned.setdefault(u, []).append(r); return r
      raise RuntimeError(f"AGX renderer: out of registers at {u.op} (need r <= {limit}, {len(free)} free); spilling is not implemented")
    def expire(i:int):
      for u in dying.get(i, []):
        for r in owned.pop(u, []): bisect.insort(free, r)

    def val(u:UOp) -> int: # register holding u, materializing constants (movimm) on first use
      u = strip(u)
      if u in reg: return reg[u]
      if u.op is Ops.CONST:
        d = alloc(u); reg[u] = d
        lines.append(f"movimm r{d}, #0x{u.arg & 0xffffffff:08x}" if u.dtype in dtypes.ints or u.dtype == dtypes.bool else f"movimm r{d}, #{float(u.arg)}")
        return d
      if u.op is Ops.LOAD and u.src[0].op is Ops.INDEX and strip(u.src[0].src[0]).op is Ops.BUFFER: return reg[strip(u.src[0].src[0])]
      raise RuntimeError(f"AGX renderer: {u.op} has no register")
    def fsrc(u:UOp) -> str: # float operand: E4M3 immediate or register
      return f"#{float(cval(u))}" if cval(u) is not None and is_e4m3(cval(u)) else f"r{val(u)}"

    done:set[UOp] = set()
    def intop(u:UOp) -> int: # int ALU op -> register holding its value, via addr (shift-add). emits code.
      if u in reg: return reg[u]
      a, b = u.src[0], u.src[1]
      if cval(a) is not None: a, b = b, a
      assert cval(a) is None, f"int op on two constants should have folded: {u}"
      ra = intop(a) if strip(a).op in {Ops.ADD, Ops.MUL} else val(a)
      if u.op is Ops.MUL:
        n = cval(b); assert n is not None and n > 0 and n & (n - 1) == 0 and n <= 16, f"only power-of-two strides up to 16 for now, got {n}"
        d = alloc(u); lines.append(f"addr r{d}, r{ra}, {n.bit_length() - 1}, #0")
      elif u.op is Ops.ADD:
        rb = None if cval(b) is not None else (intop(b) if strip(b).op in {Ops.ADD, Ops.MUL} else val(b))
        d = alloc(u); lines.append(f"addr r{d}, r{ra}, 0, " + (f"#{cval(b)}" if rb is None else f"r{rb}"))
      else: raise NotImplementedError(f"int {u.op}")
      reg[u] = d; done.add(u); return d
    def idxreg(idx:UOp) -> int: return intop(idx) if strip(idx).op in {Ops.ADD, Ops.MUL} else val(idx)
    def acc_of(store_or_load:UOp) -> UOp|None: # the register accumulator behind an INDEX, or None for a memory buffer
      b = strip(store_or_load.src[0].src[0]); return b if b.op is Ops.BUFFER else None

    i = 0
    while i < len(uops):
      u = uops[i]
      if u in done or u in fused or u.op in {Ops.PARAM, Ops.CONST, Ops.CAST, Ops.AFTER, Ops.SINK, Ops.NOOP, Ops.INDEX}: expire(i); i += 1; continue
      if u.op is Ops.BUFFER: reg[u] = alloc(u)                  # a register-resident accumulator
      elif u.op is Ops.RANGE:
        n = cval(u.src[0]); assert n is not None, "symbolic loop bounds are not supported yet"
        c = alloc(u, COUNTER_MAX)
        lines.append(f"loop r{c}, #{n}")
        # the counter reads 0..n-1 only in the pre section; copy it there so inner loops (where it already reads 1..n) see the 0-based value
        v = alloc(u); reg[u] = v; lines.append(f"addr r{v}, r{c}, 0, #0")
        # pre section: every int op inside this loop that depends only on constants, outer values and this counter
        j = i + 1
        while uops[j] is not end_of[u]:
          w = uops[j]; j += 1
          if w.dtype == dtypes.int and w.op in {Ops.ADD, Ops.MUL} and all(cval(s) is not None or strip(s) in reg for s in w.src): intop(w)
        lines.append("body")
      elif u.op is Ops.END: lines.append("endloop")
      elif u.op is Ops.LOAD:
        ix = u.src[0]
        if (acc:=acc_of(u)) is not None: reg[u] = reg[acc]      # accumulator read: alias
        else:
          d = alloc(u); reg[u] = d
          lines.append(f"load r{d}, {slot[strip(ix.src[0])]}, r{idxreg(ix.src[1])}"); lines.append("wait")
      elif u.op is Ops.STORE:
        ix, v = u.src[0], u.src[1]
        if (acc:=acc_of(u)) is not None:                        # accumulator write
          if cval(v) is not None: lines.append(f"movimm r{reg[acc]}, #{float(cval(v))}")
          else: assert reg[strip(v)] == reg[acc], "value must have been computed into the accumulator"
        else: lines.append(f"store r{val(v)}, {slot[strip(ix.src[0])]}, r{idxreg(ix.src[1])}")
      elif u.op in {Ops.ADD, Ops.MUL} and u.dtype == dtypes.float:
        us = uses.get(u, [])
        # destination: the accumulator when this value's only use is an accumulator store, else a fresh register
        if len(us) == 1 and us[0].op is Ops.STORE and (acc:=acc_of(us[0])) is not None: dst = reg[acc]
        else: dst = alloc(u)
        reg[u] = dst
        a, b = u.src
        if u.op is Ops.ADD and fused.get(b) is u: a, b = b, a
        if u.op is Ops.ADD and fused.get(a) is u: lines.append(f"fma r{dst}, {fsrc(a.src[0])}, r{val(a.src[1])}, {fsrc(b)}")   # a*b + c fused
        else: lines.append(f"{'fadd' if u.op is Ops.ADD else 'fmul'} r{dst}, r{val(a)}, {fsrc(b)}")
      elif u.op in {Ops.ADD, Ops.MUL} and u.dtype == dtypes.int: intop(u)
      else: raise NotImplementedError(f"AGX renderer: {u.op} {u.dtype}")
      expire(i); i += 1
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
