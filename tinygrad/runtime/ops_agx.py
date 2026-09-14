# AGX: hand-assembled Apple GPU machine code on the HCQ2 Metal device. Metal only mints the pipeline state; the GPU runs our bytes.
from __future__ import annotations
import ctypes, struct, functools, bisect
from tinygrad.helpers import Target
from tinygrad.device import Compiler
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.runtime.autogen import metal
import tinygrad.runtime.support.objc as objc
from tinygrad.runtime.ops_metal import MetalDevice, MetalQueue, checked, to_ns_str
from tinygrad.runtime.support.hcq2 import encode_submit
from tinygrad.runtime.support.agx import asm
from tinygrad.renderer.agx import dsl
from tinygrad.codegen.decomp.transcendental import xsin

# *****************
# renderer: linear UOps -> our assembly (tinygrad/runtime/support/agx/asm.py syntax). SPECIAL gidx/lidx read the threadgroup / local index registers.
# registers (dsl.REGISTER_FILE): r1 is the thread index; every value takes the lowest free register in r0, r2..r63 and gives it back after
# its last read. Loop counters must fit the 4-bit loop_init field (r0..r15) and an addr's shifted operand its 5-bit field (r0..r31): the
# lowest-free policy keeps both small; the DSL asserts if not. r64..r95 are unused until sources above r63 are hardware-verified.

def cval(u:UOp): # constant value behind CASTs, or None
  while u.op is Ops.CAST: u = u.src[0]
  return u.arg if u.op is Ops.CONST else None
def is_convert(u:UOp) -> bool: # a CAST that changes the bit pattern (int <-> float); other CASTs (bool <-> int, weak types) are free
  return u.op is Ops.CAST and dtypes.is_float(u.dtype) != dtypes.is_float(u.src[0].dtype)
def is_free(u:UOp) -> bool: # CAST/BITCAST that keep the register's bits: bool <-> int, weak types, 32-bit reinterpretation
  return (u.op is Ops.CAST and not is_convert(u)) or (u.op is Ops.BITCAST and u.dtype.itemsize == u.src[0].dtype.itemsize)
def strip(u:UOp) -> UOp: # the value-carrying UOp behind free casts, AFTER and NOOP
  while is_free(u) or u.op in {Ops.AFTER, Ops.NOOP}: u = u.src[0]
  return u
def is_int(u:UOp) -> bool: return dtypes.is_int(u.dtype) or u.dtype == dtypes.bool   # dtypes.is_int includes the weak int of constants
def is_e4m3(v) -> bool:
  try: dsl.e4m3(float(v)); return True
  except ValueError: return False

class AGXCompiler(Compiler):
  def __init__(self): super().__init__(None) # no disk cache: assembling is instant and a stale binary hides ISA changes
  def compile(self, src:str) -> bytes: return asm.build(src)[0]
  def disassemble(self, lib:bytes): print(dsl.disasm_str(text_section(split_archive(lib)[0])))

class AGXRenderer(Renderer):
  has_local, has_shared, supports_float4 = True, False, False   # threadgroups yes, threadgroup memory not yet, scalar only
  local_max = (1024, 1024, 1024)                                # MetalQueue: global_size = threadgroups, local_size = threads per group
  compiler = AGXCompiler()
  # hardware transcendentals (dsl.TRANS); tinygrad decomposes every float op missing here (SIN: codegen/decomp/transcendental.py). Values unused.
  # SHL/SHR here turn the decompositions' x * 2**n and x // 2**n into shifts (codegen/decomp/op.py gates those rewrites on the renderer's ops).
  # SIN: tinygrad's default decomposition reduces large arguments through 64-bit ints (Payne-Hanek); the fast variant is Cody-Waite in float32
  # (1 ulp to |x| ~ 4e4, degrading beyond). Apple's hardware sin (probes/trans/sin.metal) is a reduction sequence through undecoded ops.
  code_for_op = {Ops.EXP2: None, Ops.LOG2: None, Ops.SQRT: None, Ops.RECIPROCAL: None, Ops.SIN: None, Ops.SHL: None, Ops.SHR: None}
  extra_matcher = PatternMatcher([(UPat(Ops.SIN, dtype=dtypes.float, src=(UPat.var("d"),)), lambda d: xsin(d, fast=True))])

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
      # an int compare whose only use is a WHERE folds into the select (isel compares and selects in one op)
      if u.op is Ops.WHERE and (c:=strip(u.src[0])).op is Ops.CMPLT and uses.get(c) == [u.src[0]] and (u.src[0] is c or uses.get(u.src[0]) == [u]):
        fused[c] = u   # int or float compare

    # lazy loads: a global LOAD whose first use is in the same straight-line region (no store, loop edge, branch or barrier in between) is emitted
    # right before that use. tinygrad issues every load of an unrolled reduce first; lowering them in place needs a register per load.
    BOUNDARY = {Ops.STORE, Ops.RANGE, Ops.END, Ops.IF, Ops.ENDIF, Ops.BARRIER}
    lazy:set[UOp] = set(); emit_at:dict[UOp, int] = {}   # a lazy load reads its address at emit_at (liveness below uses it)
    def epos(c:UOp) -> int: return pos[fused[c]] if c in fused else emit_at.get(c, pos[c])   # where consumer c is actually emitted
    for u in uops:
      if u.op is Ops.LOAD and strip(u.src[0].src[0]).op is not Ops.BUFFER and (us:=uses.get(u)):
        first = min(epos(x) for x in us)
        if not any(w.op in BOUNDARY for w in uops[pos[u] + 1:first]): lazy.add(u); emit_at[u] = first
    # deferred address math: an int ADD/MUL used only by INDEXes (or by other deferred ops) is computed on demand by the load/store that
    # needs it (intop), so its register is not held from its linear position. Its operands are then read at that consumer's position.
    deferred:set[UOp] = set()
    for u in reversed(uops):
      if is_int(u) and u.op in {Ops.ADD, Ops.MUL} and (us:=uses.get(u)) and all(x.op is Ops.INDEX or x in deferred for x in us):
        ats = [epos(c) for x in us for c in (uses.get(x, []) if x.op is Ops.INDEX else [x])]
        if ats: deferred.add(u); emit_at[u] = max(ats)
    # liveness: the last position each value is read, looking through INDEX/CAST/AFTER. a value read inside a loop that did not define it
    # is live until that loop's END (it is read again next iteration).
    def roots(u:UOp) -> list[UOp]: return [r for s in u.src for r in roots(s)] if u.op in {Ops.INDEX, Ops.AFTER, Ops.NOOP} or is_free(u) else [u]
    end_of = {e.src[1]: e for e in uops if e.op is Ops.END}
    loops = [(pos[l], pos[e]) for l, e in end_of.items()]
    last:dict[UOp, int] = {}
    for i, u in enumerate(uops):
      at = pos[fused[u]] if u in fused else emit_at.get(u, i)   # fused products and lazy loads read their operands later
      for s in u.src:
        for r in roots(s):
          j = at
          for ls, le in loops:   # the loops open at the read, not at u's own position (a product defined before a loop can be fused into an add inside it)
            if ls < at < le and pos[r] < ls: j = max(j, le)
          last[r] = max(last.get(r, -1), j)
      if u.op is Ops.RANGE: last[u] = pos[end_of[u]]   # the counter lives for the whole loop
    # a LOAD from a register accumulator aliases the accumulator's register: the accumulator must live as long as that value does
    for u in uops:
      if u.op is Ops.LOAD and (b:=strip(u.src[0].src[0])).op is Ops.BUFFER: last[b] = max(last.get(b, -1), last.get(u, pos[u]))
    dying:dict[int, list[UOp]] = {}
    for u in uops: dying.setdefault(last.get(u, pos[u]), []).append(u)

    # the register file
    NREGS, COUNTER_MAX = 64, 15
    free = sorted(set(range(NREGS)) - {1})
    owned:dict[UOp, list[int]] = {}    # registers a value gives back when it dies (aliases of an accumulator own nothing)
    reg:dict[UOp, int] = {}
    # classes: loop counters and special-register reads need r0..r15 (4-bit dst), addr's shifted operand r0..r31 (5-bit), ALU results r0..r63.
    # each class takes the lowest free register at or above its preferred start, then falls back below it, so the small ranges stay free.
    def alloc(u:UOp, limit:int=NREGS - 1, prefer:int=32) -> int:
      for r in [r for r in free if prefer <= r <= limit] + [r for r in free if r < prefer]:
        free.remove(r); owned.setdefault(u, []).append(r); return r
      raise RuntimeError(f"AGX renderer: out of registers at {u.op} (need r <= {limit}, {len(free)} free); spilling is not implemented")
    def alloc_int(u:UOp) -> int: return alloc(u, 31, 8)
    scratch:list[int] = []   # registers that live only until the end of the current step (materialized constants)
    def alloc_scratch(u:UOp) -> int:
      r = alloc(u, 31, 8) if is_int(u) else alloc(u); owned[u].remove(r); scratch.append(r); return r
    def alloc_low(u:UOp) -> int: return alloc(u, COUNTER_MAX, 0)
    def expire(i:int):
      for u in dying.get(i, []):
        for r in owned.pop(u, []): bisect.insort(free, r)
      while scratch: bisect.insort(free, scratch.pop())

    INT2 = {Ops.SUB: "isub", Ops.AND: "iand", Ops.OR: "ior", Ops.XOR: "ixor", Ops.MAX: "imax", Ops.CMPLT: "icmp.lt", Ops.CMPNE: "icmp.ne", Ops.CMPEQ: "icmp.eq"}
    def val(u:UOp) -> int: # register holding u, materializing constants (movimm) on first use
      u = strip(u)
      if u in reg: return reg[u]
      if u.op is Ops.CONST:   # rematerialized at every use: a movimm is cheaper than holding a register across an unrolled kernel
        d = alloc_scratch(u)
        lines.append(f"movimm r{d}, #0x{int(u.arg) & 0xffffffff:08x}" if is_int(u) else f"movimm r{d}, #{float(u.arg)}")   # weak int consts count
        return d
      if u.op is Ops.LOAD and (a:=acc_of(u)) is not None: return acc_reg(*a)
      if u in lazy: return emit_load(u)
      raise RuntimeError(f"AGX renderer: {u.op} has no register")
    def emit_load(u:UOp) -> int:
      ix = u.src[0]; d = alloc_int(u) if is_int(u) else alloc(u); reg[u] = d
      lines.append(f"load r{d}, {slot[strip(ix.src[0])]}, r{idxreg(ix.src[1])}"); lines.append("wait")
      return d
    def fsrc(u:UOp) -> str: # float operand: E4M3 immediate or register
      return f"#{float(cval(u))}" if cval(u) is not None and is_e4m3(cval(u)) else f"r{val(u)}"

    done:set[UOp] = set()
    def intop(u:UOp) -> int: # int ADD/MUL -> register holding its value: addr (shift-add) for address math, imul otherwise. emits code.
      u = strip(u)   # a free CAST may wrap the op
      if u in reg: return reg[u]
      a, b = u.src[0], u.src[1]
      if cval(a) is not None: a, b = b, a
      assert cval(a) is None, f"int op on two constants should have folded: {u}"
      ra = intop(a) if strip(a).op in {Ops.ADD, Ops.MUL} else val(a)
      n = cval(b)
      if u.op is Ops.MUL and n is not None and n > 0 and n & (n - 1) == 0:
        d = alloc_int(u); lines.append(f"addr r{d}, r{ra}, {n.bit_length() - 1}, #0" if n <= 16 else f"ishl r{d}, r{ra}, #{n.bit_length() - 1}")
      elif u.op is Ops.MUL:
        rb = intop(b) if strip(b).op in {Ops.ADD, Ops.MUL} else val(b)
        d = alloc_int(u); lines.append(f"imul r{d}, r{ra}, r{rb}")
      elif n is not None and 0 <= n < 128:
        d = alloc_int(u); lines.append(f"addr r{d}, r{ra}, 0, #{n}")
      else:
        rb = intop(b) if strip(b).op in {Ops.ADD, Ops.MUL} else val(b)
        d = alloc_int(u); lines.append(f"iadd r{d}, r{ra}, r{rb}")
      reg[u] = d; done.add(u); return d
    def idxreg(idx:UOp) -> int: return intop(idx) if strip(idx).op in {Ops.ADD, Ops.MUL} else val(idx)
    acc_regs:dict[UOp, list[int]] = {}   # register-array accumulators: BUFFER (AddrSpace.REG, size n) -> n registers, indexed by constants
    def acc_of(store_or_load:UOp) -> tuple[UOp, int]|None: # (accumulator, element) behind an INDEX, or None for a memory buffer
      ix = store_or_load.src[0]; b = strip(ix.src[0])
      if b.op is not Ops.BUFFER: return None
      k = cval(ix.src[1]); assert k is not None, "register accumulators need constant indices (they are registers, not memory)"
      return b, k
    def acc_reg(b:UOp, k:int) -> int: return acc_regs[b][k]

    i = 0
    while i < len(uops):
      u = uops[i]
      if u in done or u in fused or u in deferred or is_free(u) or u.op in {Ops.PARAM, Ops.CONST, Ops.AFTER, Ops.SINK, Ops.NOOP, Ops.GROUP, Ops.INDEX}:
        expire(i); i += 1; continue
      if u.op is Ops.BUFFER:                                    # register-resident accumulators: one register per element
        assert u.arg.addrspace is AddrSpace.REG, f"{u.arg.addrspace} buffers are not implemented (threadgroup memory needs has_shared)"
        acc_regs[u] = [alloc(u) for _ in range(u.arg.size)]
      elif u.op is Ops.RANGE:
        n = cval(u.src[0]); assert n is not None, "symbolic loop bounds are not supported yet"
        c = alloc_low(u)
        lines.append(f"loop r{c}, #{n}")
        # the counter reads 0..n-1 only in the pre section; copy it there so inner loops (where it already reads 1..n) see the 0-based value
        v = alloc_int(u); reg[u] = v; lines.append(f"addr r{v}, r{c}, 0, #0")
        lines.append("body")   # address math stays in the body in program order (hoisting it kept every address live for the whole loop)
      elif u.op is Ops.END: lines.append("endloop")
      elif u.op is Ops.LOAD:
        ix = u.src[0]
        if (acc:=acc_of(u)) is not None: reg[u] = acc_reg(*acc)  # accumulator read: alias
        elif u not in lazy: emit_load(u)                        # lazy loads are emitted by val() at their first use
      elif u.op is Ops.STORE:
        ix, v = u.src[0], u.src[1]
        if (acc:=acc_of(u)) is not None:                        # accumulator write
          ra = acc_reg(*acc)
          if cval(v) is not None: lines.append(f"movimm r{ra}, #{float(cval(v)) if dtypes.is_float(v.dtype) else '0x%08x' % (int(cval(v)) & 0xffffffff)}")
          elif (rv:=val(v)) != ra: lines.append(f"ior r{ra}, r{rv}, r{rv}")   # register move (bitwise, any 32-bit type)
        else:
          assert v.dtype.itemsize == 4, f"only 32-bit stores for now, not {v.dtype} (float16/bfloat16 are not supported: dsl.DTYPES)"
          lines.append(f"store r{val(v)}, {slot[strip(ix.src[0])]}, r{idxreg(ix.src[1])}")
      elif u.op in {Ops.ADD, Ops.MUL} and u.dtype == dtypes.float:
        us = uses.get(u, [])
        # destination: the accumulator when this value's only use is an accumulator store, else a fresh register
        if len(us) == 1 and us[0].op is Ops.STORE and (acc:=acc_of(us[0])) is not None: dst = acc_reg(*acc)
        else: dst = alloc(u)
        reg[u] = dst
        a, b = u.src
        if u.op is Ops.ADD and fused.get(b) is u: a, b = b, a
        if u.op is Ops.ADD and fused.get(a) is u: lines.append(f"fma r{dst}, {fsrc(a.src[0])}, r{val(a.src[1])}, {fsrc(b)}")   # a*b + c fused
        else: lines.append(f"{'fadd' if u.op is Ops.ADD else 'fmul'} r{dst}, r{val(a)}, {fsrc(b)}")
      elif u.op in {Ops.ADD, Ops.MUL} and u.dtype in dtypes.ints: intop(u)
      elif u.op in INT2 and is_int(u.src[0]):                  # int binary ops and compares: registers only (constants are materialized)
        ra, rb = val(u.src[0]), val(u.src[1])
        mn = "icmp.ult" if u.op is Ops.CMPLT and u.src[0].dtype in dtypes.uints else INT2[u.op]
        d = alloc_int(u); reg[u] = d; lines.append(f"{mn} r{d}, r{ra}, r{rb}")
      elif u.op in {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ, Ops.MAX} and u.src[0].dtype == dtypes.float:   # float compares (0/1 int result) and max
        ra, rb = val(u.src[0]), val(u.src[1])
        mn = {Ops.CMPLT: "fcmp.lt", Ops.CMPNE: "fcmp.ne", Ops.CMPEQ: "fcmp.eq", Ops.MAX: "fmax"}[u.op]
        d = alloc(u) if u.op is Ops.MAX else alloc_int(u); reg[u] = d; lines.append(f"{mn} r{d}, r{ra}, r{rb}")
      elif u.op in {Ops.EXP2, Ops.LOG2, Ops.RECIPROCAL, Ops.SQRT} and u.dtype == dtypes.float:   # one hardware op each; sqrt(a) = a * rsqrts(a)
        ra = val(u.src[0]); d = alloc(u); reg[u] = d
        if u.op is Ops.SQRT: lines.append(f"rsqrts r{d}, r{ra}"); lines.append(f"fmul r{d}, r{ra}, r{d}")
        else: lines.append(f"{ {Ops.EXP2: 'exp2', Ops.LOG2: 'log2', Ops.RECIPROCAL: 'rcp'}[u.op] } r{d}, r{ra}")
      elif u.op is Ops.IF: lines.append(f"if r{val(u.src[0])}")     # per-thread predication of the region up to ENDIF
      elif u.op is Ops.ENDIF: lines.append("endif")
      elif u.op is Ops.SPECIAL:                                 # ("g"|"l", axis): threadgroup_position_in_grid / thread_position_in_threadgroup
        d = alloc_low(u); reg[u] = d; lines.append(f"getsr r{d}, {'gid' if u.arg[0] == 'g' else 'lid'}.{'xyz'[int(u.arg[-1])]}")
      elif u.op is Ops.CAST:                                    # int <-> float convert (a constant source is folded by tinygrad; else it is in a register)
        src = u.src[0]
        assert u.dtype.itemsize == 4 and src.dtype.itemsize == 4 or src.dtype == dtypes.bool, f"only 32-bit converts for now: {src.dtype} -> {u.dtype} (float16 is not supported: dsl.DTYPES)"
        mn = ("u2f" if src.dtype in dtypes.uints else "i2f") if dtypes.is_float(u.dtype) else ("f2u" if u.dtype in dtypes.uints else "f2i")
        ra = val(src); d = alloc_int(u) if is_int(u) else alloc(u); reg[u] = d; lines.append(f"{mn} r{d}, r{ra}")
      elif u.op in {Ops.SHL, Ops.SHR} and is_int(u):
        mn = "ishl" if u.op is Ops.SHL else "ushr" if u.dtype in dtypes.uints else "ishr"
        ra, n = val(u.src[0]), cval(u.src[1])
        cnt = f"#{n}" if n is not None and 0 <= n < 32 else f"r{val(u.src[1])}"
        d = alloc_int(u); reg[u] = d; lines.append(f"{mn} r{d}, r{ra}, {cnt}")
      elif u.op is Ops.NEG and is_int(u):
        ra = val(u.src[0]); d = alloc_int(u); reg[u] = d; lines.append(f"ineg r{d}, r{ra}")
      elif u.op is Ops.WHERE and u.dtype.itemsize == 4:         # select of 32-bit values (int or float alike: the select moves bits)
        c, x, y = u.src
        mn = "isel"
        if fused.get(cc:=strip(c)) is u: ra, rb = val(cc.src[0]), val(cc.src[1]); mn = "fsel" if cc.src[0].dtype == dtypes.float else "isel"   # (a < b) ? x : y
        else: ra, rb = None, val(c)                                                              # (0 < cond) ? x : y, zero register below
        rx, ry = val(x), val(y)
        if ra is None: ra = alloc_int(u); lines.append(f"movimm r{ra}, #0x00000000")
        d = alloc_int(u) if is_int(u) else alloc(u); reg[u] = d; lines.append(f"{mn} r{d}, r{ra}, r{rb}, r{rx}, r{ry}")
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
