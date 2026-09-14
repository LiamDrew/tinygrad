# Apple G15 (M3) GPU instruction table. Every entry is hardware-verified by a kernel that runs it (test/device/test_agx.py).
# An instruction is a little-endian integer of size*8 bits: `fixed` holds the constant bits, fields carve out the variable ones.
# encode and decode read the same table, so a byte we emit that we cannot read back is a bug in our understanding.
from __future__ import annotations
from dataclasses import dataclass, field
import struct

@dataclass(frozen=True)
class Field:
  name:str
  lo:int      # bit offset from the start of the instruction
  width:int
  @property
  def mask(self) -> int: return ((1 << self.width) - 1) << self.lo

@dataclass(frozen=True)
class Inst:
  mnemonic:str
  fixed:bytes                       # template; field bits inside it must be zero
  fields:tuple[Field, ...] = ()
  doc:str = ""
  @property
  def size(self) -> int: return len(self.fixed)
  @property
  def care(self) -> int: # bits that identify this instruction
    m = (1 << 8 * self.size) - 1
    for f in self.fields: m &= ~f.mask
    return m
  def encode(self, **vals:int) -> bytes:
    v = int.from_bytes(self.fixed, "little")
    for f in self.fields:
      x = vals.pop(f.name, 0)
      assert 0 <= x < (1 << f.width), f"{self.mnemonic}.{f.name}={x} does not fit {f.width} bits"
      v |= x << f.lo
    assert not vals, f"{self.mnemonic}: unknown fields {list(vals)}"
    return v.to_bytes(self.size, "little")
  def matches(self, data:bytes, off:int=0) -> bool:
    if off + self.size > len(data): return False
    v = int.from_bytes(data[off:off + self.size], "little")
    return (v & self.care) == (int.from_bytes(self.fixed, "little") & self.care)
  def decode(self, data:bytes, off:int=0) -> dict[str, int]:
    v = int.from_bytes(data[off:off + self.size], "little")
    return {f.name: (v & f.mask) >> f.lo for f in self.fields}

def B(byte:int, bit:int=0, width:int=8) -> int: return 8 * byte + bit # bit offset helper: (byte index, bit within byte)

# ---- operand encodings ----------------------------------------------------------------------------------------------------------
# register fields hold the 32-bit register number << 1 (probes/sums_live.metal: loads into fields 00,04,08,0c, sums stored from 10,14,18 with
# ALU dst nibbles 8,a,c). Apple's compiler prefers even registers. Hardware-verified (test_agx): dst nibble 2/4 <-> store field 2/4.
# dst-style fields (load dst, store src) sit at bit 1 of their byte and hold the register number directly
def reg_operand(r:int) -> int: return (r << 1) | 1     # 32-bit source operand: bit0 = 32-bit width
def operand_reg(o:int) -> int: return o >> 1
def e4m3(v:float) -> tuple[int, bool]: # unsigned E4M3 minifloat (exponent bias 11) in bits[7:1], bit0 = 32-bit flag; sign rides in the modifier
  if v == 0: raise ValueError("0.0 is not E4M3; use movimm")
  neg, v = v < 0, abs(v)
  for e in range(-11, 5):
    for m in range(8):
      if 2.0 ** e * (1 + m / 8.0) == v: return (((e + 11) << 3 | m) << 1) | 1, neg
  raise ValueError(f"{v} is not E4M3-representable; use movimm")
def e4m3_value(b:int) -> float:
  b >>= 1
  return 2.0 ** ((b >> 3) - 11) * (1 + (b & 7) / 8.0)

# ---- the table --------------------------------------------------------------------------------------------------------------------

LOAD = Inst("load", bytes.fromhex("6700440000012000"), (
  Field("first", B(1, 4), 1),       # set on the first load of the kernel
  Field("more", B(2, 4), 1),        # another load follows (scoreboard)
  Field("dst", B(3, 1), 7),         # destination register (reg_field)
  Field("slot", B(4), 8)),          # dense buffer binding index
  doc="dst = buf[slot][thread_index] (32-bit)")
STORE = Inst("store", bytes.fromhex("e700540000012100"), (
  Field("first", B(1, 4), 1),       # set when no load preceded (probes/movimm.metal)
  Field("hint", B(2, 1), 1),        # 0x56 vs 0x54 in byte 2: seen on stores straight from a load, meaning unknown; 0 works everywhere
  Field("src", B(3, 1), 7),         # source register (reg_field)
  Field("slot", B(4), 8)),
  doc="buf[slot][thread_index] = src (32-bit)")
WAIT = Inst("wait", bytes.fromhex("510100404600"), doc="wait for one outstanding load")
# float ALU. 4-byte minimal form: [09 | dst<<4] [srcb] [mod] [srca]. mod bit2 adds two bytes (byte4 = 0 so far, byte5 in {c0,20,40}:
# scheduling-looking, undecoded); mod bit1 adds an fma third operand. Verified on hardware in test_agx with explicit and odd dst.
FALU = Inst("falu", bytes.fromhex("09000000"), (
  Field("dst", B(0, 4), 4),         # destination register r0..r15 (probes/sums_live.metal)
  Field("srcb", B(1), 8),           # register operand, or E4M3 immediate when bmode has bit 7
  Field("mul", B(2, 0), 1),         # 0 = fadd, 1 = fmul (probes/fmul.metal)
  Field("fma", B(2, 1), 1),         # three-operand form, 8 bytes (probes/mul_add.metal), not encodable here yet
  Field("killb", B(2, 3), 1),       # srcb dies after this op (compiler hint; every probe agrees, semantics unverified)
  Field("killa", B(2, 4), 1),       # srca dies after this op
  Field("fwd", B(2, 5), 1),         # result feeds another ALU op rather than a store
  Field("srca", B(3), 8)),
  doc="fadd32/fmul32: dst = srca op srcb. Apple emits this short form, but on hardware a store right after it reads 0: use FALU_LONG")
FALU_LONG = Inst("falu", bytes.fromhex("090004000000"), tuple(FALU.fields) + (Field("ext", B(4), 8), Field("tag", B(5), 8)),
  doc="six-byte float ALU: mod bit2 (fixed here) set, byte4 = 0 or 0x80 for an immediate srcb, byte5 = c0/20/40 in Apple's output")
FALU_IMM_NEG, FALU_IMM_ADD = 1, 0    # for an E4M3 immediate srcb the sign lives in killb: 0x14 = add, 0x1c = negate (prog_add vs probes)
MOVIMM = Inst("movimm", bytes.fromhex("0c80020000000000"), (
  Field("hi7", B(3, 1), 7),         # fp32 bits[31:25]
  Field("mid21", B(4, 3), 21),      # fp32 bits[20:0]
  Field("lo4", B(7, 0), 4),         # fp32 bits[24:21]
  Field("dst", B(7, 4), 4)),        # destination register (probes/movimm.metal: r0 -> 0; unit unverified beyond r0)
  doc="dst = fp32 immediate")
NOP = Inst("nop", bytes.fromhex("0600"))
STOP = Inst("stop", bytes.fromhex("0e000000"))
GET_TID = Inst("get_tid", bytes.fromhex("1ca01006"), doc="read thread_position_in_grid (prologue)")
BARRIER = Inst("barrier", bytes.fromhex("110000901100"), doc="end-of-kernel barrier (epilogue, followed by stop)")

TABLE = (LOAD, STORE, WAIT, FALU_LONG, FALU, MOVIMM, GET_TID, BARRIER, STOP, NOP)

def movimm_fields(f:float) -> dict[str, int]:
  v = struct.unpack("<I", struct.pack("<f", f))[0]
  return dict(hi7=v >> 25, mid21=v & 0x1FFFFF, lo4=(v >> 21) & 0xf)
def movimm_value(d:dict[str, int]) -> float:
  return struct.unpack("<f", struct.pack("<I", (d["hi7"] << 25) | (d["lo4"] << 21) | d["mid21"]))[0]

# ---- disassembler -----------------------------------------------------------------------------------------------------------------
def fmt(inst:Inst, d:dict[str, int]) -> str:
  if inst is LOAD: return f"load r{d['dst']}, {d['slot']}" + (" ; first" if d["first"] else "") + (" ; more" if d["more"] else "")
  if inst is STORE: return f"store r{d['src']}, {d['slot']}" + (" ; first" if d["first"] else "") + (" ; hint" if d["hint"] else "")
  if inst is FALU or inst is FALU_LONG:
    op, dst, a = "fmul" if d["mul"] else "fadd", f"r{d['dst']}", f"r{operand_reg(d['srca'])}"
    flags = "".join(f" ; {n}" for n in ("fma", "killa", "killb", "fwd") if d[n]) + (f" ; tag {d['tag']:02x}" if inst is FALU_LONG and d["tag"] != 0xc0 else "")
    if inst is FALU_LONG and d["ext"] & 0x80: return f"{op} {dst}, {a}, #{'-' if d['killb'] else ''}{e4m3_value(d['srcb']):g}{flags}"
    return f"{op} {dst}, {a}, r{operand_reg(d['srcb'])}{flags}"
  if inst is MOVIMM: return f"movimm r{d['dst']}, #{movimm_value(d):g}"
  return inst.mnemonic

def disassemble(code:bytes, entry:int=0x40) -> list[tuple[int, bytes, str]]:
  """(offset, bytes, text) per instruction. Unknown 16-bit words become .short; decoding starts at `entry` (the preamble is opaque)."""
  out, pc = [], 0
  if entry: out.append((0, code[:entry], f".preamble {entry} bytes")); pc = entry
  while pc < len(code):
    for inst in TABLE:
      if inst.matches(code, pc):
        out.append((pc, code[pc:pc + inst.size], fmt(inst, inst.decode(code, pc)))); pc += inst.size; break
    else:
      out.append((pc, code[pc:pc + 2], f".short 0x{int.from_bytes(code[pc:pc + 2], 'little'):04x}")); pc += 2
  return out

def disasm_str(code:bytes, entry:int=0x40) -> str:
  return "\n".join(f"{off:04x}: {bs.hex(' '):<24} {txt}" for off, bs, txt in disassemble(code, entry))
