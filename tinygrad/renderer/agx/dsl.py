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
def reg_operand(r:int) -> int: return ((r << 2) | 1) & 0xff # 32-bit register N as a source operand: bits[7:1] = N<<1, bit0 = 32-bit width
def operand_reg(o:int) -> int: return o >> 2
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
FADD_ADD_IMM, FADD_NEG = 0x14, 0x1c # fadd modifier byte: add the immediate / negate-or-register srcB
STORE_RESULT = 0x54                 # store src byte naming the last fadd result

LOAD = Inst("load", bytes.fromhex("6700440000012000"), (
  Field("first", B(1, 4), 1),       # set on the first load of the kernel
  Field("more", B(2, 4), 1),        # another load follows (scoreboard)
  Field("dst", B(3, 2), 6),         # destination register
  Field("slot", B(4), 8)),          # dense buffer binding index
  doc="dst = buf[slot][thread_index] (32-bit)")
STORE = Inst("store", bytes.fromhex("e700000000012100"), (
  Field("src", B(2), 8),            # 0x54 = fadd result, else 0x56 for r0 or 0x54 + 2*reg
  Field("slot", B(4), 8)),
  doc="buf[slot][thread_index] = src (32-bit)")
WAIT = Inst("wait", bytes.fromhex("510100404600"), doc="wait for one outstanding load")
FALU = Inst("falu", bytes.fromhex("0900000000c0"), (
  Field("srcb", B(1), 8),           # register operand, or E4M3 immediate when bmode has bit 7
  Field("mul", B(2, 0), 1),         # 0 = fadd, 1 = fmul (probes/fmul.metal vs sum3.metal: the only differing bit)
  Field("mod", B(2, 1), 7),         # FADD_ADD_IMM / FADD_NEG, stored >> 1
  Field("srca", B(3), 8),
  Field("bmode", B(4), 8)),         # 0x80 = srcb is an immediate
  doc="fadd32/fmul32: result = srca op srcb (result register is implicit for now)")
FADD = FALU # the assembler's old name
MOVIMM = Inst("movimm", bytes.fromhex("0c80020000000000"), (
  Field("hi7", B(3, 1), 7),         # fp32 bits[31:25]
  Field("mid21", B(4, 3), 21),      # fp32 bits[20:0]
  Field("lo4", B(7, 0), 4),         # fp32 bits[24:21]
  Field("dst", B(7, 4), 4)),
  doc="dst = fp32 immediate")
NOP = Inst("nop", bytes.fromhex("0600"))
STOP = Inst("stop", bytes.fromhex("0e000000"))
GET_TID = Inst("get_tid", bytes.fromhex("1ca01006"), doc="read thread_position_in_grid (prologue)")
BARRIER = Inst("barrier", bytes.fromhex("110000901100"), doc="end-of-kernel barrier (epilogue, followed by stop)")

TABLE = (LOAD, STORE, WAIT, FALU, MOVIMM, GET_TID, BARRIER, STOP, NOP)

def movimm_fields(f:float) -> dict[str, int]:
  v = struct.unpack("<I", struct.pack("<f", f))[0]
  return dict(hi7=v >> 25, mid21=v & 0x1FFFFF, lo4=(v >> 21) & 0xf)
def movimm_value(d:dict[str, int]) -> float:
  return struct.unpack("<f", struct.pack("<I", (d["hi7"] << 25) | (d["lo4"] << 21) | d["mid21"]))[0]

# ---- disassembler -----------------------------------------------------------------------------------------------------------------
def fmt(inst:Inst, d:dict[str, int]) -> str:
  if inst is LOAD: return f"load r{d['dst']}, {d['slot']}" + (" ; first" if d["first"] else "") + (" ; more" if d["more"] else "")
  if inst is STORE:
    src = "result" if d["src"] == STORE_RESULT else f"r{0 if d['src'] == 0x56 else (d['src'] - STORE_RESULT) // 2}"
    return f"store {src}, {d['slot']}"
  if inst is FALU:
    op, a, mod = "fmul" if d["mul"] else "fadd", f"r{operand_reg(d['srca'])}", d["mod"] << 1
    if d["bmode"] & 0x80: return f"{op} {a}, #{'-' if mod == FADD_NEG else ''}{e4m3_value(d['srcb']):g}"
    return f"{op} {a}, r{operand_reg(d['srcb'])}"
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
