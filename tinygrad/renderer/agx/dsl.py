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
def dst_fields(r:int) -> dict[str, int]: # 6-bit ALU destination r0..r63 split into the byte-0 nibble and the mod-byte extension
  assert 0 <= r < 64, f"ALU destination r{r} out of range (r0..r63)"
  return dict(dst=r & 0xf, dst_hi=r >> 4)
def dst_reg(d:dict[str, int]) -> int: return d["dst"] | d.get("dst_hi", 0) << 4
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

REGISTER_FILE = """128 32-bit registers, r0..r127 (hardware-verified 2026-09-14, M3):
  - the prologue leaves the thread index in r1; everything else is free.
  - float ALU destinations (FALU, FMA, MOVIMM) reach r63: nibble in byte 0 plus DST_HI in the mod byte. Apple's compiler never writes
    above r63 with these forms and spills instead (probes/live96.metal declares 96 registers, all its fadd destinations are <= r63).
  - load/store/addr fields are 7 bits; loads into r64..r125 work.
  - HAZARD: r126 and r127. As the first kernel of a process a load into either returns 0 or stale data (they worked only after a
    warm-up kernel; declaring 132 registers did not help). The display froze minutes after those runs, the project's second freeze,
    the first being an undeclared register count (asm.gen_metadata). Never emit r126 or r127. Apple's largest observed count is 96.
  - the metadata register count (fn.f0) must cover every register named; Apple writes max register + 1..+7; we declare min(128, 2*(max+1))."""

# ---- the table --------------------------------------------------------------------------------------------------------------------

# loads and stores index a buffer slot by a 32-bit element offset register. The "plain" form Apple emits for buf[t] is areg = r1, the thread
# index the prologue leaves there. byte5 bit7 is areg's kill bit (KILL_MODEL), which an earlier reading mistook for an addressing-mode bit.
LOAD = Inst("load", bytes.fromhex("6700440000002000"), (
  Field("first", B(1, 4), 1),       # set on the first load of the kernel
  Field("more", B(2, 4), 1),        # another load follows (scoreboard)
  Field("dst", B(3, 1), 7),         # destination register
  Field("slot", B(4), 8),           # dense buffer binding index
  Field("areg", B(5, 0), 7),        # element offset register (32-bit)
  Field("killa", B(5, 7), 1)),      # areg dies after this load
  doc="dst = buf[slot][areg] (32-bit)")
LOAD_IDX = LOAD # old name
# element offset arithmetic on the thread index, which the prologue leaves in r1 (byte 4 = 02 names it; keep r1 reserved).
# shift-add: dst = (r1 << shift) + (imm | rB). shift bits: byte7 bit6 = bit0, byte9 bit4 = bit1, both zero means 4; byte8 bit0 = no shift.
# byte9 bit0: byte5 is a register (unit: field<<1 | 1 for 32-bit) instead of imm<<1. All hardware-verified (test_agx); byte7 bit5 does nothing.
ADDR = Inst("addr", bytes.fromhex("9f115400020000881004"), (
  Field("dst", B(3, 1), 7),
  Field("b", B(5), 8),              # imm<<1, or (reg<<2 | 1) when breg is set
  Field("a", B(6, 3), 5),           # register A (hardware-verified with r1 = thread index and r4 = a loaded value)
  Field("shift0", B(7, 6), 1),
  Field("noshift", B(8, 0), 1),
  Field("breg", B(9, 0), 1),
  Field("shift1", B(9, 4), 1)),
  doc="dst = (a << shift) + b: integer shift-add for address math. b as a register is only safe for small non-negative values (IADD doc)")
# multiply: dst = r1 * stride. 12-byte form, stride in bits [15:2] (probes: 7t -> 1c 00, 100t -> 90 01); the low two bits are ignored.
IMUL = Inst("imul", bytes.fromhex("9f1054000204000050200200"), (
  Field("dst", B(3, 1), 7),
  Field("stride", B(6, 2), 14)),
  doc="dst = r1 * stride (elements)")
# counted loop (probes/loop*.metal, hardware-verified in test_agx). Layout the assembler emits:
#   LOOP_INIT c=0 | L: LOOP_INC c += step | LOOP_CMP c, n | NOP | LOOP_ENTER | body | LOOP_TAIL | BRANCH L | LOOP_EXIT
# the loop exits when c == n after the increment, so the body sees c = step, 2*step, ..., n. n must be even in LOOP_CMP; LOOP_ENTER's
# `extra` bit adds one iteration for odd counts. Counts of 0 and 1 in LOOP_CMP hang the GPU.
LOOP_INIT = Inst("loop_init", bytes.fromhex("0b0020c0"), (Field("dst", B(0, 4), 4),), doc="counter = 0; carries the load wait (tail c0)")
LOOP_INC = Inst("loop_inc", bytes.fromhex("9f01040003000088 1504".replace(" ", "")), (
  Field("dst", B(3, 1), 7), Field("step", B(5, 1), 7), Field("src", B(6, 3), 5)),   # src: counter register << 3 (r3 -> 18, r6 -> 30)
  doc="counter = counter + step")
LOOP_CMP = Inst("loop_cmp", bytes.fromhex("0a002380"), (Field("depth2", B(0, 4), 4), Field("src", B(1), 8), Field("n", B(3, 1), 6)),
  doc="compare counter with n (even). src: (reg<<1)|1; bit0 of byte3 ignored; depth2 = 2 * loop nesting depth (probes/mm_shape.metal)")
LOOP_ENTER = Inst("loop_enter", bytes.fromhex("07020000"), (Field("extra", B(0, 4), 1),), doc="loop control; extra = one more iteration")
LOOP_TAIL = Inst("loop_tail", bytes.fromhex("8f045422"), (Field("depth4", B(3, 2), 6),), doc="loop control before the back branch; byte3 = 0x22 + 4 * nesting depth")
BRANCH = Inst("branch", bytes.fromhex("0f00540000000000ff00"), (Field("off", B(3), 40),), doc="pc += off (relative to this instruction's address); 10 bytes")
LOOP_EXIT = Inst("loop_exit", bytes.fromhex("0f0604020000"), doc="after the loop; stores do not land without it")
STORE = Inst("store", bytes.fromhex("e700540000002000"), (
  Field("first", B(1, 4), 1),       # set when no load preceded (probes/movimm.metal)
  Field("wait", B(2, 1), 1),        # wait for outstanding loads first (0x56). Required on the first consumer of load data, see LOAD_WAIT_MODEL
  Field("src", B(3, 1), 7),         # source register
  Field("slot", B(4), 8),
  Field("areg", B(5, 0), 7),        # element offset register (32-bit); r1 = thread index for buf[t]
  Field("killa", B(5, 7), 1),       # areg dies after this store
  Field("last", B(6, 0), 1)),       # Apple sets it on a kernel's final store only; no observable effect on hardware (probes/two_stores.metal)
  doc="buf[slot][areg] = src (32-bit). Must be followed by STORE_WAIT before the next store")
STORE_IDX = STORE # old name
WAIT = Inst("wait", bytes.fromhex("510100404600"), doc="wait for one outstanding load")
LOAD_WAIT_MODEL = """Loads are asynchronous and the synchronization is explicit, in the consumer (hardware-verified, test_agx):
  1. every load is followed by WAIT (51 01 00 40 46 00); it must directly follow its load.
  2. the first consumer of loaded data must be a waiting form: STORE with wait=1 (0x56) or the six-byte FALU_LONG with tail 00 c0.
     Non-waiting consumers (four-byte FALU, STORE 0x54) read stale registers (zeros) if they come first.
  3. one waiting consumer covers all outstanding loads: after it, four-byte FALU and 0x54 stores are fine (Apple emits exactly this).
Apple's later six-byte ops carry tails 00 20 / 00 40; those do not work as the first consumer and are not understood."""
KILL_MODEL = """Operand kill bits are real semantics, not hints (hardware-verified 2026-09-14): after an op with a source's kill bit set, that
register reads as 0. Apple's compiler sets them on the last use of every value (register release). Ours: the assembler sets none by default; a
value read twice (fadd r4, r0, r2 / fadd r6, r0, r2) came back 0 while every kill bit was set. Kill bits found so far: FALU/FMA killa, killb;
FMA c_hi (its register c); LOAD/STORE byte5 bit7 (areg); IADD/ISUB/INEG byte8 bits 1,2; IMUL_RR byte9 bits 1,2; the 0x0b bitwise and 0x02
min/max/select/compare families byte2 bits 3,4 (FALU layout), ISEL c byte4 bit7 and e byte8 bit7; 12-byte shifts byte9 bits 0 (value) and 1
(count); 10-byte arithmetic shifts byte7 bits 6 (value) and 7 (count). Every probe-derived template must be checked
by running the op twice on the same operands. The old renderer only survived because every register was read exactly once."""
# float ALU. 4-byte form: [09 | dst<<4] [srcb] [mod] [srca]. mod bit2 adds the two-byte tail (FALU_LONG); mod bit1 adds an fma third operand.
# destination registers: the 4-bit nibble in byte 0 holds bits [3:0]; bits [5:4] sit in the mod byte's top two bits (probes/live24, live48:
# fadd r21 = 59 b1 54, r40 = 89 b1 94; Apple never writes above r63 with these forms, even with 96 values live). Same split for MOVIMM.
DST_HI = Field("dst_hi", B(2, 6), 2)
FALU = Inst("falu", bytes.fromhex("09000000"), (
  Field("dst", B(0, 4), 4),         # destination register bits [3:0] (probes/sums_live.metal)
  DST_HI,
  Field("srcb", B(1), 8),           # register operand, or E4M3 immediate when bmode has bit 7
  Field("mul", B(2, 0), 1),         # 0 = fadd, 1 = fmul (probes/fmul.metal); mod bit1 (fixed 0 here) selects FMA
  Field("killb", B(2, 3), 1),       # srcb dies after this op: it reads as 0 afterwards (KILL_MODEL)
  Field("killa", B(2, 4), 1),       # srca dies after this op
  Field("fwd", B(2, 5), 1),         # result feeds another ALU op rather than a store
  Field("srca", B(3), 8)),
  doc="fadd32/fmul32: dst = srca op srcb. Non-waiting form: only valid after a waiting consumer (LOAD_WAIT_MODEL)")
FALU_LONG = Inst("falu", bytes.fromhex("090004000000"), tuple(FALU.fields) + (Field("ext", B(4), 8), Field("tag", B(5), 8)),
  doc="six-byte float ALU: byte4 = 0, or 0x80 for an immediate srcb; tail byte 0xc0 = wait for outstanding loads (LOAD_WAIT_MODEL)")
FALU_IMM_NEG, FALU_IMM_ADD = 1, 0    # for an E4M3 immediate srcb the sign lives in killb: 0x14 = add, 0x1c = negate (prog_add vs probes)
# fused multiply-add, 8 bytes: dst = (+-)(srca * srcb) + (+-)c. Hardware-verified (test_agx) for register/immediate c, negated c,
# negated product, immediate srca, and dst != r0. byte4: bit5 = c is an E4M3 immediate, bit4 = negate register c, bit7 = kill register c
# (Apple sets it on a last use, and on a negated immediate c; probes/mul_add.metal and friends). byte7: bit3 = negate the product, bit1 = srca is an immediate.
FMA = Inst("fma", bytes.fromhex("090006000100 02c0"), (
  Field("dst", B(0, 4), 4), DST_HI,
  Field("srcb", B(1), 8),
  Field("killb", B(2, 3), 1), Field("killa", B(2, 4), 1), Field("fwd", B(2, 5), 1),
  Field("srca", B(3), 8),
  Field("c_hi", B(4, 7), 1),        # register c: its kill bit (KILL_MODEL; hardware-verified: c survives with 0). Apple also sets it on a negated immediate c
  Field("c_imm", B(4, 5), 1), Field("c_neg", B(4, 4), 1),
  Field("c", B(5), 8),              # register << 1, or E4M3 immediate without its low bit
  Field("negp", B(7, 3), 1), Field("aimm", B(7, 1), 1)),
  doc="dst = srca * srcb + c")
# ---- integer ALU (probes/int/*.metal, 2026-09-14; every form hardware-verified with a in r0, b in r2, plus re-reads for the kill bits) ----
# add/sub/mul are the ADDR family (10 or 12 bytes, 7-bit dst in byte 3, b in byte 5, a in byte 6 bits 3..7). Opcode byte 0x1f subtracts: dst = b - a.
# int add reg-reg and reg-imm are ADDR with shift 0. Apple's iadd (9f 01 56 ...) differs from ADDR only in flag bits we do not need.
# byte 7 bit 5 = b is a 32-bit register (without it -3 read as 65533, so ADDR's register form is 16-bit); byte 8 bits 1,2 = operand kill bits
# (Apple's 17: both operands died; re-reading them gave 0). We emit a8 11.
IADD = Inst("iadd", bytes.fromhex("9f015600020000a81105"), (
  Field("dst", B(3, 1), 7), Field("breg", B(5, 2), 6), Field("a", B(6, 3), 5), Field("killb", B(8, 1), 1), Field("killa", B(8, 2), 1)),
  doc="dst = a + b, 32-bit. Use this, not ADDR's register form, for values")
ISUB = Inst("isub", bytes.fromhex("1f01560002000 0a81105".replace(" ", "")), tuple(IADD.fields), doc="dst = b - a")
INEG = Inst("ineg", bytes.fromhex("1f015600020000881104"), (Field("dst", B(3, 1), 7), Field("a", B(6, 3), 5), Field("killa", B(8, 2), 1)),
  doc="dst = 0 - a (ISUB with immediate b = 0; Apple's 15 kills a)")
IMUL_RR = Inst("imul", bytes.fromhex("9f00560002000000d0200a00"), (
  Field("dst", B(3, 1), 7), Field("breg", B(5, 2), 6), Field("a", B(6, 3), 5), Field("killb", B(9, 1), 1), Field("killa", B(9, 2), 1)),
  doc="dst = a * b (32-bit); IMUL is the thread-index-times-stride form. Apple's byte 9 = 26: both operands killed")
# shifts (probes/int/ishl_imm*, ishr_imm*, ushr_imm*, *_reg): a 12-byte family. byte 0: 27 = left, a7 = right; byte 1 bit 0 = arithmetic, bit 4 = count in
# a register; byte 5 = value register << 2; byte 6 = count << 2 (immediate) or count register << 3; byte 9 bit 0 kills the value, bit 1 the count.
# Counts of 32 and up give 0 (Apple masks with & 31 first). Apple emits shift-add pairs for small left shifts and half-register moves for 16.
_SHF = (Field("dst", B(3, 1), 7), Field("a", B(5, 2), 6), Field("killa", B(9, 0), 1))
SHL_IMM  = Inst("ishl", bytes.fromhex("270056000200000 0f0100100".replace(" ", "")), _SHF + (Field("n", B(6, 2), 5),), doc="dst = a << n")
LSHR_IMM = Inst("ushr", bytes.fromhex("a70056000200000 0f0100100".replace(" ", "")), _SHF + (Field("n", B(6, 2), 5),), doc="dst = a >> n, logical")
# arithmetic right by an immediate is Apple's 10-byte form: its kill bit is byte 7 bit 6 (78 -> 38). The 12-byte a7 01 layout only worked with its kill set.
ASHR_IMM = Inst("ishr", bytes.fromhex("a70156000200003862 00".replace(" ", "")), (Field("dst", B(3, 1), 7), Field("a", B(5, 2), 6), Field("n", B(6, 2), 5), Field("killa", B(7, 6), 1)),
  doc="dst = a >> n, arithmetic")
SHL_REG  = Inst("ishl", bytes.fromhex("271056000200000 0f0100500".replace(" ", "")), _SHF + (Field("breg", B(6, 3), 5), Field("killb", B(9, 1), 1)), doc="dst = a << b")
LSHR_REG = Inst("ushr", bytes.fromhex("a71056000200000 0f0100500".replace(" ", "")), _SHF + (Field("breg", B(6, 3), 5), Field("killb", B(9, 1), 1)), doc="dst = a >> b, logical")
ASHR_REG = Inst("ishr", bytes.fromhex("a7115600020000 18e200".replace(" ", "")), (Field("dst", B(3, 1), 7), Field("a", B(5, 2), 6), Field("breg", B(6, 3), 5),
  Field("killa", B(7, 6), 1), Field("killb", B(7, 7), 1)), doc="dst = a >> b, arithmetic (10-byte form like ASHR_IMM; Apple: d8 = both killed)")
SHIFTS = {("ishl", False): SHL_IMM, ("ushr", False): LSHR_IMM, ("ishr", False): ASHR_IMM, ("ishl", True): SHL_REG, ("ushr", True): LSHR_REG, ("ishr", True): ASHR_REG}
# converts (probes/cvt/*.metal): source register in byte 5 << 2, 7-bit dst in byte 3, byte 6 bit 5 kills the source (Apple's ac/b4 -> our 8c/94).
# int -> float is 8 bytes (byte 7 bit 6 = signed); float -> int is 10 bytes, truncating toward zero (-7.9 -> -7), byte 7 bit 6 / byte 8 bit 0 = signed.
_CVT = (Field("dst", B(3, 1), 7), Field("a", B(5, 2), 6), Field("killa", B(6, 5), 1))
I2F = Inst("i2f", bytes.fromhex("a70756000200 8c60".replace(" ", "")), _CVT, doc="dst = float(int32 a)")
U2F = Inst("u2f", bytes.fromhex("a70756000200 8c20".replace(" ", "")), _CVT, doc="dst = float(uint32 a)")
F2I = Inst("f2i", bytes.fromhex("270756000200 94480300".replace(" ", "")), _CVT, doc="dst = int32(a), truncating toward zero")
F2U = Inst("f2u", bytes.fromhex("270756000200 94080200".replace(" ", "")), _CVT, doc="dst = uint32(a), truncating")
CVTS = {"i2f": I2F, "u2f": U2F, "f2i": F2I, "f2u": F2U}
# the 0x0b bitwise family, 10 bytes: FALU-style operands (byte 1 = srcb, byte 3 = srca, dst nibble + DST_HI), byte 2 bits 3,4 = kills. Bytes 2, 4, 5
# select the function (Apple: and 1f/00 00, or 1f/02 08, xor 1e/02 08 with the kills set); byte 5 is not an operand (moving b left it unchanged).
_BIT = lambda name, hexs, doc: Inst(name, bytes.fromhex(hexs), (Field("dst", B(0, 4), 4), Field("srcb", B(1), 8), Field("killb", B(2, 3), 1), Field("killa", B(2, 4), 1),
                                                                 DST_HI, Field("srca", B(3), 8)), doc=doc)
IAND = _BIT("iand", "0b000700000000800000", "dst = srca & srcb")
IOR  = _BIT("ior",  "0b000700020800800000", "dst = srca | srcb")
IXOR = _BIT("ixor", "0b000600020800800000", "dst = srca ^ srcb")
# the 0x02 family: byte 2 bits 0..2 pick the form. 1e = min/max (6 bytes, byte 4: 06 max, 07 min), 1f = select (8 or 10 bytes), 1d = compare (8 bytes).
IMAX = Inst("imax", bytes.fromhex("020006000 6c0".replace(" ", "")), (Field("dst", B(0, 4), 4), Field("srcb", B(1), 8), Field("killb", B(2, 3), 1), Field("killa", B(2, 4), 1),
                                                                    DST_HI, Field("srca", B(3), 8)), doc="dst = max(srca, srcb), signed")
IMIN = Inst("imin", bytes.fromhex("0200060007c0"), tuple(IMAX.fields), doc="dst = min(srca, srcb), signed")
# select: dst = (srcb < srca) ? c : srcb (8 bytes), or ? c : e with byte 4 bit 1 and two more bytes (10). Kills: c byte 4 bit 7, e byte 8 bit 7.
# byte 4 bit 0 is required in the 8-byte form (cleared: wrong results); the 10-byte form has bit 1 instead
ISEL = Inst("isel", bytes.fromhex("0200070001000 7c0".replace(" ", "")), (Field("dst", B(0, 4), 4), Field("srcb", B(1), 8), Field("killb", B(2, 3), 1), Field("killa", B(2, 4), 1),
  DST_HI, Field("srca", B(3), 8), Field("killc", B(4, 7), 1), Field("c", B(5, 1), 7)),
  doc="dst = srcb < srca ? c : (else). Apple's 8-byte form; with the kill bit off the false case read 0, so the assembler only emits ISEL4")
ISEL4 = Inst("isel", bytes.fromhex("0200070002000 7c00000".replace(" ", "")), tuple(ISEL.fields) + (Field("kille", B(8, 7), 1), Field("e", B(9, 1), 7)),
  doc="dst = srcb < srca ? c : e (signed)")
# compare: dst = srcb OP srca as 0/1. The 8-byte compare must be followed by the 6-byte 20 8x 13 00 00 01 (its nibble does nothing; without it the
# register stays 0), so the table carries both as one 14-byte instruction. Codes (byte 4, byte 5 bit 0, byte 6, byte 9 bit 0):
CMP_CODES = {"lt": (0x22, 1, 7, 0), "gt": (0x22, 1, 6, 0), "ult": (0x22, 1, 5, 0), "eq": (0x26, 1, 7, 0), "ne": (0x26, 0, 7, 1)}
ICMP = Inst("icmp", bytes.fromhex("020005000080 00c0 2080 13000001".replace(" ", "")), (Field("dst", B(0, 4), 4), Field("srcb", B(1), 8), Field("killb", B(2, 3), 1),
  Field("killa", B(2, 4), 1), DST_HI, Field("srca", B(3), 8), Field("k4", B(4), 8), Field("k5", B(5, 0), 1), Field("code", B(6), 8), Field("inv", B(9, 0), 1)),
  doc="dst = (srcb OP srca) ? 1 : 0; OP from CMP_CODES (lt gt eq ne signed, ult unsigned)")
def cmp_fields(op:str) -> dict[str, int]:
  k4, k5, code, inv = CMP_CODES[op]; return dict(k4=k4, k5=k5, code=code, inv=inv)
def cmp_name(d:dict[str, int]) -> str: return next((n for n, v in CMP_CODES.items() if v == (d["k4"], d["k5"], d["code"], d["inv"])), f"?{d['k4']:02x}{d['k5']}{d['code']}{d['inv']}")

# 32-bit immediate, scattered (single-bit probes against 42.0 = 0x42280000, 2026-09-14; bits 0..8 come from Apple's 3.14159 encoding ac d0 42 40 1e 04 48 02,
# the compiler switches to another form when only low bits change). Earlier layout lost bits 0..12: movimm r26, #3.14159 stored 3.1420898.
MOVIMM = Inst("movimm", bytes.fromhex("0c80020000000000"), (
  Field("dst", B(0, 4), 4),         # destination register, top nibble of byte 0 like the float ALU (probe: two constants -> 2c.., 0c..)
  DST_HI,                           # probes/live24_const: r26 -> ac d0 42 (byte2 bit6)
  Field("v0_6", B(1, 0), 7),        # imm[6:0]
  Field("v25_31", B(3, 1), 7),      # imm[31:25]
  Field("v7_8", B(4, 1), 2),        # imm[8:7]
  Field("v9_10", B(4, 3), 2),       # imm[10:9]
  Field("v11_12", B(5, 2), 2),      # imm[12:11]
  Field("v13_20", B(6, 0), 8),      # imm[20:13]
  Field("v21_24", B(7, 0), 4)),     # imm[24:21]
  doc="dst = 32-bit immediate (any bit pattern; used for floats and for a loop counter's -1 start)")
_MOVIMM_SLICES = (("v0_6", 0, 7), ("v7_8", 7, 2), ("v9_10", 9, 2), ("v11_12", 11, 2), ("v13_20", 13, 8), ("v21_24", 21, 4), ("v25_31", 25, 7))
NOP = Inst("nop", bytes.fromhex("0600"))
STOP = Inst("stop", bytes.fromhex("0e000000"))
GET_TID = Inst("get_tid", bytes.fromhex("1ca01006"), doc="read thread_position_in_grid (prologue)")
STORE_WAIT = Inst("store_wait", bytes.fromhex("110000901100"),
  doc="wait for the preceding store. Apple emits one after every store; two stores back to back lose data without it (test_agx)")
BARRIER = STORE_WAIT # old name

TABLE = (LOAD, I2F, U2F, F2I, F2U, SHL_IMM, LSHR_IMM, ASHR_IMM, SHL_REG, LSHR_REG, ASHR_REG, IADD, ISUB, INEG, IMUL_RR, IAND, IOR, IXOR, IMAX, IMIN, ISEL4, ISEL, ICMP, IMUL, LOOP_INC, ADDR, LOOP_INIT, LOOP_CMP, LOOP_ENTER, LOOP_TAIL, BRANCH, LOOP_EXIT, STORE, WAIT, FMA, FALU_LONG, FALU, MOVIMM, GET_TID, STORE_WAIT, STOP, NOP)

def movimm_bits_fields(v:int) -> dict[str, int]: return {n: (v >> lo) & ((1 << w) - 1) for n, lo, w in _MOVIMM_SLICES}
def movimm_fields(f:float) -> dict[str, int]: return movimm_bits_fields(struct.unpack("<I", struct.pack("<f", f))[0])
def movimm_bits(d:dict[str, int]) -> int: return sum(d[n] << lo for n, lo, _ in _MOVIMM_SLICES)
def movimm_value(d:dict[str, int]) -> float: return struct.unpack("<f", struct.pack("<I", movimm_bits(d)))[0]

# ---- disassembler -----------------------------------------------------------------------------------------------------------------
def fmt(inst:Inst, d:dict[str, int]) -> str:
  if inst is LOAD: return f"load r{d['dst']}, {d['slot']}, r{d['areg']}" + "".join(f" ; {n}" for n in ("first", "more", "killa") if d[n])
  if inst is ADDR:
    sh = 0 if d["noshift"] else (d["shift0"] | d["shift1"] << 1) or 4
    return f"addr r{d['dst']}, r{d['a']}, {sh}, " + (f"r{d['b'] >> 2}" if d["breg"] else f"#{d['b'] >> 1}")
  if inst is IMUL: return f"imul r{d['dst']}, #{d['stride']}"
  if inst is LOOP_INIT: return f"loop_init r{d['dst']}"
  if inst is LOOP_INC: return f"loop_inc r{d['dst']}, r{d['src']}, #{d['step']}"
  if inst is LOOP_CMP: return f"loop_cmp r{d['src'] >> 1}, #{d['n'] << 1}" + (f" ; depth {d['depth2'] >> 1}" if d["depth2"] else "")
  if inst is LOOP_TAIL: return "loop_tail" + (f" ; depth {d['depth4'] - 8}" if d["depth4"] != 8 else "")
  if inst is LOOP_ENTER: return "loop_enter" + (" ; extra" if d["extra"] else "")
  if inst is BRANCH: return f"branch {d['off'] - (1 << 40) if d['off'] >> 39 else d['off']:+d}"
  if inst is STORE: return f"store r{d['src']}, {d['slot']}, r{d['areg']}" + "".join(f" ; {n}" for n in ("first", "wait", "killa", "last") if d[n])
  if inst is FALU or inst is FALU_LONG:
    op, dst, a = "fmul" if d["mul"] else "fadd", f"r{dst_reg(d)}", f"r{operand_reg(d['srca'])}"
    flags = "".join(f" ; {n}" for n in ("killa", "killb", "fwd") if d[n]) + (f" ; tag {d['tag']:02x}" if inst is FALU_LONG and d["tag"] != 0xc0 else "")
    if inst is FALU_LONG and d["ext"] & 0x80: return f"{op} {dst}, {a}, #{'-' if d['killb'] else ''}{e4m3_value(d['srcb']):g}{flags}"
    return f"{op} {dst}, {a}, r{operand_reg(d['srcb'])}{flags}"
  if inst is FMA:
    a = f"#{e4m3_value(d['srca'] | 1):g}" if d["aimm"] else f"r{operand_reg(d['srca'])}"
    c = f"#{'-' if d['c_hi'] else ''}{e4m3_value(d['c'] | 1):g}" if d["c_imm"] else f"{'-' if d['c_neg'] else ''}r{d['c'] >> 1}"
    return f"fma r{dst_reg(d)}, {'-' if d['negp'] else ''}{a}, r{operand_reg(d['srcb'])}, {c}" + "".join(f" ; {n}" for n in ("killa", "killb", "fwd") if d[n])
  if inst is MOVIMM: return f"movimm r{dst_reg(d)}, #{movimm_value(d):g}"
  if inst in CVTS.values(): return f"{inst.mnemonic} r{d['dst']}, r{d['a']}" + (" ; killa" if d["killa"] else "")
  if inst in SHIFTS.values(): return f"{inst.mnemonic} r{d['dst']}, r{d['a']}, " + (f"r{d['breg']}" if "breg" in d else f"#{d['n']}") + "".join(f" ; {n}" for n in ("killa", "killb") if d.get(n))
  if inst is IADD: return f"iadd r{d['dst']}, r{d['a']}, r{d['breg']}"
  if inst is ISUB: return f"isub r{d['dst']}, r{d['breg']}, r{d['a']}"
  if inst is INEG: return f"ineg r{d['dst']}, r{d['a']}"
  if inst is IMUL_RR: return f"imul r{d['dst']}, r{d['a']}, r{d['breg']}"
  if inst in (IAND, IOR, IXOR, IMAX, IMIN):
    return f"{inst.mnemonic} r{dst_reg(d)}, r{operand_reg(d['srca'])}, r{operand_reg(d['srcb'])}" + "".join(f" ; {n}" for n in ("killa", "killb") if d[n])
  if inst is ISEL or inst is ISEL4:
    e = f"r{d['e']}" if inst is ISEL4 else f"r{operand_reg(d['srcb'])}"
    return f"isel r{dst_reg(d)}, r{operand_reg(d['srcb'])}, r{operand_reg(d['srca'])}, r{d['c']}, {e}" + "".join(f" ; {n}" for n in ("killa", "killb", "killc", "kille") if d.get(n))
  if inst is ICMP: return f"icmp.{cmp_name(d)} r{dst_reg(d)}, r{operand_reg(d['srcb'])}, r{operand_reg(d['srca'])}" + "".join(f" ; {n}" for n in ("killa", "killb") if d[n])
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
