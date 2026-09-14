# Apple G15 (M3) GPU — ISA reference

Everything here is traceable to one of three places, and each claim says which:

| source | what it is | status |
|---|---|---|
| `tinygrad/renderer/agx/dsl.py` | the instruction table: templates, fields, bit offsets | **authoritative**; where this file and `QUIRKS.md` disagree, `dsl.py` wins |
| `tinygrad/runtime/support/agx/asm.py` | the assembler (`assemble_text`), the metadata builder, the Mach-O/archive container | authoritative for mnemonics and syntax |
| `test/device/test_agx.py` | one test per hardware behaviour | the proof that an encoding runs |
| `extra/agx/probes/*.metal` | the Metal kernels whose compiled output produced each encoding | the evidence trail |

`extra/agx/QUIRKS.md` is the companion document: war stories, freezes, hangs, and the measurement protocol.
This file is the reference. Anything **inferred but not run on hardware** is marked *(inferred)*.

Tooling: `python3 extra/agx/probe.py <file.metal>` compiles a Metal kernel with Apple's compiler and disassembles the
result through this table (`--diff a b`, `--hex`, `--dump out.bin`). `python3 extra/agx/metadata.py <file.metal>` prints
Apple's `__GPU_METADATA` function table.

---

## A. Overview

**The machine.** Apple G15, as shipped in the M3. Verified 2026-09-14 on macOS 15.

- **128 32-bit registers**, `r0`..`r127`. There is no separate float/int file; the same registers hold both.
  `r126`/`r127` are a hazard — see `dsl.REGISTER_FILE` and QUIRKS "Registers". Never emit them.
- **32-lane SIMD groups.** The matrix unit and the fragment layout are defined over exactly 32 lanes.
- **Threadgroups** of up to 1024 threads (`AGXRenderer.local_max`), with a threadgroup memory allocation declared in
  the kernel metadata. Dispatch dimensions are `(groups x, y, z, threads x, y, z)`.
- **Only 32-bit types are decoded** (`dsl.DTYPES`): float32, int32, uint32, and bool as a 32-bit 0/1. 16-bit values live
  in half registers with different opcodes; 64-bit needs register pairs. Neither is decoded.

**Instruction format.** Little-endian, variable length, 2 to 14 bytes. `dsl.Inst` models one instruction as a fixed
template plus `Field(name, bit offset from byte 0, width)` holes. `encode` and `decode` read the same table, so a byte
the assembler emits that the disassembler cannot read back is by construction a bug in our understanding.

Instruction identity is `Inst.care`: all template bits except the field holes. `dsl.TABLE` is searched in order, so
overlapping templates must be ordered most-specific first (`FALU_LONG` before `FALU`, `ISEL4` before `ISEL`).
Anything that matches nothing disassembles as `.short 0xNNNN`.

**Kernel image layout** (`asm.assemble_text`):

```
0x00  PREAMBLE_IDX   03 00 07 00 02 00 00 00 60 00 0e 00 00 00   (opaque; copied verbatim)
0x0e  nop padding    06 00 ...                                   (up to 0x40)
0x40  getsr r1, tid.x                        <- entry point, symbol _agc.main
      ...body...
      stop           0e 00 00 00
```

The prologue leaves `thread_position_in_grid.x` in `r1` and the assembler reserves it: the plain `buf[t]` addressing
form is the indexed form with the address register set to `r1`. The entry symbol is fixed at 0x40 in `build_inner`.

**Container.** `asm.build(program)` returns a fat Metal binary archive: an applegpu slice holding an object Mach-O with
our `__text` and `__GPU_METADATA`, plus an air64 metallib slice. `ops_agx` hooks Metal's binary-archive lookup so the
driver takes our compute object instead of compiling anything.

---

## B. Assembler reference

Syntax notes that apply throughout:

- One instruction per line. `;` starts a comment (`#` introduces an immediate, so it cannot be the comment character).
- Commas are whitespace: `iadd r4, r0, r2` and `iadd r4 r0 r2` are the same.
- Registers are `rN`. Immediates are `#N`, `#1.5`, or `#0xdeadbeef` for raw bits.
- `MAXREG` tracks the highest register named anywhere and is declared to the driver in the metadata. Multi-register
  forms (`load4`, `mma`) register their implied neighbours too.

### Directives

| directive | meaning |
|---|---|
| `.buffers N` | number of device buffers bound. Also raised implicitly by any `load`/`store` slot index. |
| `.tgmem BYTES` | threadgroup memory allocation. Required before any `tgload`/`tgstore`; drives metadata (§D). |

### Memory

| mnemonic | operands | semantics |
|---|---|---|
| `load` / `load2` / `load4` | `rD, slot [, rA [, #off]]` | `rD..rD+N-1 = buf[slot][rA + off ..]`, 32-bit elements. `rA` defaults to `r1`. |
| `store` / `store2` / `store4` | `rS, slot [, rA [, #off]]` | `buf[slot][rA + off ..] = rS..rS+N-1`. `rA` defaults to `r1`. |
| `wait` | — | accepted and ignored. Legacy: the wait word is part of the load now (§C2, `dsl.LOAD_WAIT_MODEL`). |

Offsets are in 32-bit elements, and the field widths differ (§C2): loads 13 bits, stores 11 bits. Larger constants must
go into the index register.

### Threadgroup memory

| mnemonic | operands | semantics |
|---|---|---|
| `tgload` / `tgload2` / `tgload4` | `rD, rA [, #off]` | `rD..rD+N-1 = threadgroup[rA + off ..]`, 32-bit elements. |
| `tgstore` / `tgstore2` / `tgstore4` | `rS, rA [, #off]` | `threadgroup[rA + off ..] = rS..rS+N-1`. |
| `barrier` | — | `threadgroup_barrier(mem_threadgroup)`. |

Threadgroup load offsets are 9 bits (max 511); store offsets are 11 bits (verified to 1024). The index register field is
6 bits, so `rA` must be `r0`..`r63`. Tested in `test_threadgroup_memory`.

### Address arithmetic

| mnemonic | operands | semantics |
|---|---|---|
| `addr` | `rD, [rA,] shift, #imm \| rB` | `rD = (rA << shift) + (imm \| rB)`; `rA` defaults to `r1`. `shift` 0..4. |
| `imul` | `rD, #stride` | `rD = r1 * stride` (the thread-index-times-stride form). |

`addr`'s register addend is 16-bit: `-3` read back as 65533. Use `iadd` for real 32-bit values (`dsl.ADDR` doc).
The `a` field is 5 bits, so `rA` must be `r0`..`r31`; the immediate is 7 bits (`imm << 1` in an 8-bit byte).

### Integer ALU

| mnemonic | operands | semantics | register limits |
|---|---|---|---|
| `iadd` | `rD, rA, rB` | `rD = rA + rB` | `rA` ≤ r31, `rB` ≤ r63, `rD` ≤ r127 |
| `isub` | `rD, rA, rB` | `rD = rA - rB` | same |
| `ineg` | `rD, rA` | `rD = -rA` | `rA` ≤ r31 |
| `imul` | `rD, rA, rB` | `rD = rA * rB` | `rA` ≤ r31, `rB` ≤ r63 |
| `iand` / `ior` / `ixor` | `rD, rA, rB` | bitwise | `rD` ≤ r63, sources ≤ r63 |
| `imax` / `imin` | `rD, rA, rB` | signed min/max | `rD` ≤ r63, sources ≤ r63 |

`ior rD, rA, rA` is the idiomatic register move: it copies bits and works for any 32-bit type.

### Shifts

| mnemonic | operands | semantics |
|---|---|---|
| `ishl` | `rD, rA, #n \| rB` | `rD = rA << count` |
| `ishr` | `rD, rA, #n \| rB` | arithmetic right shift |
| `ushr` | `rD, rA, #n \| rB` | logical right shift |

Immediate counts are 5 bits. Counts of 32 and above produce 0 — the hardware masks with `& 31` first
(`dsl` shift comment). Verified in `test_int_shifts`.

### Converts

| mnemonic | operands | semantics |
|---|---|---|
| `i2f` | `rD, rA` | `rD = float(int32 rA)` |
| `u2f` | `rD, rA` | `rD = float(uint32 rA)` |
| `f2i` | `rD, rA` | `rD = int32(rA)`, truncating toward zero (`-7.9 → -7`) |
| `f2u` | `rD, rA` | `rD = uint32(rA)`, truncating |

Verified in `test_converts`.

### Float ALU

| mnemonic | operands | semantics |
|---|---|---|
| `fadd` | `rD, rA, rB \| #imm` | `rD = rA + rB` |
| `fmul` | `rD, rA, rB \| #imm` | `rD = rA * rB` |
| `fma` | `rD, rA \| #imm, rB, rC \| -rC \| #imm` | `rD = rA * rB + rC` |
| `fmax` / `fmin` | `rD, rA, rB` | float32 min/max |
| `movimm` | `rD, #1.5` or `rD, #0xdeadbeef` | 32-bit immediate, any bit pattern |

Float immediates other than in `movimm` must be E4M3-representable (§C); anything else raises and needs `movimm`.
All float ALU destinations and sources are limited to `r0`..`r63` (§C, operand encodings).

### Compare and select

| mnemonic | operands | semantics |
|---|---|---|
| `icmp.lt` / `.gt` / `.ult` / `.eq` / `.ne` | `rD, rA, rB` | `rD = (rA OP rB) ? 1 : 0`, integer (`ult` unsigned) |
| `fcmp.lt` / `.gt` / `.le` / `.eq` / `.ne` | `rD, rA, rB` | same, float32 |
| `isel` | `rD, rA, rB, rC [, rE]` | `rD = (rA < rB) ? rC : rE`, integer compare; `rE` defaults to `rA` |
| `fsel` | `rD, rA, rB, rC [, rE]` | same with a float compare |

The assembler always emits the explicit-else 10-byte select (`ISEL4`); the 8-byte implicit-else form read 0 in the
false case unless its kill bit was set, so it is decoded but never emitted (`dsl.ISEL` doc).

### Transcendentals

| mnemonic | operands | semantics |
|---|---|---|
| `rcp` | `rD, rA` | `1 / rA`, hardware approximation |
| `rsqrt` | `rD, rA` | `1 / sqrt(rA)`; `rsqrt(0) = inf` |
| `rsqrts` | `rD, rA` | `1 / sqrt(rA)` but `rsqrts(0) = 1`, so `a * rsqrts(a)` is a `sqrt` with `sqrt(0) = 0` |
| `exp2` | `rD, rA` | `2 ** rA` |
| `log2` | `rD, rA` | `log2(rA)` |

All within 1 ulp of numpy over a wide range (`test_transcendental_tensors`). There is no hardware `sin`, `pow` or
`precise` reciprocal in the table; see §E.

### Matrix

| mnemonic | operands | semantics |
|---|---|---|
| `mma` | `rD, rA, rB [, rC]` | 8×8×8 fp32 `D = A * B [+ C]` across the SIMD group; every operand is a **register pair** `(r, r+1)` |

Each of the 32 lanes holds two consecutive elements of one row:

```
lane t holds row    ((t >> 1) & 3) + 4 * (t >> 4)
        columns   2 * (t & 1) + 4 * ((t >> 3) & 1)   and that + 1
```

This is exactly tinygrad's `tc.metal` swizzle. Established by `probes/simd/frag_elems.metal`, exercised in
`test_simdgroup_mma`. `C` is consumed unless `D == C` — accumulate in place (§D, kill model).

### Control flow

| mnemonic | operands | semantics |
|---|---|---|
| `loop` | `rC, #n` or `rC, rN` | opens a counted loop with counter `rC` |
| `body` | — | marks the end of the pre section and the start of the body (§D) |
| `endloop` | — | closes the loop |
| `if` | `rC` | the region runs per-thread where `rC != 0` |
| `endif` | — | closes `if` |
| `getsr` | `rD, name` | `rD =` a special register; `rD` ≤ r15 |

`getsr` names: `tid.x/y/z` (thread_position_in_grid), `gid.x/y/z` (threadgroup_position_in_grid),
`lid.x/y/z` (thread_position_in_threadgroup), `tpg.x/y/z` (threads_per_threadgroup). Verified in
`test_special_registers` under multi-threadgroup dispatch.

Immediate loop counts are 2..255; counts of 0 and 1 **hang the GPU**. Counts above 255 must use the register form.
The assembler rejects `if` inside a loop (`assert not loops`) because the depth nibble is unverified there.

### Escape

| mnemonic | operands | meaning |
|---|---|---|
| `raw` | `<hex bytes>` | emit bytes verbatim — for encodings under investigation only |

### Worked example

```
.buffers 3
.tgmem 1024
getsr r2, lid.x
addr r4, r2, 2, #0        ; r4 = lid.x * 4
load4 r8, 1, r4           ; r8..r11 = buf1[r4 .. r4+3]
tgstore4 r8, r4           ; threadgroup[r4 .. r4+3] = r8..r11
barrier
tgload2 r12, r4, #2       ; r12, r13 = threadgroup[r4+2], threadgroup[r4+3]
fadd r16, r12, r13
store r16, 0, r2
```

assembles and disassembles back to:

```
0040: 1c a0 10 06                                getsr r1, tid.x
0044: 2c a4 10 06                                getsr r2, lid.x
0048: 9f 11 54 08 02 00 10 88 10 14              addr r4, r2, 2, #0
0052: 67 10 44 10 01 04 20 00 57 01 00 40 46 00  load4 r8, 1, r4 ; first
0060: e7 02 56 10 10 10 00 00 5c 02 00 30 02 00  tgstore4 r8, r4 ; wait ; base 10
006e: 07 04 54 61 09 00                          barrier
0074: 67 02 54 18 10 10 00 00 64 0d 01 c0 08 00  tgload2 r12, r4, #2 ; base 10
0082: 09 1b 44 19 00 c0                          fadd r16, r12, r13
0088: e7 00 54 20 00 02 21 00 11 00 00 90 11 00  store r16, 0, r2 ; last
0096: 0e 00 00 00                                stop
```

Note the `fadd` came out as the 6-byte waiting form because a `tgload` was outstanding (§D).

---

## C. Encoding reference

### C.1 Operand encodings — the most error-prone part of the ISA

Six different register encodings are in use, and two of them appear **in the same instruction**. The table below gives
the *byte* value for register `r`, and the resulting reachable range.

| encoding | byte value | reach | used by |
|---|---|---|---|
| 8-bit source operand | `(r << 1) \| 1` | **r0..r63** — bit 7 is a flag, not a register bit | `FALU`/`FMA` `srca`/`srcb`, bitwise, min/max, select, compare, `IF_BEGIN.cond`, `LOOP_CMP.src` |
| split ALU destination | nibble in byte 0 bits 4..7, bits [5:4] in byte 2 bits 6..7 (`DST_HI`) | r0..r63 | `FALU`, `FALU_LONG`, `FMA`, `MOVIMM`, bitwise, min/max, select, compare |
| 7-bit destination at bit 1 | `r << 1` | r0..r127 | `LOAD.dst`, `STORE.src`, `ADDR.dst`, shifts, converts, transcendentals, `TG_LOAD.dst`, `TG_STORE.src`, `LOOP_INC.dst` |
| 6-bit source at bit 2 | `r << 2` | r0..r63 | shift `a`, convert `a`, transcendental `a`, `TG_LOAD`/`TG_STORE` `a`, `IADD`/`ISUB`/`IMUL_RR` `breg` |
| 5-bit source at bit 3 | `r << 3` | r0..r31 | `ADDR.a`, `IADD`/`ISUB`/`INEG`/`IMUL_RR` `a`, `LOOP_INC.src`, register shift counts |
| 7-bit raw | `r` | r0..r127 | `LOAD.areg`, `STORE.areg`, `MMA.dst` (8-bit field) |

Three things follow that cost real debugging time:

1. **An 8-bit source byte cannot name a register above r63.** `(70 << 1) | 1 = 0x8d` sets bit 7, which the hardware reads
   as a flag; `r70` as a float source read 0 (`dsl.REGISTER_FILE`). This is why the float ALU is confined to the low half
   of the register file even though the file is 128 deep.
2. **`addr`'s addend byte** is `imm << 1` for an immediate or `(r << 2) | 1` for a register (`asm.i_addr`), so immediates
   reach 127 and register addends reach r63 — but the register form is 16-bit arithmetic (§B).
3. **The matrix op uses three different encodings at once** (next section).

### C.2 The matrix operand encodings

| operand | field | byte value for `r` | note |
|---|---|---|---|
| `D` | byte 3, 8 bits | `r` | plain register number |
| `A` | byte 5 bits 1..7 | `r << 1` | bit 0 of the byte is 0 |
| `B` | byte 6 bits 1..7 | `r << 1` | bit 0 of the byte is 0 |
| `C` | byte 7, 8 bits | `mma_c(r)` = adjacent bit pairs swapped | `r4 → 0x08`, `r6 → 0x09`, `r8 → 0x04`, `r20 → 0x28` |

`dsl.mma_c(reg) = ((reg & 0x55) << 1) | ((reg & 0xaa) >> 1)`. It is an involution, so the same function decodes.
Found by stepping the field against loaded candidate registers on hardware; all four encodings are exercised in
`test_simdgroup_mma`.

### C.3 Immediate encodings

**E4M3 minifloat** (`dsl.e4m3`), used for float ALU immediate operands:

```
byte = ((e + 11) << 3 | m) << 1 | 1      value = 2 ** e * (1 + m / 8)      e in [-11, 4], m in [0, 7]
```

bit 0 is the 32-bit width flag; the **sign does not live in the byte** — it rides in the `killb` bit of the modifier byte
for `FALU`, and in `c_hi` for `FMA`'s third operand. Representable magnitudes run from 2⁻¹¹ (0.00048828125) to 30.0.
Zero is not representable; use `movimm`.

**32-bit immediate** (`MOVIMM`), scattered across seven fields. An earlier layout silently lost bits 0..12
(`3.14159` stored as `3.1420898`), so the slice table matters:

| bits | field | position |
|---|---|---|
| [6:0] | `v0_6` | byte 1 bits 0..6 |
| [8:7] | `v7_8` | byte 4 bits 1..2 |
| [10:9] | `v9_10` | byte 4 bits 3..4 |
| [12:11] | `v11_12` | byte 5 bits 2..3 |
| [20:13] | `v13_20` | byte 6 |
| [24:21] | `v21_24` | byte 7 bits 0..3 |
| [31:25] | `v25_31` | byte 3 bits 1..7 |

### C.4 Instruction templates

Field positions are written `byte.bit`. Every template below is `Inst.fixed` from `dsl.py` verbatim.

#### Memory

| mnemonic | size | fixed |
|---|---|---|
| `load` | 14 | `67 00 44 00 00 00 20 00 51 01 00 40 40 00` |
| `store` | 14 | `e7 00 54 00 00 00 20 00 11 00 00 10 10 00` |
| `tgload` | 14 | `67 02 54 00 00 00 00 00 44 0d 00 c0 08 00` |
| `tgstore` | 14 | `e7 02 54 00 00 00 00 00 44 02 00 00 02 00` |
| `barrier` | 6 | `07 04 54 61 09 00` |

See §C2 below for the full field breakdown of the four 14-byte forms.

#### Address arithmetic

| mnemonic | size | fixed | fields |
|---|---|---|---|
| `addr` | 10 | `9f 11 54 00 02 00 00 88 10 04` | `dst` 3.1 w7; `b` 5.0 w8; `a` 6.3 w5; `shift0` 7.6; `noshift` 8.0; `breg` 9.0; `shift1` 9.4 |
| `imul` (stride) | 12 | `9f 10 54 00 02 04 00 00 50 20 02 00` | `dst` 3.1 w7; `stride` 6.2 w14 |

Shift amount is split: `shift0` is bit 0, `shift1` is bit 1, both zero means **4**, and `noshift` forces 0.

#### Integer ALU

| mnemonic | size | fixed | fields |
|---|---|---|---|
| `iadd` | 10 | `9f 01 56 00 02 00 00 a8 11 05` | `dst` 3.1 w7; `breg` 5.2 w6; `a` 6.3 w5; `killb` 8.1; `killa` 8.2 |
| `isub` | 10 | `1f 01 56 00 02 00 00 a8 11 05` | same as `iadd`; computes `b - a` |
| `ineg` | 10 | `1f 01 56 00 02 00 00 88 11 04` | `dst` 3.1 w7; `a` 6.3 w5; `killa` 8.2 |
| `imul` (reg) | 12 | `9f 00 56 00 02 00 00 00 d0 20 0a 00` | `dst` 3.1 w7; `breg` 5.2 w6; `a` 6.3 w5; `killb` 9.1; `killa` 9.2 |
| `iand` | 10 | `0b 00 07 00 00 00 00 80 00 00` | `dst` 0.4 w4 + `dst_hi` 2.6 w2; `srcb` 1.0 w8; `killb` 2.3; `killa` 2.4; `srca` 3.0 w8 |
| `ior` | 10 | `0b 00 07 00 02 08 00 80 00 00` | same shape |
| `ixor` | 10 | `0b 00 06 00 02 08 00 80 00 00` | same shape |
| `imax` | 6 | `02 00 06 00 06 c0` | same shape |
| `imin` | 6 | `02 00 06 00 07 c0` | same shape |
| `fmax` | 6 | `02 00 06 00 00 c0` | same shape |
| `fmin` | 6 | `02 00 06 00 01 c0` | same shape |

The `0x0b` bitwise family and the `0x02` min/max family share the `FALU` operand layout. In the `0x02` family byte 2
bits 0..2 pick the form: `1e` min/max, `1f` select, `1d` compare (`dsl` comment).

#### Shifts

| mnemonic | size | fixed | fields |
|---|---|---|---|
| `ishl` imm | 12 | `27 00 56 00 02 00 00 00 f0 10 01 00` | `dst` 3.1 w7; `a` 5.2 w6; `n` 6.2 w5; `killa` 9.0 |
| `ushr` imm | 12 | `a7 00 56 00 02 00 00 00 f0 10 01 00` | same |
| `ishr` imm | 10 | `a7 01 56 00 02 00 00 38 62 00` | `dst` 3.1 w7; `a` 5.2 w6; `n` 6.2 w5; `killa` 7.6 |
| `ishl` reg | 12 | `27 10 56 00 02 00 00 00 f0 10 05 00` | `dst`; `a`; `breg` 6.3 w5; `killa` 9.0; `killb` 9.1 |
| `ushr` reg | 12 | `a7 10 56 00 02 00 00 00 f0 10 05 00` | same |
| `ishr` reg | 10 | `a7 11 56 00 02 00 00 18 e2 00` | `dst`; `a`; `breg` 6.3 w5; `killa` 7.6; `killb` 7.7 |

Byte 0 picks direction (`27` left, `a7` right); byte 1 bit 0 selects arithmetic, bit 4 selects a register count.
Arithmetic right shift uses Apple's 10-byte layout in both forms — the 12-byte `a7 01` layout only worked with its kill
bit set, so it is not used.

#### Converts

| mnemonic | size | fixed | fields |
|---|---|---|---|
| `i2f` | 8 | `a7 07 56 00 02 00 8c 60` | `dst` 3.1 w7; `a` 5.2 w6; `killa` 6.5 |
| `u2f` | 8 | `a7 07 56 00 02 00 8c 20` | same |
| `f2i` | 10 | `27 07 56 00 02 00 94 48 03 00` | same |
| `f2u` | 10 | `27 07 56 00 02 00 94 08 02 00` | same |

Byte 7 bit 6 is the signedness bit in the int→float direction; float→int also uses byte 8 bit 0 (`dsl` comment).

#### Transcendentals

All five share one 10-byte layout in the convert style. Byte 0 bit 7 and byte 1 pick the function.

| mnemonic | fixed | kill bit |
|---|---|---|
| `rcp` | `af 00 54 00 02 00 00 48 20 00` | byte 6 **bit 4** |
| `rsqrt` | `af 01 54 00 02 00 90 40 00 00` | byte 6 bit 5 |
| `rsqrts` | `2f 01 54 00 02 00 90 40 00 00` | byte 6 bit 5 |
| `exp2` | `af 02 54 00 02 00 90 40 00 00` | byte 6 bit 5 |
| `log2` | `2f 02 54 00 02 00 90 40 00 00` | byte 6 bit 5 |

Common fields: `wait` 2.1; `dst` 3.1 w7; `hint` 4.0; `a` 5.2 w6; `killa` 6.4 or 6.5; `h` 6.1.
`hint` is set by Apple whenever `dst != r0` and had no observable effect; `h` is clear in Apple's killing forms and set
otherwise, also with no effect — both were tested on hardware.

`2f 03` is the tail of Apple's hardware `sin` and is **not** in the table (§E).

#### Float ALU

| mnemonic | size | fixed | fields |
|---|---|---|---|
| `falu` (4-byte) | 4 | `09 00 00 00` | `dst` 0.4 w4 + `dst_hi` 2.6 w2; `srcb` 1.0 w8; `mul` 2.0; `killb` 2.3; `killa` 2.4; `fwd` 2.5; `srca` 3.0 w8 |
| `falu` (6-byte) | 6 | `09 00 04 00 00 00` | the above plus `ext` 4.0 w8, `tag` 5.0 w8 |
| `fma` | 8 | `09 00 06 00 01 00 02 c0` | `dst`+`dst_hi`; `srcb` 1.0; `killb` 2.3; `killa` 2.4; `fwd` 2.5; `srca` 3.0; `c_neg` 4.4; `c_imm` 4.5; `c_hi` 4.7; `c` 5.0 w8; `aimm` 7.1; `negp` 7.3 |
| `movimm` | 8 | `0c 80 02 00 00 00 00 00` | `dst` 0.4 w4 + `dst_hi` 2.6 w2; the seven immediate slices of §C.3 |

`mul` selects `fmul` over `fadd`. In the 6-byte form `ext = 0x80` marks an immediate `srcb`, and `tag = 0xc0` is the
waiting form (§D). `FMA`'s `c_hi` doubles as the kill bit for a register `c` and as the sign of a negated immediate `c`.

#### Compare and select

| mnemonic | size | fixed | fields |
|---|---|---|---|
| `isel` (8-byte) | 8 | `02 00 07 00 01 00 07 c0` | FALU operand shape; `killc` 4.7; `c` 5.1 w7 |
| `isel` (10-byte) | 10 | `02 00 07 00 02 00 00 c0 00 00` | the above plus `code` 6.0 w8; `kille` 8.7; `e` 9.1 w7 |
| `icmp` | 14 | `02 00 05 00 00 80 00 c0 20 80 13 00 00 01` | FALU operand shape; `k4` 4.0 w8; `k5` 5.0; `code` 6.0 w8; `inv` 9.0 |

`SEL_CODES`: `isel` = 7 (signed int compare), `fsel` = 3 (float compare).

The compare is really an 8-byte instruction that **must** be followed by the 6-byte word `20 8x 13 00 00 01`, or the
destination stays 0, so the table carries both as one 14-byte instruction. The comparison is chosen by four fields:

| name | `k4` | `k5` | `code` | `inv` |
|---|---|---|---|---|
| `lt` | 0x22 | 1 | 7 | 0 |
| `gt` | 0x22 | 1 | 6 | 0 |
| `ult` | 0x22 | 1 | 5 | 0 |
| `eq` | 0x26 | 1 | 7 | 0 |
| `ne` | 0x26 | 0 | 7 | 1 |
| `flt` | 0x22 | 1 | 3 | 0 |
| `fgt` | 0x22 | 1 | 2 | 0 |
| `fle` | 0x26 | 1 | 3 | 0 |
| `feq` | 0x26 | 1 | 0 | 0 |
| `fne` | 0x26 | 0 | 0 | 1 |

#### Matrix

| mnemonic | size | fixed | fields |
|---|---|---|---|
| `mma` | 12 | `cf 02 54 00 00 00 00 00 54 40 24 00` | `wait` 2.1; `dst` 3.0 w8; `a` 5.1 w7; `b` 6.1 w7; `c` 7.0 w8; `f8` 8.7; `f9` 9.0; `creg` 9.1; `acc` 11.0 |

`acc` (byte 11 bit 0) enables the accumulate; on its own it is sufficient. `creg` (byte 9 bit 1) says the `c` byte names a
register — with it clear the byte is a small immediate (`0x08` → `+0.125`). `f8` and `f9` are set in Apple's code on last
uses but **did not** zero `A` or `B` when tested, so the assembler never sets them.

#### Control flow

| mnemonic | size | fixed | fields |
|---|---|---|---|
| `loop_init` | 4 | `0b 00 20 c0` | `dst` 0.4 w4 (counter, r0..r15) |
| `loop_inc` | 10 | `9f 01 04 00 03 00 00 88 15 04` | `dst` 3.1 w7; `step` 5.1 w7; `src` 6.3 w5 |
| `loop_cmp` | 4 | `0a 00 23 00` | `depth2` 0.4 w4; `src` 1.0 w8; `n_hi` 2.4; `nreg` 3.0; `n` 3.1 w7 |
| `loop_enter` | 4 | `07 00 00 00` | `extra` 0.4; `imm` 1.1 |
| `loop_tail` | 4 | `8f 04 54 22` | `depth4` 3.2 w6 |
| `branch` | 10 | `0f 00 54 00 00 00 00 00 ff 00` | `off` 3.0 w40, signed, relative to this instruction |
| `loop_exit` | 6 | `0f 06 04 02 00 00` | — |
| `if` | 14 | `0a 00 2b 80 06 00 07 c2 00 00 0f 05 54 21` | `cond` 1.0 w8 |
| `endif` | 6 | `0f 06 04 01 00 00` | — |
| `getsr` | 4 | `0c 00 10 06` | `dst` 0.4 w4; `sr` 1.0 w8 |
| `nop` | 2 | `06 00` | — |
| `stop` | 4 | `0e 00 00 00` | — |

Special register numbers (`dsl.SR`): `tpg.x/y/z` = 0x98/0x99/0x9a, `gid.x/y/z` = 0x9c/0x9d/0x9e,
`tid.x/y/z` = 0xa0/0xa1/0xa2, `lid.x/y/z` = 0xa4/0xa5/0xa6.

The `if` opener is four separate words glued into one table entry: a compare-with-zero (the `loop_cmp` word with byte 2
bit 3 meaning "not equal" and `n = 0`), then `06 00`, `07 c2 00 00`, `0f 05 54 21`. `endif` is `loop_exit` with byte 3 = 01.

---

## C2. The 14-byte load and store

This section exists because the shape was misread for a week. A memory instruction is **8 address bytes plus a 6-byte
tail**, and the tail must directly follow — two loads followed by two tails read zeros. What used to be called a
separate `WAIT` instruction (`51 01 00 40 46 00`) and `STORE_WAIT` (`11 00 00 90 11 00`) are those tails.

### Device load — `67 00 44 00 00 00 20 00 | 51 01 00 40 40 00`

| field | position | meaning |
|---|---|---|
| `first` | 1.4 | set on the first load of the kernel |
| `more` | 2.4 | another load follows (scoreboard-related) |
| `dst` | 3.1 w7 | destination register, first of `count` consecutive |
| `slot` | 4.0 w8 | dense buffer binding index |
| `areg` | 5.0 w7 | element-offset register |
| `killa` | 5.7 | `areg` dies after this load |
| `wcode` | 8.1 w3 | vector width code |
| `sb0` `sb1` `sb2` | 8.6 w2, 9.0, 9.3 | scoreboard slot; Apple varies these per load, we emit 1, 1, 0 (§E) |
| `off0` `off1` `off9` | 9.7, 10.0 w8, 11.0 w4 | immediate element offset, **13 bits** |
| `fmt` | 12.0 w4 | stride format |

### Device store — `e7 00 54 00 00 00 20 00 | 11 00 00 10 10 00`

| field | position | meaning |
|---|---|---|
| `first` | 1.4 | set when no load preceded |
| `wait` | 2.1 | wait for outstanding loads (byte 2 becomes 0x56) |
| `src` | 3.1 w7 | source register, first of `count` |
| `slot` | 4.0 w8 | buffer index |
| `areg` | 5.0 w7 | element-offset register |
| `killa` | 5.7 | `areg` dies |
| `last` | 6.0 | Apple sets it on the final store; no observable effect |
| `wcode` | 8.1 w3 | width code |
| `sb0` | 8.6 w2 | scoreboard |
| `off0` `off3` | 9.5 w3, 10.0 w8 | immediate element offset, **11 bits** |
| `s7` | 11.7 | set in Apple's 1-wide stores, clear in its wide ones; hardware runs wide stores with it set |
| `fmt` | 12.0 w4 | stride format |

### Threadgroup load and store

Same 14-byte shape, byte 1 = `02`, byte 6 = `00`, and the index register moves to byte 5 as `r << 2` (6 bits, so r0..r63).

| | `tgload` | `tgstore` |
|---|---|---|
| template | `67 02 54 00 00 00 00 00 44 0d 00 c0 08 00` | `e7 02 54 00 00 00 00 00 44 02 00 00 02 00` |
| index | `a` 5.2 w6 | `a` 5.2 w6 |
| kill | `killa` 6.7 | `killa` 6.7 |
| offset | `off0` 9.7, `off1` 10.0 w8 → **9 bits** | `off0` 9.5 w3, `off3` 10.0 w8 → **11 bits** |
| width | `w4` 8.3 w2, `w2` 8.5 | same |
| stride format | always element units | `s30` 11.4 w2 |
| base | `base` 4.0 w8 | `base` 4.0 w8 |

`base` is the **uniform register holding the threadgroup pointer, in 16-bit units**, placed right after the buffer
addresses: `2 * tg_base_words(N)` where `tg_base_words(N) = 2N + (2 if N odd else 0)`. That is `0x08` for two buffers and
`0x10` for three.

### Width codes

| count | device `wcode` | threadgroup `(w4, w2)` |
|---|---|---|
| 1 | 0 | (0, 0) — tail byte 8 = 0x44 |
| 2 | 4 | (0, 1) — tail byte 8 = 0x64 |
| 3 | 6 | — |
| 4 | 3 | (3, 0) — tail byte 8 = 0x5c |

Count 3 appears in `dsl.WCODE` (from `probes/mem/ld3.metal`, a `packed_float3` load) but the assembler exposes only
1, 2 and 4: `load3` is rejected as an unknown op.

### Stride format — the silent-corruption field

| instruction | field | value | meaning |
|---|---|---|---|
| device load | `fmt` byte 12 | 6 | index counts 4-byte elements (**what we emit**) |
| | | 8 | index counts float2s |
| | | 0 | index counts float4s (Apple's `device float4*`) |
| device store | `fmt` byte 12 | 1 / 2 / 0 | elements / float2 / float4 |
| threadgroup store | `s30` byte 11 bits 4,5 | 3 | index and offset count elements (**what we emit**) |
| | | 0 | index counts float4s — Apple's form |

With the element stride a count-4 load reads elements `i..i+3` into `dst..dst+3` at any destination alignment. Getting
this wrong is silent: with `s30` clear the staged matmul produced a maximum error of 152 rather than an error message.

---

## D. Semantic models

### D.1 The load/wait model (`dsl.LOAD_WAIT_MODEL`)

Loads are asynchronous and the synchronisation is explicit, **in the consumer**:

1. The 6-byte tail belongs to the load and must directly follow its address bytes.
2. The **first** consumer of loaded data must be a waiting form. Non-waiting consumers read stale registers (zeros).
3. One waiting consumer covers **all** outstanding loads; after it, non-waiting forms are fine. Apple emits exactly this.

The waiting forms are:

| form | how it waits |
|---|---|
| `store` | `wait` field, byte 2 becomes 0x56 |
| 6-byte float ALU | `tag = 0xc0` |
| `iadd`/`isub`/`ineg`/`imul` (reg)/shifts/converts | byte 2 = 0x56 **in the template** — these always wait |
| `addr`/`imul` (stride)/`rcp`/`exp2`/…/`mma`/`tgload`/`tgstore` | byte 2 bit 1, a settable `wait` field |

The assembler tracks this with a single `pending` flag: set by any load, cleared by any op whose template or `wait` field
waits. Verified in `test_load_wait_model`.

Apple's later 6-byte ops carry tails `00 20` / `00 40`; these do **not** work as a first consumer and are not understood.

### D.2 The kill model (`dsl.KILL_MODEL`)

Operand "kill" bits are **semantics, not hints**: after an op with a source's kill bit set, that register reads 0. Apple's
compiler sets them on the last use of every value, so every template copied from Apple's output arrives with kill bits
set. The assembler emits **none** by default.

Positions, all verified by running an op twice on the same operands:

| family | kill bits |
|---|---|
| `FALU` / `FMA` | `killa` 2.4, `killb` 2.3; `FMA`'s register `c` uses `c_hi` 4.7 |
| `LOAD` / `STORE` | `killa` 5.7 (the address register) |
| `IADD` / `ISUB` / `INEG` | 8.1 (b), 8.2 (a) |
| `IMUL` (reg) | 9.1, 9.2 |
| bitwise, min/max, select, compare | 2.3, 2.4 |
| `ISEL4` | `killc` 4.7, `kille` 8.7 |
| 12-byte shifts | 9.0 (value), 9.1 (count) |
| 10-byte shifts | 7.6 (value), 7.7 (count) |
| converts | 6.5 |
| transcendentals | 6.5, except `rcp` at 6.4 |
| threadgroup load/store | 6.7 |

**The matrix op is the exception that is not fully understood.** `C` is consumed unless `D == C`: after
`mma rD, rA, rB, rC` with `D != C`, the pair `rC` read back 0 for r14, r16 and r20 but survived for r4, r8 and r24.
No bit pattern explains the split. The safe rules, both verified across every register tried: **accumulate in place**
(`D == C`), or reload `C`. `A` and `B` always survive and can feed several `mma`s back to back.

### D.3 Loops

The assembler's `loop` / `body` / `endloop` produces:

```
        loop_init rC                  <- rC = 0
L:      [pre section]                 <- sees rC = 0, 1, ..., n-1
        loop_inc  rC += step
        loop_cmp  rC, n
        nop
        loop_enter
        [body section]                <- sees rC = 1, 2, ..., n
        loop_tail
        branch L
        loop_exit
```

The branch target is the **start of the pre section**, so text written between `loop` and `body` is part of the loop and
re-executed, just before the increment. That is Apple's layout, and it is why address arithmetic that wants a 0-based
counter goes in the pre section (`test_reduce`).

| aspect | encoding |
|---|---|
| immediate count | `loop_cmp.n = ((n >> 1) & 0x3f) \| 0x40`, `n_hi = n >> 7`; `loop_enter.extra = n & 1`, `loop_enter.imm = 1` |
| register count | `loop_cmp.nreg = 1`, `loop_cmp.n = register`; `loop_enter.imm = 0` |
| counter operand | `loop_cmp.src = (rC << 1) \| 1` |
| nesting depth | encoded twice: `loop_cmp.depth2 = 2 * depth`, `loop_tail.depth4 = 8 + depth` (byte 3 = 0x22 + 4·depth) |

Immediate counts stop at 255 (the field holds n/2 in 7 bits plus the odd bit). Counts of 0 and 1 **hang the GPU**.
With a register count, `loop_enter.imm` **must** be 0 — leaving Apple's `02` there hung the GPU. Stores placed after a
loop do not land without `loop_exit`. Verified in `test_loop`, `test_reduce`, `test_long_loop`.

### D.4 The `if` region

`if rC` runs the region per-thread where `rC != 0`, and is hardware-verified with divergent threads inside one SIMD group,
across threadgroups, and after a loop (`test_if`). Only **loop depth 0** is verified: an `if` nested inside a loop almost
certainly needs the depth nibble in byte 0, and the assembler refuses to emit one.

### D.5 Metadata

A kernel is not runnable without its `__GPU_METADATA` flatbuffer (`asm.gen_metadata`). Every object in it is
dereferenced by the driver; omitting one segfaults `ProgramBindingRemap`. The load-bearing fields:

| location | field | contents |
|---|---|---|
| `fn` slot 0 | register count | **must cover every register named.** Unset, a kernel touching r24 froze the display. We declare `min(128, (MAXREG + 9) // 8 * 8 + 8)` |
| `fn` slot 1 | uniform word count | `4`, or `tg_base_words(N) + 4` with threadgroup memory |
| `fn` slot 2 | uniform-register spans | `2N` address words, a pad span when `N` is odd, plus a type-6 span for the threadgroup base |
| `fn` slot 3 | `8 * N` | |
| `fn` slot 4 | buffer descriptors | one per buffer: kind 5, index, address-register offset `2i`, written flag |
| `fn` slot 28 | threadgroup bytes | present with threadgroup memory |
| `fn` slot 32 | u8 1 | when the register count exceeds 10 |
| **`nameh` slot 2** | **threadgroup words** (`bytes / 4`) | **the field the hardware actually reads.** Without it every threadgroup load returns 0, even with slot 28 set |

The threadgroup allocation living outside the function table is the single most surprising thing in the metadata. It was
found by running Apple's compiled binary through our dispatch and splicing our flatbuffer sub-structures into Apple's one
reference at a time (memory note `agx-metadata-bisect-method`); `fn` slot 28, the type-6 span and fields 15/16/18 were all
individually removable from Apple's metadata without breaking it.

---

## E. Unresolved and not decoded

The most useful section for whoever picks this up next. Each entry says what is known, what was tried, and the next
experiment.

### E.1 Scoreboard slots in the load tail

**Known.** `LOAD` carries `sb0` (8.6, 2 bits), `sb1` (9.0) and `sb2` (9.3); `STORE` carries `sb0` only. Apple varies them
across the loads of one kernel — its 1024 matmul uses six distinct combinations across eight loads (`sb` values 111, 001,
101, 201, 011, 301 as our disassembler prints them). Consumer-side, Apple's 6-byte ALU ops carry tails `20`, `40`, `60`,
`80`, `a0` where we always emit `c0`, and `dsl.LOAD_WAIT_MODEL` records that those tails do not work as a first consumer.

**Inferred, not verified:** the pair is a per-load scoreboard slot, so a consumer can wait on one outstanding load
instead of all of them. That is the one latency-hiding mechanism Apple has that we do not use.

**Tried.** Nothing directly. We emit a constant `sb0=1, sb1=1, sb2=0` and rely on the wait-everything `c0` tail.

**Next.** Assemble two loads into different slots, then a consumer whose tail names one slot, and check whether it reads
the other load's register as stale. A kernel with a deliberately slow load (a long dependent address chain) would make
the difference observable.

### E.2 The matrix op's undecoded variants

**Known.** Apple's matrix instructions vary byte 1's high nibble (`02`, `12`, `22`, `42`, `82` all observed in one loop),
byte 2 bit 0, and byte 6 bit 0. Our template fixes byte 1 = `02`, byte 2 bit 0 = 0 and byte 6 bit 0 = 0, so **instructions
with byte 6 bit 0 set do not decode at all** — they fall through to `.short`. `dsl.MMA`'s comment states these are not
`C`'s high bits.

**Tried.** Setting byte 1 to `12` and `22`, and byte 2 bit 0, while trying to reach a `C` register above r15: all three
produced the non-accumulating result, so none of them is a `C` extension.

**Next.** Diff two Apple kernels that differ only in which registers a single `mma` uses, and correlate the nibble with
the operand register numbers. The 1024 matmul disassembly in the scratchpad is a good corpus: 16 `mma`s, 6 of which our
table decodes.

### E.3 The A-operand bit 7 flag

**Known.** `dsl.MMA.a` is a **7-bit field at bit 1**, which absorbs bit 7 of the byte. `dsl.REGISTER_FILE` states bit 7
of a matrix A/B operand byte is a flag, not a register bit (r70 as a source read 0). The two statements contradict each
other in effect: disassembling Apple's code prints `r118` where the real register is r54.

In Apple's 1024 matmul the flag is **clear on exactly the last use of each A fragment** and set on the other three uses,
which reads as "this register is used again later" *(inferred from 16 instructions in one kernel; never run)*.

**Next.** Narrow `MMA.a`/`MMA.b` to 6 bits and add an explicit flag field, then re-disassemble Apple's kernels and check
every A register against the loads that produced it. Then run an `mma` with the flag set and clear on the same operands
and see whether the operand survives.

### E.4 Why the matrix `C` operand is consumed for some registers

**Known.** With `D != C`, `C`'s pair read back 0 for r14, r16, r20 and survived for r4, r8, r24 (§D.2).

**Tried.** Stepping the `c` byte across registers, and toggling `f8`, `f9`, `creg`, `acc`. No pattern in the register
number, the encoded byte, or any flag bit explains the split.

**Next.** Test the full r0..r31 range in one sweep to see whether the surviving set has a structure (the sample so far is
consistent with "even multiples of 4 survive", which r20 contradicts, so a larger sample is needed). Also worth testing
whether the *reader* matters: all the failing cases were read back by a `store2`, the surviving ones by a later `mma`.

### E.5 Apple's spill mechanism

**Known.** Apple's compiler never writes above r63 with ALU forms; `probes/live96.metal` declares 96 live values and all
its `fadd` destinations are ≤ r63, so it spills. We have no spilling at all, which is what caps the tinygrad tensor-core
path at 64×64 (`AGXRenderer.tensor_cores` is opt-in behind `AGX_TC=1`).

**Next.** Disassemble `probes/live96.metal` and `probes/live128.metal` and look for the store/load pair that is not a
device buffer access — the spill slot has to be addressed somehow, and the candidates are threadgroup memory or a stack
pointer in a uniform register.

### E.6 Half registers and 16-bit types

**Known** (`dsl.DTYPES`): Apple addresses 16-bit values as half registers — operand bit 0 **clear** instead of set,
distinct opcodes (`09 c2 9b 06` is a half `fadd`), and loads with byte 6 = 0x47. `probes/livehalf192.metal` and
`livehalf256.metal` were captured but never decoded. The renderer asserts on any non-4-byte store or convert.

**Next.** The operand encoding is the whole story: if bit 0 of a source byte is the width flag, then a half operand is
`reg << 1` with a half-selector somewhere. Start with `livehalf192.metal` diffed against a float32 equivalent.

### E.7 64-bit register pairs

**Known.** Nothing beyond `dsl.DTYPES`: 64-bit types need register pairs and none of it is decoded. tinygrad's sine
decomposition needs 64-bit integer arithmetic, which is exactly why `SIN` uses the float32 Cody-Waite variant instead
(`AGXRenderer.extra_matcher`).

### E.8 Apple's hardware `sin` and precise reciprocal/sqrt

**Known.** `2f 03` is the last step of Apple's `sin`, reached after a range reduction through undecoded ops
(`probes/trans/sin.metal`, `fast_sin.metal`). Apple's precise `1/x`, `sqrt` and `rsqrt` are the fast op plus Newton steps,
also through undecoded ops (`recip.metal`, `sqrt.metal`, `precise_sqrt.metal`).

**Next.** For `sin`, the reduction is an fma chain plus two 8-byte ops of the `0x32` family and two 14-byte select
fixups; diff `fast_sin.metal` against `sin.metal` to separate the Cody-Waite path from the Payne-Hanek one. For the
precise forms, one Newton step of our own using the ops we already have would close the gap without decoding anything.

### E.9 `if` at loop depth > 0

**Known.** Only depth 0 is verified. `LOOP_CMP` has a `depth2` nibble and `LOOP_TAIL` a `depth4` field, and the `if`
opener embeds a `loop_cmp`-shaped word, so an `if` inside a loop most likely needs the depth nibble set. Apple hoisted the
`if` out of the loop in `probes/if/if_in_loop.metal`, so we have no reference bytes.

**Next.** Write a probe whose `if` cannot be hoisted (a condition depending on the loop counter and a side effect inside),
then compare its opener's byte 0 against the depth-0 form.

### E.10 The register-held loop count speedup

**Known.** Switching a loop from an immediate count to a register count of the same size took the 1024 unstaged matmul
from 624 µs to 491 µs, reproducibly (QUIRKS "simdgroup matrix"). The two differ in `loop_cmp.nreg`/`n` and in
`loop_enter.imm`.

**Next.** Isolate it: a long empty loop timed both ways. If it reproduces, `loop_enter.imm` is the suspect, since the
`extra` bit path is the only behavioural difference.

### E.11 Smaller loose ends

| item | status |
|---|---|
| `STORE.last` | Apple sets it on a kernel's final store; no observable hardware effect (`probes/two_stores.metal`) |
| `STORE.s7` (11.7) | set in Apple's 1-wide stores, clear in its wide ones; hardware runs wide stores with it set |
| `LOAD.first` / `more` | scoreboard-related; we set them positionally and have never varied them alone |
| transcendental `hint` (4.0) and `h` (6.1) | both tested on hardware in both states with no effect |
| `FALU.fwd` (2.5) | Apple sets it when the result feeds another ALU op rather than a store; never varied alone |
| `ADDR` byte 7 bit 5 | tested, does nothing |
| `TG_LOAD`/`TG_STORE` byte 4 | `base`, the uniform register index; the value `08` appeared in every early probe before the formula was derived |
| `IMUL` stride low bits | the 14-bit field sits at bit 2; the low two bits of the byte pair are ignored |
| `r126` / `r127` | never emit: stale reads as a process's first kernel, and a display freeze followed |
