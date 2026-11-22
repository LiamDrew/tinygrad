# AGX backend: next task is tier 2, the transcendentals

You are continuing a native Apple GPU (M3, G15 ISA) backend for tinygrad on branch `agx` (HEAD `bebda3975`). Read the memory notes
first (they load automatically), then `tinygrad/renderer/agx/dsl.py` top to bottom: it is the ISA table and every quirk we know
(REGISTER_FILE, KILL_MODEL, LOAD_WAIT_MODEL, DTYPES). Then `tinygrad/runtime/ops_agx.py` (renderer + device) and
`tinygrad/runtime/support/agx/asm.py` (assembler). Tests: `DEV=AGX python3 -m unittest test.device.test_agx` (35 pass, ~5 s).

## State
Tier 1 is complete: float add/mul/fma, all int ops, shifts, int<->float converts, compares (int and float), min/max, select,
IF/ENDIF, loops (counts to 255), SPECIAL thread indices (real threadgroup dispatch; NOOPT no longer needed), a register allocator
with classes (counters/specials r0-15, address ints r8-31, floats r32-63), lazy loads, deferred address math, constants
rematerialized per use. Matmuls verified to 256x256 with tinygrad's optimizer. Group reduces are gated off (no threadgroup memory
yet: has_shared=False; one-line check added in codegen/opt/postrange.py). No spilling. FP16/bf16/64-bit deliberately unsupported.

## The task
tinygrad's float32 vocabulary still lacks: EXP2, LOG2, SIN, SQRT, RECIPROCAL (Unary) and FDIV/POW paths (division is usually
MUL by RECIPROCAL). Two routes, decide per op after probing:
1. Hardware instructions. Probe `o[t] = exp2(a)`, `log2`, `rsqrt`, `sqrt`, `1.0f/a`, `sin`, `precise::sin` etc. with one-op kernels
   (`extra/agx/probes/<family>/*.metal`, list with `python3 extra/agx/probe.py <file>`). Apple's fast-math `exp2/log2/rsqrt/rcp`
   are likely single ops; `sqrt` may be rsqrt+mul; `sin` will be a sequence (range reduction + polynomial).
2. tinygrad's own decompositions in `tinygrad/uop/transcendental.py` (`TRANSCENDENTAL=2` forces them). They need only ops we have
   (bitcast, shifts, compares, where, fma). This is the fastest path to correctness; hardware ops are the performance path.
   Check how a Renderer opts in/out (look for `code_for_op` / `TRANSCENDENTAL` handling in codegen) and what happens for an
   ISA renderer whose op set lacks EXP2 etc.
Deliverables: table entries with hardware-verified fields, mnemonics, renderer lowering, tests (unit + tensor-level, e.g.
`Tensor.exp()`, `.log()`, `.sin()`, `.sqrt()`, `1/x`, `softmax`), one commit per batch, ask before committing.

## Method (do not skip)
- One-thing probes, pairwise diffs against a known kernel. Print op bytes with a small scanner (loads 67 xx / stores e7 / waits are
  known; everything between them is the op). Keep probes in `extra/agx/probes/`.
- Hardware verification harness: patch `AGXRenderer.render` to return your assembly (see `run_asm`, `run_asm_int`, `run_asm_dims`
  in test_agx.py); `raw <hex>` in the assembler emits bytes under investigation. Data: a in r0, b in r2 via `load r0, 1 / wait /
  load r2, 2 / wait`; store the result register to slot 0.
- Every Apple-derived template carries KILL BITS on last-use operands (reading a killed register gives 0). Always run the op twice
  on the same operands (double read) and clear the bits until both reads are right. Positions found so far are in KILL_MODEL.
- The first consumer of loaded data must be a waiting form (byte 2 = 56 in the addr-family ops, tail c0 in float ALU); see
  LOAD_WAIT_MODEL. Non-waiting ops as first consumer read zeros.
- Bash timeouts 10 s, device wait 10 s, 2-4 kernels per call. If a variant returns garbage (not 0, not the answer), stop varying
  that field: mis-encoded instructions execute silently and preceded two display freezes. Never emit r126/r127.
- Verify destination fields (nibble + DST_HI at byte 2 bits 6,7 for float-ALU-style ops; 7-bit byte 3 for addr-family ops) with a
  high register (r22) and both operand registers swapped.

## Tinygrad plumbing you will touch
`AGXRenderer.render` is a linear walk over UOps with a dispatch chain (`elif u.op is ...`). Add float unary ops there; results are
floats (alloc(u)); operands via `val(u)` (materializes constants). Assembler: add a mnemonic branch in `assemble_text`. Table:
`Inst(...)` + add to `TABLE` (decode order matters when templates overlap) + a `fmt` line for the disassembler. Tests: offline
round-trip in `TestAGXISA`, hardware in `TestAGX`.
