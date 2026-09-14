# Apple G15 (M3) GPU: hardware quirks learned the hard way

Everything here was verified on hardware (M3, macOS 15, 2026-09-14) with hand-assembled kernels through `tinygrad/runtime/support/agx/asm.py`.
The instruction table with per-field notes is `tinygrad/renderer/agx/dsl.py`; probes that produced the bytes are under `extra/agx/probes/`.
Freezes and hangs are marked. Never emit an instruction whose fields you have not run twice (see "kill bits").

## Registers
- 128 x 32-bit registers r0..r127. Float ALU, movimm and the mma's A/B operands reach only r0..r63 (6-bit fields; bit 7 of an operand byte is a
  flag, and r70 as a float source reads 0). Loads, stores, shifts, converts and the mma's C and D fields have 7 or 8-bit fields and reach r125.
- HAZARD r126/r127: read stale or zero as the first kernel of a process; the display froze minutes later (freeze #2). Never emit them.
- FREEZE #1: the metadata register count (`__GPU_METADATA` fn slot 0) must cover every register named. Unset it and a kernel touching r24 froze the
  whole display. We declare max register + one 8-block. Declaring 2x made no timing difference.
- HANG: an mma accumulating in place in the far bank (D = C = r80, pair zero-initialized by `ishl`) hung the GPU; recovered by the 10 s device
  timeout. D = r80 with C = r84 had worked. The renderer does not use the far bank.
- Apple's compiler never writes above r63 with ALU forms and spills instead (its spill mechanism is undecoded).

## Kill bits (`dsl.KILL_MODEL`)
- Operand "kill" bits are semantics, not hints: after an op with a source's kill bit set, that register reads 0. Apple sets them on every last use,
  so EVERY template copied from Apple's code carries them. Run every new encoding twice on the same operands and clear bits until both reads agree.
- Positions found: FALU/FMA killa/killb; FMA c_hi; LOAD/STORE byte 5 bit 7 (address register); IADD/ISUB/INEG byte 8 bits 1,2; IMUL byte 9 bits
  1,2; bitwise/min/max/select/compare byte 2 bits 3,4; select c byte 4 bit 7, e byte 8 bit 7; shifts byte 9 bits 0,1 (10-byte forms byte 7 bits
  6,7); converts byte 6 bit 5; transcendentals byte 6 bit 5 (rcp: bit 4); threadgroup loads/stores byte 6 bit 7.
- The mma's C operand is consumed unless D = C: after `mma rD, rA, rB, rC` (D != C) the pair rC read back 0 for r14, r16, r20 but survived for
  r4, r8, r24 (unexplained). Accumulate in place or reload C. A and B survive and feed several mmas back to back.

## Loads, stores, waits (`dsl.LOAD_WAIT_MODEL`)
- A load or store is 14 bytes: 8 address bytes + a 6-byte tail that MUST directly follow (two loads then two tails read 0). The tail was
  mistaken for a separate "wait" instruction for a week. It holds the vector width (byte 8 bits 1..3: 1->0, 2->4, 3->6, 4->3), the immediate
  element offset (loads: byte 9 bit 7 | byte 10 << 1 | byte 11 low << 9; stores: byte 9 bits 5..7 | byte 10 << 3) and a stride format nibble
  (byte 12: loads 6 = element stride, 8 = float2, 0 = float4; stores 1/2/0). With the element stride a count-4 load reads elements i..i+3 into
  dst..dst+3, any dst alignment; with the float4 stride the index counts 16-byte vectors (Apple's `device float4*`).
- Loads are asynchronous. The FIRST consumer of loaded data must be a waiting form: STORE byte 2 = 0x56, six-byte float ALU with tail 00 c0,
  or byte 2 bit 1 on the addr-family / convert / transcendental / mma forms. Non-waiting consumers read zeros. One waiting consumer covers all
  outstanding loads. Apple uses per-load scoreboard slots instead (consumer tails 20/40/60/80/a0 wait on one slot); undecoded, not needed so far.
- Two stores back to back lose data without each store's tail (the old "store_wait"); the tail is now part of the store.
- Issuing loads early and consuming later hides latency: the 256 matmul went 430 -> 99 us with a 24-register early-load budget; 131 us all-early.

## Loops and branches
- Counted loop layout: LOOP_INIT | LOOP_INC | LOOP_CMP | NOP | LOOP_ENTER | body | LOOP_TAIL | BRANCH back | LOOP_EXIT. Immediate counts hold n/2 in
  7 bits (odd counts via LOOP_ENTER's extra bit) and stop at 255; counts of 0 and 1 HANG the GPU.
- Counts above 255: LOOP_CMP byte 3 = (reg << 1) | 1 names a register with the full count, and LOOP_ENTER byte 1 must be 00 (02 with an
  immediate). The 02 with a register count HUNG the GPU (recovered by the timeout). Verified 256, 999, 1000.
- Stores after a loop do not land without LOOP_EXIT. `if` regions are only verified at loop depth 0 (nesting needs the depth nibble).
- Nesting depth is encoded twice: LOOP_CMP byte 0 nibble = 2 * depth, LOOP_TAIL byte 3 = 0x22 + 4 * depth.

## Immediates and constants
- E4M3 immediates in float ALU operands (sign in the killb position); anything else needs movimm, whose 32-bit immediate is scattered across seven
  fields (first layout lost bits 0..12 silently: 3.14159 stored as 3.1420898).
- ADDR's register form is 16-bit (-3 read as 65533); use IADD (byte 7 bit 5) for 32-bit values. Shift counts of 32 and up give 0.
- The mma's C byte with byte 9 bit 1 clear is a small immediate (08 -> +0.125).

## Threadgroup memory
- tgload/tgstore address `threadgroup[(rA + off) * 4]` with the threadgroup base in a uniform register whose index (16-bit units) is byte 4:
  right after the buffer addresses (08 for two buffers, 10 for three, with the odd-N padding span).
- The allocation is metadata, and the load-bearing field is NOT in the function table: name-header table slot 2 = bytes / 4. fn slot 28 (bytes),
  the type-6 uniform span and fields 15/16/18 were all removable from Apple's metadata; without slot 2 every threadgroup load read 0. Found by
  running Apple's binary through our dispatch and splicing our flatbuffer sub-structures into it one reference at a time (memory note
  `agx-metadata-bisect-method`).
- Dispatch dims are (threadgroups x,y,z, threads x,y,z): 8 groups of 1 thread each have private threadgroup memory and look like a broken barrier.
- Offset field widths: threadgroup loads 9 bits (511 elements), threadgroup and device stores 11 bits (verified to 1024), device loads 13 bits.
  Fold anything larger into the index register.
- Wide threadgroup stores: byte 11 bits 4,5 (0x30) are the stride format. Set, the index and offset count elements (what the assembler emits);
  Apple clears them on its float4 stores, whose index then counts float4s (thread 1's float4 landed at element 16); with them clear the staged
  matmul is wrong (maxerr 152). tgload2/tgload4 always count elements.

## Measuring
- Never time two kernels concurrently: a 32-tile matmul measured 3646 us while another process used the GPU and 796 us alone.
- The GPU clock is bimodal under Python-paced dispatch: the same correct kernel measured min 84 us / median 335 us over 30 runs (stock Metal's
  512 matmul: min 116 / median 337). Compare kernels by the minimum over many runs, or use problems large enough (>= 1 ms) that one kernel
  ramps the clock itself. Single-run numbers earlier in the log (264, 123, 283 us) are not comparable with each other.

## Transcendentals
- rcp, rsqrt, rsqrt_special, exp2, log2 are single 10-byte ops in the convert layout (af/2f + function byte). rsqrt(0) = inf; rsqrt_special(0) = 1 so
  `a * rsqrt_special(a)` is Apple's fast sqrt with sqrt(0) = 0. All within 1 ulp. Apple's sin, precise 1/x and sqrt are sequences through undecoded ops.
- Only 32-bit types: 16-bit values are half registers with different opcodes (undecoded).

## simdgroup matrix (mma)
- One 12-byte instruction per 8x8x8 fp32 product (a 16x16x16 compiles to eight). Three register encodings in ONE instruction: D = register number,
  A/B = register << 1, C = register with adjacent bit pairs swapped (r4 -> 08, r6 -> 09, r8 -> 04, r20 -> 28). byte 11 bit 0 = accumulate.
- Lane layout: lane t holds row ((t>>1)&3) + 4*(t>>4), columns 2*(t&1) + 4*((t>>3)&1) as a register pair. Same as tinygrad's `tc.metal` swizzle.
- Hand-written kernels live in `extra/agx/mm_hand.py` (16x16 tile per SIMD group, no staging), `mm_staged.py` (32x32 tile per threadgroup of
  4 SIMD groups, A 32x16 / B 16x32 staged through threadgroup memory) and `mm_staged64.py` (64x64 tile, 16 accumulator pairs per thread).
- Timings, minimum over 20 runs of one process at a time (fp32, us):

  | N    | mm_hand | mm_staged | mm_staged64 | stock tinygrad on Metal |
  |------|---------|-----------|-------------|-------------------------|
  | 1024 | 491     | 2119      | 476         | 431                     |
  | 2048 | 4463    | -         | 4103        | 3154                    |

  Metal's own kernel is still 10-30% ahead. Accumulators per thread dominate: the 64x64 tile (32 accumulator registers) beats the 32x32 tile
  (8 registers) by 4x, and staging only pays once the tile is large enough to reuse what was staged. An earlier reading that a hand kernel
  matched Apple's compiler and beat Metal came from single 512 runs and did not survive the measurement protocol below.
- Switching a loop from an immediate count to a register count (same iteration count) took the 1024 unstaged kernel from 624 to 491 us,
  reproducibly. Unexplained; possibly the immediate form's `extra` bit path.

## Method (what actually worked)
- One field per probe, pairwise diffs against a known kernel, then a minimal GPU kernel with a double read.
- A variant that returns garbage (neither 0 nor the answer) means STOP varying that field: mis-encoded instructions execute silently and preceded
  both freezes. Bash timeouts 10 s, device wait 10 s, 2-4 kernels per call.
- When bytes match Apple's but results don't: run Apple's compiled binary through our dispatch, then swap text and metadata between the archives.
