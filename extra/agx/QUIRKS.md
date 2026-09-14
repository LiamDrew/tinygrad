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

## The memory model (`dsl.SCOREBOARD`)
- There are exactly **two scoreboard slots**. A load names its slot in **byte 8 bit 7** of its tail. Byte 9 bits 0 and 3, which Apple also
  varies, have no effect on waiting; byte 8 bit 6 likewise.
- Consumers choose which slot to wait on: the six-byte float ALU's **tail byte** is `0xc0` for slot 0 and `0x80` for slot 1 (`00 20 40 60 a0 e0`
  never wait at all); **addr-family** ops (load, store, convert, mma) use **byte 2 bit 1** for slot 0 and **byte 1 bit 7** for slot 1. The two are
  independent bits, so one instruction can wait on both slots.
- A wait covers *every* outstanding load on that slot, verified to 16 loads, not just the oldest one.
- With **no wait at all** a load still lands after roughly 33-38 issue slots. The wait is a blocking scoreboard check, not the only path to the
  data, which is why the WAR hazard below exists.
- What this does NOT buy: splitting the matmul's loads across both slots so the first four mmas start early measures 3155.8 us against 3156.6
  for one slot at N=2048, i.e. nothing. The kernel is matrix-issue bound and starting an mma earlier does not help.

## Kill bits: what they are and what they are not
- Semantics are settled (`dsl.KILL_MODEL`): setting a source's kill bit makes that register read as 0 afterwards. Apple's compiler sets them on
  the last use of every value. We emit none, and thousands of kernels have run correctly without them, so they are not functionally required.
- **No measurable performance effect.** A deliberately memory-latency-bound kernel (16 widely strided loads per thread, summed) with the kill bit
  on every fadd's last-use operand measured 57.9 / 55.5 us against 57.4 / 56.1 without: inside the noise.
- They are **not** a dynamic occupancy mechanism. Occupancy comes from the *declared* register count in the metadata, which is static, and
  varying it over its valid range produced no reliable difference on either a matmul (issue-bound, so none is expected: 3157 us at every value
  from 64 to 128) or the latency-bound kernel once measurement artifacts were controlled.
- The matrix op is the exception to the kill model: neither the bits Apple sets on last uses (byte 8 bit 7, byte 9 bit 0) nor the `live_a` /
  `live_b` flags zero their operands. We have no encoding that kills an mma operand.
- **Never declare fewer registers than the kernel names.** That is what froze the machine (Registers, above); `asm.build` now asserts it.

## Where mm_direct's last 8% goes
Same 16-mma stream every time, one thing ablated at a time (scratch ablate.py), N=2048, min of 10:

  | variant                                   | time    | cost of the part added |
  |-------------------------------------------|---------|------------------------|
  | 16 mmas alone (ceiling)                    | 2920 us | -                      |
  | + the 5 address `iadd`s                    | 2971 us | 51 us                  |
  | + the 8 `load2`s, first mma not waiting    | 3013 us | 42 us                  |
  | + the wait on the first mma (= mm_direct)  | 3145 us | 132 us                 |

So the 225 us gap is 59% memory stall, 23% address arithmetic, 19% load issue. Every way to attack it needs more registers:
prefetching the next K step needs a second set of 16 fragment registers; unrolling to amortise the address adds needs the same
(and is otherwise wrong, see the WAR hazard); loading less per mma needs a bigger tile, i.e. more accumulators. All 64 usable
registers are already spoken for (32 accumulators, 16 fragments, ~11 addresses), so the only source is the far bank r64..r125,
where the mma's C and D fields do reach but an in-place accumulate once hung the GPU.

## Pipelining the stall away does not work (and why)
`extra/agx/mm_pipelined.py` moves the 16 accumulator pairs into the far bank r64..r95, which frees r16..r47 for two complete sets of fragment
registers, and software-pipelines the loop: while one K step's 16 mmas run, the next step's eight loads are already in flight on the other
scoreboard slot, writing the other register set, so the WAR hazard cannot bite. It is correct and it is **slower**: 3167 us against mm_direct's
3156 at N=2048.
So the 132 us the ablation table attributes to the wait is not a stall that prefetching can recover. With four SIMD groups per threadgroup and
thousands of threadgroups, the hardware already hides that latency with other groups; removing the wait only looked faster because it removed
synchronization the kernel actually needs. Ablation prices an instruction, it does not prove the time is recoverable.
Useful by-product: the far bank is safe for mma accumulators up to r95 and dangerous above it (r112 hung, r124 read zeros).

## No in-kernel cycle counter
All 256 special-register indices were read twice with an 8- and a 256-instruction gap between the reads; none returned a
changing value, so there is no timer or cycle counter to build an instruction-level profiler on. `0x9b` and `0xab` showed a
delta of 1 in the scan but read 0 four times in a row directly, so that was an artifact of the surrounding instructions.
Ablation - delete instructions from a real kernel and measure - is the profiler, and the table above is what it produces.

## Apple's compiler on our own algorithm
- `extra/agx/mm_metal_ref.py` writes mm_direct's exact algorithm in Metal and runs Apple's compiled output through the same dispatch.
  At N=2048, min of 10, alternated: ours 3157 us, Apple's compiler on the same algorithm 3274 us (3.7% slower), and the hardware
  ceiling is ~2900. Careful hand assembly is slightly ahead of the compiler, not behind it.
- **How the source is written matters 3.6x.** Expressing the 4x4 accumulators as `simdgroup_float8x8 c[4][4]` with `for` loops, which is
  the natural way, gives 11500 us: the compiler does not unroll those loops, so its inner loop holds 4 mmas instead of 16. Writing every
  fragment and accumulator as its own named variable gives 3274. tinygrad's Metal renderer emits the unrolled form, which is why stock
  Metal was competitive in the earlier comparisons.
- Apple's unrolled loop body is 502 bytes with 11 loads and heavy address arithmetic against our 354 bytes with 8 loads and 5 `iadd`s,
  because it recomputes addresses from the loop counter where we increment pointers. It uses both scoreboard slots and sets kill bits
  throughout, and is still slower, which is more evidence that neither of those is worth anything here.

## Measuring, part two: cache residency
- The same latency-bound kernel measured 21 us in two process launches and 55 in a third, with identical bytes. Its working set is 4.4 MB, right
  at the edge of what stays resident, so whether the buffer survives between launches changes the answer by 2.6x. An earlier "12% faster at 64
  declared registers" on an elementwise kernel was the same artifact: eight launches of one fixed configuration spread 408-422 us and the fast
  mode never recurred. Any effect below ~10% on a memory-bound kernel is not measurable this way. Effects on the matmul are trustworthy because
  min and median agree to 0.1% across launches.

## Loads, stores, waits (`dsl.LOAD_WAIT_MODEL`)
- A load or store is 14 bytes: 8 address bytes + a 6-byte tail that MUST directly follow (two loads then two tails read 0). The tail was
  mistaken for a separate "wait" instruction for a week. It holds the vector width (byte 8 bits 1..3: 1->0, 2->4, 3->6, 4->3), the immediate
  element offset (loads: byte 9 bit 7 | byte 10 << 1 | byte 11 low << 9; stores: byte 9 bits 5..7 | byte 10 << 3) and a stride format nibble
  (byte 12: loads 6 = element stride, 8 = float2, 0 = float4; stores 1/2/0). With the element stride a count-4 load reads elements i..i+3 into
  dst..dst+3, any dst alignment; with the float4 stride the index counts 16-byte vectors (Apple's `device float4*`).
- Loads are asynchronous. The FIRST consumer of loaded data must be a waiting form: STORE byte 2 = 0x56, six-byte float ALU with tail 00 c0,
  or byte 2 bit 1 on the addr-family / convert / transcendental / mma forms. Non-waiting consumers read zeros. One waiting consumer covers all
  outstanding loads. Apple uses per-load scoreboard slots instead (consumer tails 20/40/60/80/a0 wait on one slot); undecoded, not needed so far.
- A register with a load in flight is overwritten when that load lands, so a `movimm` into it is silently lost: `load r0,.. | movimm r0,#10 |
  iadd r6,r0,r0` returns 2x the LOADED value, not 20. Putting any waiting instruction (`iadd` and friends always wait, byte 2 bit 1 is set in
  their templates) before the `movimm` drains the queue and the write sticks. r0 is not special; any destination register behaves this way.
- The same async landing blocks loop unrolling. Reusing fragment registers for a second K sub-step is WRONG on hardware even though the
  instruction stream is correct: the second sub-step's loads land before the first sub-step's later `mma`s have read those registers. The error
  pattern proves it - the fragment consumed by the first four mmas is mostly right, the fragments consumed by mmas 5-16 are entirely wrong.
  Unrolling needs either disjoint fragment registers (we have only 16 free) or the scoreboard slots, which are not decoded.
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
- Hand-written kernels: `extra/agx/mm_direct.py` (the fast one: no threadgroup memory at all, 4 SIMD groups, 32x32 tile of C per SIMD group in
  16 accumulator pairs, four A and four B fragments loaded straight from device memory per K step of 8, 4x4 outer product = 8 loads -> 16 mmas),
  `mm_hand.py` (16x16 tile per SIMD group, no staging), `mm_staged.py` and `mm_staged64.py` (tiles staged through threadgroup memory).
- The shape is the whole game. Staging through threadgroup memory LOSES: the kernel that matches Metal has no barriers and lets the cache do the
  reuse, exactly like Apple's own compiled kernel (scratch mm_metal_1024.dis). What matters is accumulators per thread (32) and reusing each
  loaded fragment four times.
- Timings, minimum over 20 runs of one process at a time (fp32, us):

  | N    | mm_direct | mm_hand | mm_staged | mm_staged64 | stock tinygrad on Metal |
  |------|-----------|---------|-----------|-------------|-------------------------|
  | 1024 | 436       | 491     | 2119      | 476         | 431                     |
  | 2048 | 3156      | 4463    | -         | 4103        | 3196 (median 3834)      |

  mm_direct matches Metal: 1.2% behind at 1024, level at 2048 and with a far tighter median (3159 vs 3834). It uses NO scoreboard slots and NO
  kill bits, so both of those remain untapped headroom. An earlier reading that a hand kernel matched Apple's compiler and beat Metal at N=512
  came from single runs and did not survive the measurement protocol below.
- Switching a loop from an immediate count to a register count (same iteration count) took the 1024 unstaged kernel from 624 to 491 us,
  reproducibly. Unexplained; possibly the immediate form's `extra` bit path.

## How much headroom is left in the matmul
- `mm_direct` at N=2048 runs 3156 us. The identical kernel with every load and address instruction deleted runs 2936 us, and with the
  accumulator chains collapsed to one, 2854 us. So perfect memory overlap is worth at most 7%, and scoreboard slots and kill bits are
  bounded by that.
- The wall is the **matrix unit**, not instruction issue. A pure-mma loop with 16, 32, 64 or 128 mmas sharing one set of loop-control
  instructions runs 2917 / 2896 / 2896 / 2904 us: flat, even though loop control falls from 27% of the instruction stream to 4.5%. The unit
  sustains 16.78M mma instructions in ~2900 us = 5.8e9 mma/s = 5.9 TFLOP/s of fp32, and nothing else in the kernel is competing for issue.
  That number is the ceiling for any fp32 8x8x8 matmul on this chip, so the ~9% between it and mm_direct is all that tuning can ever reach.
- Accumulator dependencies cost nothing: 16, 4 and 1 independent accumulator chains run 2935 / 2896 / 2854 us, i.e. fewer chains is marginally
  FASTER. In-place accumulation has no latency penalty, so there is nothing to win from more chains or from ping-ponging accumulators.

## Method (what actually worked)
- One field per probe, pairwise diffs against a known kernel, then a minimal GPU kernel with a double read.
- A variant that returns garbage (neither 0 nor the answer) means STOP varying that field: mis-encoded instructions execute silently and preceded
  both freezes. Bash timeouts 10 s, device wait 10 s, 2-4 kernels per call.
- When bytes match Apple's but results don't: run Apple's compiled binary through our dispatch, then swap text and metadata between the archives.
