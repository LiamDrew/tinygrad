# Apple G15 (M3) GPU: what actually determines matmul speed

Companion to `ISA.md` (encodings) and `QUIRKS.md` (hardware traps). This file is the performance story: what the hardware's
limit is, what moves a kernel toward it, what does not, and how to measure any of it without fooling yourself.
Everything below is fp32 and was measured on 2026-09-14 on an M3 (10 GPU cores).

## 1. The ceiling is the matrix unit

A loop of nothing but `mma` instructions, with no loads and no address arithmetic, runs a 2048-cube's worth of work
(16.78M matrix instructions) in **~2900 us**. That is **5.8e9 mma/s = 5.9 TFLOP/s**.

It is genuinely the unit and not instruction issue: putting 16, 32, 64 or 128 matrix ops behind one set of loop-control
instructions gives 2917 / 2896 / 2896 / 2904 us, flat, while loop control falls from 27% of the instruction stream to 4.5%.
Everything else in a real kernel issues for free alongside the matrix ops.

## 2. Where everyone lands

| N    | ours (`mm_tile` 2x4 SG=2) | ours (`mm_direct` 4x4 SG=4) | Apple MPS | Apple's compiler, our algorithm | tinygrad on Metal | ceiling |
|------|---------------------------|------------------------------|-----------|---------------------------------|-------------------|---------|
| 1024 | 416 us                    | 436 us                       | 415 us    | -                               | 431 us            | -       |
| 2048 | 3161 us                   | 3156 us                      | 3211 us   | 3274 us                         | 3196 us           | 2920 us |

Three independent implementations land between 3156 and 3274 against a 2920 floor. **Apple's own tuned library is also ~9%
above the ceiling.** That is the strongest evidence that the gap is the irreducible cost of feeding the unit rather than
something anyone has failed to find.

## 3. What moves the needle

**Accumulators per thread, and the matrix ops per load that follow from it.** The single biggest jump all day was
32 accumulator registers per thread instead of 8: at N=1024 a 32x32 tile per SIMD group is 476 us where a 16x16 tile is
2119. A 4x4 fragment outer product reuses each loaded fragment four times (8 loads -> 16 mmas); a 2x2 halves that and is
50% slower on the same tile.

**Threadgroup count, once the tile is big enough.** At N=1024 a 32x128 threadgroup tile yields only 256 threadgroups and
measures 436 us; a 16x64 tile yields 1024 threadgroups and measures 416, matching MPS. At N=2048 the big tile already gives
1024 threadgroups and every shape lands within 0.7%. Use `extra/agx/mm_tile.py` to sweep; `MI=2 MJ=4 SG=2` is the best
default found (matches MPS at 1024, within 0.2% of the best at 2048).

**Issuing loads early.** Letting tinygrad's renderer emit loads at their first use instead of up front cost 4x on an
unrolled matmul (430 us vs 99 at N=256, an early measurement). Loads want to be in flight well before their consumer.

## 4. What does not move the needle

Each of these was built, verified correct, and measured:

| idea | result |
|---|---|
| staging tiles through threadgroup memory | **much slower**: 2119 us vs 436 at N=1024. The cache already does the reuse |
| the two scoreboard slots, splitting loads so some mmas start early | 3155.8 vs 3156.6 us: nothing |
| kill bits on every last use, as Apple's compiler emits | within noise on a latency-bound kernel (57.9/55.5 vs 57.4/56.1) |
| declared register count (occupancy), swept over its valid range | flat from 64 to 128 on the matmul |
| software pipelining: far-bank accumulators + double-buffered fragments | **slower**: 3167 vs 3156 us |
| unrolling K by 2 with shared fragment registers | incorrect on hardware (WAR hazard), and no headroom to win anyway |

The pipelining result is the instructive one. Ablation says the wait on the first matrix op costs 132 us, but building a
kernel that removes that wait does not recover it: with four SIMD groups per threadgroup and thousands of threadgroups,
the hardware already hides load latency with other groups. **Ablation prices an instruction; it does not prove the time is
recoverable.**

## 5. Where the last 8% goes

Same 16-mma stream, one thing ablated at a time, N=2048, min of 10:

| variant                                   | time    | cost of the part added |
|-------------------------------------------|---------|------------------------|
| 16 mmas alone                              | 2920 us | -                      |
| + the 5 address `iadd`s                    | 2971 us | 51 us                  |
| + the 8 `load2`s, first mma not waiting    | 3013 us | 42 us                  |
| + the wait on the first mma                | 3145 us | 132 us                 |

Read this as an inventory, not a to-do list: see section 4 on why the 132 us is not recoverable.

## 6. How to measure on this machine

The measurement protocol cost more time than any single experiment, and three separate "findings" evaporated under it.

- **One process at a time.** A kernel measured 3646 us with another process on the GPU and 796 us alone.
- **The clock is bimodal.** The same correct kernel can read 84 us or 335 us at N=512. Compare by the **minimum over
  many runs**, and trust a result only when min and median agree. At N=2048 they agree to 0.1%; below N=1024 they do not.
- **Alternate conditions.** A 12% "win" from a register-count change reversed when the conditions were interleaved, and
  eight launches of one fixed configuration then spread 408-422 us with the fast mode never recurring.
- **Watch cache residency.** A kernel with a 4.4 MB working set read 21 us twice and 55 us on the third launch with
  identical bytes. Effects below ~10% on memory-bound kernels are not measurable this way.
- **Keep the GPU busy for short kernels.** MPS at N=1024 reads 484-923 us one dispatch per command buffer and 415 us with
  20 per buffer. Comparing a steady state against a non-steady state is how the MPS comparison was first got wrong.
- **There is no in-kernel cycle counter.** All 256 special-register indices were read twice with an 8- and a
  256-instruction gap; none changed. Ablation is the profiler.

## 7. Open

- fp16 / bf16: does the matrix unit issue faster for 16-bit inputs? The whole backend is 32-bit only today (`dsl.DTYPES`),
  so this needs the half-register operand convention, 16-bit loads, and the fp16 matrix variant decoded first. The
  deciding measurement is small: a pure fp16 mma loop against section 1's 5.8e9 mma/s.
- Small N: our harness cannot measure below 1024 because it dispatches one kernel per launch. Batching several dispatches
  into one submission would fix it.
- Tile sweep at non-power-of-two and rectangular shapes; `mm_tile.py` currently requires powers of two.
