# AGX backend handoff (hand-written Apple G15/M3 machine code on tinygrad HCQ2)

Branch `agx` = nimlgen's draft PR #18006 "move metal to hcq2" (head a69f95f, fetched 2026-09-13 as local branch
`metal_hcq2_pr`) with nothing else on top yet. Keep ALL new work in new files so rebasing onto the merged PR is a no-op:
`tinygrad/runtime/ops_agx.py`, `tinygrad/renderer/isa/agx.py`, tests under `test/`. Do not edit `ops_metal.py`.

## The standalone toolchain (reference implementation, keep in sync by reading, not importing)
`~/development/Metal/working-tools/` — 831 lines, stdlib only, README explains every stage:
- `tinyasm2.py`  assembler (text asm -> G15 bytes), `__GPU_METADATA` flatbuffer, applegpu Mach-O compute object, archive
- `air.py`       from-scratch LLVM bitstream writer -> AIR module `kernel void k(device float* b0..bN-1, uint t)` -> MTLB metallib
- `run.py`       in-memory dispatch. THIS is the body of the `pipeline()` override below.
Kernels in `~/development/Metal/kernels/prog_*.txt`; `python3 check.py` verifies against captured references.

## How dispatch works (verified 2026-09-13, incl. through an indirect command buffer)
1. `newBinaryArchiveWithDescriptor:` with NO url -> empty in-memory archive.
2. `newLibraryWithData:` with our metallib bytes (Metal needs the AIR only for the signature/reflection; the GPU never runs it).
3. Swizzle `-[_MTLBinaryArchive getBinaryDataForKey:reflectionType:]`: on first lookup of one of OUR archives, build
   `MTLBinaryEntry initWithData:reflectionBlock:binaryPosition:` (data = compute object = slice0 `__compute` section,
   block = 16 zero bytes, pos = UINT64_MAX), poke ivar `_reflectionFlags` = 2, `addBinaryEntryInternal:forKey:` under the
   key Metal just asked for, then call the original. We never compute the key.
4. `newComputePipelineStateWithDescriptor:options:reflection:error:` with `MTLPipelineOptionFailOnBinaryArchiveMiss` (=4)
   and `setSupportIndirectCommandBuffers:YES`. Assert the insertion happened.
Gotcha: the archive key derives from the AIR only. Two kernels with the same signature MUST have different AIR or Metal's
in-process pipeline cache serves one for the other. `air.gen_air(N, srcname)` puts a per-program SOURCE_FILENAME in; keep that.

## Where it plugs into PR #18006 (`tinygrad/runtime/ops_metal.py` on this branch)
- `MetalDevice.pipeline(self, lib: bytes, name: str)` — `functools.cache`d on lib bytes; the ONLY place bytes become a
  pipeline state. `class AGXDevice(MetalDevice)` overrides it with steps 1-4 above. `lib` is whatever our compiler returns.
- `super().__init__(device, MetalAllocator(self), [MetalRenderer], None, arch=...)` — pass our renderer instead.
  `Renderer.compiler.compile(src) -> bytes` is where asm text becomes the archive (identity-ish: tinyasm2.build).
  No Program class: HCQ2 owns execution.
- Register the device name in `tinygrad/device.py` ALL_DEVICES (and `Device` import path convention `ops_agx.py` -> "AGX").

## Conventions the MetalQueue forces on codegen (read `MetalQueue.exec/binding` in ops_metal.py)
- Buffers: `setKernelBuffer:offset:atIndex:` in order of `prg.arg.globals`. Matches our N-buffer AIR.
- Scalars (vals): NOT setBytes. Packed into one scalar buffer, bound at index len(globals)+i with an offset. Our AIR must
  declare scalar params as buffer-typed args; machine code loads them from memory.
- Dispatch: `concurrentDispatchThreadgroups:threadsPerThreadgroup:` (global_size, local_size). Kernels must index by
  threadgroup id + local id; current test kernels use a flat 8-thread grid.
- Every command has `setBarrier` (in-order within a batch). Nothing to do.

## State of the PR branch on this machine (M3, macOS 15)
`DEV=METAL python3 -m unittest test.device.test_hcq2` -> 18 pass, 1 skip, 1 FAIL (test_rt_patches_are_inputs_and_vars_only:
runtime patch reads mtl_handles/cmdbuf params — a draft-side issue, not ours). Tensor ops work. `pytest` is not installed;
use unittest. `METAL=1` is deprecated -> `DEV=METAL`. `timeout` is not on this Mac.

## Suggested order
1. `ops_agx.py`: AGXDevice(MetalDevice) with `pipeline()` override; compiler = tinyasm2-equivalent; run prog_add through
   tinygrad's HCQ2 path via a custom kernel first (see test_hcq2 custom kernel sources test).
2. Scalars in AIR + metadata; threadgroup indexing in the assembler.
3. `renderer/isa/agx.py`: ISARenderer subclass (model: `renderer/isa/x86.py`, bitfield DSL idea: `renderer/amd/dsl.py`).
Memory notes for Claude sessions live in `~/.claude/projects/-Users-liamdrew-development-Metal/memory/`.
