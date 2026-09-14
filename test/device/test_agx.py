import unittest, struct, math
from unittest.mock import patch
from tinygrad import Tensor, Device
from tinygrad.renderer.agx import dsl
from tinygrad.runtime.support.agx import asm

class TestAGXISA(unittest.TestCase): # no device needed: the table must read back what it writes
  def test_roundtrip(self):
    src = "load r0, 1\nwait\nload r2, 2\nwait\nfadd r4, r0, r2\nstore r4, 0\n"
    text, n, written = asm.assemble_text(src)
    self.assertEqual((n, written), (3, {0}))
    body = [t for _, _, t in dsl.disassemble(text)]
    self.assertEqual(body, [".preamble 64 bytes", "getsr r1, tid.x", "load r0, 1, r1 ; first ; more", "load r2, 2, r1", "fadd r4, r0, r2",
                            "store r4, 0, r1 ; last", "stop"])
    self.assertFalse(any(t.startswith(".short") for t in body), "every emitted byte decodes")

  def test_apple_bytes_decode(self): # bytes captured from probes/*.metal
    for hexs, want in [("09011c05 00c0".replace(" ", ""), "fadd r0, r2, r0 ; killa ; killb"),   # sum3: b2 = b0 + b1
                       ("09011d0500c0", "fmul r0, r2, r0 ; killa ; killb"),                       # fmul
                       ("c9110015", "fadd r12, r10, r8"),                                        # sums_live: 4-byte form, explicit dst
                       ("3905040100c0", "fadd r3, r0, r2"),                                      # keep_inputs: odd dst
                       ("09c9140180c0", "fadd r0, r0, #3 ; killa"),                              # add_const: E4M3 immediate
                       ("6700440400012000 510100404600".replace(" ", ""), "load r2, 0, r1"), ("e700540800012100 110000901100".replace(" ", ""), "store r4, 0, r1 ; last"),
                       ("e700560000012000 110000901100".replace(" ", ""), "store r0, 0, r1 ; wait"),
                       ("6700440000822000 510100404600".replace(" ", ""), "load r0, 0, r2 ; killa"),   # probes/off1.metal: byte5 82 = areg r2 with its kill bit
                       ("6710440001042000 570100404000".replace(" ", ""), "load4 r0, 1, r4 ; first ; fmt 0"),   # probes/mem/ld4: a float4 pointer, index scaled by 16 bytes
                       ("6710440001022000 590100404800".replace(" ", ""), "load2 r0, 1, r2 ; first ; fmt 8"),
                       ("e700540000042100 170000101000".replace(" ", ""), "store4 r0, 0, r4 ; last ; fmt 0"),
                       ("6700541601892000 570900404600".replace(" ", ""), "load4 r11, 1, r9 ; more ; killa ; sb 111"),   # Apple's NOOPT matmul: count 4, element stride
                       ("af02560002 00b0400000".replace(" ", ""), "exp2 r0, r0 ; killa ; wait"), ("2f02560403009240 0000".replace(" ", ""), "log2 r2, r0 ; wait"),   # probes/trans
                       ("af00560002001048 2000".replace(" ", ""), "rcp r0, r0 ; killa ; wait"), ("2f01540403109240 0000".replace(" ", ""), "rsqrts r2, r4")]:
      b = bytes.fromhex(hexs)
      inst = next(i for i in dsl.TABLE if i.matches(b))
      self.assertEqual(dsl.fmt(inst, inst.decode(b)), want, hexs)

  def test_fma_decodes_apple(self): # bytes from probes: b0*b1 + b2 and variants
    for hexs, want in [("09011e05810802c0", "fma r0, r2, r0, r4 ; killa ; killb"), ("09011e05910802c0", "fma r0, r2, r0, -r4 ; killa ; killb"),
                       ("09011e0521b002c0", "fma r0, r2, r0, #1 ; killa ; killb"), ("09011e05a1b002c0", "fma r0, r2, r0, #-1 ; killa ; killb"),
                       ("09051e01910802c8", "fma r0, -r0, r2, -r4 ; killa ; killb"), ("09010ec1810402c2", "fma r0, #2, r0, r2 ; killb")]:
      b = bytes.fromhex(hexs); inst = next(i for i in dsl.TABLE if i.matches(b))
      self.assertEqual((inst.size, dsl.fmt(inst, inst.decode(b))), (8, want), hexs)

  def test_immediates(self):
    text, _, _ = asm.assemble_text("movimm r0, #42.0\nfadd r0, r0, #-3.5\nstore r0, 0\n")
    body = [t for _, _, t in dsl.disassemble(text)]
    self.assertIn("movimm r0, #42", body)
    self.assertIn("fadd r0, r0, #-3.5 ; killb ; tag 00", body) # no loads pending: not a waiting op. killb carries the immediate's sign

  def test_int_roundtrip(self): # the int mnemonics assemble to the hardware-verified bytes and read back
    src = ("load r0, 1\nwait\nload r2, 2\nwait\nisub r4, r0, r2\nineg r5, r0\nimul r6, r0, r2\niand r7, r0, r2\nior r8, r0, r2\nixor r9, r0, r2\n"
           "imax r10, r0, r2\nimin r22, r0, r2\nisel r12, r0, r2, r4\nisel r13, r0, r2, r4, r5\nicmp.lt r14, r0, r2\nicmp.ne r30, r0, r2\nstore r4, 0\n")
    text, _, _ = asm.assemble_text(src)
    body = [(bs.hex(), t) for _, bs, t in dsl.disassemble(text)][4:-2]
    self.assertEqual([t for _, t in body], ["isub r4, r0, r2", "ineg r5, r0", "imul r6, r0, r2", "iand r7, r0, r2", "ior r8, r0, r2", "ixor r9, r0, r2",
                                            "imax r10, r0, r2", "imin r22, r0, r2", "isel r12, r0, r2, r4, r0", "isel r13, r0, r2, r4, r5",
                                            "icmp.lt r14, r0, r2", "icmp.ne r30, r0, r2"])
    text, _, _ = asm.assemble_text("load r0, 1\nwait\nload r2, 2\nwait\nishl r4, r0, #5\nishr r5, r2, #3\nushr r6, r0, #1\nishl r7, r0, r2\nishr r8, r2, r0\nushr r9, r0, r2\nstore r4, 0\n")
    self.assertEqual([t for _, _, t in dsl.disassemble(text)][4:-2], ["ishl r4, r0, #5", "ishr r5, r2, #3", "ushr r6, r0, #1", "ishl r7, r0, r2", "ishr r8, r2, r0", "ushr r9, r0, r2"])
    text, _, _ = asm.assemble_text("load r0, 1\nwait\ni2f r4, r0\nu2f r5, r0\nf2i r6, r0\nf2u r22, r0\nstore r4, 0\n")
    self.assertEqual([t for _, _, t in dsl.disassemble(text)][3:-2], ["i2f r4, r0", "u2f r5, r0", "f2i r6, r0", "f2u r22, r0"])
    text, _, _ = asm.assemble_text("load r0, 1\nwait\nexp2 r4, r0\nlog2 r5, r0\nrcp r6, r0\nrsqrt r7, r0\nrsqrts r22, r0\nstore r4, 0\n")
    self.assertEqual([t for _, _, t in dsl.disassemble(text)][3:-2], ["exp2 r4, r0 ; wait", "log2 r5, r0 ; wait", "rcp r6, r0 ; wait", "rsqrt r7, r0 ; wait", "rsqrts r22, r0 ; wait"])
    self.assertEqual(body[0][0], "1f01560802001 0a81105".replace(" ", ""))   # verified on hardware: a - b into r4, kill bits clear (Apple: 17)
    self.assertEqual(body[3][0], "7b050701000000800000")                     # and, kill bits clear (07, Apple: 1f)

@unittest.skipUnless(Device.DEFAULT == "AGX", "AGX device required")
class TestAGX(unittest.TestCase):
  def test_add_two_numbers(self):
    a, b = Tensor([1.5]).contiguous().realize(), Tensor([2.25]).contiguous().realize()
    self.assertEqual((a + b).item(), 3.75)

  def test_mul_two_numbers(self):
    a, b = Tensor([1.5]).contiguous().realize(), Tensor([2.25]).contiguous().realize()
    self.assertEqual((a * b).item(), 3.375)

  def test_matmul(self): # the renderer lowers tinygrad's naive matmul (NOOPT): three loops, indexed loads, fma, indexed store
    from tinygrad.helpers import Context
    A = [[float(i * 4 + j) for j in range(4)] for i in range(4)]; B = [[float((i + j) % 3) for j in range(4)] for i in range(4)]
    with Context(NOOPT=1):
      got = (Tensor(A).contiguous().realize() @ Tensor(B).contiguous().realize()).tolist()
    self.assertEqual(got, [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)])

  def run_asm(self, src:str, a:float, b:float) -> float: # any three-buffer kernel: data0 = f(data1, data2)
    from tinygrad.runtime import ops_agx
    from tinygrad import codegen
    from tinygrad.runtime.support import hcq2
    from tinygrad.engine import realize
    # every call has the same kernel AST; without this the first compiled program would be reused for all of them
    for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
    with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: src):
      x, y = Tensor([a]).contiguous().realize(), Tensor([b]).contiguous().realize()
      return (x + y).item() # the op here is irrelevant: render is patched

  def run_asm8(self, src:str, a:list[float], b:list[float]) -> list[float]: # 8 threads, data0[t] = f(data1, data2); inputs may be longer
    from tinygrad.runtime import ops_agx, ops_metal
    from tinygrad import codegen
    from tinygrad.runtime.support import hcq2
    from tinygrad.engine import realize
    for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
    Device[Device.DEFAULT].wait_timeout_ms = 10000.0 # a hand-built kernel either finishes at once or hung the GPU
    with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: src), \
         patch.object(ops_metal.MetalQueue, "dims", staticmethod(lambda prg: (8, 1, 1, 1, 1, 1))):
      x, y = Tensor(a).contiguous().realize(), Tensor(b).contiguous().realize()
      return (x[:8] + y[:8]).contiguous().tolist() if False else (x + y).tolist()[:8]

  def run_asm_list(self, src:str, a:list[float], b:list[float]) -> list[float]: # tinygrad's own dispatch (one thread), whole output
    from tinygrad.runtime import ops_agx
    from tinygrad import codegen
    from tinygrad.runtime.support import hcq2
    from tinygrad.engine import realize
    for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
    Device[Device.DEFAULT].wait_timeout_ms = 10000.0
    with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: src):
      x, y = Tensor(a).contiguous().realize(), Tensor(b).contiguous().realize()
      return (x + y).tolist()

  def run_asm_int(self, src:str, a:int, b:int) -> int: # int32 data: data0 = f(data1, data2), a in r0 and b in r2 by the kernel's own loads
    from tinygrad import dtypes
    from tinygrad.runtime import ops_agx
    from tinygrad import codegen
    from tinygrad.runtime.support import hcq2
    from tinygrad.engine import realize
    for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
    Device[Device.DEFAULT].wait_timeout_ms = 10000.0
    with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: ".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\n" + src):
      x, y = Tensor([a], dtype=dtypes.int32).contiguous().realize(), Tensor([b], dtype=dtypes.int32).contiguous().realize()
      return (x + y).item()

  def test_int_ops(self): # probes/int/*.metal: one op each on a=5, b=3 (and 3, 5 where the order matters)
    for src, want in [("isub r4, r0, r2\nstore r4, 0\n", 2), ("ineg r4, r0\nstore r4, 0\n", -5), ("imul r4, r0, r2\nstore r4, 0\n", 15),
                      ("iand r4, r0, r2\nstore r4, 0\n", 1), ("ior r4, r0, r2\nstore r4, 0\n", 7), ("ixor r4, r0, r2\nstore r4, 0\n", 6),
                      ("imax r4, r0, r2\nstore r4, 0\n", 5), ("imin r22, r0, r2\nstore r22, 0\n", 3), ("icmp.gt r4, r0, r2\nstore r4, 0\n", 1),
                      ("icmp.eq r4, r0, r2\nstore r4, 0\n", 0), ("icmp.ne r4, r0, r2\nstore r4, 0\n", 1)]:
      with self.subTest(src=src.split("\n")[0]): self.assertEqual(self.run_asm_int(src, 5, 3), want)

  def test_int_shifts(self): # probes/int/ishl_imm*.metal and friends; value in r0 or r2, count immediate or in a register
    for a, b, src, want in [(5, 3, "ishl r4, r0, #5\nstore r4, 0\n", 160), (5, 3, "ishl r4, r0, #16\nstore r4, 0\n", 327680), (3, 5, "ishl r4, r2, #2\nstore r4, 0\n", 20),
                            (-16, 3, "ishr r4, r0, #3\nstore r4, 0\n", -2), (-16, 3, "ushr r4, r0, #3\nstore r4, 0\n", 536870910),
                            (5, 3, "ishl r4, r0, r2\nstore r4, 0\n", 40), (-16, 3, "ishr r4, r0, r2\nstore r4, 0\n", -2), (-16, 3, "ushr r4, r0, r2\nstore r4, 0\n", 536870910),
                            (5, 3, "ishl r4, r0, r2\nishl r6, r0, r2\nstore r6, 0\n", 40)]:   # operands survive (no kill bits)
      with self.subTest(a=a, b=b, src=src.split("\n")[0]): self.assertEqual(self.run_asm_int(src, a, b), want)

  def run_asm_cvt(self, src:str, a, from_int:bool): # data1 (int32 or float32) in r0, the stored result read back as the other type
    from tinygrad import dtypes
    from tinygrad.runtime import ops_agx
    from tinygrad import codegen
    from tinygrad.runtime.support import hcq2
    from tinygrad.engine import realize
    for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
    Device[Device.DEFAULT].wait_timeout_ms = 10000.0
    with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: ".buffers 2\nload r0, 1\nwait\n" + src):
      x = Tensor([a], dtype=dtypes.int32 if from_int else dtypes.float32).contiguous().realize()
      return (x.float() if from_int else x.cast(dtypes.int32)).item()

  def test_converts(self): # probes/cvt: int <-> float, signed and unsigned, truncation toward zero, source re-read (no kill)
    self.assertEqual(self.run_asm_cvt("i2f r4, r0\nstore r4, 0\n", 7, True), 7.0)
    self.assertEqual(self.run_asm_cvt("i2f r4, r0\nstore r4, 0\n", -7, True), -7.0)
    self.assertEqual(self.run_asm_cvt("u2f r4, r0\nstore r4, 0\n", -1, True), 4294967296.0)
    self.assertEqual(self.run_asm_cvt("f2i r4, r0\nstore r4, 0\n", -7.9, False), -7)
    self.assertEqual(self.run_asm_cvt("f2u r22, r0\nstore r22, 0\n", 7.9, False), 7)
    self.assertEqual(self.run_asm_cvt("i2f r4, r0\ni2f r6, r0\nstore r6, 0\n", 5, True), 5.0)

  def test_convert_tensors(self): # the renderer lowers CAST
    from tinygrad import dtypes
    from tinygrad.helpers import Context
    A, F = [5, -2, 7, 0], [1.5, -2.7, 3.0, 0.25]
    a, f = Tensor(A, dtype=dtypes.int32).contiguous().realize(), Tensor(F).contiguous().realize()
    with Context(NOOPT=1):
      for name, t, want in [("i2f", a.float(), [float(x) for x in A]), ("f2i", f.cast(dtypes.int32), [int(x) for x in F]),
                            ("cmp2f", (a < 3).float(), [float(x < 3) for x in A]), ("mixed", a.float() * f, [x * y for x, y in zip(A, F)])]:
        with self.subTest(op=name): self.assertEqual(t.tolist(), [struct.unpack("f", struct.pack("f", w))[0] if isinstance(w, float) else w for w in want])

  def run_asm_dims(self, src:str, dims:tuple) -> list[int]: # int32 out[0..7] under a (gx, gy, gz, lx, ly, lz) dispatch; store via r1 = tid.x unless src stores itself
    from tinygrad import dtypes
    from tinygrad.runtime import ops_agx, ops_metal
    from tinygrad import codegen
    from tinygrad.runtime.support import hcq2
    from tinygrad.engine import realize
    for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
    Device[Device.DEFAULT].wait_timeout_ms = 10000.0
    with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: ".buffers 3\n" + src), patch.object(ops_metal.MetalQueue, "dims", staticmethod(lambda prg: dims)):
      x, y = Tensor(list(range(8)), dtype=dtypes.int32).contiguous().realize(), Tensor(list(range(8)), dtype=dtypes.int32).contiguous().realize()
      return (x + y).tolist()

  def test_special_registers(self): # probes/sr: 4 threadgroups of 2 threads; the prologue's tid.x is the flat index
    self.assertEqual(self.run_asm_dims("getsr r2, tid.x\nstore r2, 0\n", (4, 1, 1, 2, 1, 1)), list(range(8)))
    self.assertEqual(self.run_asm_dims("getsr r2, gid.x\nstore r2, 0\n", (4, 1, 1, 2, 1, 1)), [0, 0, 1, 1, 2, 2, 3, 3])
    self.assertEqual(self.run_asm_dims("getsr r2, lid.x\nstore r2, 0\n", (4, 1, 1, 2, 1, 1)), [0, 1, 0, 1, 0, 1, 0, 1])
    self.assertEqual(self.run_asm_dims("getsr r2, tpg.x\nstore r2, 0\n", (4, 1, 1, 2, 1, 1)), [2] * 8)
    # 2D: out[tid.y * 2 + tid.x] = gid.y under 2x2 threadgroups of one thread, then lid.y under one group of 2x2
    src = "getsr r2, tid.y\naddr r3, r2, 1, r1\ngetsr r4, %s\nstore r4, 0, r3\n"
    self.assertEqual(self.run_asm_dims(src % "gid.y", (2, 2, 1, 1, 1, 1))[:4], [0, 0, 1, 1])
    self.assertEqual(self.run_asm_dims(src % "lid.y", (1, 1, 1, 2, 2, 1))[:4], [0, 0, 1, 1])

  def test_if(self): # probes/if: the region runs only where the condition register is nonzero; threads diverge within a SIMD group
    # 8 threads in one group: out[t] = t != 0 ? 10 + t : 99 (r0 = t from data1 = range(8), r2 = data2 = range(8))
    src = "load r0, 1\nwait\nload r2, 2\nwait\nmovimm r7, #0x00000063\nmovimm r5, #0x0000000a\niadd r5, r5, r2\nif r0\nior r7, r5, r5\nendif\nstore r7, 0\n"
    self.assertEqual(self.run_asm_dims(src, (1, 1, 1, 8, 1, 1)), [99] + [10 + t for t in range(1, 8)])
    self.assertEqual(self.run_asm_dims(src, (8, 1, 1, 1, 1, 1)), [99] + [10 + t for t in range(1, 8)])
    # after a loop, like tinygrad's guarded final store: r2 doubled 4 times, stored only where t != 0
    src = "load r0, 1\nwait\nload r2, 2\nwait\nmovimm r7, #0x00000063\nloop r3, #4\nbody\niadd r2, r2, r2\nendloop\nif r0\nior r7, r2, r2\nendif\nstore r7, 0\n"
    self.assertEqual(self.run_asm_dims(src, (1, 1, 1, 8, 1, 1)), [99] + [16 * t for t in range(1, 8)])

  def test_optimized_kernels(self): # the default optimizer: SPECIAL indices, threadgroups, upcasts; no NOOPT
    A = [[float(i * 8 + j) for j in range(8)] for i in range(8)]; B = [[float((i * j) % 5) for j in range(8)] for i in range(8)]
    a, b = Tensor(A).contiguous().realize(), Tensor(B).contiguous().realize()
    self.assertEqual((a + b).tolist(), [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(A, B)])
    self.assertEqual((a @ b).tolist(), [[sum(A[i][k] * B[k][j] for k in range(8)) for j in range(8)] for i in range(8)])
    self.assertEqual(a.sum(axis=1).tolist(), [sum(r) for r in A])
    self.assertEqual(Tensor([float(i) for i in range(1000)]).contiguous().realize().sum().item(), sum(range(1000)))
    # a fully unrolled reduce: 128 loads and their addresses are issued up front by tinygrad; lazy loads and deferred address math keep it in 64 registers
    import random; random.seed(0)
    for n in (32, 64):   # 64 uses a 16-element register-array accumulator (4x4 upcast)
      A = [[random.random() for _ in range(n)] for _ in range(n)]; B = [[random.random() for _ in range(n)] for _ in range(n)]
      got = (Tensor(A).contiguous().realize() @ Tensor(B).contiguous().realize()).tolist()
      self.assertLess(max(abs(g - sum(A[i][k] * B[k][j] for k in range(n))) for i, row in enumerate(got) for j, g in enumerate(row)), 1e-3)

  def test_float_compare_select(self): # probes/fcmp: the compare instruction with float codes, fmax/fmin, and the float-compare select
    k = ".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\nmovimm r5, #9.0\n%s\nstore r4, 0\n"
    for a, b, body, want in [(1.5, 2.25, "fcmp.lt r4, r0, r2", 1.0), (2.25, 1.5, "fcmp.lt r4, r0, r2", 0.0), (2.25, 1.5, "fcmp.gt r4, r0, r2", 1.0), (1.5, 1.5, "fcmp.le r4, r0, r2", 1.0),
                             (1.5, 1.5, "fcmp.eq r4, r0, r2", 1.0), (1.5, 2.25, "fcmp.ne r4, r0, r2", 1.0), (1.5, 1.5, "fcmp.ne r4, r0, r2", 0.0),
                             (1.5, 2.25, "fmax r4, r0, r2", 2.25), (1.5, 2.25, "fmin r4, r0, r2\nfmin r6, r0, r2\nior r4, r6, r6", 1.5),
                             (1.5, 2.25, "fsel r4, r0, r2, r5, r0", 9.0), (2.25, 1.5, "fsel r4, r0, r2, r5, r0", 2.25)]:
      with self.subTest(a=a, b=b, body=body.split("\n")[0]):
        got = self.run_asm(k % body, a, b)
        # compares store an int 0/1 into a float buffer: read the bits back
        self.assertEqual(struct.unpack("i", struct.pack("f", got))[0] if body.startswith("fcmp") else got, int(want) if body.startswith("fcmp") else want)

  def test_float_compare_tensors(self):
    from tinygrad import dtypes
    from tinygrad.helpers import Context
    A, B = [1.5, -2.0, 7.0, 0.0], [1.5, 3.0, -7.0, 1.0]
    a, b = Tensor(A).contiguous().realize(), Tensor(B).contiguous().realize()
    with Context(NOOPT=1):
      for name, t, want in [("lt", (a < b).cast(dtypes.int32), [int(x < y) for x, y in zip(A, B)]), ("ne", (a != b).cast(dtypes.int32), [int(x != y) for x, y in zip(A, B)]),
                            ("max", a.maximum(b), [max(x, y) for x, y in zip(A, B)]), ("where", (a < b).where(a, b), [x if x < y else y for x, y in zip(A, B)]),
                            ("relu", a.relu(), [max(x, 0.0) for x in A]), ("where_f", (a < b).where(2.5, -1.0), [2.5 if x < y else -1.0 for x, y in zip(A, B)])]:
        with self.subTest(op=name): self.assertEqual(t.tolist(), want)

  def test_int_compare_select(self):
    for a, b, src, want in [(3, 5, "icmp.lt r4, r0, r2\nstore r4, 0\n", 1), (5, 3, "icmp.lt r4, r0, r2\nstore r4, 0\n", 0), (-1, 5, "icmp.lt r4, r0, r2\nstore r4, 0\n", 1),
                            (-1, 5, "icmp.ult r4, r0, r2\nstore r4, 0\n", 0), (3, 3, "icmp.eq r4, r0, r2\nstore r4, 0\n", 1),
                            (3, 5, "movimm r4, #0x00000009\nisel r6, r0, r2, r4, r2\nstore r6, 0\n", 9), (5, 3, "movimm r4, #0x00000009\nisel r6, r0, r2, r4, r2\nstore r6, 0\n", 3)]:
      with self.subTest(a=a, b=b, src=src.split("\n")[-2]): self.assertEqual(self.run_asm_int(src, a, b), want)

  def test_int_tensors(self): # the renderer lowers tinygrad's int ops
    from tinygrad import dtypes
    from tinygrad.helpers import Context
    A, B = [5, -2, 7, 0], [3, 3, -7, 1]
    a, b = Tensor(A, dtype=dtypes.int32).contiguous().realize(), Tensor(B, dtype=dtypes.int32).contiguous().realize()
    with Context(NOOPT=1):
      for name, t, want in [("sub", a - b, [x - y for x, y in zip(A, B)]), ("mul", a * b, [x * y for x, y in zip(A, B)]), ("and", a & b, [x & y for x, y in zip(A, B)]),
                            ("or", a | b, [x | y for x, y in zip(A, B)]), ("xor", a ^ b, [x ^ y for x, y in zip(A, B)]), ("max", a.maximum(b), [max(x, y) for x, y in zip(A, B)]),
                            ("where", (a < b).where(a, b), [x if x < y else y for x, y in zip(A, B)]),
                            ("ne", (a != b).cast(dtypes.int32), [int(x != y) for x, y in zip(A, B)]), ("mul7", a * 7, [x * 7 for x in A]),
                            ("shl", a << 5, [x << 5 for x in A]), ("shr", a >> 1, [x >> 1 for x in A]), ("shl_reg", a << (b & 3), [x << (y & 3) for x, y in zip(A, B)])]:
        with self.subTest(op=name): self.assertEqual(t.tolist(), want)

  def test_kill_model(self): # dsl.KILL_MODEL: with no kill bits set, a register can be read again after an op, a load and a store
    self.assertEqual(self.run_asm(".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\nfadd r4, r0, r2\nfadd r6, r0, r2\nstore r6, 0\n", 1.5, 2.25), 3.75)
    self.assertEqual(self.run_asm(".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\nmovimm r5, #3.0\nfma r4, r0, r2, r5\nfadd r6, r5, r4\nstore r6, 0\n", 1.5, 2.25), 9.375)
    # one address register shared by the load and the store of every iteration (the elementwise kernel shape); with the load's kill bit set only element 0 landed
    a = [float(i) for i in range(8)]
    src = ".buffers 3\nloop r2, #8\naddr r3, r2, 0, #0\nbody\nload r4, 1, r3\nwait\nload r5, 2, r3\nwait\nfmul r6, r4, r5\nstore r6, 0, r3\nendloop\n"
    self.assertEqual(self.run_asm_list(src, a, a), [x * x for x in a])
    # a killed register reads as 0 (the fact the model rests on)
    from tinygrad.runtime.support.agx import asm
    from tinygrad.renderer.agx import dsl
    killing = lambda dst, a, bb, mul=0, wait=True: dsl.FALU_LONG.encode(**dsl.dst_fields(dst), srcb=dsl.reg_operand(bb), mul=mul, killa=1, killb=1, srca=dsl.reg_operand(a), tag=0xc0)
    with patch.object(asm, "i_falu_reg", killing):
      self.assertEqual(self.run_asm(".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\nfadd r4, r0, r2\nfadd r6, r0, r2\nstore r6, 0\n", 1.5, 2.25), 0.0)

  def run_capturing(self, t:Tensor) -> tuple[list, list[str]]: # t.tolist() and the renderer's output for every kernel it ran
    from tinygrad.runtime import ops_agx
    from tinygrad import codegen
    codegen.to_program_cache.clear()
    srcs, orig = [], ops_agx.AGXRenderer.render
    with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: (srcs.append(orig(self, uops)), srcs[-1])[1]): got = t.tolist()
    return got, srcs

  def test_register_reuse(self): # the allocator gives registers back at a value's last read; sequential allocation needed 80+ here
    import re
    from tinygrad.helpers import Context
    A = [float(i) / 8 for i in range(8)]; B = [float(i % 3) - 1 for i in range(8)]
    a, b = Tensor(A).contiguous().realize(), Tensor(B).contiguous().realize()
    x, want = a, A[:]
    for _ in range(40): x = x * b + a; want = [w * bb + aa for w, bb, aa in zip(want, B, A)]
    with Context(NOOPT=1): got, srcs = self.run_capturing(x)
    self.assertEqual(got, want)
    self.assertLess(len(set(re.findall(r"\br(\d+)", srcs[-1]))), 10, srcs[-1])   # distinct registers, not the highest number

  def test_many_live_values(self): # 24 loads live at once, then a chain of fmas: registers into the twenties, all correct
    from tinygrad.helpers import Context
    with Context(NOOPT=1):
      ins = [Tensor([float(i + j) for j in range(8)]).contiguous().realize() for i in range(24)]
      y = sum(ins[2 * i] * ins[2 * i + 1] for i in range(12))
      self.assertEqual(y.tolist(), [sum((2 * i + j) * (2 * i + 1 + j) for i in range(12)) for j in range(8)])

  def test_addr_offset(self): # addr r0 = t + imm; indexed load through r0 (dsl.ADDR, dsl.LOAD_IDX)
    a = [float(i) for i in range(16)]
    self.assertEqual(self.run_asm8(".buffers 3\naddr r0, 0, #1\nload r0, 1, r0\nwait\nstore r0, 0\n", a, a), a[1:9])
    self.assertEqual(self.run_asm8(".buffers 3\naddr r0, 0, #5\nload r0, 1, r0\nwait\nstore r0, 0\n", a, a), a[5:13])

  def test_addr_shift(self): # addr r0 = t << shift
    a = [float(i) for i in range(64)]
    self.assertEqual(self.run_asm8(".buffers 3\naddr r0, 1, #0\nload r0, 1, r0\nwait\nstore r0, 0\n", a, a), a[0:16:2])
    self.assertEqual(self.run_asm8(".buffers 3\naddr r0, 2, #0\nload r0, 1, r0\nwait\nstore r0, 0\n", a, a), a[0:32:4])
    self.assertEqual(self.run_asm8(".buffers 3\naddr r0, 3, #1\nload r0, 1, r0\nwait\nstore r0, 0\n", a, a), a[1:64:8])

  def test_addr_multiply(self): # imul, shift-add with a register, and their composition (row * stride + col)
    a = [float(i) for i in range(1024)]
    self.assertEqual(self.run_asm8(".buffers 3\nimul r4, #7\nload r0, 1, r4\nwait\nstore r0, 0\n", a, a), a[0:56:7])
    self.assertEqual(self.run_asm8(".buffers 3\nimul r4, #100\nload r0, 1, r4\nwait\nstore r0, 0\n", a, a), a[0:800:100])
    self.assertEqual(self.run_asm8(".buffers 3\naddr r4, 1, r1\nload r0, 1, r4\nwait\nstore r0, 0\n", a, a), a[0:24:3])   # (t<<1) + t
    self.assertEqual(self.run_asm8(".buffers 3\naddr r4, 4, #0\nload r0, 1, r4\nwait\nstore r0, 0\n", a, a), a[0:128:16]) # shift 4
    self.assertEqual(self.run_asm8(".buffers 3\nimul r4, #5\naddr r6, 1, r4\nload r0, 1, r6\nwait\nstore r0, 0\n", a, a), a[0:56:7]) # 2t + 5t

  def test_loop(self): # counted loop: s = a[t]; n times s += 1
    a = [float(i) for i in range(8)]
    for c, n in ((3, 4), (6, 4), (3, 3), (3, 5), (3, 16)):
      with self.subTest(counter=c, n=n):
        self.assertEqual(self.run_asm8(f".buffers 3\nload r2, 1\nwait\nloop r{c}, #{n}\nbody\nfadd r2, r2, #1.0\nendloop\nstore r2, 0\n", a, a), [x + n for x in a])

  def test_reduce(self): # the first reduction: out[t] = sum of a[8t .. 8t+7]. the address math sits in the pre section (counter 0..n-1)
    a = [float(i * i % 17) for i in range(64)]
    src = ".buffers 3\nloop r3, #8\naddr r6, 3, r3\nbody\nload r8, 1, r6\nwait\nfadd r2, r2, r8\nendloop\nstore r2, 0\n"
    self.assertEqual(self.run_asm8(src, a, a), [sum(a[8 * t:8 * t + 8]) for t in range(8)])

  def test_addr_any_register(self): # rD = (rA << s) + b with rA a counter, and a store through a computed address
    a = [float(i) for i in range(64)]
    # out[t] = a[2*(t+3)]: r4 = t + 3 (addr on r1), r6 = (r4 << 1) + 0
    self.assertEqual(self.run_asm8(".buffers 3\naddr r4, 0, #3\naddr r6, r4, 1, #0\nload r0, 1, r6\nwait\nstore r0, 0\n", a, a), a[6:22:2])
    # indexed store: out[t+1] = a[t] for t < 7, out[0] untouched (0)
    self.assertEqual(self.run_asm8(".buffers 3\nload r0, 1\nwait\naddr r4, 0, #1\nstore r0, 0, r4\n", a, a)[1:], a[0:7])

  def test_addr_register(self): # the offset can live in another register (probe with r0 busy: 9f ... 04 / load byte5 82)
    a = [float(i) for i in range(16)]
    self.assertEqual(self.run_asm8(".buffers 3\naddr r2, 0, #1\nload r0, 1, r2\nwait\nstore r0, 0\n", a, a), a[1:9])
    self.assertEqual(self.run_asm8(".buffers 3\naddr r5, 0, #2\nload r3, 1, r5\nwait\nstore r3, 0\n", a, a), a[2:10])

  def test_dst_register(self): # the destination nibble in byte 0 of the float ALU op, verified on hardware
    for d in (0, 1, 3, 4, 7, 15):
      with self.subTest(dst=d):
        self.assertEqual(self.run_asm(f".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\nfadd r{d}, r0, r2\nstore r{d}, 0\n", 1.5, 2.25), 3.75)

  def test_odd_registers(self): # loads and operands numbered in 32-bit registers, odd ones included
    self.assertEqual(self.run_asm(".buffers 3\nload r1, 1\nwait\nload r3, 2\nwait\nfmul r5, r1, r3\nstore r5, 0\n", 1.5, 2.25), 3.375)

  def test_load_wait_model(self): # dsl.LOAD_WAIT_MODEL
    # copy: the store is the first consumer and must wait
    self.assertEqual(self.run_asm(".buffers 3\nload r0, 1\nwait\nstore r0, 0\n", 1.5, 2.25), 1.5)
    # first op waits (six-byte, tail c0), the next two are four-byte: (1.5 + 1) + 2.25, squared
    self.assertEqual(self.run_asm(".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\nfadd r0, r0, #1.0\nfadd r0, r0, r2\nfmul r0, r0, r0\nstore r0, 0\n", 1.5, 2.25), 22.5625)
    # one waiting op on r0 covers r2 too: store r2 plainly afterwards
    self.assertEqual(self.run_asm(".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\nfadd r4, r0, #1.0\nstore r2, 0\n", 1.5, 2.25), 2.25)

  def test_fma(self): # dst = a*b + c on hardware, through the assembler
    k = ".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\n%s\nstore r4, 0\n"
    self.assertEqual(self.run_asm(k % "fma r4, r0, r2, r2", 1.5, 2.25), 5.625)
    self.assertEqual(self.run_asm(k % "fma r4, r0, r2, -r2", 1.5, 2.25), 1.125)
    self.assertEqual(self.run_asm(k % "fma r4, r0, r2, #1.0", 1.5, 2.25), 4.375)
    self.assertEqual(self.run_asm(k % "fma r4, r0, r2, #-3.0", 1.5, 2.25), 0.375)
    self.assertEqual(self.run_asm(k % "fma r4, #2.0, r0, r2", 1.5, 2.25), 5.25)

  def test_transcendentals(self): # probes/trans: one hardware op each, the source re-read afterwards (its kill bit is clear), dst above r15 and src r2
    k = ".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\n%s\nstore r6, 0\n"
    for a, b, body, want in [(3.0, 2.0, "exp2 r4, r0\nfadd r6, r4, r0", 11.0), (3.0, 2.0, "exp2 r22, r2\nfadd r6, r22, r2", 6.0), (8.0, 2.0, "log2 r4, r0\nfadd r6, r4, r0", 11.0),
                             (8.0, 4.0, "rsqrt r4, r2\nfadd r6, r4, r2", 4.5), (8.0, 4.0, "rcp r4, r2\nfadd r6, r4, r2", 4.25), (4.0, 0.0, "rsqrts r4, r0\nfadd r6, r4, r0", 4.5),
                             (4.0, 0.0, "rsqrts r4, r2\nfmul r6, r4, r2", 0.0), (4.0, 0.0, "rsqrts r4, r2\nstore r4, 0\nfadd r6, r4, r4", 2.0)]:   # rsqrts(0) = 1: sqrt(0) = 0 * 1
      with self.subTest(a=a, b=b, body=body.split("\n")[0]): self.assertEqual(self.run_asm(k % body, a, b), want)
    self.assertTrue(math.isnan(self.run_asm(k % "rsqrt r4, r2\nfmul r6, r4, r2", 4.0, 0.0)))   # plain rsqrt(0) = inf, 0 * inf = nan: why sqrt uses rsqrts
    # Apple's killing forms zero the source (dsl.KILL_MODEL): exp2's byte 6 bit 5, rcp's bit 4
    self.assertEqual(self.run_asm(k % "raw af 02 56 08 02 00 b0 40 00 00\nfadd r6, r4, r0", 3.0, 2.0), 8.0)
    self.assertEqual(self.run_asm(k % "raw af 00 56 08 02 08 10 48 20 00\nfadd r6, r4, r2", 8.0, 4.0), 0.25)

  def test_transcendental_tensors(self): # exp2/log2/sqrt/1/x on hardware (1 ulp of numpy), sin through the float32 decomposition, softmax
    import numpy as np
    from tinygrad.helpers import Context
    x = np.concatenate([np.linspace(0.01, 40, 41, dtype=np.float32), np.array([0.0, 1e-3, 1e3, 1e-30, 1e30, 3.0], np.float32)])
    t = Tensor(x).contiguous().realize()
    def ulp(got, want): return np.max(np.where(np.isfinite(want) & (want != 0), np.abs(got.astype(np.float64) - want) / np.spacing(np.abs(want)), 0))
    for ctx in ({"NOOPT": 1}, {}):
      with Context(**ctx):
        for name, got, want in [("exp2", t.exp2().numpy(), np.exp2(x)), ("log2", t.log2().numpy(), np.log2(x)), ("sqrt", t.sqrt().numpy(), np.sqrt(x)),
                                ("recip", (1 / t).numpy(), 1 / x), ("div", (t / (t + 3)).numpy(), x / (x + 3)), ("exp", t.exp().numpy(), np.exp(x)), ("log", t.log().numpy(), np.log(x)),
                                ("sin", (t - 20).sin().numpy(), np.sin(x.astype(np.float64) - 20).astype(np.float32))]:
          with self.subTest(op=name, **ctx):
            if name == "sin": self.assertLess(np.abs(got - want)[np.abs(x - 20) < 4e4].max(), 1e-6)   # ulps blow up near the zeros of sin; Cody-Waite range
            elif name in ("exp", "log"): self.assertLess(np.max(np.where(np.isfinite(want) & (want != 0), np.abs(got / want - 1), 0)), 1e-5)   # exp2(x * log2 e) loses ulps as x grows
            else: self.assertLessEqual(ulp(got, want), 1.0, (got, want))
            keep = np.abs(x - 20) < 4e4 if name == "sin" else np.ones_like(x, bool)   # sin(1e30) is outside the float32 reduction's range
            np.testing.assert_array_equal(np.isnan(got[keep]), np.isnan(want[keep])); np.testing.assert_array_equal(np.isinf(got[keep]), np.isinf(want[keep]))
        self.assertEqual((t.sqrt().numpy()[41], (1 / t).numpy()[41]), (0.0, np.inf))   # sqrt(0) = 0, 1/0 = inf
        A = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 0.5, 2.5]], np.float32)
        ref = np.exp(A) / np.exp(A).sum(1, keepdims=True)
        self.assertLess(np.abs(Tensor(A).contiguous().realize().softmax(axis=1).numpy() - ref).max(), 1e-6)
    self.assertTrue(np.isnan(Tensor([np.inf, np.nan]).sin().numpy()).all())

  def test_vector_memory(self): # probes/mem: 14-byte loads/stores with a width and an immediate element offset (dsl.LOAD), 8 threads, x[i] = 2^i
    k = ".buffers 3\n%s\n"
    for body, want in [("load4 r4, 1, r1\nfadd r8, r4, r5\nfadd r8, r8, r6\nfadd r8, r8, r7\nstore r8, 0", [15.0 * 2 ** t for t in range(8)]),   # elements t..t+3 (unscaled index)
                       ("load4 r5, 1, r1\nstore r8, 0", [8.0 * 2 ** t for t in range(8)]),                                                       # any dst, 4th element
                       ("load2 r4, 1, r1, #2\nfadd r8, r4, r5\nstore r8, 0", [12.0 * 2 ** t for t in range(8)]),                                    # offset 2: elements t+2, t+3
                       ("load r4, 1, r1, #1000\nstore r4, 0", [2.0 ** ((1000 + t) % 32) for t in range(8)]),                                       # 10-bit offset
                       ("load4 r4, 1, r1\nstore4 r4, 0, r1", [2.0 ** t for t in range(8)]),                                                         # overlapping 4-wide stores agree
                       ("load4 r4, 1, r1\nstore4 r4, 0, r1, #1", [2.0 ** (t - 1) for t in range(1, 8)])]:                                          # store offset
      with self.subTest(body=body.split("\n")[0]):
        got = self.run_asm8v(k % body)
        self.assertEqual(got[8 - len(want):8], want)
    self.assertEqual(self.run_asm8v(k % "load r4, 1, r1\nstore r4, 0, r1, #8")[8:16], [2.0 ** t for t in range(8)])   # store offset bit 3 (byte 10)

  def run_asm8v(self, src:str, group:bool=False) -> list[float]: # 8 threads (one threadgroup if group) over 2048-element inputs data1[i] = 2^(i mod 32), data2[i] = 100 + i; data0[:16]
    from tinygrad.runtime import ops_agx, ops_metal
    from tinygrad import codegen
    from tinygrad.runtime.support import hcq2
    from tinygrad.engine import realize
    for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
    Device[Device.DEFAULT].wait_timeout_ms = 10000.0
    with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: src), \
         patch.object(ops_metal.MetalQueue, "dims", staticmethod(lambda prg: (1, 1, 1, 8, 1, 1) if group else (8, 1, 1, 1, 1, 1))):
      x, y = Tensor([2.0 ** (i % 32) for i in range(2048)]).contiguous().realize(), Tensor([100.0 + i for i in range(2048)]).contiguous().realize()
      return (x + y).tolist()[:16]

  def test_float4_tensors(self): # the renderer's vector loads/stores and immediate offsets under the optimizer (supports_float4)
    import numpy as np
    A = np.arange(1024, dtype=np.float32).reshape(32, 32); B = (np.arange(1024, dtype=np.float32).reshape(32, 32) % 7) - 3
    a, b = Tensor(A).contiguous().realize(), Tensor(B).contiguous().realize()
    np.testing.assert_allclose((a @ b).numpy(), A @ B, rtol=1e-6)
    np.testing.assert_array_equal((a + b).numpy(), A + B)
    np.testing.assert_array_equal((a * 2 + 1)[:, 4:20].numpy(), (A * 2 + 1)[:, 4:20])
    np.testing.assert_array_equal(a.sum(axis=1).numpy(), A.sum(axis=1))

  def test_threadgroup_memory(self): # probes/tg: tgstore/tgload/barrier with the metadata allocation; one group of 8 threads, x[i] = 2^i
    k = ".buffers 3\n.tgmem 4096\nload r0, 1\n%s\nstore r4, 0\n"   # 1024 floats: the offset cases index past the first 64
    for body, want in [("tgstore r0, r1\nbarrier\ntgload r4, r1", [2.0 ** t for t in range(8)]),
                       ("movimm r3, #0x00000001\nixor r2, r1, r3\ntgstore r0, r1\nbarrier\ntgload r4, r2", [2.0 ** (t ^ 1) for t in range(8)]),   # a neighbour's slot
                       ("tgstore r0, r1\nbarrier\ntgload r4, r1, #2", [2.0 ** (t + 2) for t in range(6)]),                                        # load offset
                       ("tgstore r0, r1, #8\nbarrier\ntgload r4, r1, #8", [2.0 ** t for t in range(8)]),                                          # store offset
                       ("tgstore r0, r1\nbarrier\ntgload4 r4, r1\nfadd r4, r4, r5\nfadd r4, r4, r6\nfadd r4, r4, r7", [15.0 * 2 ** t for t in range(5)]),   # 4-wide
                       ("addr r2, r1, 2, #0\nload4 r4, 1, r2\ntgstore4 r4, r2\nbarrier\ntgload2 r4, r2, #2\nior r4, r5, r5", [2.0 ** (4 * t + 3) for t in range(8)]),   # 4-wide store, 2-wide load, element units
                       ("addr r2, r1, 2, #0\nload4 r4, 1, r2\ntgstore4 r4, r2, #256\nbarrier\nmovimm r8, #0x00000100\niadd r3, r2, r8\ntgload r4, r3, #1", [2.0 ** (4 * t + 1) for t in range(8)])]:   # 11-bit store offset
      with self.subTest(body=body.split("\n")[-1]): self.assertEqual(self.run_asm8v(k % body, group=True)[:len(want)], want)

  def test_group_reduce_tensors(self): # the optimizer's GROUP_REDUCE kernels: threadgroup buffer, barrier, second-stage loop, IF-guarded store
    import numpy as np
    A = np.random.default_rng(0).random((64, 4096), dtype=np.float32)
    t = Tensor(A).contiguous().realize()
    np.testing.assert_allclose(t.sum(axis=1).numpy(), A.sum(1), rtol=1e-5)
    np.testing.assert_array_equal(t.max(axis=1).numpy(), A.max(1))
    self.assertAlmostEqual(Tensor(A.ravel()).contiguous().realize().sum().item() / A.sum(), 1.0, places=5)

  def test_long_loop(self): # counts above 255 hold the count in a register (probes/loop1000: loop_cmp byte 3 = register, loop_enter byte 1 = 00)
    for n in (1000, 999, 256):
      self.assertEqual(self.run_asm(f".buffers 3\nmovimm r3, #0x{n:08x}\nmovimm r4, #0.0\nloop r5, r3\nbody\nfadd r4, r4, #1.0\nendloop\nstore r4, 0\n", 1.5, 2.0), float(n))
    a = [float(i % 7) for i in range(1000)]
    from tinygrad.helpers import Context
    with Context(NOOPT=1): self.assertEqual(Tensor(a).contiguous().realize().sum().item(), sum(a))

  def test_simdgroup_mma(self): # probes/simd: 8x8x8 multiply-accumulate on register pairs; lane layout from frag_elems.metal; sources survive
    import numpy as np
    row = lambda t: ((t >> 1) & 3) + 4 * (t >> 4); col = lambda t: 2 * (t & 1) + 4 * ((t >> 3) & 1)
    def perm(M): return [float(M[row(t), col(t) + k]) for t in range(32) for k in range(2)]
    def unperm(v):
      M = np.zeros((8, 8), np.float32)
      for t in range(32):
        for k in range(2): M[row(t), col(t) + k] = v[2 * t + k]
      return M
    rng = np.random.default_rng(1); A, B, C = (rng.integers(-3, 4, (8, 8)).astype(np.float32) for _ in range(3))
    x, y = perm(A) + perm(B), perm(C) + [0.0] * 64
    k = ".buffers 3\naddr r12, r1, 1, #0\nload2 r0, 1, r12\nload2 r2, 1, r12, #64\nload2 r20, 2, r12\n%s\nstore2 r6, 0, r12\n"
    for body, want in [("mma r6, r0, r2", A @ B), ("mma r6, r0, r2, r20", A @ B + C),
                       ("load2 r22, 2, r12\nmma r40, r0, r2, r20\nmma r6, r40, r2, r22", (A @ B + C) @ B + C),   # a result feeds the next mma's A; C reloaded (dsl.MMA: C is consumed)
                       ("mma r20, r0, r2, r20\nmma r20, r0, r2, r20\nior r6, r20, r20\nior r7, r21, r21", 2 * (A @ B) + C),   # in place: D over C, chained
                       ("mma r8, r0, r2, r20\nmma r6, r0, r2, r8", 2 * (A @ B) + C),   # D as the next C
                       ("mma r8, r0, r2, r20\nior r6, r2, r2\nior r7, r3, r3", B), ("mma r40, r0, r2, r20\nmma r6, r0, r40", A @ (A @ B + C))]:   # B survives; D as B
      with self.subTest(body=body.split("\n")[0]):
        got = unperm(self.run_asm_lanes(k % body, x, y))
        np.testing.assert_array_equal(got, want)

  def run_asm_lanes(self, src:str, x:list[float], y:list[float]) -> list[float]: # 32 threads in one SIMD group; data0[:64]
    from tinygrad.runtime import ops_agx, ops_metal
    from tinygrad import codegen
    from tinygrad.runtime.support import hcq2
    from tinygrad.engine import realize
    for c in (codegen.to_program_cache, hcq2.hcq_compile_cache, hcq2.link_linear_cache, realize.runtime_cache): c.clear()
    Device[Device.DEFAULT].wait_timeout_ms = 10000.0
    with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: src), \
         patch.object(ops_metal.MetalQueue, "dims", staticmethod(lambda prg: (1, 1, 1, 32, 1, 1))):
      return (Tensor(x).contiguous().realize() + Tensor(y).contiguous().realize()).tolist()[:64]

  def test_tensor_core_matmul(self): # tinygrad's tensor-core path (tc.metal layout) lowered to mma; small sizes only (registers), see AGXRenderer.tensor_cores
    import numpy as np
    from tinygrad.renderer import tc
    from tinygrad.runtime import ops_agx
    with patch.object(ops_agx.AGXRenderer, "tensor_cores", tc.metal[:1]):
      for n in (16, 64):
        A, B = np.random.default_rng(n).random((n, n), dtype=np.float32), np.random.default_rng(n + 1).random((n, n), dtype=np.float32)
        with self.subTest(n=n): np.testing.assert_allclose((Tensor(A).contiguous().realize() @ Tensor(B).contiguous().realize()).numpy(), A @ B, rtol=1e-5, atol=1e-4)

  def test_scoreboard_slots(self): # dsl.SCOREBOARD: two slots; a load names its slot, a consumer picks which slot to wait on
    ld = lambda slot, reg=4: "raw " + dsl.LOAD.encode(first=1, more=0, dst=reg, slot=1, areg=1, wcode=dsl.WCODE[1], fmt=6, sb0=2 * slot).hex(" ")
    fa = lambda tag: "raw " + dsl.FALU_LONG.encode(**dsl.dst_fields(8), srcb=dsl.reg_operand(4), mul=0, srca=dsl.reg_operand(4), tag=tag).hex(" ")
    st = lambda reg=8: "raw " + dsl.STORE.encode(first=0, wait=0, src=reg, slot=0, areg=1, last=1, wcode=dsl.WCODE[1], s7=1, fmt=1).hex(" ")
    landed = [2.0 ** (t + 1) for t in range(8)]
    for slot, tag, waits in [(0, 0xc0, True), (0, 0x80, False), (1, 0x80, True), (1, 0xc0, False)]:   # float ALU: tail c0 = slot 0, 80 = slot 1
      with self.subTest(slot=slot, tag=hex(tag)):
        got = self.run_asm8v(".buffers 3\n" + ld(slot) + "\n" + fa(tag) + "\n" + st() + "\n")[:8]
        self.assertEqual(got == landed, waits, got)
    for slot, b1, b2, waits in [(0, 0x02, 0x56, True), (0, 0x02, 0x54, False), (1, 0x82, 0x54, True), (1, 0x02, 0x56, False)]:   # addr family: byte 2 bit 1 = slot 0, byte 1 bit 7 = slot 1
      with self.subTest(slot=slot, b1=hex(b1), b2=hex(b2)):
        e = bytearray(dsl.MMA.encode(dst=8, a=4, b=4, c=0, acc=0, creg=0, wait=0)); e[1], e[2] = b1, b2
        got = self.run_asm8v(".buffers 3\n" + ld(slot) + "\nraw " + bytes(e).hex(" ") + "\n" + st(4) + "\n")[:8]
        self.assertEqual(got == [2.0 ** t for t in range(8)], waits, got)   # the mma's own result is ignored; it is here only to wait

  def test_two_stores(self): # each store followed by STORE_WAIT; the second overwrites the first
    self.assertEqual(self.run_asm(".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\nstore r0, 0\nstore r2, 0\n", 1.5, 2.25), 2.25)

  def test_operand_order(self): # subtraction would tell a from b; with add/mul only, check srca and srcb both read the right registers
    self.assertEqual(self.run_asm(".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\nfadd r4, r2, r0\nstore r4, 0\n", 1.5, 2.25), 3.75)
    self.assertEqual(self.run_asm(".buffers 3\nload r6, 1\nwait\nload r2, 2\nwait\nfmul r0, r6, r2\nstore r0, 0\n", 1.5, 2.25), 3.375)

if __name__ == "__main__": unittest.main()
