import unittest
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
    self.assertEqual(body, [".preamble 64 bytes", "get_tid", "load r0, 1, r1 ; first ; more", "wait", "load r2, 2, r1", "wait", "fadd r4, r0, r2",
                            "store r4, 0, r1 ; last", "store_wait", "stop"])
    self.assertFalse(any(t.startswith(".short") for t in body), "every emitted byte decodes")

  def test_apple_bytes_decode(self): # bytes captured from probes/*.metal
    for hexs, want in [("09011c05 00c0".replace(" ", ""), "fadd r0, r2, r0 ; killa ; killb"),   # sum3: b2 = b0 + b1
                       ("09011d0500c0", "fmul r0, r2, r0 ; killa ; killb"),                       # fmul
                       ("c9110015", "fadd r12, r10, r8"),                                        # sums_live: 4-byte form, explicit dst
                       ("3905040100c0", "fadd r3, r0, r2"),                                      # keep_inputs: odd dst
                       ("09c9140180c0", "fadd r0, r0, #3 ; killa"),                              # add_const: E4M3 immediate
                       ("6700440400012000", "load r2, 0, r1"), ("e700540800012100", "store r4, 0, r1 ; last"), ("e700560000012000", "store r0, 0, r1 ; wait"),
                       ("6700440000822000", "load r0, 0, r2 ; killa")]:   # probes/off1.metal: byte5 82 = areg r2 with its kill bit
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
    body = [(bs.hex(), t) for _, bs, t in dsl.disassemble(text)][6:-3]
    self.assertEqual([t for _, t in body], ["isub r4, r0, r2", "ineg r5, r0", "imul r6, r0, r2", "iand r7, r0, r2", "ior r8, r0, r2", "ixor r9, r0, r2",
                                            "imax r10, r0, r2", "imin r22, r0, r2", "isel r12, r0, r2, r4, r0", "isel r13, r0, r2, r4, r5",
                                            "icmp.lt r14, r0, r2", "icmp.ne r30, r0, r2"])
    text, _, _ = asm.assemble_text("load r0, 1\nwait\nload r2, 2\nwait\nishl r4, r0, #5\nishr r5, r2, #3\nushr r6, r0, #1\nishl r7, r0, r2\nishr r8, r2, r0\nushr r9, r0, r2\nstore r4, 0\n")
    self.assertEqual([t for _, _, t in dsl.disassemble(text)][6:-3], ["ishl r4, r0, #5", "ishr r5, r2, #3", "ushr r6, r0, #1", "ishl r7, r0, r2", "ishr r8, r2, r0", "ushr r9, r0, r2"])
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
    self.assertLess(max(int(r) for r in re.findall(r"\br(\d+)", srcs[-1])), 10, srcs[-1])

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

  def test_two_stores(self): # each store followed by STORE_WAIT; the second overwrites the first
    self.assertEqual(self.run_asm(".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\nstore r0, 0\nstore r2, 0\n", 1.5, 2.25), 2.25)

  def test_operand_order(self): # subtraction would tell a from b; with add/mul only, check srca and srcb both read the right registers
    self.assertEqual(self.run_asm(".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\nfadd r4, r2, r0\nstore r4, 0\n", 1.5, 2.25), 3.75)
    self.assertEqual(self.run_asm(".buffers 3\nload r6, 1\nwait\nload r2, 2\nwait\nfmul r0, r6, r2\nstore r0, 0\n", 1.5, 2.25), 3.375)

if __name__ == "__main__": unittest.main()
