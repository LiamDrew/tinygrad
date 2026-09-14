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
    self.assertEqual(body, [".preamble 64 bytes", "get_tid", "load r0, 1 ; first ; more", "wait", "load r2, 2", "wait", "fadd r4, r0, r2 ; killa ; killb",
                            "store r4, 0 ; last", "store_wait", "stop"])
    self.assertFalse(any(t.startswith(".short") for t in body), "every emitted byte decodes")

  def test_apple_bytes_decode(self): # bytes captured from probes/*.metal
    for hexs, want in [("09011c05 00c0".replace(" ", ""), "fadd r0, r2, r0 ; killa ; killb"),   # sum3: b2 = b0 + b1
                       ("09011d0500c0", "fmul r0, r2, r0 ; killa ; killb"),                       # fmul
                       ("c9110015", "fadd r12, r10, r8"),                                        # sums_live: 4-byte form, explicit dst
                       ("3905040100c0", "fadd r3, r0, r2"),                                      # keep_inputs: odd dst
                       ("09c9140180c0", "fadd r0, r0, #3 ; killa"),                              # add_const: E4M3 immediate
                       ("6700440400012000", "load r2, 0"), ("e700540800012100", "store r4, 0 ; last"), ("e700560000012000", "store r0, 0 ; wait")]:
      b = bytes.fromhex(hexs)
      inst = next(i for i in dsl.TABLE if i.matches(b))
      self.assertEqual(dsl.fmt(inst, inst.decode(b)), want, hexs)

  def test_immediates(self):
    text, _, _ = asm.assemble_text("movimm r0, #42.0\nfadd r0, r0, #-3.5\nstore r0, 0\n")
    body = [t for _, _, t in dsl.disassemble(text)]
    self.assertIn("movimm r0, #42", body)
    self.assertIn("fadd r0, r0, #-3.5 ; killa ; killb ; tag 20", body) # no loads pending: not a waiting op

@unittest.skipUnless(Device.DEFAULT == "AGX", "AGX device required")
class TestAGX(unittest.TestCase):
  def test_add_two_numbers(self):
    a, b = Tensor([1.5]).contiguous().realize(), Tensor([2.25]).contiguous().realize()
    self.assertEqual((a + b).item(), 3.75)

  def test_mul_two_numbers(self):
    a, b = Tensor([1.5]).contiguous().realize(), Tensor([2.25]).contiguous().realize()
    self.assertEqual((a * b).item(), 3.375)

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
    with patch.object(ops_agx.AGXRenderer, "render", lambda self, uops: src), \
         patch.object(ops_metal.MetalQueue, "dims", staticmethod(lambda prg: (8, 1, 1, 1, 1, 1))):
      x, y = Tensor(a).contiguous().realize(), Tensor(b).contiguous().realize()
      return (x[:8] + y[:8]).contiguous().tolist() if False else (x + y).tolist()[:8]

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

  def test_two_stores(self): # each store followed by STORE_WAIT; the second overwrites the first
    self.assertEqual(self.run_asm(".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\nstore r0, 0\nstore r2, 0\n", 1.5, 2.25), 2.25)

  def test_operand_order(self): # subtraction would tell a from b; with add/mul only, check srca and srcb both read the right registers
    self.assertEqual(self.run_asm(".buffers 3\nload r0, 1\nwait\nload r2, 2\nwait\nfadd r4, r2, r0\nstore r4, 0\n", 1.5, 2.25), 3.75)
    self.assertEqual(self.run_asm(".buffers 3\nload r6, 1\nwait\nload r2, 2\nwait\nfmul r0, r6, r2\nstore r0, 0\n", 1.5, 2.25), 3.375)

if __name__ == "__main__": unittest.main()
