import unittest
from tinygrad import Tensor, Device
from tinygrad.renderer.agx import dsl
from tinygrad.runtime.support.agx import asm

class TestAGXISA(unittest.TestCase): # no device needed: the table must read back what it writes
  def test_roundtrip(self):
    src = "load r0, 1\nwait\nload r1, 2\nwait\nfadd r2, r0, r1\nstore r2, 0\n"
    text, n, written = asm.assemble_text(src)
    self.assertEqual((n, written), (3, {0}))
    body = [t for _, _, t in dsl.disassemble(text)]
    self.assertEqual(body, [".preamble 64 bytes", "get_tid", "load r0, 1 ; first ; more", "wait", "load r1, 2", "wait", "fadd r0, r1",
                            "store result, 0", "barrier", "stop"])
    self.assertFalse(any(t.startswith(".short") for t in body), "every emitted byte decodes")

  def test_fmul_matches_apple(self): # probes/fmul.metal: Apple's fmul differs from fadd by the low bit of the modifier byte
    text, _, _ = asm.assemble_text("load r0, 0\nwait\nload r1, 1\nwait\nfmul r2, r0, r1\nstore r2, 2\n")
    self.assertIn(bytes.fromhex("09011d0500c0"), text)
    self.assertIn("fmul r0, r1", [t for _, _, t in dsl.disassemble(text)])

  def test_immediates(self):
    text, _, _ = asm.assemble_text("movimm r0, #42.0\nfadd r0, r0, #-3.5\nstore r0, 0\n")
    body = [t for _, _, t in dsl.disassemble(text)]
    self.assertIn("movimm r0, #42", body)
    self.assertIn("fadd r0, #-3.5", body)

@unittest.skipUnless(Device.DEFAULT == "AGX", "AGX device required")
class TestAGX(unittest.TestCase):
  def test_add_two_numbers(self):
    a, b = Tensor([1.5]).contiguous().realize(), Tensor([2.25]).contiguous().realize()
    self.assertEqual((a + b).item(), 3.75)

  def test_mul_two_numbers(self):
    a, b = Tensor([1.5]).contiguous().realize(), Tensor([2.25]).contiguous().realize()
    self.assertEqual((a * b).item(), 3.375)

if __name__ == "__main__": unittest.main()
