import unittest
from tinygrad import Tensor, Device

@unittest.skipUnless(Device.DEFAULT == "AGX", "AGX device required")
class TestAGX(unittest.TestCase):
  def test_add_two_numbers(self):
    a, b = Tensor([1.5]).contiguous().realize(), Tensor([2.25]).contiguous().realize()
    self.assertEqual((a + b).item(), 3.75)

if __name__ == "__main__": unittest.main()
