# multiply two numbers with hand-assembled Apple GPU machine code
#   PYTHONPATH=. DEV=AGX DEBUG=4 python3 examples/agx/mul.py
from tinygrad import Tensor

a = Tensor([1.5]).contiguous().realize()
b = Tensor([2.25]).contiguous().realize()
print("1.5 * 2.25 =", (a * b).item())
