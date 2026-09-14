# the simplest AGX program: add two numbers with hand-assembled Apple GPU machine code
#   PYTHONPATH=. DEV=AGX DEBUG=4 python3 examples/agx/add.py
# DEBUG=4 prints the assembly text the renderer emitted and the disassembly of the bytes handed to the GPU.
# (DEBUG>=5 currently fails inside hcq2's generic printers for every HCQ2 device, Metal included.)
from tinygrad import Tensor

a = Tensor([1.5]).contiguous().realize()
b = Tensor([2.25]).contiguous().realize()
print("1.5 + 2.25 =", (a + b).item())
