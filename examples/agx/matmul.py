# 4x4 matmul through hand-assembled Apple GPU machine code: three nested loops, indexed loads, one fma, one thread
#   PYTHONPATH=. DEV=AGX NOOPT=1 python3 examples/agx/matmul.py          (add DEBUG=4 for the assembly, DEBUG=7 for the disassembly)
# NOOPT=1: the renderer lowers the naive kernel only (no upcasts, no vector loads, no threads yet)
from tinygrad import Tensor, Device
Device[Device.DEFAULT].wait_timeout_ms = 10000.0

a = Tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.],
            [12., 13., 14., 15.]]).contiguous().realize()
b = Tensor([[0., 1., 2., 0.],
            [1., 2., 0., 1.],
            [2., 0., 1., 2.],
            [0., 1., 2., 0.]]).contiguous().realize()

want = [[ 5.,  5.,  8.,  5.],
        [17., 21., 28., 17.],
        [29., 37., 48., 29.],
        [41., 53., 68., 41.]]

got = (a @ b).tolist()
for row in got: print(row)
print("correct" if got == want else f"WRONG, want {want}")
