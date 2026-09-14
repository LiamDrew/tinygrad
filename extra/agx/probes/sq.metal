// simplest: one op on a loaded value. b1 = b0*b0  (expect six-byte form)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, uint t [[thread_position_in_grid]]) { b1[t] = b0[t] * b0[t]; }
