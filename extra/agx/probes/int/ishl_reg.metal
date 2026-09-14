// int primitive: o[t] = a << b (diff against iadd_imm.metal / the register form)
#include <metal_stdlib>
using namespace metal;
kernel void k(device int* o, device int* ba, device int* bb, uint t [[thread_position_in_grid]]) { int a = ba[t], b = bb[t]; o[t] = a << b; }
