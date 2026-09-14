// int primitive: o[t] = a * b. two int loads, one op, one store (pairwise diff against iadd.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device int* o, device int* ba, device int* bb, uint t [[thread_position_in_grid]]) { int a = ba[t], b = bb[t]; o[t] = a * b; }
