// arithmetic shift right by 16 (diff against ishr_imm.metal = by 3)
#include <metal_stdlib>
using namespace metal;
kernel void k(device int* o, device int* ba, device int* bb, uint t [[thread_position_in_grid]]) { int a = ba[t], b = bb[t]; o[t] = a >> 16; }
