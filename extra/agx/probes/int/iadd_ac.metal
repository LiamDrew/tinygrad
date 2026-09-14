// int primitive: o[t] = a + c, three int inputs (a=r0? b=r2 c=r4): register fields (diff against iadd_ac.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device int* o, device int* ba, device int* bb, device int* bc, uint t [[thread_position_in_grid]]) { int a = ba[t], b = bb[t], c = bc[t]; o[t] = a + c; }
