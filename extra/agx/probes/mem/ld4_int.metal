// 4-wide int load: o[t] = sum of the int4 b[t]
#include <metal_stdlib>
using namespace metal;
kernel void k(device int* o, device int4* b, uint t [[thread_position_in_grid]]) { int4 v = b[t]; o[t] = v.x + v.y + v.z + v.w; }
