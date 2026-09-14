// load with an immediate element offset: o[t] = b[t].f[4] (diff against off0.metal; base = t * 8 elements)
#include <metal_stdlib>
using namespace metal;
struct S { float f[1024]; };
kernel void k(device float* o, device S* b, uint t [[thread_position_in_grid]]) { o[t] = b[t].f[4]; }
