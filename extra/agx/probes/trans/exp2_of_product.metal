// exp2 of a computed value: o[t] = exp2(a * b) (source register moves; diff against exp2.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* ba, device float* bb, uint t [[thread_position_in_grid]]) { float a = ba[t], b = bb[t]; o[t] = exp2(a * b); }
