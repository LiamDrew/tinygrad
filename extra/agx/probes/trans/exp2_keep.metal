// exp2 with a kept alive: o[t] = exp2(a) + a (dst moves off r0, a's kill bit clears; diff against exp2.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* ba, device float* bb, uint t [[thread_position_in_grid]]) { float a = ba[t], b = bb[t]; o[t] = exp2(a) + a; }
