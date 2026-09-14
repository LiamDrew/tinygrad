// fast reciprocal with a kept alive: o[t] = fast::divide(1.0f, a) + a (diff against fast_divide_recip.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* ba, device float* bb, uint t [[thread_position_in_grid]]) { float a = ba[t], b = bb[t]; o[t] = fast::divide(1.0f, a) + a; }
