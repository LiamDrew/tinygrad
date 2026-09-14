// fast rsqrt with a kept alive: o[t] = fast::rsqrt(a) + a (diff against fast_rsqrt.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* ba, device float* bb, uint t [[thread_position_in_grid]]) { float a = ba[t], b = bb[t]; o[t] = fast::rsqrt(a) + a; }
