// transcendental probe: o[t] = exp2(a) (diff against ../copy.metal / each other)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* ba, device float* bb, uint t [[thread_position_in_grid]]) { float a = ba[t], b = bb[t]; o[t] = exp2(a); }
