// convert: o[t] = (uint)ba[t]; (one load, one convert, one store)
#include <metal_stdlib>
using namespace metal;
kernel void k(device uint* o, device float* ba, uint t [[thread_position_in_grid]]) { o[t] = (uint)ba[t]; }
