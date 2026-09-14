// convert: o[t] = (float)ba[t]; (one load, one convert, one store)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device int* ba, uint t [[thread_position_in_grid]]) { o[t] = (float)ba[t]; }
