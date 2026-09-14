// special register: o[t] = t (diff against tid.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device uint* o, uint t [[thread_position_in_grid]]) { o[t] = t; }
