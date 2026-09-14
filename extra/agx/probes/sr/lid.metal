// special register: o[t] = l (diff against tid.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device uint* o, uint t [[thread_position_in_grid]], uint l [[thread_position_in_threadgroup]]) { o[t] = l; }
