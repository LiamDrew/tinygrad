// special register: o[t.x] = l.y (diff against tid.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device uint* o, uint3 t [[thread_position_in_grid]], uint3 l [[thread_position_in_threadgroup]]) { o[t.x] = l.y; }
