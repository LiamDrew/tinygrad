// special register: o[t.x] = g.y (diff against tid.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device uint* o, uint3 t [[thread_position_in_grid]], uint3 g [[threadgroup_position_in_grid]]) { o[t.x] = g.y; }
