// special register: o[t.x] = t.z (diff against tid.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device uint* o, uint3 t [[thread_position_in_grid]]) { o[t.x] = t.z; }
