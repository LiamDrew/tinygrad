// special register: o[t] = n (diff against tid.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device uint* o, uint t [[thread_position_in_grid]], uint n [[threads_per_threadgroup]]) { o[t] = n; }
