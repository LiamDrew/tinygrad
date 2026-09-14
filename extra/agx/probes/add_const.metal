// reference: prog_add. buf[t] += 3.0 (E4M3 immediate)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, uint t [[thread_position_in_grid]]) { b0[t] = b0[t] + 3.0f; }
