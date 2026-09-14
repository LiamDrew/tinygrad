// scalar load with an offset in the index: o[t] = b[t + 1] (diff against ../copy.metal, off1.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* b, uint t [[thread_position_in_grid]]) { o[t] = b[t + 1]; }
