// control flow: if (ba[t] < bb[t]) o[t] = ba[t] + 1.0f; (a store under a condition; diff against ../copy.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* ba, device float* bb, uint t [[thread_position_in_grid]]) { if (ba[t] < bb[t]) o[t] = ba[t] + 1.0f; }
