// control flow: if (bc[t] != 0) o[t] = ba[t]; (a store under a condition; diff against ../copy.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* ba, device int* bc, uint t [[thread_position_in_grid]]) { if (bc[t] != 0) o[t] = ba[t]; }
