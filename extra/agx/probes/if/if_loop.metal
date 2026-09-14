// control flow: if (bc[t] != 0) o[t] = s; (a store under a condition; diff against ../copy.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* ba, device int* bc, uint t [[thread_position_in_grid]]) { float s = 0; 
  #pragma clang loop unroll(disable)
  for (int i = 0; i < 4; i++) s += ba[4*t+i];
  if (bc[t] != 0) o[t] = s; }
