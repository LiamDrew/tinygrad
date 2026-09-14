// control flow: a guarded store inside a loop body (depth bytes of the if sequence; diff against if_int.metal and ../loop4.metal)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o, device float* ba, device int* bc, uint t [[thread_position_in_grid]]) {
  #pragma clang loop unroll(disable)
  for (int i = 0; i < 4; i++) if (bc[t] != 0) o[4*t+i] = ba[4*t+i]; }
