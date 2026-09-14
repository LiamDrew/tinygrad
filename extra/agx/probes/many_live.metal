// (kept as a record: this probe mixes indexed addressing with ALU discovery; prefer one-thing probes)
// target: register numbering. six loads all live, stored in reverse -> six distinct registers in load dst and store src fields
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, device float* b2, device float* b3, device float* b4, device float* b5, device float* o, uint t [[thread_position_in_grid]]) {
  float x0 = b0[t], x1 = b1[t], x2 = b2[t], x3 = b3[t], x4 = b4[t], x5 = b5[t];
  o[6*t+0] = x5; o[6*t+1] = x4; o[6*t+2] = x3; o[6*t+3] = x2; o[6*t+4] = x1; o[6*t+5] = x0; }
