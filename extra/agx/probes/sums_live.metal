// (kept as a record: this probe mixes indexed addressing with ALU discovery; prefer one-thing probes)
// target: fadd dst nibble vs store src field. three sums with all inputs kept live
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, device float* b1, device float* b2, device float* b3, device float* o, uint t [[thread_position_in_grid]]) {
  float x0 = b0[t], x1 = b1[t], x2 = b2[t], x3 = b3[t];
  float s0 = x0 + x1, s1 = x2 + x3, s2 = s0 + s1;
  o[7*t+0] = s2; o[7*t+1] = s1; o[7*t+2] = s0; o[7*t+3] = x3; o[7*t+4] = x2; o[7*t+5] = x1; o[7*t+6] = x0; }
