// reference: prog_movimm. buf[t] = 42.0 (full fp32 via mov_imm)
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* b0, uint t [[thread_position_in_grid]]) { b0[t] = 42.0f; }
