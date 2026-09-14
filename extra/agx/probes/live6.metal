// target: ALU destination registers above r15. 6 values live at once (y_i = b[t].f[i] + 1.0, stored in reverse); one struct buffer each way
#include <metal_stdlib>
using namespace metal;
struct S { float f[32]; };
kernel void k(device S* b, device S* o, uint t [[thread_position_in_grid]]) {
  float y0 = b[t].f[0] + 1.0f; float y1 = b[t].f[1] + 1.0f; float y2 = b[t].f[2] + 1.0f; float y3 = b[t].f[3] + 1.0f; float y4 = b[t].f[4] + 1.0f; float y5 = b[t].f[5] + 1.0f;
  o[t].f[5] = y5; o[t].f[4] = y4; o[t].f[3] = y3; o[t].f[2] = y2; o[t].f[1] = y1; o[t].f[0] = y0; }
