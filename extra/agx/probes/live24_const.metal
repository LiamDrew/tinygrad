// target: movimm destination above r15. live24 plus one constant stored first, so the constant's register is r24 or above
#include <metal_stdlib>
using namespace metal;
struct S { float f[32]; };
kernel void k(device S* b, device S* o, uint t [[thread_position_in_grid]]) {
  float y0 = b[t].f[0] + 1.0f; float y1 = b[t].f[1] + 1.0f; float y2 = b[t].f[2] + 1.0f; float y3 = b[t].f[3] + 1.0f; float y4 = b[t].f[4] + 1.0f; float y5 = b[t].f[5] + 1.0f; float y6 = b[t].f[6] + 1.0f; float y7 = b[t].f[7] + 1.0f; float y8 = b[t].f[8] + 1.0f; float y9 = b[t].f[9] + 1.0f; float y10 = b[t].f[10] + 1.0f; float y11 = b[t].f[11] + 1.0f; float y12 = b[t].f[12] + 1.0f; float y13 = b[t].f[13] + 1.0f; float y14 = b[t].f[14] + 1.0f; float y15 = b[t].f[15] + 1.0f; float y16 = b[t].f[16] + 1.0f; float y17 = b[t].f[17] + 1.0f; float y18 = b[t].f[18] + 1.0f; float y19 = b[t].f[19] + 1.0f; float y20 = b[t].f[20] + 1.0f; float y21 = b[t].f[21] + 1.0f; float y22 = b[t].f[22] + 1.0f; float y23 = b[t].f[23] + 1.0f;
  float c = 3.14159f; o[t].f[24] = c;
  o[t].f[23] = y23; o[t].f[22] = y22; o[t].f[21] = y21; o[t].f[20] = y20; o[t].f[19] = y19; o[t].f[18] = y18; o[t].f[17] = y17; o[t].f[16] = y16; o[t].f[15] = y15; o[t].f[14] = y14; o[t].f[13] = y13; o[t].f[12] = y12; o[t].f[11] = y11; o[t].f[10] = y10; o[t].f[9] = y9; o[t].f[8] = y8; o[t].f[7] = y7; o[t].f[6] = y6; o[t].f[5] = y5; o[t].f[4] = y4; o[t].f[3] = y3; o[t].f[2] = y2; o[t].f[1] = y1; o[t].f[0] = y0; }
