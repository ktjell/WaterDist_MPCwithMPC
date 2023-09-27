/* This file was automatically generated by CasADi 3.6.2.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) open_mapping_f1_tank_filler_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[52] = {48, 1, 0, 48, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
static const casadi_int casadi_s1[223] = {219, 1, 0, 219, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218};

/* open_mapping_f1_tank_filler:(i0[48],i1[219])->(o0[48]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a4, a5, a6, a7, a8, a9;
  a0=arg[1]? arg[1][171] : 0;
  a1=240.;
  a2=arg[1]? arg[1][168] : 0;
  a3=(a1*a2);
  a4=arg[0]? arg[0][0] : 0;
  a3=(a3+a4);
  a5=arg[0]? arg[0][24] : 0;
  a3=(a3+a5);
  a6=arg[1]? arg[1][24] : 0;
  a3=(a3-a6);
  a0=(a0*a3);
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[1]? arg[1][172] : 0;
  a3=(a1*a2);
  a7=arg[0]? arg[0][1] : 0;
  a8=(a7+a4);
  a3=(a3+a8);
  a9=arg[0]? arg[0][25] : 0;
  a9=(a9+a5);
  a3=(a3+a9);
  a5=arg[1]? arg[1][25] : 0;
  a5=(a5+a6);
  a3=(a3-a5);
  a0=(a0*a3);
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[1]? arg[1][173] : 0;
  a3=(a1*a2);
  a6=arg[0]? arg[0][2] : 0;
  a8=(a6+a8);
  a3=(a3+a8);
  a10=arg[0]? arg[0][26] : 0;
  a10=(a10+a9);
  a3=(a3+a10);
  a9=arg[1]? arg[1][26] : 0;
  a9=(a9+a5);
  a3=(a3-a9);
  a0=(a0*a3);
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[1]? arg[1][174] : 0;
  a3=(a1*a2);
  a5=arg[0]? arg[0][3] : 0;
  a8=(a5+a8);
  a3=(a3+a8);
  a11=arg[0]? arg[0][27] : 0;
  a11=(a11+a10);
  a3=(a3+a11);
  a10=arg[1]? arg[1][27] : 0;
  a10=(a10+a9);
  a3=(a3-a10);
  a0=(a0*a3);
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[1]? arg[1][175] : 0;
  a3=(a1*a2);
  a9=arg[0]? arg[0][4] : 0;
  a8=(a9+a8);
  a3=(a3+a8);
  a12=arg[0]? arg[0][28] : 0;
  a12=(a12+a11);
  a3=(a3+a12);
  a11=arg[1]? arg[1][28] : 0;
  a11=(a11+a10);
  a3=(a3-a11);
  a0=(a0*a3);
  if (res[0]!=0) res[0][4]=a0;
  a0=arg[1]? arg[1][176] : 0;
  a3=(a1*a2);
  a10=arg[0]? arg[0][5] : 0;
  a8=(a10+a8);
  a3=(a3+a8);
  a13=arg[0]? arg[0][29] : 0;
  a13=(a13+a12);
  a3=(a3+a13);
  a12=arg[1]? arg[1][29] : 0;
  a12=(a12+a11);
  a3=(a3-a12);
  a0=(a0*a3);
  if (res[0]!=0) res[0][5]=a0;
  a0=arg[1]? arg[1][177] : 0;
  a3=(a1*a2);
  a11=arg[0]? arg[0][6] : 0;
  a8=(a11+a8);
  a3=(a3+a8);
  a14=arg[0]? arg[0][30] : 0;
  a14=(a14+a13);
  a3=(a3+a14);
  a13=arg[1]? arg[1][30] : 0;
  a13=(a13+a12);
  a3=(a3-a13);
  a0=(a0*a3);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][178] : 0;
  a3=(a1*a2);
  a12=arg[0]? arg[0][7] : 0;
  a8=(a12+a8);
  a3=(a3+a8);
  a15=arg[0]? arg[0][31] : 0;
  a15=(a15+a14);
  a3=(a3+a15);
  a14=arg[1]? arg[1][31] : 0;
  a14=(a14+a13);
  a3=(a3-a14);
  a0=(a0*a3);
  if (res[0]!=0) res[0][7]=a0;
  a0=arg[1]? arg[1][179] : 0;
  a3=(a1*a2);
  a13=arg[0]? arg[0][8] : 0;
  a8=(a13+a8);
  a3=(a3+a8);
  a16=arg[0]? arg[0][32] : 0;
  a16=(a16+a15);
  a3=(a3+a16);
  a15=arg[1]? arg[1][32] : 0;
  a15=(a15+a14);
  a3=(a3-a15);
  a0=(a0*a3);
  if (res[0]!=0) res[0][8]=a0;
  a0=arg[1]? arg[1][180] : 0;
  a3=(a1*a2);
  a14=arg[0]? arg[0][9] : 0;
  a8=(a14+a8);
  a3=(a3+a8);
  a17=arg[0]? arg[0][33] : 0;
  a17=(a17+a16);
  a3=(a3+a17);
  a16=arg[1]? arg[1][33] : 0;
  a16=(a16+a15);
  a3=(a3-a16);
  a0=(a0*a3);
  if (res[0]!=0) res[0][9]=a0;
  a0=arg[1]? arg[1][181] : 0;
  a3=(a1*a2);
  a15=arg[0]? arg[0][10] : 0;
  a8=(a15+a8);
  a3=(a3+a8);
  a18=arg[0]? arg[0][34] : 0;
  a18=(a18+a17);
  a3=(a3+a18);
  a17=arg[1]? arg[1][34] : 0;
  a17=(a17+a16);
  a3=(a3-a17);
  a0=(a0*a3);
  if (res[0]!=0) res[0][10]=a0;
  a0=arg[1]? arg[1][182] : 0;
  a3=(a1*a2);
  a16=arg[0]? arg[0][11] : 0;
  a8=(a16+a8);
  a3=(a3+a8);
  a19=arg[0]? arg[0][35] : 0;
  a19=(a19+a18);
  a3=(a3+a19);
  a18=arg[1]? arg[1][35] : 0;
  a18=(a18+a17);
  a3=(a3-a18);
  a0=(a0*a3);
  if (res[0]!=0) res[0][11]=a0;
  a0=arg[1]? arg[1][183] : 0;
  a3=(a1*a2);
  a17=arg[0]? arg[0][12] : 0;
  a8=(a17+a8);
  a3=(a3+a8);
  a20=arg[0]? arg[0][36] : 0;
  a20=(a20+a19);
  a3=(a3+a20);
  a19=arg[1]? arg[1][36] : 0;
  a19=(a19+a18);
  a3=(a3-a19);
  a0=(a0*a3);
  if (res[0]!=0) res[0][12]=a0;
  a0=arg[1]? arg[1][184] : 0;
  a3=(a1*a2);
  a18=arg[0]? arg[0][13] : 0;
  a8=(a18+a8);
  a3=(a3+a8);
  a21=arg[0]? arg[0][37] : 0;
  a21=(a21+a20);
  a3=(a3+a21);
  a20=arg[1]? arg[1][37] : 0;
  a20=(a20+a19);
  a3=(a3-a20);
  a0=(a0*a3);
  if (res[0]!=0) res[0][13]=a0;
  a0=arg[1]? arg[1][185] : 0;
  a3=(a1*a2);
  a19=arg[0]? arg[0][14] : 0;
  a8=(a19+a8);
  a3=(a3+a8);
  a22=arg[0]? arg[0][38] : 0;
  a22=(a22+a21);
  a3=(a3+a22);
  a21=arg[1]? arg[1][38] : 0;
  a21=(a21+a20);
  a3=(a3-a21);
  a0=(a0*a3);
  if (res[0]!=0) res[0][14]=a0;
  a0=arg[1]? arg[1][186] : 0;
  a3=(a1*a2);
  a20=arg[0]? arg[0][15] : 0;
  a8=(a20+a8);
  a3=(a3+a8);
  a23=arg[0]? arg[0][39] : 0;
  a23=(a23+a22);
  a3=(a3+a23);
  a22=arg[1]? arg[1][39] : 0;
  a22=(a22+a21);
  a3=(a3-a22);
  a0=(a0*a3);
  if (res[0]!=0) res[0][15]=a0;
  a0=arg[1]? arg[1][187] : 0;
  a3=(a1*a2);
  a21=arg[0]? arg[0][16] : 0;
  a8=(a21+a8);
  a3=(a3+a8);
  a24=arg[0]? arg[0][40] : 0;
  a24=(a24+a23);
  a3=(a3+a24);
  a23=arg[1]? arg[1][40] : 0;
  a23=(a23+a22);
  a3=(a3-a23);
  a0=(a0*a3);
  if (res[0]!=0) res[0][16]=a0;
  a0=arg[1]? arg[1][188] : 0;
  a3=(a1*a2);
  a22=arg[0]? arg[0][17] : 0;
  a8=(a22+a8);
  a3=(a3+a8);
  a25=arg[0]? arg[0][41] : 0;
  a25=(a25+a24);
  a3=(a3+a25);
  a24=arg[1]? arg[1][41] : 0;
  a24=(a24+a23);
  a3=(a3-a24);
  a0=(a0*a3);
  if (res[0]!=0) res[0][17]=a0;
  a0=arg[1]? arg[1][189] : 0;
  a3=(a1*a2);
  a23=arg[0]? arg[0][18] : 0;
  a8=(a23+a8);
  a3=(a3+a8);
  a26=arg[0]? arg[0][42] : 0;
  a26=(a26+a25);
  a3=(a3+a26);
  a25=arg[1]? arg[1][42] : 0;
  a25=(a25+a24);
  a3=(a3-a25);
  a0=(a0*a3);
  if (res[0]!=0) res[0][18]=a0;
  a0=arg[1]? arg[1][190] : 0;
  a3=(a1*a2);
  a24=arg[0]? arg[0][19] : 0;
  a8=(a24+a8);
  a3=(a3+a8);
  a27=arg[0]? arg[0][43] : 0;
  a27=(a27+a26);
  a3=(a3+a27);
  a26=arg[1]? arg[1][43] : 0;
  a26=(a26+a25);
  a3=(a3-a26);
  a0=(a0*a3);
  if (res[0]!=0) res[0][19]=a0;
  a0=arg[1]? arg[1][191] : 0;
  a3=(a1*a2);
  a25=arg[0]? arg[0][20] : 0;
  a8=(a25+a8);
  a3=(a3+a8);
  a28=arg[0]? arg[0][44] : 0;
  a28=(a28+a27);
  a3=(a3+a28);
  a27=arg[1]? arg[1][44] : 0;
  a27=(a27+a26);
  a3=(a3-a27);
  a0=(a0*a3);
  if (res[0]!=0) res[0][20]=a0;
  a0=arg[1]? arg[1][192] : 0;
  a3=(a1*a2);
  a26=arg[0]? arg[0][21] : 0;
  a8=(a26+a8);
  a3=(a3+a8);
  a29=arg[0]? arg[0][45] : 0;
  a29=(a29+a28);
  a3=(a3+a29);
  a28=arg[1]? arg[1][45] : 0;
  a28=(a28+a27);
  a3=(a3-a28);
  a0=(a0*a3);
  if (res[0]!=0) res[0][21]=a0;
  a0=arg[1]? arg[1][193] : 0;
  a3=(a1*a2);
  a27=arg[0]? arg[0][22] : 0;
  a8=(a27+a8);
  a3=(a3+a8);
  a30=arg[0]? arg[0][46] : 0;
  a30=(a30+a29);
  a3=(a3+a30);
  a29=arg[1]? arg[1][46] : 0;
  a29=(a29+a28);
  a3=(a3-a29);
  a0=(a0*a3);
  if (res[0]!=0) res[0][22]=a0;
  a0=arg[1]? arg[1][194] : 0;
  a1=(a1*a2);
  a2=arg[0]? arg[0][23] : 0;
  a8=(a2+a8);
  a1=(a1+a8);
  a8=arg[0]? arg[0][47] : 0;
  a8=(a8+a30);
  a1=(a1+a8);
  a8=arg[1]? arg[1][47] : 0;
  a8=(a8+a29);
  a1=(a1-a8);
  a0=(a0*a1);
  if (res[0]!=0) res[0][23]=a0;
  a0=arg[1]? arg[1][195] : 0;
  a1=arg[1]? arg[1][48] : 0;
  a1=(a4+a1);
  a0=(a0*a1);
  if (res[0]!=0) res[0][24]=a0;
  a0=arg[1]? arg[1][196] : 0;
  a7=(a7+a4);
  a4=arg[1]? arg[1][49] : 0;
  a4=(a7+a4);
  a0=(a0*a4);
  if (res[0]!=0) res[0][25]=a0;
  a0=arg[1]? arg[1][197] : 0;
  a6=(a6+a7);
  a7=arg[1]? arg[1][50] : 0;
  a7=(a6+a7);
  a0=(a0*a7);
  if (res[0]!=0) res[0][26]=a0;
  a0=arg[1]? arg[1][198] : 0;
  a5=(a5+a6);
  a6=arg[1]? arg[1][51] : 0;
  a6=(a5+a6);
  a0=(a0*a6);
  if (res[0]!=0) res[0][27]=a0;
  a0=arg[1]? arg[1][199] : 0;
  a9=(a9+a5);
  a5=arg[1]? arg[1][52] : 0;
  a5=(a9+a5);
  a0=(a0*a5);
  if (res[0]!=0) res[0][28]=a0;
  a0=arg[1]? arg[1][200] : 0;
  a10=(a10+a9);
  a9=arg[1]? arg[1][53] : 0;
  a9=(a10+a9);
  a0=(a0*a9);
  if (res[0]!=0) res[0][29]=a0;
  a0=arg[1]? arg[1][201] : 0;
  a11=(a11+a10);
  a10=arg[1]? arg[1][54] : 0;
  a10=(a11+a10);
  a0=(a0*a10);
  if (res[0]!=0) res[0][30]=a0;
  a0=arg[1]? arg[1][202] : 0;
  a12=(a12+a11);
  a11=arg[1]? arg[1][55] : 0;
  a11=(a12+a11);
  a0=(a0*a11);
  if (res[0]!=0) res[0][31]=a0;
  a0=arg[1]? arg[1][203] : 0;
  a13=(a13+a12);
  a12=arg[1]? arg[1][56] : 0;
  a12=(a13+a12);
  a0=(a0*a12);
  if (res[0]!=0) res[0][32]=a0;
  a0=arg[1]? arg[1][204] : 0;
  a14=(a14+a13);
  a13=arg[1]? arg[1][57] : 0;
  a13=(a14+a13);
  a0=(a0*a13);
  if (res[0]!=0) res[0][33]=a0;
  a0=arg[1]? arg[1][205] : 0;
  a15=(a15+a14);
  a14=arg[1]? arg[1][58] : 0;
  a14=(a15+a14);
  a0=(a0*a14);
  if (res[0]!=0) res[0][34]=a0;
  a0=arg[1]? arg[1][206] : 0;
  a16=(a16+a15);
  a15=arg[1]? arg[1][59] : 0;
  a15=(a16+a15);
  a0=(a0*a15);
  if (res[0]!=0) res[0][35]=a0;
  a0=arg[1]? arg[1][207] : 0;
  a17=(a17+a16);
  a16=arg[1]? arg[1][60] : 0;
  a16=(a17+a16);
  a0=(a0*a16);
  if (res[0]!=0) res[0][36]=a0;
  a0=arg[1]? arg[1][208] : 0;
  a18=(a18+a17);
  a17=arg[1]? arg[1][61] : 0;
  a17=(a18+a17);
  a0=(a0*a17);
  if (res[0]!=0) res[0][37]=a0;
  a0=arg[1]? arg[1][209] : 0;
  a19=(a19+a18);
  a18=arg[1]? arg[1][62] : 0;
  a18=(a19+a18);
  a0=(a0*a18);
  if (res[0]!=0) res[0][38]=a0;
  a0=arg[1]? arg[1][210] : 0;
  a20=(a20+a19);
  a19=arg[1]? arg[1][63] : 0;
  a19=(a20+a19);
  a0=(a0*a19);
  if (res[0]!=0) res[0][39]=a0;
  a0=arg[1]? arg[1][211] : 0;
  a21=(a21+a20);
  a20=arg[1]? arg[1][64] : 0;
  a20=(a21+a20);
  a0=(a0*a20);
  if (res[0]!=0) res[0][40]=a0;
  a0=arg[1]? arg[1][212] : 0;
  a22=(a22+a21);
  a21=arg[1]? arg[1][65] : 0;
  a21=(a22+a21);
  a0=(a0*a21);
  if (res[0]!=0) res[0][41]=a0;
  a0=arg[1]? arg[1][213] : 0;
  a23=(a23+a22);
  a22=arg[1]? arg[1][66] : 0;
  a22=(a23+a22);
  a0=(a0*a22);
  if (res[0]!=0) res[0][42]=a0;
  a0=arg[1]? arg[1][214] : 0;
  a24=(a24+a23);
  a23=arg[1]? arg[1][67] : 0;
  a23=(a24+a23);
  a0=(a0*a23);
  if (res[0]!=0) res[0][43]=a0;
  a0=arg[1]? arg[1][215] : 0;
  a25=(a25+a24);
  a24=arg[1]? arg[1][68] : 0;
  a24=(a25+a24);
  a0=(a0*a24);
  if (res[0]!=0) res[0][44]=a0;
  a0=arg[1]? arg[1][216] : 0;
  a26=(a26+a25);
  a25=arg[1]? arg[1][69] : 0;
  a25=(a26+a25);
  a0=(a0*a25);
  if (res[0]!=0) res[0][45]=a0;
  a0=arg[1]? arg[1][217] : 0;
  a27=(a27+a26);
  a26=arg[1]? arg[1][70] : 0;
  a26=(a27+a26);
  a0=(a0*a26);
  if (res[0]!=0) res[0][46]=a0;
  a0=arg[1]? arg[1][218] : 0;
  a2=(a2+a27);
  a27=arg[1]? arg[1][71] : 0;
  a2=(a2+a27);
  a0=(a0*a2);
  if (res[0]!=0) res[0][47]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int open_mapping_f1_tank_filler(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int open_mapping_f1_tank_filler_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int open_mapping_f1_tank_filler_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void open_mapping_f1_tank_filler_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int open_mapping_f1_tank_filler_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void open_mapping_f1_tank_filler_release(int mem) {
}

CASADI_SYMBOL_EXPORT void open_mapping_f1_tank_filler_incref(void) {
}

CASADI_SYMBOL_EXPORT void open_mapping_f1_tank_filler_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int open_mapping_f1_tank_filler_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int open_mapping_f1_tank_filler_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real open_mapping_f1_tank_filler_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* open_mapping_f1_tank_filler_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* open_mapping_f1_tank_filler_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* open_mapping_f1_tank_filler_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* open_mapping_f1_tank_filler_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int open_mapping_f1_tank_filler_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif