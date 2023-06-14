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
  #define CASADI_PREFIX(ID) open_phi_tank_filler_ ## ID
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
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_sq CASADI_PREFIX(sq)

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

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[52] = {48, 1, 0, 48, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
static const casadi_int casadi_s1[4] = {0, 1, 0, 0};
static const casadi_int casadi_s2[29] = {25, 1, 0, 25, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
static const casadi_int casadi_s3[5] = {1, 1, 0, 1, 0};

/* open_phi_tank_filler:(i0[48],i1[0],i2[25])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a4, a5, a6, a7, a8, a9;
  a0=arg[2]? arg[2][24] : 0;
  a1=3.6000000000000001e+00;
  a2=arg[2]? arg[2][0] : 0;
  a3=2.7006172839506173e-10;
  a4=arg[0]? arg[0][0] : 0;
  a5=casadi_sq(a4);
  a5=(a4*a5);
  a5=(a3*a5);
  a6=30.;
  a7=6.9999999999999996e-01;
  a8=(a7*a4);
  a8=(a6*a8);
  a5=(a5+a8);
  a5=(a2*a5);
  a5=(a1*a5);
  a8=5.;
  a5=(a5/a8);
  a5=(a5*a4);
  a4=10000.;
  a5=(a5/a4);
  a9=arg[2]? arg[2][1] : 0;
  a10=arg[0]? arg[0][1] : 0;
  a11=casadi_sq(a10);
  a11=(a10*a11);
  a11=(a3*a11);
  a12=(a7*a10);
  a12=(a6*a12);
  a11=(a11+a12);
  a11=(a9*a11);
  a11=(a1*a11);
  a11=(a11/a8);
  a11=(a11*a10);
  a11=(a11/a4);
  a5=(a5+a11);
  a11=arg[2]? arg[2][2] : 0;
  a10=arg[0]? arg[0][2] : 0;
  a12=casadi_sq(a10);
  a12=(a10*a12);
  a12=(a3*a12);
  a13=(a7*a10);
  a13=(a6*a13);
  a12=(a12+a13);
  a12=(a11*a12);
  a12=(a1*a12);
  a12=(a12/a8);
  a12=(a12*a10);
  a12=(a12/a4);
  a5=(a5+a12);
  a12=arg[2]? arg[2][3] : 0;
  a10=arg[0]? arg[0][3] : 0;
  a13=casadi_sq(a10);
  a13=(a10*a13);
  a13=(a3*a13);
  a14=(a7*a10);
  a14=(a6*a14);
  a13=(a13+a14);
  a13=(a12*a13);
  a13=(a1*a13);
  a13=(a13/a8);
  a13=(a13*a10);
  a13=(a13/a4);
  a5=(a5+a13);
  a13=arg[2]? arg[2][4] : 0;
  a10=arg[0]? arg[0][4] : 0;
  a14=casadi_sq(a10);
  a14=(a10*a14);
  a14=(a3*a14);
  a15=(a7*a10);
  a15=(a6*a15);
  a14=(a14+a15);
  a14=(a13*a14);
  a14=(a1*a14);
  a14=(a14/a8);
  a14=(a14*a10);
  a14=(a14/a4);
  a5=(a5+a14);
  a14=arg[2]? arg[2][5] : 0;
  a10=arg[0]? arg[0][5] : 0;
  a15=casadi_sq(a10);
  a15=(a10*a15);
  a15=(a3*a15);
  a16=(a7*a10);
  a16=(a6*a16);
  a15=(a15+a16);
  a15=(a14*a15);
  a15=(a1*a15);
  a15=(a15/a8);
  a15=(a15*a10);
  a15=(a15/a4);
  a5=(a5+a15);
  a15=arg[2]? arg[2][6] : 0;
  a10=arg[0]? arg[0][6] : 0;
  a16=casadi_sq(a10);
  a16=(a10*a16);
  a16=(a3*a16);
  a17=(a7*a10);
  a17=(a6*a17);
  a16=(a16+a17);
  a16=(a15*a16);
  a16=(a1*a16);
  a16=(a16/a8);
  a16=(a16*a10);
  a16=(a16/a4);
  a5=(a5+a16);
  a16=arg[2]? arg[2][7] : 0;
  a10=arg[0]? arg[0][7] : 0;
  a17=casadi_sq(a10);
  a17=(a10*a17);
  a17=(a3*a17);
  a18=(a7*a10);
  a18=(a6*a18);
  a17=(a17+a18);
  a17=(a16*a17);
  a17=(a1*a17);
  a17=(a17/a8);
  a17=(a17*a10);
  a17=(a17/a4);
  a5=(a5+a17);
  a17=arg[2]? arg[2][8] : 0;
  a10=arg[0]? arg[0][8] : 0;
  a18=casadi_sq(a10);
  a18=(a10*a18);
  a18=(a3*a18);
  a19=(a7*a10);
  a19=(a6*a19);
  a18=(a18+a19);
  a18=(a17*a18);
  a18=(a1*a18);
  a18=(a18/a8);
  a18=(a18*a10);
  a18=(a18/a4);
  a5=(a5+a18);
  a18=arg[2]? arg[2][9] : 0;
  a10=arg[0]? arg[0][9] : 0;
  a19=casadi_sq(a10);
  a19=(a10*a19);
  a19=(a3*a19);
  a20=(a7*a10);
  a20=(a6*a20);
  a19=(a19+a20);
  a19=(a18*a19);
  a19=(a1*a19);
  a19=(a19/a8);
  a19=(a19*a10);
  a19=(a19/a4);
  a5=(a5+a19);
  a19=arg[2]? arg[2][10] : 0;
  a10=arg[0]? arg[0][10] : 0;
  a20=casadi_sq(a10);
  a20=(a10*a20);
  a20=(a3*a20);
  a21=(a7*a10);
  a21=(a6*a21);
  a20=(a20+a21);
  a20=(a19*a20);
  a20=(a1*a20);
  a20=(a20/a8);
  a20=(a20*a10);
  a20=(a20/a4);
  a5=(a5+a20);
  a20=arg[2]? arg[2][11] : 0;
  a10=arg[0]? arg[0][11] : 0;
  a21=casadi_sq(a10);
  a21=(a10*a21);
  a21=(a3*a21);
  a22=(a7*a10);
  a22=(a6*a22);
  a21=(a21+a22);
  a21=(a20*a21);
  a21=(a1*a21);
  a21=(a21/a8);
  a21=(a21*a10);
  a21=(a21/a4);
  a5=(a5+a21);
  a21=arg[2]? arg[2][12] : 0;
  a10=arg[0]? arg[0][12] : 0;
  a22=casadi_sq(a10);
  a22=(a10*a22);
  a22=(a3*a22);
  a23=(a7*a10);
  a23=(a6*a23);
  a22=(a22+a23);
  a22=(a21*a22);
  a22=(a1*a22);
  a22=(a22/a8);
  a22=(a22*a10);
  a22=(a22/a4);
  a5=(a5+a22);
  a22=arg[2]? arg[2][13] : 0;
  a10=arg[0]? arg[0][13] : 0;
  a23=casadi_sq(a10);
  a23=(a10*a23);
  a23=(a3*a23);
  a24=(a7*a10);
  a24=(a6*a24);
  a23=(a23+a24);
  a23=(a22*a23);
  a23=(a1*a23);
  a23=(a23/a8);
  a23=(a23*a10);
  a23=(a23/a4);
  a5=(a5+a23);
  a23=arg[2]? arg[2][14] : 0;
  a10=arg[0]? arg[0][14] : 0;
  a24=casadi_sq(a10);
  a24=(a10*a24);
  a24=(a3*a24);
  a25=(a7*a10);
  a25=(a6*a25);
  a24=(a24+a25);
  a24=(a23*a24);
  a24=(a1*a24);
  a24=(a24/a8);
  a24=(a24*a10);
  a24=(a24/a4);
  a5=(a5+a24);
  a24=arg[2]? arg[2][15] : 0;
  a10=arg[0]? arg[0][15] : 0;
  a25=casadi_sq(a10);
  a25=(a10*a25);
  a25=(a3*a25);
  a26=(a7*a10);
  a26=(a6*a26);
  a25=(a25+a26);
  a25=(a24*a25);
  a25=(a1*a25);
  a25=(a25/a8);
  a25=(a25*a10);
  a25=(a25/a4);
  a5=(a5+a25);
  a25=arg[2]? arg[2][16] : 0;
  a10=arg[0]? arg[0][16] : 0;
  a26=casadi_sq(a10);
  a26=(a10*a26);
  a26=(a3*a26);
  a27=(a7*a10);
  a27=(a6*a27);
  a26=(a26+a27);
  a26=(a25*a26);
  a26=(a1*a26);
  a26=(a26/a8);
  a26=(a26*a10);
  a26=(a26/a4);
  a5=(a5+a26);
  a26=arg[2]? arg[2][17] : 0;
  a10=arg[0]? arg[0][17] : 0;
  a27=casadi_sq(a10);
  a27=(a10*a27);
  a27=(a3*a27);
  a28=(a7*a10);
  a28=(a6*a28);
  a27=(a27+a28);
  a27=(a26*a27);
  a27=(a1*a27);
  a27=(a27/a8);
  a27=(a27*a10);
  a27=(a27/a4);
  a5=(a5+a27);
  a27=arg[2]? arg[2][18] : 0;
  a10=arg[0]? arg[0][18] : 0;
  a28=casadi_sq(a10);
  a28=(a10*a28);
  a28=(a3*a28);
  a29=(a7*a10);
  a29=(a6*a29);
  a28=(a28+a29);
  a28=(a27*a28);
  a28=(a1*a28);
  a28=(a28/a8);
  a28=(a28*a10);
  a28=(a28/a4);
  a5=(a5+a28);
  a28=arg[2]? arg[2][19] : 0;
  a10=arg[0]? arg[0][19] : 0;
  a29=casadi_sq(a10);
  a29=(a10*a29);
  a29=(a3*a29);
  a30=(a7*a10);
  a30=(a6*a30);
  a29=(a29+a30);
  a29=(a28*a29);
  a29=(a1*a29);
  a29=(a29/a8);
  a29=(a29*a10);
  a29=(a29/a4);
  a5=(a5+a29);
  a29=arg[2]? arg[2][20] : 0;
  a10=arg[0]? arg[0][20] : 0;
  a30=casadi_sq(a10);
  a30=(a10*a30);
  a30=(a3*a30);
  a31=(a7*a10);
  a31=(a6*a31);
  a30=(a30+a31);
  a30=(a29*a30);
  a30=(a1*a30);
  a30=(a30/a8);
  a30=(a30*a10);
  a30=(a30/a4);
  a5=(a5+a30);
  a30=arg[2]? arg[2][21] : 0;
  a10=arg[0]? arg[0][21] : 0;
  a31=casadi_sq(a10);
  a31=(a10*a31);
  a31=(a3*a31);
  a32=(a7*a10);
  a32=(a6*a32);
  a31=(a31+a32);
  a31=(a30*a31);
  a31=(a1*a31);
  a31=(a31/a8);
  a31=(a31*a10);
  a31=(a31/a4);
  a5=(a5+a31);
  a31=arg[2]? arg[2][22] : 0;
  a10=arg[0]? arg[0][22] : 0;
  a32=casadi_sq(a10);
  a32=(a10*a32);
  a32=(a3*a32);
  a33=(a7*a10);
  a33=(a6*a33);
  a32=(a32+a33);
  a32=(a31*a32);
  a32=(a1*a32);
  a32=(a32/a8);
  a32=(a32*a10);
  a32=(a32/a4);
  a5=(a5+a32);
  a32=arg[2]? arg[2][23] : 0;
  a10=arg[0]? arg[0][23] : 0;
  a33=casadi_sq(a10);
  a33=(a10*a33);
  a33=(a3*a33);
  a34=(a7*a10);
  a6=(a6*a34);
  a33=(a33+a6);
  a33=(a32*a33);
  a33=(a1*a33);
  a33=(a33/a8);
  a33=(a33*a10);
  a33=(a33/a4);
  a5=(a5+a33);
  a33=arg[0]? arg[0][24] : 0;
  a10=casadi_sq(a33);
  a10=(a33*a10);
  a10=(a3*a10);
  a6=40.;
  a34=(a7*a33);
  a34=(a6*a34);
  a10=(a10+a34);
  a2=(a2*a10);
  a2=(a1*a2);
  a2=(a2/a8);
  a2=(a2*a33);
  a2=(a2/a4);
  a5=(a5+a2);
  a2=arg[0]? arg[0][25] : 0;
  a33=casadi_sq(a2);
  a33=(a2*a33);
  a33=(a3*a33);
  a10=(a7*a2);
  a10=(a6*a10);
  a33=(a33+a10);
  a9=(a9*a33);
  a9=(a1*a9);
  a9=(a9/a8);
  a9=(a9*a2);
  a9=(a9/a4);
  a5=(a5+a9);
  a9=arg[0]? arg[0][26] : 0;
  a2=casadi_sq(a9);
  a2=(a9*a2);
  a2=(a3*a2);
  a33=(a7*a9);
  a33=(a6*a33);
  a2=(a2+a33);
  a11=(a11*a2);
  a11=(a1*a11);
  a11=(a11/a8);
  a11=(a11*a9);
  a11=(a11/a4);
  a5=(a5+a11);
  a11=arg[0]? arg[0][27] : 0;
  a9=casadi_sq(a11);
  a9=(a11*a9);
  a9=(a3*a9);
  a2=(a7*a11);
  a2=(a6*a2);
  a9=(a9+a2);
  a12=(a12*a9);
  a12=(a1*a12);
  a12=(a12/a8);
  a12=(a12*a11);
  a12=(a12/a4);
  a5=(a5+a12);
  a12=arg[0]? arg[0][28] : 0;
  a11=casadi_sq(a12);
  a11=(a12*a11);
  a11=(a3*a11);
  a9=(a7*a12);
  a9=(a6*a9);
  a11=(a11+a9);
  a13=(a13*a11);
  a13=(a1*a13);
  a13=(a13/a8);
  a13=(a13*a12);
  a13=(a13/a4);
  a5=(a5+a13);
  a13=arg[0]? arg[0][29] : 0;
  a12=casadi_sq(a13);
  a12=(a13*a12);
  a12=(a3*a12);
  a11=(a7*a13);
  a11=(a6*a11);
  a12=(a12+a11);
  a14=(a14*a12);
  a14=(a1*a14);
  a14=(a14/a8);
  a14=(a14*a13);
  a14=(a14/a4);
  a5=(a5+a14);
  a14=arg[0]? arg[0][30] : 0;
  a13=casadi_sq(a14);
  a13=(a14*a13);
  a13=(a3*a13);
  a12=(a7*a14);
  a12=(a6*a12);
  a13=(a13+a12);
  a15=(a15*a13);
  a15=(a1*a15);
  a15=(a15/a8);
  a15=(a15*a14);
  a15=(a15/a4);
  a5=(a5+a15);
  a15=arg[0]? arg[0][31] : 0;
  a14=casadi_sq(a15);
  a14=(a15*a14);
  a14=(a3*a14);
  a13=(a7*a15);
  a13=(a6*a13);
  a14=(a14+a13);
  a16=(a16*a14);
  a16=(a1*a16);
  a16=(a16/a8);
  a16=(a16*a15);
  a16=(a16/a4);
  a5=(a5+a16);
  a16=arg[0]? arg[0][32] : 0;
  a15=casadi_sq(a16);
  a15=(a16*a15);
  a15=(a3*a15);
  a14=(a7*a16);
  a14=(a6*a14);
  a15=(a15+a14);
  a17=(a17*a15);
  a17=(a1*a17);
  a17=(a17/a8);
  a17=(a17*a16);
  a17=(a17/a4);
  a5=(a5+a17);
  a17=arg[0]? arg[0][33] : 0;
  a16=casadi_sq(a17);
  a16=(a17*a16);
  a16=(a3*a16);
  a15=(a7*a17);
  a15=(a6*a15);
  a16=(a16+a15);
  a18=(a18*a16);
  a18=(a1*a18);
  a18=(a18/a8);
  a18=(a18*a17);
  a18=(a18/a4);
  a5=(a5+a18);
  a18=arg[0]? arg[0][34] : 0;
  a17=casadi_sq(a18);
  a17=(a18*a17);
  a17=(a3*a17);
  a16=(a7*a18);
  a16=(a6*a16);
  a17=(a17+a16);
  a19=(a19*a17);
  a19=(a1*a19);
  a19=(a19/a8);
  a19=(a19*a18);
  a19=(a19/a4);
  a5=(a5+a19);
  a19=arg[0]? arg[0][35] : 0;
  a18=casadi_sq(a19);
  a18=(a19*a18);
  a18=(a3*a18);
  a17=(a7*a19);
  a17=(a6*a17);
  a18=(a18+a17);
  a20=(a20*a18);
  a20=(a1*a20);
  a20=(a20/a8);
  a20=(a20*a19);
  a20=(a20/a4);
  a5=(a5+a20);
  a20=arg[0]? arg[0][36] : 0;
  a19=casadi_sq(a20);
  a19=(a20*a19);
  a19=(a3*a19);
  a18=(a7*a20);
  a18=(a6*a18);
  a19=(a19+a18);
  a21=(a21*a19);
  a21=(a1*a21);
  a21=(a21/a8);
  a21=(a21*a20);
  a21=(a21/a4);
  a5=(a5+a21);
  a21=arg[0]? arg[0][37] : 0;
  a20=casadi_sq(a21);
  a20=(a21*a20);
  a20=(a3*a20);
  a19=(a7*a21);
  a19=(a6*a19);
  a20=(a20+a19);
  a22=(a22*a20);
  a22=(a1*a22);
  a22=(a22/a8);
  a22=(a22*a21);
  a22=(a22/a4);
  a5=(a5+a22);
  a22=arg[0]? arg[0][38] : 0;
  a21=casadi_sq(a22);
  a21=(a22*a21);
  a21=(a3*a21);
  a20=(a7*a22);
  a20=(a6*a20);
  a21=(a21+a20);
  a23=(a23*a21);
  a23=(a1*a23);
  a23=(a23/a8);
  a23=(a23*a22);
  a23=(a23/a4);
  a5=(a5+a23);
  a23=arg[0]? arg[0][39] : 0;
  a22=casadi_sq(a23);
  a22=(a23*a22);
  a22=(a3*a22);
  a21=(a7*a23);
  a21=(a6*a21);
  a22=(a22+a21);
  a24=(a24*a22);
  a24=(a1*a24);
  a24=(a24/a8);
  a24=(a24*a23);
  a24=(a24/a4);
  a5=(a5+a24);
  a24=arg[0]? arg[0][40] : 0;
  a23=casadi_sq(a24);
  a23=(a24*a23);
  a23=(a3*a23);
  a22=(a7*a24);
  a22=(a6*a22);
  a23=(a23+a22);
  a25=(a25*a23);
  a25=(a1*a25);
  a25=(a25/a8);
  a25=(a25*a24);
  a25=(a25/a4);
  a5=(a5+a25);
  a25=arg[0]? arg[0][41] : 0;
  a24=casadi_sq(a25);
  a24=(a25*a24);
  a24=(a3*a24);
  a23=(a7*a25);
  a23=(a6*a23);
  a24=(a24+a23);
  a26=(a26*a24);
  a26=(a1*a26);
  a26=(a26/a8);
  a26=(a26*a25);
  a26=(a26/a4);
  a5=(a5+a26);
  a26=arg[0]? arg[0][42] : 0;
  a25=casadi_sq(a26);
  a25=(a26*a25);
  a25=(a3*a25);
  a24=(a7*a26);
  a24=(a6*a24);
  a25=(a25+a24);
  a27=(a27*a25);
  a27=(a1*a27);
  a27=(a27/a8);
  a27=(a27*a26);
  a27=(a27/a4);
  a5=(a5+a27);
  a27=arg[0]? arg[0][43] : 0;
  a26=casadi_sq(a27);
  a26=(a27*a26);
  a26=(a3*a26);
  a25=(a7*a27);
  a25=(a6*a25);
  a26=(a26+a25);
  a28=(a28*a26);
  a28=(a1*a28);
  a28=(a28/a8);
  a28=(a28*a27);
  a28=(a28/a4);
  a5=(a5+a28);
  a28=arg[0]? arg[0][44] : 0;
  a27=casadi_sq(a28);
  a27=(a28*a27);
  a27=(a3*a27);
  a26=(a7*a28);
  a26=(a6*a26);
  a27=(a27+a26);
  a29=(a29*a27);
  a29=(a1*a29);
  a29=(a29/a8);
  a29=(a29*a28);
  a29=(a29/a4);
  a5=(a5+a29);
  a29=arg[0]? arg[0][45] : 0;
  a28=casadi_sq(a29);
  a28=(a29*a28);
  a28=(a3*a28);
  a27=(a7*a29);
  a27=(a6*a27);
  a28=(a28+a27);
  a30=(a30*a28);
  a30=(a1*a30);
  a30=(a30/a8);
  a30=(a30*a29);
  a30=(a30/a4);
  a5=(a5+a30);
  a30=arg[0]? arg[0][46] : 0;
  a29=casadi_sq(a30);
  a29=(a30*a29);
  a29=(a3*a29);
  a28=(a7*a30);
  a28=(a6*a28);
  a29=(a29+a28);
  a31=(a31*a29);
  a31=(a1*a31);
  a31=(a31/a8);
  a31=(a31*a30);
  a31=(a31/a4);
  a5=(a5+a31);
  a31=arg[0]? arg[0][47] : 0;
  a30=casadi_sq(a31);
  a30=(a31*a30);
  a3=(a3*a30);
  a7=(a7*a31);
  a6=(a6*a7);
  a3=(a3+a6);
  a32=(a32*a3);
  a1=(a1*a32);
  a1=(a1/a8);
  a1=(a1*a31);
  a1=(a1/a4);
  a5=(a5+a1);
  a0=(a0*a5);
  if (res[0]!=0) res[0][0]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int open_phi_tank_filler(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int open_phi_tank_filler_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int open_phi_tank_filler_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void open_phi_tank_filler_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int open_phi_tank_filler_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void open_phi_tank_filler_release(int mem) {
}

CASADI_SYMBOL_EXPORT void open_phi_tank_filler_incref(void) {
}

CASADI_SYMBOL_EXPORT void open_phi_tank_filler_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int open_phi_tank_filler_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int open_phi_tank_filler_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real open_phi_tank_filler_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* open_phi_tank_filler_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* open_phi_tank_filler_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* open_phi_tank_filler_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* open_phi_tank_filler_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int open_phi_tank_filler_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
