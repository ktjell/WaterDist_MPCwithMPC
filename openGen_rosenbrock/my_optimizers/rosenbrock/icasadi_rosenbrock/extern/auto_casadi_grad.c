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
  #define CASADI_PREFIX(ID) open_grad_phi_rosenbrock_ ## ID
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
#define casadi_fmax CASADI_PREFIX(fmax)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
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

casadi_real casadi_fmax(casadi_real x, casadi_real y) {
/* Pre-c99 compatibility */
#if __STDC_VERSION__ < 199901L
  return x>y ? x : y;
#else
  return fmax(x, y);
#endif
}

static const casadi_int casadi_s0[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};

/* open_grad_phi_rosenbrock:(i0[5],i1,i2[5])->(o0[5]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=1.5000000000000000e+00;
  a1=arg[2]? arg[2][3] : 0;
  a2=arg[0]? arg[0][0] : 0;
  a3=(a0*a2);
  a4=arg[0]? arg[0][1] : 0;
  a3=(a3-a4);
  a3=(a1*a3);
  a3=(a3+a3);
  a5=5.0000000000000000e-01;
  a6=arg[1]? arg[1][0] : 0;
  a5=(a5*a6);
  a3=(a3*a5);
  a1=(a1*a3);
  a0=(a0*a1);
  a3=arg[2]? arg[2][0] : 0;
  a6=(a3-a2);
  a6=(a6+a6);
  a7=arg[2]? arg[2][2] : 0;
  a6=(a6*a7);
  a0=(a0-a6);
  a6=(a2+a2);
  a2=casadi_sq(a2);
  a2=(a4-a2);
  a2=(a2+a2);
  a8=arg[2]? arg[2][1] : 0;
  a9=(a8*a7);
  a2=(a2*a9);
  a6=(a6*a2);
  a0=(a0-a6);
  if (res[0]!=0) res[0][0]=a0;
  a0=(a3-a4);
  a0=(a0+a0);
  a0=(a0*a7);
  a1=(a1+a0);
  a0=(a4+a4);
  a6=arg[0]? arg[0][2] : 0;
  a4=casadi_sq(a4);
  a4=(a6-a4);
  a4=(a4+a4);
  a9=(a8*a7);
  a4=(a4*a9);
  a0=(a0*a4);
  a1=(a1+a0);
  a2=(a2-a1);
  if (res[0]!=0) res[0][1]=a2;
  a2=0.;
  a1=arg[0]? arg[0][3] : 0;
  a0=(a6-a1);
  a9=1.0000000000000001e-01;
  a0=(a0+a9);
  a9=(a2<=a0);
  a10=(a0<=a2);
  a10=(a10+a9);
  a9=(a9/a10);
  a10=arg[2]? arg[2][4] : 0;
  a2=casadi_fmax(a2,a0);
  a2=(a10*a2);
  a2=(a2+a2);
  a2=(a2*a5);
  a10=(a10*a2);
  a9=(a9*a10);
  a10=(a3-a6);
  a10=(a10+a10);
  a10=(a10*a7);
  a10=(a9-a10);
  a2=(a6+a6);
  a6=casadi_sq(a6);
  a6=(a1-a6);
  a6=(a6+a6);
  a5=(a8*a7);
  a6=(a6*a5);
  a2=(a2*a6);
  a10=(a10-a2);
  a10=(a10+a4);
  if (res[0]!=0) res[0][2]=a10;
  a3=(a3-a1);
  a3=(a3+a3);
  a3=(a3*a7);
  a9=(a9+a3);
  a3=(a1+a1);
  a10=arg[0]? arg[0][4] : 0;
  a1=casadi_sq(a1);
  a10=(a10-a1);
  a10=(a10+a10);
  a8=(a8*a7);
  a10=(a10*a8);
  a3=(a3*a10);
  a9=(a9+a3);
  a6=(a6-a9);
  if (res[0]!=0) res[0][3]=a6;
  if (res[0]!=0) res[0][4]=a10;
  return 0;
}

CASADI_SYMBOL_EXPORT int open_grad_phi_rosenbrock(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int open_grad_phi_rosenbrock_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int open_grad_phi_rosenbrock_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void open_grad_phi_rosenbrock_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int open_grad_phi_rosenbrock_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void open_grad_phi_rosenbrock_release(int mem) {
}

CASADI_SYMBOL_EXPORT void open_grad_phi_rosenbrock_incref(void) {
}

CASADI_SYMBOL_EXPORT void open_grad_phi_rosenbrock_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int open_grad_phi_rosenbrock_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int open_grad_phi_rosenbrock_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real open_grad_phi_rosenbrock_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* open_grad_phi_rosenbrock_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* open_grad_phi_rosenbrock_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* open_grad_phi_rosenbrock_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* open_grad_phi_rosenbrock_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int open_grad_phi_rosenbrock_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
