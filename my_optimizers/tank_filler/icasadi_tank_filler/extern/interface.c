/*
 * Interface/Wrapper for C functions generated by CasADi
 *
 * CasADi generated the following four files:
 * - auto_casadi_cost.c
 * - auto_casadi_grad.c
 * - auto_casadi_mapping_f1.c
 * - auto_casadi_mapping_f2.c
 * - auto_preconditioning_functions.c
 *
 * This file is autogenerated by Optimization Engine
 * See http://doc.optimization-engine.xyz
 *
 *
 * Metadata:
 * + Optimizer
 *   + name: tank_filler
 *   + version: 0.0.0
 *   + licence: MIT
 * + Problem
 *   + vars: 48
 *   + parameters: 24
 *   + n1: 0
 *   + n2: 0
 *
 */
#include <stdlib.h>

/*
 * This is to be used ONLY for DEBUG purposes
 * Compile with -DTEST_INTERFACE
 */
#ifdef TEST_INTERFACE
#include <stdio.h>
#endif

#include "casadi_memory.h"

/* Number of input variables */
#define NU_TANK_FILLER 48

/* Number of static parameters */
#define NP_TANK_FILLER 24

/* Dimension of F1 (number of ALM constraints) */
#define N1_TANK_FILLER 0

/* Dimension of F2 (number of PM constraints) */
#define N2_TANK_FILLER 0

/* Dimension of xi = (c, y) */
#define NXI_TANK_FILLER 0

/* Preconditioning Flag */



#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif


/* ------EXTERNAL FUNCTIONS (DEFINED IN C FILES)-------------------------------- */

/*
 * CasADi interface for the cost function
 */
extern int open_phi_tank_filler(
    const casadi_real** arg,
    casadi_real** res,
    casadi_int* iw,
    casadi_real* w,
    void* mem);

/*
 * CasADi interface for the gradient of the cost
 */
extern int open_grad_phi_tank_filler(
    const casadi_real** arg,
    casadi_real** res,
    casadi_int* iw,
    casadi_real* w,
    void* mem);

/*
 * CasADi interface for the gradient of mapping F1
 */
extern int open_mapping_f1_tank_filler(
    const casadi_real** arg,
    casadi_real** res,
    casadi_int* iw,
    casadi_real* w,
    void* mem);

/*
 * CasADi interface for the gradient of mapping F2
 */
extern int open_mapping_f2_tank_filler(
    const casadi_real** arg,
    casadi_real** res,
    casadi_int* iw,
    casadi_real* w,
    void* mem);

//#ifdef PRECONDITIONING_TANK_FILLER
/*
 * CasADi interface for cost function preconditioning
 */
extern int open_preconditioning_w_cost_tank_filler(
    const casadi_real** arg,
    casadi_real** res,
    casadi_int* iw,
    casadi_real* w,
    void* mem);

/*
 * CasADi interface for f1 constraints preconditioning
 */
extern int open_preconditioning_w_f1_tank_filler(
    const casadi_real** arg,
    casadi_real** res,
    casadi_int* iw,
    casadi_real* w,
    void* mem);

/*
 * CasADi interface for f2 constraints preconditioning
 */
extern int open_preconditioning_w_f2_tank_filler(
    const casadi_real** arg,
    casadi_real** res,
    casadi_int* iw,
    casadi_real* w,
    void* mem);

/*
 * CasADi interface for initial penalty
 */
extern int open_initial_penalty_tank_filler(
    const casadi_real** arg,
    casadi_real** res,
    casadi_int* iw,
    casadi_real* w,
    void* mem);


/* ------WORKSPACES------------------------------------------------------------- */

/*
 * Integer workspaces
 */
#if COST_SZ_IW_TANK_FILLER > 0
static casadi_int allocated_i_workspace_cost[COST_SZ_IW_TANK_FILLER];  /* cost (int )  */
#else
static casadi_int *allocated_i_workspace_cost = NULL;
#endif

#if GRAD_SZ_IW_TANK_FILLER > 0
static casadi_int allocated_i_workspace_grad[GRAD_SZ_IW_TANK_FILLER];  /* grad (int )  */
#else
static casadi_int *allocated_i_workspace_grad = NULL;
#endif

#if F1_SZ_IW_TANK_FILLER > 0
static casadi_int allocated_i_workspace_f1[F1_SZ_IW_TANK_FILLER];      /* f1 (int )    */
#else
static casadi_int *allocated_i_workspace_f1 = NULL;
#endif

#if F2_SZ_IW_TANK_FILLER > 0
static casadi_int allocated_i_workspace_f2[F2_SZ_IW_TANK_FILLER];      /* f2 (int )    */
#else
static casadi_int *allocated_i_workspace_f2 = NULL;
#endif

#if W_COST_SZ_IW_TANK_FILLER > 0
static casadi_int allocated_i_workspace_w_cost[W_COST_SZ_IW_TANK_FILLER];      /* w_cost (int )    */
#else
static casadi_int *allocated_i_workspace_w_cost = NULL;
#endif

#if W1_SZ_IW_TANK_FILLER > 0
static casadi_int allocated_i_workspace_w1[W1_SZ_IW_TANK_FILLER];      /* w1 (int )    */
#else
static casadi_int *allocated_i_workspace_w1 = NULL;
#endif

#if W2_SZ_IW_TANK_FILLER > 0
static casadi_int allocated_i_workspace_w2[W2_SZ_IW_TANK_FILLER];      /* w2 (int )    */
#else
static casadi_int *allocated_i_workspace_w2 = NULL;
#endif

#if INIT_PENALTY_SZ_IW_TANK_FILLER > 0
static casadi_int allocated_i_workspace_init_penalty[INIT_PENALTY_SZ_IW_TANK_FILLER];      /* init_penalty (int )    */
#else
static casadi_int *allocated_i_workspace_init_penalty = NULL;
#endif


/*
 * Real workspaces
 */
#if COST_SZ_W_TANK_FILLER > 0
static casadi_real allocated_r_workspace_cost[COST_SZ_W_TANK_FILLER];  /* cost (real)  */
#else
static casadi_real *allocated_r_workspace_cost = NULL;
#endif


#if GRAD_SZ_W_TANK_FILLER > 0
static casadi_real allocated_r_workspace_grad[GRAD_SZ_W_TANK_FILLER];  /* grad (real ) */
#else
static casadi_real *allocated_r_workspace_grad = NULL;
#endif

#if F1_SZ_W_TANK_FILLER > 0
static casadi_real allocated_r_workspace_f1[F1_SZ_W_TANK_FILLER];      /* f1 (real )   */
#else
static casadi_real *allocated_r_workspace_f1 = NULL;
#endif

#if F2_SZ_W_TANK_FILLER > 0
static casadi_real allocated_r_workspace_f2[F2_SZ_W_TANK_FILLER];      /* f2 (real )   */
#else
static casadi_real *allocated_r_workspace_f2 = NULL;
#endif

#if W_COST_SZ_W_TANK_FILLER > 0
static casadi_real allocated_r_workspace_w_cost[W_COST_SZ_W_TANK_FILLER];      /* w_cost (real )   */
#else
static casadi_real *allocated_r_workspace_w_cost = NULL;
#endif

#if W1_SZ_W_TANK_FILLER > 0
static casadi_real allocated_r_workspace_w1[W1_SZ_W_TANK_FILLER];      /* w1 (real )   */
#else
static casadi_real *allocated_r_workspace_w1 = NULL;
#endif

#if W2_SZ_W_TANK_FILLER > 0
static casadi_real allocated_r_workspace_w2[W2_SZ_W_TANK_FILLER];      /* w2 (real )   */
#else
static casadi_real *allocated_r_workspace_w2 = NULL;
#endif

#if INIT_PENALTY_SZ_W_TANK_FILLER > 0
static casadi_real allocated_r_workspace_init_penalty[INIT_PENALTY_SZ_W_TANK_FILLER];      /* init_penalty (real )   */
#else
static casadi_real *allocated_r_workspace_init_penalty = NULL;
#endif

/*
 * Result workspaces
 */
#if COST_SZ_RES_TANK_FILLER > 0
static casadi_real *result_space_cost[COST_SZ_RES_TANK_FILLER];       /* cost (res )  */
#else
static casadi_real **result_space_cost = NULL;
#endif

#if GRAD_SZ_RES_TANK_FILLER > 0
static casadi_real *result_space_grad[GRAD_SZ_RES_TANK_FILLER];        /* grad (res )  */
#else
static casadi_real **result_space_grad = NULL;
#endif


#if F1_SZ_RES_TANK_FILLER > 0
static casadi_real *result_space_f1[F1_SZ_RES_TANK_FILLER];            /* f1 (res )    */
#else
static casadi_real **result_space_f1 = NULL;
#endif


#if F2_SZ_RES_TANK_FILLER > 0
static casadi_real *result_space_f2[F2_SZ_RES_TANK_FILLER];            /* f2 (res )    */
#else
static casadi_real **result_space_f2 = NULL;
#endif


#if W_COST_SZ_RES_TANK_FILLER > 0
static casadi_real *result_space_w_cost[W_COST_SZ_RES_TANK_FILLER];            /* w_cost (res )    */
#else
static casadi_real **result_space_w_cost = NULL;
#endif


#if W1_SZ_RES_TANK_FILLER > 0
static casadi_real *result_space_w1[W1_SZ_RES_TANK_FILLER];            /* w1 (res )    */
#else
static casadi_real **result_space_w1 = NULL;
#endif


#if W2_SZ_RES_TANK_FILLER > 0
static casadi_real *result_space_w2[W2_SZ_RES_TANK_FILLER];            /* w2 (res )    */
#else
static casadi_real **result_space_w2 = NULL;
#endif


#if INIT_PENALTY_SZ_RES_TANK_FILLER > 0
static casadi_real *result_space_init_penalty[INIT_PENALTY_SZ_RES_TANK_FILLER];            /* init_penalty (res )    */
#else
static casadi_real **result_space_init_penalty = NULL;
#endif



/* ------U, XI, P, W------------------------------------------------------------ */

/*
 * Space for storing (u, xi, p, w)
 * that is, uxip_space = [u, xi, p, w]
 *
 * The memory layout of the u-xi-p-w space is described below
 *
 * | --- | -- 0
 * |     |
 * |  u  |
 * |     |
 * | --- |
 *
 * | --- | -- NU
 * |     |
 * |  xi |
 * |     |
 * | --- |
 *
 * | --- | -- NU + NXI
 * |     |
 * |  p  |
 * |     |
 * | --- |
 *
 * | --- |
 * | wc  | -- NU + NXI + NP
 * | --- |
 *
 * | --- | -- NU + NXI + NP + 1
 * |     |
 * |  w1 |
 * |     |
 * | --- |
 *
 * | --- | -- NU + NXI + NP + N1 + 1
 * |     |
 * |  w2 |
 * |     |
 * | --- |
 *
 */

#define IDX_XI_TANK_FILLER NU_TANK_FILLER
#define IDX_P_TANK_FILLER  IDX_XI_TANK_FILLER + NXI_TANK_FILLER
#define IDX_WC_TANK_FILLER IDX_P_TANK_FILLER + NP_TANK_FILLER
#define IDX_W1_TANK_FILLER IDX_WC_TANK_FILLER + 1
#define IDX_W2_TANK_FILLER IDX_W1_TANK_FILLER + N1_TANK_FILLER
#define N_UXIPW_TANK_FILLER IDX_W2_TANK_FILLER + N2_TANK_FILLER

static casadi_real uxip_space[N_UXIPW_TANK_FILLER];

/**
 * Return scaling factor of cost function (getter)
 */
casadi_real get_w_cost_tank_filler(void) {
    return uxip_space[IDX_WC_TANK_FILLER];
}

/**
 * This function should be called upon initialisation. The sets all w's to 1.
 */
void init_interface_tank_filler(void) {
    unsigned int i;
    unsigned int offset = IDX_WC_TANK_FILLER;
    unsigned int len = N1_TANK_FILLER + N2_TANK_FILLER + 1;
    for (i = 0; i < len; i++) {
        uxip_space[offset + i] = 1.0;
    }
}

/**
 * Copy (u, xi, p) into uxip_space
 *
 * Input arguments:
 * - `arg = {u, xi, p}`, where `u`, `xi` and `p` are pointer-to-double
 */
static void copy_args_into_uxip_space(const casadi_real** arg) {
    unsigned int i;
    for (i=0; i<NU_TANK_FILLER; i++)  uxip_space[i] = arg[0][i];  /* copy u  */
    for (i=0; i<NXI_TANK_FILLER; i++) uxip_space[IDX_XI_TANK_FILLER+i] = arg[1][i];  /* copy xi */
    for (i=0; i<NP_TANK_FILLER; i++)  uxip_space[IDX_P_TANK_FILLER+i] = arg[2][i];  /* copy p  */
}


 /**
 * Copy (u, p) into uxip_space
 *
 * Input arguments:
 * - `arg = {u, p}`, where `u` and `p` are pointer-to-double
 */
static void copy_args_into_up_space(const casadi_real** arg) {
    unsigned int i;
    for (i=0; i<NU_TANK_FILLER; i++) uxip_space[i] = arg[0][i];  /* copy u  */
    for (i=0; i<NP_TANK_FILLER; i++) uxip_space[IDX_P_TANK_FILLER+i] = arg[1][i];  /* copy p  */
}


/**
 * Cost function
 *
 * Input arguments:
 * - `arg = {u, xi, p}`, where `u`, `xi`, and `p` are pointer-to-double
 * - `res = {cost}`, where `cost` is a pointer-to-double
 */
int cost_function_tank_filler(const casadi_real** arg, casadi_real** res) {
    const casadi_real* args__[COST_SZ_ARG_TANK_FILLER] =
             {uxip_space,  /* :u  */
              uxip_space + IDX_XI_TANK_FILLER,   /* :xi  */
              uxip_space + IDX_P_TANK_FILLER };  /* :p  */
    copy_args_into_uxip_space(arg);

    result_space_cost[0] = res[0];
    return open_phi_tank_filler(
        args__,
        result_space_cost,
        allocated_i_workspace_cost,
        allocated_r_workspace_cost,
        (void*) 0);
}


/**
 * Gradient function
 *
 * Input arguments:
 * - `arg = {u, xi, p}`, where `u`, `xi`, and `p` are pointer-to-double
 * - `res = {grad}`, where `grad` is a pointer-to-double
 */
int grad_cost_function_tank_filler(const casadi_real** arg, casadi_real** res) {
    const casadi_real* args__[GRAD_SZ_ARG_TANK_FILLER] =
            { uxip_space,  /* :u  */
              uxip_space + IDX_XI_TANK_FILLER,  /* :xi  */
              uxip_space + IDX_P_TANK_FILLER};  /* :p   */
    copy_args_into_uxip_space(arg);
    result_space_grad[0] = res[0];
    return open_grad_phi_tank_filler(
        args__,
        result_space_grad,
        allocated_i_workspace_grad,
        allocated_r_workspace_grad,
        (void*) 0);
}

/**
 * Mapping F1
 *
 * Input arguments:
 * - `arg = {u, p}`, where `u` and `p` are pointer-to-double
 * - `res = {F1}`, where `F1` is a pointer-to-double
 */
int mapping_f1_function_tank_filler(const casadi_real** arg, casadi_real** res) {
    /* Array of pointers to where (u, p) are stored */
    const casadi_real* args__[F1_SZ_ARG_TANK_FILLER] =
            {uxip_space,  /* :u   */
            uxip_space + IDX_P_TANK_FILLER};  /* :p  */
    /* Copy given data to variable `uxip_space` */
    copy_args_into_up_space(arg);
    /*
     * The result should be written in result_space_f1
     * (memory has been allocated - see beginning of this file)
     */
    result_space_f1[0] = res[0];
    /*
     * Call auto-generated function open_mapping_f1_tank_filler
     * Implemented in: icasadi/extern/auto_casadi_mapping_f1.c
     */
    return open_mapping_f1_tank_filler(
        args__,
        result_space_f1,
        allocated_i_workspace_f1,
        allocated_r_workspace_f1,
        (void*) 0);
}


/**
 * Mapping F2
 *
 * Input arguments:
 * - `arg = {u, p}`, where `u` and `p` are pointer-to-double
 * - `res = {F1}`, where `F1` is a pointer-to-double
 */
int mapping_f2_function_tank_filler(const casadi_real** arg, casadi_real** res) {
    /* Array of pointers to where (u, p) are stored */
    const casadi_real* args__[F2_SZ_ARG_TANK_FILLER] =
            {uxip_space,  /* :u   */
             uxip_space + IDX_P_TANK_FILLER};  /* :p   */
    /* Copy given data to variable `uxip_space` */
    copy_args_into_up_space(arg);
    /*
     * The result should be written in result_space_f2
     * (memory has been allocated - see beginning of this file)
     */
    result_space_f2[0] = res[0];
    /*
     * Call auto-generated function open_mapping_f2_tank_filler
     * Implemented in: icasadi/extern/auto_casadi_mapping_f2.c
     */
    return open_mapping_f2_tank_filler(
        args__,
        result_space_f2,
        allocated_i_workspace_f2,
        allocated_r_workspace_f2,
        (void*) 0);
}

/**
 * Interface to auto-generated CasADi function for w_cost(u, p)
 *
 * Input arguments:
 *  - arg = {u, theta}
 */
static int preconditioning_w_cost_function_tank_filler(const casadi_real** arg) {
    /* Array of pointers to where (u, p) are stored */
    const casadi_real* args__[W_COST_SZ_ARG_TANK_FILLER] =
            {uxip_space,  /* :u   */
             uxip_space + IDX_P_TANK_FILLER};  /* :p   */
    /* Copy given data to variable `uxip_space` */
    copy_args_into_up_space(arg);
    /*
     * The result should be written in result_space_w_cost
     * (memory has been allocated - see beginning of this file)
     */
    result_space_w_cost[0] = uxip_space + IDX_WC_TANK_FILLER;
    /*
     * Call auto-generated function open_preconditioning_w_cost_tank_filler
     * Implemented in: icasadi/extern/auto_preconditioning_functions.c
     */
    return open_preconditioning_w_cost_tank_filler(
        args__,
        result_space_w_cost,
        allocated_i_workspace_w_cost,
        allocated_r_workspace_w_cost,
        (void*) 0);
}

/**
 * Interface to auto-generated CasADi function for w1(u, p), which computes an
 * n1-dimensional vector of scaling parameters
 */
static int preconditioning_w1_function_tank_filler(const casadi_real** arg) {
    /* Array of pointers to where (u, p) are stored */
    const casadi_real* args__[W1_SZ_ARG_TANK_FILLER] =
            {uxip_space,  /* :u   */
             uxip_space + IDX_P_TANK_FILLER};  /* :p   */
    /* Copy given data to variable `uxip_space` */
    copy_args_into_up_space(arg);
    /*
     * The result should be written in result_space_w1
     * (memory has been allocated - see beginning of this file)
     */
    result_space_w1[0] = uxip_space + IDX_W1_TANK_FILLER;
    /*
     * Call auto-generated function open_preconditioning_w_f1_tank_filler
     * Implemented in: icasadi/extern/auto_preconditioning_functions.c
     */
    return open_preconditioning_w_f1_tank_filler(
        args__,
        result_space_w1,
        allocated_i_workspace_w1,
        allocated_r_workspace_w1,
        (void*) 0);
}

/**
 * Interface to auto-generated CasADi function for w2(u, p), which computes an
 * n2-dimensional vector of scaling parameters
 */
static int preconditioning_w2_function_tank_filler(const casadi_real** arg) {
    /* Array of pointers to where (u, p) are stored */
    const casadi_real* args__[W2_SZ_ARG_TANK_FILLER] =
            {uxip_space,  /* :u   */
             uxip_space + IDX_P_TANK_FILLER};  /* :p   */
    /* Copy given data to variable `uxip_space` */
    copy_args_into_up_space(arg);
    /*
     * The result should be written in result_space_w2
     * (memory has been allocated - see beginning of this file)
     */
    result_space_w2[0] = uxip_space + IDX_W2_TANK_FILLER;
    /*
     * Call auto-generated function open_preconditioning_w_f2_tank_filler
     * Implemented in: icasadi/extern/auto_preconditioning_functions.c
     */
    return open_preconditioning_w_f2_tank_filler(
        args__,
        result_space_w2,
        allocated_i_workspace_w2,
        allocated_r_workspace_w2,
        (void*) 0);
}

/**
 * Interface to auto-generated CasADi function for rho_1(u, theta), which computes the initial
 * penalty parameter. Note that this is a function of u and theta = (p, w_cost, w1, w2) and the
 * caller needs to provide p, w_cost, w1 and w2
 *
 * Input arguments:
 *  - (in )   arg = {u, p}     pointers to u and p (NOT theta, just p); we don't need to provide the preconditioning
 *                             parameters because they are stored in `uxipw_space`; they are only computed once and
 *                             we don't need to move their values around
 *  - (out)   res = {init_penalty}   pointer to initial penalty
 *
 * Output arguments:
 *  - status code (0: all good)
 */
int init_penalty_function_tank_filler(const casadi_real** arg, casadi_real** res) {
    /* Array of pointers to where (u, p) are stored */
    const casadi_real* args__[INIT_PENALTY_SZ_ARG_TANK_FILLER] =
            {uxip_space,  /* :u   */
             uxip_space + IDX_P_TANK_FILLER};  /* :theta   */
    /* Copy given data to variable `uxip_space` */
    copy_args_into_up_space(arg);
    /*
     * The result should be written in result_space_init_penalty
     * (memory has been allocated - see beginning of this file)
     */
    result_space_init_penalty[0] = res[0];
    /*
     * Call auto-generated function 
     * Implemented in: icasadi/extern/auto_preconditioning_functions.c
     */
    return open_initial_penalty_tank_filler(
        args__,
        result_space_init_penalty,
        allocated_i_workspace_init_penalty,
        allocated_r_workspace_init_penalty,
        (void*) 0);
}

/**
 * Compute all preconditioning/scaling factors, w
 *
 * Input arguments:
 * - `arg = {u, p}`, where `u` and `p` are pointer-to-double
 */
int preconditioning_www_tank_filler(const casadi_real** arg) {
    int status_ = 0;
    status_ += preconditioning_w1_function_tank_filler(arg);
    status_ += preconditioning_w2_function_tank_filler(arg);
    status_ += preconditioning_w_cost_function_tank_filler(arg);
    return status_;
}

/*
 * This is to be used ONLY for DEBUG purposes
 * Compile with -DTEST_INTERFACE
 */
#if defined(TEST_INTERFACE) && defined(PRECONDITIONING_TANK_FILLER)

static casadi_real u_test[NU_TANK_FILLER];
static casadi_real p_test[NP_TANK_FILLER];

static void init_up_test(void) {
    unsigned int i;
    for (i=0; i<NU_TANK_FILLER; i++){
        u_test[i] = 20 + i;
    }
    for (i=0; i<NP_TANK_FILLER; i++){
        p_test[i] = 1.5 + 15 * i;
    }
}

static void print_static_array(void){
    unsigned int i;
    for (i=0; i<NU_TANK_FILLER; i++){
        printf("u[%2d] = %4.2f\n", i, uxip_space[i]);
    }
    for (i=0; i<NXI_TANK_FILLER; i++){
        printf("xi[%2d] = %4.2f\n", i, uxip_space[IDX_XI_TANK_FILLER+i]);
    }
    for (i=0; i<NP_TANK_FILLER; i++){
        printf("p[%2d] = %4.2f\n", i, uxip_space[IDX_P_TANK_FILLER+i]);
    }
    printf("w_cost = %g\n", uxip_space[IDX_WC_TANK_FILLER]);
#if N1_TANK_FILLER > 0
     for (i=0; i<N1_TANK_FILLER; i++){
        printf("w1[%2d] = %g\n", i, uxip_space[IDX_W1_TANK_FILLER+i]);
    }
#endif /* IF N1 > 0 */
#if N2_TANK_FILLER > 0
     for (i=0; i<N2_TANK_FILLER; i++){
        printf("w2[%2d] = %g\n", i, uxip_space[IDX_W2_TANK_FILLER+i]);
    }
#endif /* IF N2 > 0 */
}

static casadi_real test_initial_penalty(void) {
    const casadi_real *args[2] = {u_test, p_test};
    casadi_real initial_penalty = -1.;
    casadi_real *res[1] = { &initial_penalty };
    init_penalty_function_tank_filler(args, res);
    return initial_penalty;
}

int main(void) {
    init_interface_tank_filler();
    init_up_test();
    const casadi_real *argz[2] = {u_test, p_test};
    preconditioning_www_tank_filler(argz);

    /*
     * Since this is invoked after `test_w_cost`, `test_w1` and `test_w2`, the ws have been computed previously
     * and are available in `uxipw_space`. The caller does need to provide them
     */
    casadi_real rho1 = test_initial_penalty();
    print_static_array();
    printf("rho1 = %g\n", rho1);
    return 0;
}

#endif /* END of TEST_INTERFACE and PRECONDITIONING_TANK_FILLER */