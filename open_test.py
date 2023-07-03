#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:08:53 2023

@author: kst
"""


import casadi.casadi as cs
import casadi as ca
import opengen as og
import numpy as np
import sys
sys.path.append('../Python_simulation')
from Python_simulation.parameters import sups, tank, simu
# from plotting import plotting



def f(x, c, sup):
    eta = 0.7
    return  c * ( (1/(simu.dt**2)) * sup.r * eta * x**3 + eta*x*(sup.Dz - sup.p0)) * 3.6 + sup.K*x
    
kappa = 1#10000

U = cs.SX.sym('U', 2*simu.M)
p = cs.SX.sym('p', 4*simu.M + 1)


price, demand, h0, extr = (p[:simu.M], p[simu.M:2*simu.M], p[2*simu.M], p[2*simu.M+1:] )

# ext = cs.SX.sym('ext', (simu.M, simu.N))

cost = 0
for i in range(simu.N):
    for k in range(simu.M):
        cost += f(U[i*simu.M+k], price[k],sups[i])
    for k in range(1, simu.M):
        cost += (U[i*simu.M+k] - U[i*simu.M + k-1])**2
 

cost += kappa* (ca.sum1(U) - ca.sum1(demand))**2 

   
#constraints:
seg_ids = [simu.M-1, 2*simu.M-1]

# 0< U < Qmax
rect1 = og.constraints.Rectangle(xmin=[0]*simu.M, xmax=[sups[0].Qmax]*simu.M)
rect2 = og.constraints.Rectangle(xmin=[0]*simu.M, xmax=[sups[1].Qmax]*simu.M)
bounds = og.constraints.CartesianProduct(seg_ids, [rect1,rect2])

# Vmin < V < Vmax

F1 = np.ones(simu.M)*h0*tank.area + ca.cumsum(U[:simu.M]) + ca.cumsum(U[simu.M:]) - ca.cumsum(demand)
C1 = og.constraints.Rectangle(xmin=[tank.hmin*tank.area]*simu.M, xmax=[tank.hmax*tank.area]*simu.M)

# Extraction constraint
max_extr = np.append(np.ones(simu.M)*sups[0].Vmax, np.ones(simu.M)*sups[1].Vmax )
F2 = ca.cumsum(U) + extr
# C2 = og.constraints.Rectangle(xmin=[0]*2*simu.M  , xmax = max_extr)

#Collect Lagrange constraints into "one":
F = cs.vertcat(F1, F2)

min_list = np.append(np.ones(simu.M)*tank.hmin*tank.area, np.zeros(2*simu.M)).flatten().tolist()
max_list = np.append(np.ones(simu.M)*tank.hmax*tank.area, max_extr).flatten().tolist() 
C = og.constraints.Rectangle(xmin=min_list, xmax=max_list)


#Problem definition
problem = og.builder.Problem(U, p, cost)\
    .with_constraints(bounds)\
    .with_aug_lagrangian_constraints(F, C)\
    # .with_aug_lagrangian_constraints(F2, C1)

##### with tcp interface 
# build_config = og.config.BuildConfiguration()\
#     .with_build_directory("my_optimizers")\
#     .with_build_mode("debug")\
#     .with_tcp_interface_config()
# meta = og.config.OptimizerMeta()\
#     .with_optimizer_name("tank_filler")
# solver_config = og.config.SolverConfiguration()\
#     .with_tolerance(1e-5)
# builder = og.builder.OpEnOptimizerBuilder(problem,
#                                           meta,
#                                           build_config,
#                                           solver_config)
# builder.build()

### with callable python module
build_config = og.config.BuildConfiguration() \
    .with_build_directory("my_optimizers") \
    .with_build_mode("debug") \
    .with_build_python_bindings()
meta = og.config.OptimizerMeta() \
    .with_optimizer_name("tank_filler")
builder = og.builder.OpEnOptimizerBuilder(problem, meta,
                                          build_config)
builder.build()

