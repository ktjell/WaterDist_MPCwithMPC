#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:37:34 2023

@author: kst
"""



import casadi.casadi as cs
import casadi as ca
import opengen as og
import numpy as np
import os
from ip_config import ipconfigs as ips
from parameters import sups, tank, simu


ipv4 = os.popen('ip addr show eth1').read().split("inet ")[1].split("/")[0]
pnr = ips.addr_dict['local_ctr'].index([ipv4, ips.port])


def f(x, c, sup):
    eta = 0.7
    return  c * ( (1/(simu.dt**2)) * sup.r * eta * x**3 + eta*x*(sup.Dz - sup.p0)) * 3.6 + sup.K*x
    
kappa = 1#10000

U = cs.SX.sym('U', 2*simu.M)
p = cs.SX.sym('p', 7*simu.M + 2)


price, demand, extr, lamb, Uglobal, h0, rho = (p[:simu.M], 
                                               p[simu.M:2*simu.M], 
                                               p[2*simu.M:3*simu.M], 
                                               p[3*simu.M:5*simu.M], 
                                               p[5*simu.M:7*simu.M],
                                               p[7*simu.M],
                                               p[7*simu.M+1])
cost = 0
for k in range(simu.M):
    cost += f(U[pnr*simu.M+k], price[k],sups[pnr]) #cost of producing the water at station i
for k in range(1, simu.M):
    cost += (U[pnr*simu.M+k] - U[pnr*simu.M + k-1])**2 #Norm of changes in Ui
cost += kappa* (ca.sum1(U) - ca.sum1(demand))**2  #Norm of V0 - V(M)
cost += ca.sum1( lamb * (U-Uglobal)) \
      + rho/2 * ca.norm_1(U-Uglobal)**2 
   
#constraints:
seg_ids = [simu.M-1, 2*simu.M-1]

# 0< U < Qmax
rect1 = og.constraints.Rectangle(xmin=[0]*simu.M, xmax=[sups[pnr].Qmax]*simu.M)
rect2 = og.constraints.Rectangle(xmin=[0]*simu.M, xmax=None)
if pnr == 0:
    rect = [rect1, rect2]
else:
    rect = [rect2, rect1]

bounds = og.constraints.CartesianProduct(seg_ids, rect)

# Vmin < V < Vmax

F1 = np.ones(simu.M)*h0*tank.area + ca.cumsum(U[:simu.M]) + ca.cumsum(U[simu.M:]) - ca.cumsum(demand)
C1 = og.constraints.Rectangle(xmin=[tank.hmin*tank.area]*simu.M, xmax=[tank.hmax*tank.area]*simu.M)

# Extraction constraint
max_extr = np.ones(simu.M)*sups[pnr].Vmax
F2 = ca.cumsum(U[pnr*simu.M:(pnr+1)*simu.M]) + extr
# C2 = og.constraints.Rectangle(xmin=[0]*2*simu.M  , xmax = max_extr)

#Collect Lagrange constraints into "one":
F = cs.vertcat(F1, F2)

min_list = np.append(np.ones(simu.M)*tank.hmin*tank.area, np.zeros(simu.M)).flatten().tolist()
max_list = np.append(np.ones(simu.M)*tank.hmax*tank.area, max_extr).flatten().tolist() 
C = og.constraints.Rectangle(xmin=min_list, xmax=max_list)


#Problem definition
problem = og.builder.Problem(U, p, cost)\
    .with_constraints(bounds)\
    .with_aug_lagrangian_constraints(F, C)\
    # .with_aug_lagrangian_constraints(F2, C1)
    
    
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

    

### with tcp interface
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

