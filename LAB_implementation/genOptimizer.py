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
from parameters import p_sts, tank, model


ipv4 = os.popen('ip addr show eth1').read().split("inet ")[1].split("/")[0]
pnr = ips.addr_dict['local_ctr'].index([ipv4, ips.port])


def f(x, c, pump):
    eta = 0.7
    return  c * ( (1/(model.dt**2)) * pump.r * eta * x**3\
                 + eta*x*(pump.Dz - pump.p0)) * 3.6 + pump.K*x
    
kappa = 1#10000

U = cs.SX.sym('U', 2*model.M)
p = cs.SX.sym('p', 7*model.M + 2)


price, demand, extr, lamb, Uglobal, h0, rho = (p[:model.M], 
                                               p[model.M:2*model.M], 
                                               p[2*model.M:3*model.M], 
                                               p[3*model.M:5*model.M], 
                                               p[5*model.M:7*model.M],
                                               p[7*model.M],
                                               p[7*model.M+1])
cost = 0
for k in range(model.M):
    cost += f(U[pnr*model.M+k], price[k],p_sts[pnr]) #cost of producing the water at station i
for k in range(1, model.M):
    cost += (U[pnr*model.M+k] - U[pnr*model.M + k-1])**2 #Norm of changes in Ui
cost += kappa* (ca.sum1(U) - ca.sum1(demand))**2  #Norm of V0 - V(M)
cost += ca.sum1( lamb * (U-Uglobal)) \
      + rho/2 * ca.norm_1(U-Uglobal)**2 
   
#constraints:
seg_ids = [model.M-1, 2*model.M-1]

# 0< U < Qmax
rect1 = og.constraints.Rectangle(xmin=[0]*model.M, xmax=[p_sts[pnr].Qmax]*model.M)
rect2 = og.constraints.Rectangle(xmin=[0]*model.M, xmax=None)
if pnr == 0:
    rect = [rect1, rect2]
else:
    rect = [rect2, rect1]

bounds = og.constraints.CartesianProduct(seg_ids, rect)

# Vmin < V < Vmax
F1 = np.ones(model.M)*h0*tank.area + ca.cumsum(U[:model.M])\
    + ca.cumsum(U[model.M:]) - ca.cumsum(demand)
C1 = og.constraints.Rectangle(xmin=[tank.hmin*tank.area]*model.M,\
                              xmax=[tank.hmax*tank.area]*model.M)

# Extraction constraint
max_extr = np.ones(model.M)*p_sts[pnr].Vmax
F2 = ca.cumsum(U[pnr*model.M:(pnr+1)*model.M]) + extr

#Collect Lagrange constraints into "one":
F = cs.vertcat(F1, F2)

min_list = np.append(np.ones(model.M)*tank.hmin*tank.area,\
                     np.zeros(model.M)).flatten().tolist()
max_list = np.append(np.ones(model.M)*tank.hmax*tank.area,\
                     max_extr).flatten().tolist() 
C = og.constraints.Rectangle(xmin=min_list, xmax=max_list)


#Problem definition
problem = og.builder.Problem(U, p, cost)\
    .with_constraints(bounds)\
    .with_aug_lagrangian_constraints(F, C)\

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

    
