#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 08:33:42 2023

@author: kst
"""


import casadi.casadi as cs
import casadi as ca
import opengen as og
import numpy as np
import os



def f(x, y):
    return x**2+ 5*x*y  
    

x = cs.SX.sym('x', 1)
y = cs.SX.sym('p', 1)


cost = f(x,y)
   


#Problem definition
problem = og.builder.Problem(x,y, cost)#\
    # .with_constraints(bounds)\
    # .with_aug_lagrangian_constraints(F, C)\
    # .with_aug_lagrangian_constraints(F2, C1)
    
    
### with callable python module
build_config = og.config.BuildConfiguration() \
    .with_build_directory("my_optimizers") \
    .with_build_mode("debug") \
    .with_build_python_bindings()
meta = og.config.OptimizerMeta() \
    .with_optimizer_name("tester")
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

