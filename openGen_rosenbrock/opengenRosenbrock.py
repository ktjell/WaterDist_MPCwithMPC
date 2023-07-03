#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:30:27 2023

@author: kst
"""

import opengen as og
import casadi.casadi as cs

u = cs.SX.sym("u", 5)
p = cs.SX.sym("p", 2)
phi = og.functions.rosenbrock(u, p)
c = cs.vertcat(1.5 * u[0] - u[1],
               cs.fmax(0.0, u[2] - u[3] + 0.1))
bounds = og.constraints.Ball2(radius=1.5)
problem = og.builder.Problem(u, p, phi) \
    .with_penalty_constraints(c) \
    .with_constraints(bounds)
build_config = og.config.BuildConfiguration() \
    .with_build_directory("my_optimizers") \
    .with_build_mode("debug") \
    .with_build_python_bindings()
meta = og.config.OptimizerMeta() \
    .with_optimizer_name("rosenbrock")
builder = og.builder.OpEnOptimizerBuilder(problem, meta,
                                          build_config)
builder.build()