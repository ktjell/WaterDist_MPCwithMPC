#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:02:33 2023

@author: kst
"""

import sys
sys.path.insert(1, "/home/pi/WaterDist_MPCwithMPC/my_optimizers/tester")
import tester

y = 16

solver = tester.solver()
print('Solver succesfully started.')

print('Optimizing...')
result = solver.run(p = y)
s = result.solution
print('got result: ', s)

