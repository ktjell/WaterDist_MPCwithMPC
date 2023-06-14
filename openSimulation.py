#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:42:27 2023

@author: kst
"""




import casadi as ca
import numpy as np
import opengen as og
# import sys
# sys.path.append('../Python_simulation')
from parameters import sups, tank, simu
from plotting import plotting



mng = og.tcp.OptimizerTcpManager('my_optimizers/tank_filler')
mng.start()

pong = mng.ping()                 # check if the server is alive
print(pong)

# price = simu.c[:simu.M].flatten().tolist()
# response =  mng.call(price)
    
# if response.is_ok():
#     # Solver returned a solution
#     solution_data = response.get()
#     u_star = solution_data.solution
#     exit_status = solution_data.exit_status
#     solver_time = solution_data.solve_time_ms
#     print(u_star)
# else:
#     # Invocation failed - an error report is returned
#     solver_error = response.get()
#     error_code = solver_error.code
#     error_msg = solver_error.message


   

       
## Simulation and plotting ###################################
q = np.zeros((simu.ite, simu.N))                  #The optimized flows from pumps
qE = np.zeros((simu.ite, simu.N))
h,hE,V = np.zeros(simu.ite), np.zeros(simu.ite), np.zeros(simu.ite+1)    #Tank level and Volume
cum_q = np.zeros((simu.ite,simu.N))
V[0] = tank.h0*tank.area                          #Start Volume
p = np.zeros((simu.ite,simu.N))    #Pressures
A = np.tril(np.ones((simu.M,simu.M)))
plot = plotting('Plot1')
cost = np.zeros(simu.N)
Qextr = np.zeros((simu.M, simu.N))
U = np.zeros((simu.M,simu.N))


sample_pr_day = int(24/simu.sample_hourly)

for k in range(0,simu.ite): 

      h[k] = V[k]/tank.area   #Level of water in tank: Volume divided by area of tank.
      hE[k] = h[k]
     
     
      par = simu.c[k:k+simu.M].flatten().tolist()
      par.extend(simu.d[:simu.M].flatten().tolist())
      par.append(h[k])
      response =  mng.call(par)
      if response.is_ok():
          # Solver returned a solution
          solution_data = response.get()
          s = solution_data.solution
          U[:,0] = s[:simu.M]
          U[:,1] = s[simu.M:]
      else:
          print('opti error')
          break
    
    
      q[ k,:] = U[0,:]        #Delivered water from pump 1 and 2
      qE[k,:] = U[0,:]
      Q = U[0,:]
      #Calculate cummulated consumption for last 23 hours to use at next iteration
      prevq = q[max(k-(sample_pr_day-2),0):k+1, :]
      prevq_pad = np.pad(prevq, ((simu.M-1 - len(prevq),0),(0,0))) #pad with zeros in front, so array has length M
      Qextr = np.pad(np.cumsum(prevq_pad[::-1], axis = 0)[::-1],((0,1),(0,0)))

      for i in range(simu.N):
          p[k,i] = sups[i].r * q[k,i]**2 + sups[i].Dz   #Calculate the pressure

     
      dV = sum(q[k,:]) - simu.d[k]      #Change of volume in the tank: the sum of supply minus consumption.
      V[k+1] = V[k] + dV                  #Volume in tank: volume of last time stem + change in volume
      ## Done with simulation
     
      #Calculate the commulated cost of solution. (to compare with other solutions)
      for i in range(simu.N):
          cost[i] += simu.c[k] * (1/(simu.dt**2) * sups[i].r * 0.7 * Q[i]**3 + 0.7*Q[i]*(sups[i].Dz -sups[i].p0)) * 3.6 \
          + sups[i].K*Q[i]
     
     
      #Calculating moving cumulated sum for the last 24 hours to check sattisfaction of constraints.
      cum_q[k] = np.sum(q[max(k-(sample_pr_day-1),0):k+1,:], axis = 0)

     
      #Calculating how good the optimator estimates "what happens"
      Ve = np.ones((simu.M,1))*(V[k]) + A @ (U @ np.ones((simu.N,1)) - simu.d[k:k+simu.M])
      he = Ve / tank.area
     

##### TRY TO PLOT MAYBE PREDITION OF EXTRACTION
      prevq = q[max(k-(sample_pr_day-1),0):k, :] #go to k since index k+1 already is in U.
      prevq1 = np.pad(prevq, ((simu.M - len(prevq),0),(0,0))) #pad with zeros in front, so array has length M
      extr = np.vstack((prevq1, U))
      extr = extr.cumsum(axis = 0)
      extr = extr[simu.M:,:]-extr[:simu.M,:] #take only the last simu.M values

      plot.updatePlot(k+1, h[:k+1], q[:k+1,:],simu.d[:k+1],cum_q[:k+1,:], p[:k+1,:], he, extr)

 
print('Commulated cost: ', sum(cost))  
mng.kill()   