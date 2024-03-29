#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:48:43 2023

@author: kst
"""

import numpy as np
import cvxpy as cp
import sys
sys.path.append('../Python_simulation')
from Python_simulation.parameters import sups, tank, simu
from Python_simulation.plotting import plotting


# Cost function
def E(x, r, Dz, p0):
    eta = 0.7
    return cp.inv_pos(simu.dt**2) * r * eta * cp.power(x,3) + eta*x*(Dz - p0) 

################################################
## MPC optimization #########################
def opti(sups, g, c, h0, extr):
    kappa = 10000
    U = cp.Variable((simu.M, simu.N))
    A = np.tril(np.ones((simu.M,simu.M)))
    cost = 0
    constr = [ U >= np.zeros((simu.M,simu.N)),
              np.ones((simu.M,1))*(h0*tank.area) + A @ (U @ np.ones((simu.N,1)) - g) >= np.ones((simu.M,1))*tank.hmin*tank.area,
              np.ones((simu.M,1))*(h0*tank.area) + A @ (U @ np.ones((simu.N,1)) - g) <= np.ones((simu.M,1))*tank.hmax*tank.area
              ]
    
    for i in range(simu.N):
        for k in range(simu.M):
            cost += c[k] * E(U[k,i],sups[i].r, sups[i].Dz, sups[i].p0)* 3.6 \
                 + sups[i].K*U[k,i] 
        for k in range(1, simu.M):
            cost += cp.power(cp.norm(U[k,i] - U[k-1,i],2),2)
        # cost += 2*cp.power(cp.norm(U[:,i],2),2)
      
        constr.extend([
                    U[:,i] <= sups[i].Qmax,
                    cp.cumsum(U[:,i]) <= np.ones(simu.M)*sups[i].Vmax - extr[:,i] 
                    # cp.sum(U[:,i]) <= sups[i].Vmax
                    ])

    cost += kappa* cp.power(cp.norm(np.ones((1,simu.M)) @ (U @ np.ones((simu.N,1)) - g),2),2)
    
    problem = cp.Problem(cp.Minimize(cost) , constr)
    problem.solve()#solver = cp.MOSEK)
    status = problem.status

    return U.value[0,:], U.value

       
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

disturbance = np.random.normal(0,5,(simu.ite))

for k in range(0,simu.ite): 
    try:
        h[k] = V[k]/tank.area   #Level of water in tank: Volume divided by area of tank.
        hE[k] = h[k]
        
        
    
        Q , U = opti(sups, simu.d[k:k+simu.M], simu.c[k:k+simu.M], h[k], Qextr)
        q[ k,:] = Q        #Delivered water from pump 1 and 2
        qE[k,:] = Q
        #Calculate cummulated consumption for last 23 hours to use at next iteration
        prevq = q[max(k-22,0):k+1, :]
        prevq_pad = np.pad(prevq, ((simu.M-1 - len(prevq),0),(0,0))) #pad with zeros in front, so array has length M
        Qextr = np.pad(np.cumsum(prevq_pad[::-1], axis = 0)[::-1],((0,1),(0,0)))

        for i in range(simu.N):
            p[k,i] = sups[i].r * q[k,i]**2 + sups[i].Dz   #Calculate the pressure

        
        dV = sum(q[k,:]) - simu.d[k] + disturbance[k]      #Change of volume in the tank: the sum of supply minus consumption.
        V[k+1] = V[k] + dV                  #Volume in tank: volume of last time stem + change in volume
        ## Done with simulation
        
        #Calculate the commulated cost of solution. (to compare with other solutions)
        for i in range(simu.N):
            cost[i] += simu.c[k] * (1/(simu.dt**2) * sups[i].r * 0.7 * Q[i]**3 + 0.7*Q[i]*(sups[i].Dz -sups[i].p0)) * 3.6 \
             + sups[i].K*Q[i]
        
        
        #Calculating moving cumulated sum for the last 24 hours to check sattisfaction of constraints.
        cum_q[k] = np.sum(q[max(k-23,0):k+1,:], axis = 0)

        
        #Calculating how good the optimator estimates "what happens"
        Ve = np.ones((simu.M,1))*(V[k]) + A @ (U @ np.ones((simu.N,1)) - simu.d[k:k+simu.M])
        he = Ve / tank.area
        

   ##### TRY TO PLOT MAYBE PREDITION OF EXTRACTION
        prevq = q[max(k-23,0):k, :] #go to k since index k+1 already is in U.
        prevq1 = np.pad(prevq, ((simu.M - len(prevq),0),(0,0))) #pad with zeros in front, so array has length M
        extr = np.vstack((prevq1, U))
        extr = extr.cumsum(axis = 0)
        extr = extr[simu.M:,:]-extr[:simu.M,:] #take only the last simu.M values
        # extr = np.sum(U, axis = 0)
        
        # cumsum1 = cum_q1[k-1]
        # cumsum2 = cum_q2[k-1]
        # if k % simu.M == 0:     #Reset cumsum at beginning of each day
        #     cumsum1 = 0
        #     cumsum2 = 0
        # cum_q1[k] = cumsum1 + q1[k] 
        # cum_q2[k] = cumsum2 + q2[k] 
        # print(simu.c[k], simu.d[k])
        plot.updatePlot(k+1, h[:k+1], q[:k+1,:],simu.d[:k+1],cum_q[:k+1,:], p[:k+1,:], he, extr)
        # print(Q)
        # print(U)
    except KeyboardInterrupt:
        break
print('Commulated cost: ', sum(cost))  
       