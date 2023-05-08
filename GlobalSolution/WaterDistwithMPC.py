#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:48:43 2023

@author: kst
"""

import numpy as np
import cvxpy as cp
from parameters import sups, tank, simu, E
from plotting import plotting

################################################
## MPC optimization #########################
def opti(sups, g, c, h0):
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
            #* 
        constr.extend([
                    U[:,i] <= np.ones(simu.M)*sups[i].Qmax,
                    cp.sum(U[:,i]) <= sups[i].Vmax
                    ])

    cost += kappa* cp.power(cp.norm(np.ones((1,simu.M)) @ (U @ np.ones((simu.N,1)) - g),2),2)
    
    problem = cp.Problem(cp.Minimize(cost) , constr)
    problem.solve()#solver = cp.MOSEK)
    status = problem.status

    return U.value[0,:], U.value

       
## Simulation and plotting ###################################
q1,q2 = np.zeros(simu.ite),np.zeros(simu.ite)     #The optimized flows from pumps
q1E, q2E = np.zeros(simu.ite),np.zeros(simu.ite)
h,hE,V = np.zeros(simu.ite), np.zeros(simu.ite), np.zeros(simu.ite+1)    #Tank level and Volume
cum_q1, cum_q2 = np.zeros(simu.ite), np.zeros(simu.ite)
V[0] = tank.h0*tank.area                          #Start Volume
p1,p2 = np.zeros(simu.ite), np.zeros(simu.ite)    #Pressures
plot = plotting('Plot1')
cost = np.zeros(simu.N)
for k in range(0,simu.ite): 
    try:
        h[k] = V[k]/tank.area   #Level of water in tank: Volume divided by area of tank.
        hE[k] = h[k]
    
        Q , U= opti(sups, simu.d[k:k+simu.M], simu.c[k:k+simu.M], h[k])
        q1[k] = Q[0]        #Delivered water from pump 1
        q2[k] = Q[1]        #Delivered water from pump 2
        q1E[k] = Q[0]
        q2E[k]=Q[1]
        
        p1[k] = sups[0].r * q1[k]**2 + sups[0].Dz   #Calculate the pressure
        p2[k] = sups[1].r * q2[k]**2 + sups[1].Dz
        
        dV = q1[k] + q2[k] - simu.d[k]      #Change of volume in the tank: the sum of supply minus consumption.
        V[k+1] = V[k] + dV                  #Volume in tank: volume of last time stem + change in volume
        ## Done with simulation
        
        #Calculate the commulated cost of solution. (to compare with other solutions)
        for i in range(simu.N):
            cost[i] += simu.c[k] * (1/(simu.dt**2) * sups[i].r * 0.7 * Q[i]**3 + 0.7*Q[i]*(sups[i].Dz -sups[i].p0)) * 3.6 \
             + sups[i].K*Q[i]
        
        
        #Calculating cumulated sum to check sattisfaction of constraints.
        cumsum1 = cum_q1[k-1]
        cumsum2 = cum_q2[k-1]
        if k % simu.M == 0:     #Reset cumsum at beginning of each day
            cumsum1 = 0
            cumsum2 = 0
        cum_q1[k] = cumsum1 + q1[k] 
        cum_q2[k] = cumsum2 + q2[k] 
        # print(simu.c[k], simu.d[k])
        plot.updatePlot(k, h[:k], q1[:k],q2[:k],simu.d[:k],cum_q1[:k],cum_q2[:k], p1[:k],p2[:k])
        # print(Q)
        # print(U)
    except KeyboardInterrupt:
        break
print('Commulated cost: ', sum(cost))  
       