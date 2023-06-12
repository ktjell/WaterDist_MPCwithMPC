#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:50:20 2023

@author: kst
"""

import numpy as np
import scipy
from scipy.optimize import NonlinearConstraint
# import sys
# sys.path.append('../Python_simulation')
from Python_simulation.parameters import sups, tank, simu
from Python_simulation.plotting import plotting


def f(x, c,g):
    eta = 0.7
    r = np.append(np.ones(simu.M)*sups[0].r, np.ones(simu.M)*sups[1].r)
    Dz = np.append(np.ones(simu.M)*sups[0].Dz, np.ones(simu.M)*sups[1].Dz)
    p0 = np.append(np.ones(simu.M)*sups[0].p0, np.ones(simu.M)*sups[1].p0)   
    K = np.append(np.ones(simu.M)*sups[0].K, np.ones(simu.M)*sups[1].K)
    normV = (np.sum(x) - np.sum(g))**2
    return (np.sum( c * ( 1/simu.dt**2 * r * eta * x**3 + eta*x*(Dz - p0)) * 3.6 + K*x)+normV)/1000000
    
################################################
## MPC optimization #########################
def opti(sups, g, c, h0, extr, x0):
    kappa = 10000
    Qmax = np.append(np.ones(simu.M)*1/sups[0].Qmax, np.ones(simu.M)*1/sups[1].Qmax)
    Mextr0 = max(sups[0].Vmax, np.max(extr[:,0]))
    Mextr1 = max(sups[1].Vmax, np.max(extr[:,1]))
    
    constr = [NonlinearConstraint(lambda x: x*Qmax, np.zeros(2*simu.M), np.ones(2*simu.M)),
              # NonlinearConstraint(lambda x: np.sum(x[:simu.M])*1/sups[0].Vmax, 0, 1),
              # NonlinearConstraint(lambda x: np.sum(x[simu.M:])*1/sups[1].Vmax, 0, 1),
              NonlinearConstraint(lambda x: np.cumsum(x[:simu.M])*1/Mextr0, 0, (np.ones(simu.M)*sups[0].Vmax - extr[:,0])*1/Mextr0 ),
              NonlinearConstraint(lambda x: np.cumsum(x[simu.M:])*1/Mextr1, 0, (np.ones(simu.M)*sups[1].Vmax - extr[:,1])*1/Mextr1 ),
              NonlinearConstraint(lambda x: np.ones(simu.M)*h0 + (np.cumsum(x[:simu.M]) + np.cumsum(x[simu.M:])- np.cumsum(g))/tank.area , tank.hmin, tank.hmax)
              ]
    
    res = scipy.optimize.minimize(f, x0, args = (c,g), constraints = constr)
    if res.status != 0:
        print(res)
    U = np.zeros((simu.M,simu.N))
    U[:,0] = res.x[:simu.M]
    U[:,1] = res.x[simu.M:]
    # print(U)
    return  U, res.x

       
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
x0 = np.ones(2*simu.M)
for k in range(0,simu.ite): 
    try:
        h[k] = V[k]/tank.area   #Level of water in tank: Volume divided by area of tank.
        hE[k] = h[k]
        
        
    
        U, x0 = opti(sups, simu.d[k:k+simu.M], simu.c[k:k+simu.M], h[k], Qextr, x0)
        q[ k,:] = U[0,:]        #Delivered water from pump 1 and 2
        qE[k,:] = U[0,:]
      
        Q = U[0,:]
        #Calculate cummulated consumption for last 23 hours to use at next iteration
        prevq = q[max(k-22,0):k+1, :]
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
       