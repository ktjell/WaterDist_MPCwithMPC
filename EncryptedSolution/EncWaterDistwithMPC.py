#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:48:43 2023

@author: kst
"""

import numpy as np
import cvxpy as cp
import matplotlib.pylab as plt
from parameters import sups, tank, simu, E
from plotting import plotting

################################################
## MPC optimization #########################

def opti(sup, i, g, c, h0, lamb, rho, Uglobal):
    
    kappa = 2#0.1
    U = cp.Variable((simu.M,simu.N))
    u = cp.Variable(simu.M)
    A = np.tril(np.ones((simu.M,simu.M)))
    cost = 0

    for k in range(simu.M):
        cost += c[k] * E(u[k],sup.r, sup.Dz, sup.p0)* 3.6 \
             + sup.K*u[k]     #*3.6 to get from kWh til kWs.
             
    cost += kappa* cp.power(cp.norm(np.ones((1,simu.M)) @ (U @ np.ones((simu.N,1)) - g),2),2)\
          + cp.norm(lamb.T @ (U-Uglobal)) \
          + rho/2 * cp.power(cp.norm(U-Uglobal,2),2)
          
    constr = [U[:,i] == u, 
              U >= np.zeros((simu.M,simu.N)), 
              U[:,i] <= np.ones(simu.M)*sup.Qmax,
              cp.sum(u) <= sups[i].Vmax,
              np.ones((simu.M,1))*(h0*tank.area) + A @ (U @ np.ones((simu.N,1)) - g) >= np.ones((simu.M,1))*tank.hmin*tank.area,
              np.ones((simu.M,1))*(h0*tank.area) + A @ (U @ np.ones((simu.N,1)) - g) <= np.ones((simu.M,1))*tank.hmax*tank.area
              ]
    
    problem = cp.Problem(cp.Minimize(cost) , constr)
    problem.solve()#solver = cp.MOSEK)
    status = problem.status
    # if status=='optimal':
    return u.value.reshape((simu.M,)), U.value

## NOTE to self, pr'v at genbruge lambda og sæt ite i opti til at stoppe når 
## PRøv at fjerne lille u fro opti problem
def optiSeparable(sups, g, c, h0, lambPrev, Uglobal):
    ite = 12
    
    lamb = np.zeros((ite+1, simu.N, simu.M, simu.N))
    lamb[0,:,:,:] = lambPrev*0.9
    LAMB = np.zeros((ite,simu.N))
    
    Utemp = np.zeros((ite, simu.N, simu.M, simu.N))
    
            
    utemp = np.zeros((ite, simu.N, simu.M))

    rho = 3
    # for j in range(ite):
    acc = True
    j = 0
    while acc and j < ite:
        print(j)
        Uavr = np.zeros((simu.M,simu.N))
        for i in range(simu.N):
            
            u, U = opti(sups[i], i, g, c, h0, lamb[j,i,:,:], rho, Uglobal)
            utemp[j,i,:] = u
            Utemp[j,i,:,:] = U
            Uavr += U + 1/rho * lamb[j,i,:,:]
        UglobalTemp = Uglobal
        Uglobal = 1/simu.N * Uavr    

        for i in range(simu.N):
            lamb[j+1,i,:,:] = lamb[j,i,:,:] + rho*( Utemp[j,i,:,:] - Uglobal )
            LAMB[j,i] = np.linalg.norm(lamb[j+1,i,:,:],2)
    # print(Uglobal)
        j+=1
        # print(Uglobal- UglobalTemp)
        # print((np.linalg.norm(Uglobal - UglobalTemp, 2)))
        acc = (np.linalg.norm(Uglobal - UglobalTemp, 2) > 5)
    
    for i in range(simu.N):
        plt.plot(LAMB[:j,i])
    
        
    return Uglobal[0,:], lamb[j,:,:,:], Uglobal

## Simulation ###################################
q1,q2 = np.zeros(simu.ite),np.zeros(simu.ite)     #The optimized flows from pumps
h,V = np.zeros(simu.ite), np.zeros(simu.ite+1)    #Tank level and Volume
cum_q1, cum_q2 = np.zeros(simu.ite), np.zeros(simu.ite)
V[0] = tank.h0*tank.area                          #Start Volume
p1,p2 = np.zeros(simu.ite), np.zeros(simu.ite)    #Pressures

Uglobal = np.zeros((simu.M,simu.N))
lamb = np.zeros((simu.N, simu.M,simu.N))
plot = plotting('Plot1')
plt.figure()
####
for k in range(0,simu.ite): 
    try:
        h[k] = V[k]/tank.area
    
        Q, lamb, Uglobal = optiSeparable(sups, simu.d[k:k+simu.M], simu.c[k:k+simu.M], h[k], lamb, Uglobal)
        q1[k] = Q[0]
        q2[k] = Q[1]
        
        p1[k] = sups[0].r * q1[k]**2 + sups[0].Dz
        p2[k] = sups[1].r * q2[k]**2 + sups[1].Dz
        
        dV = q1[k] + q2[k] - simu.d[k]
        V[k+1] = V[k] + dV
        
        cumsum1 = cum_q1[k-1]
        cumsum2 = cum_q2[k-1]
        if k % simu.M == 0:
            cumsum1 = 0
            cumsum2 = 0
        cum_q1[k] = cumsum1 + q1[k] 
        cum_q2[k] = cumsum2 + q2[k] 
    
        plot.updatePlot(k, h[:k], q1[:k],q2[:k],simu.d[:k],cum_q1[:k],cum_q2[:k], p1[:k],p2[:k])
    except KeyboardInterrupt:
        break
