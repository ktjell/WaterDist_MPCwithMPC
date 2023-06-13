#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:21:52 2023

@author: kst
"""

import numpy as np
import matplotlib.pylab as plt
import scipy
from scipy.optimize import NonlinearConstraint

from parameters import sups, tank, simu
from plotting import plotting

################################################
## MPC optimization #########################

def f(x,i,c,g,sup,rho, Uglobal, lamb):
    eta = 0.7
    normV = (np.sum(x) - np.sum(g))**2
    return ( \
        np.sum( 
            c * ( 1/simu.dt**2 * sup.r * eta * x[i*simu.M:(i+1)*simu.M]**3 + eta*x[i*simu.M:(i+1)*simu.M]*(sup.Dz - sup.p0)) * 3.6 \
            + sup.K*x[i*simu.M:(i+1)*simu.M])+normV \
            + np.sum(lamb * (x-Uglobal)) \
            + rho/2 * np.linalg.norm(x-Uglobal,2)**2 \
            ) /1000000
    
################################################
## MPC optimization #########################
def opti(sup, i, x0, g, c, h0, lamb, rho, Uglobal, extr):

    Qmax = np.ones(simu.M)*1/sup.Qmax
    Mextr = max(sup.Vmax, np.max(extr))
    
    constr = [NonlinearConstraint(lambda x: x[i*simu.M:(i+1)*simu.M]*Qmax, np.zeros(simu.M), np.ones(simu.M)),
              NonlinearConstraint(lambda x: np.cumsum(x[i*simu.M:(i+1)*simu.M])*1/Mextr, 0, (np.ones(simu.M)*sup.Vmax - extr)*1/Mextr ),
              NonlinearConstraint(lambda x: np.ones(simu.M)*h0 + (np.cumsum(x[:simu.M]) + np.cumsum(x[simu.M:])- np.cumsum(g))/tank.area , tank.hmin, tank.hmax)
              ]
    
    res = scipy.optimize.minimize(f, x0, args = (i,c,g,sup, rho, Uglobal.flatten('F'), lamb.flatten('F')), constraints = constr)
    if res.status != 0:
        print(res)
    U = np.zeros((simu.M,simu.N))
    U[:,0] = res.x[:simu.M]
    U[:,1] = res.x[simu.M:]
    # print(U)
    return  U, res.x


def optiSeparable(sups, g, c, h0, lambPrev, Uglobal,Qextr):
    ite = 100
    
    lamb = np.zeros((ite+1, simu.N, simu.M, simu.N))
    lamb[0,:,:,:] = lambPrev
    LAMB = np.zeros((ite,simu.N))
    
    Utemp = np.zeros((ite, simu.N, simu.M, simu.N))
    # Uglobal = np.zeros((simu.M,simu.N))
    u = np.zeros((simu.N, ite))
    rho = 0.3
    x0 = np.ones(2*simu.M)

    acc = True
    j = 0
    while acc and j < ite:
        # print(j)
        Uavr = np.zeros((simu.M,simu.N))
        for i in range(simu.N):
            U, x0 = opti(sups[i], i, x0, g, c, h0, lamb[j,i,:,:], rho, Uglobal, Qextr[:,i])
            Utemp[j,i,:,:] = U
            Uavr += U + 1/rho * lamb[j,i,:,:]
            u[i,j] = U[0,i]
        UglobalTemp = Uglobal
        Uglobal = (1/simu.N) * Uavr     
        
        
        for i in range(simu.N):
            lamb[j+1,i,:,:] = lamb[j,i,:,:] + rho*( Utemp[j,i,:,:] - Uglobal )
            LAMB[j,i] = np.linalg.norm(lamb[j+1,i,:,:],2)
            
            
       
        
        # acc = (np.linalg.norm(Utemp[j,0,:,:] - Utemp[j,1,:,:],2) > 0.01) #Measure the consensus of the U matrix
        acc = (np.linalg.norm(LAMB[j,:] - LAMB[j-1,:], 2) > 0.1)  #>0.9 gives 3 iterations
                                                                  #>0.5 gives 5 ite and still violations 
        j+=1                                                      #>0.3 
    for i in range(simu.N):
        plt.plot(LAMB[:j,i])
    
        
    return u[:,j-1], lamb[j,:,:,:], Uglobal, j-1 #return the u[0] calculated by the local ctr
    # return Uglobal[0,:], lamb[j,:,:,:], Uglobal, j-1   # return the u[0] from the global U
## Simulation ###################################
q = np.zeros((simu.ite,simu.N))    #The optimized flows from pumps
qS = np.zeros((simu.ite,simu.N))  #The optimized flows from pumps
h,hS,V = np.zeros(simu.ite), np.zeros(simu.ite), np.zeros(simu.ite+1)    #Tank level and Volume
cum_q= np.zeros((simu.ite,simu.N))
V[0] = tank.h0*tank.area                          #Start Volume
p = np.zeros((simu.ite, simu.N))  #Pressures
Qextr = np.zeros((simu.M, simu.N))
Uglobal = np.zeros((simu.M,simu.N))
lamb = np.ones((simu.N, simu.M,simu.N))*2000
plot = plotting('Plot1')
plt.figure()
####
javr = 0
cost = np.zeros(simu.N)
sample_pr_day = int(24/simu.sample_hourly)
for k in range(0,simu.ite): 
    try:
        h[k] = V[k]/tank.area
        hS[k] = h[k]
    
        Q, lamb, Uglobal, j = optiSeparable(sups, simu.d[k:k+simu.M], simu.c[k:k+simu.M], h[k], lamb, Uglobal, Qextr)
        q[ k,:] = Q 
        qS[k,:] = Q
        
        #Calculate cummulated consumption for last 23 hours to use at next iteration
        prevq = q[max(k-(sample_pr_day-2),0):k+1, :]
        prevq_pad = np.pad(prevq, ((simu.M-1 - len(prevq),0),(0,0))) #pad with zeros in front, so array has length M
        Qextr = np.pad(np.cumsum(prevq_pad[::-1], axis = 0)[::-1],((0,1),(0,0)))
     
        javr += j
        # print(Q,u)
        for i in range(simu.N):
            p[k,i] = sups[i].r * q[k,i]**2 + sups[i].Dz   #Calculate the pressure


        dV = sum(q[k,:]) - simu.d[k]
        V[k+1] = V[k] + dV
        
        #Calculate the commulated cost of solution. (to compare with other solutions)
        for i in range(simu.N):
            cost[i] += simu.c[k] * (1/(simu.dt**2) * sups[i].r * 0.7 * Q[i]**3 + 0.7*Q[i]*(sups[i].Dz -sups[i].p0)) * 3.6 \
             + sups[i].K*Q[i]
        
        
        #Calculating moving cumulated sum for the last 24 hours to check sattisfaction of constraints.
        cum_q[k] = np.sum(q[max(k-(sample_pr_day-1),0):k+1,:], axis = 0)


        print(k, '/', simu.ite, 'ite = ',j)

        plot.updatePlot(k+1, h[:k+1], q[:k+1,:],simu.d[:k+1],cum_q[:k+1,:], p[:k+1,:])
    except KeyboardInterrupt:
        break
print('Average iterations used in MofM: ' ,javr/simu.ite)