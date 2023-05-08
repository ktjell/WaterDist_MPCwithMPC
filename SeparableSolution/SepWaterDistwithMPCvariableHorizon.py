#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:57:50 2023

@author: kst
"""

import numpy as np
import cvxpy as cp
import matplotlib.pylab as plt
from parameters import sups, tank, simu, E
from plotting import plotting


################################################
## MPC optimization #########################

def opti(M, sup, i, g, c, h0, extracted, lamb, rho, Uglobal):
    
    kappa = 2#0.1
    U = cp.Variable((M,simu.N))
    # u = cp.Variable(simu.M)
    A = np.tril(np.ones((M,M)))
    cost = 0
    

    for k in range(M):
        cost += c[k] * E(U[k,i],sup.r, sup.Dz, sup.p0)* 3.6 \
             + sup.K*U[k,i]     #*3.6 to get from kWh til kWs.
             
    cost += kappa* cp.power(cp.norm(np.ones((1,M)) @ (U @ np.ones((simu.N,1)) - g),2),2)\
          + cp.sum(cp.multiply(lamb , (U-Uglobal))) \
          + rho/2 * cp.power(cp.norm(U-Uglobal,2),2)
          
    constr = [
              U >= np.zeros((M,simu.N)), 
              U[:,i] <= np.ones(M)*sup.Qmax,
              cp.sum(U[:,i]) <= sup.Vmax - extracted,
              np.ones((M,1))*(h0*tank.area) + A @ (U @ np.ones((simu.N,1)) - g) >= np.ones((M,1))*tank.hmin*tank.area,
              np.ones((M,1))*(h0*tank.area) + A @ (U @ np.ones((simu.N,1)) - g) <= np.ones((M,1))*tank.hmax*tank.area
              ]
    
    problem = cp.Problem(cp.Minimize(cost) , constr)
    problem.solve()#solver = cp.MOSEK, mosek_params = {'MSK_DPAR_OPTIMIZER_MAX_TIME':  10.0})
    # status = problem.status
    # if status=='optimal':
    return U.value[0,i], U.value


def optiSeparable(M, sups, g, c, h0, extracted, lambPrev, Uglobal):
    ite = 100
    
    lamb = np.zeros((ite+1, simu.N, M, simu.N))
    lamb[0,:,:,:] = lambPrev
    LAMB = np.zeros((ite,simu.N))
    
    Utemp = np.zeros((ite, simu.N, M, simu.N))
    # Uglobal = np.zeros((M,simu.N))
    u = np.zeros((simu.N, ite))
    rho = .1

    acc = True
    j = 0
    while acc and j < ite:
        # print(j)
        Uavr = np.zeros((M,simu.N))
        for i in range(simu.N):
            
            u[i,j], U = opti(M,sups[i], i, g, c, h0, extracted[i], lamb[j,i,:,:], rho, Uglobal)
            Utemp[j,i,:,:] = U
            Uavr += U + 1/rho * lamb[j,i,:,:]
        UglobalTemp = Uglobal
        Uglobal = (1/simu.N) * Uavr     
        
        
        for i in range(simu.N):
            lamb[j+1,i,:,:] = lamb[j,i,:,:] + rho*( Utemp[j,i,:,:] - Uglobal )
            LAMB[j,i] = np.linalg.norm(lamb[j+1,i,:,:],2)
            
            
       
        
        # acc = (np.linalg.norm(Utemp[j,0,:,:] - Utemp[j,1,:,:],2) > 0.01) #Measure the consensus of the U matrix
        acc = (np.linalg.norm(LAMB[j,:] - LAMB[j-1,:], 2) > 0.01)  #>0.9 gives 3 iterations
                                                                  #>0.5 gives 5 ite and still violations 
        j+=1                                                      #>0.3 
    for i in range(simu.N):
        plt.plot(LAMB[:j,i])
    
        
    return u[:,j-1], lamb[j,:,:,:], Uglobal, j-1 #return the u[0] calculated by the local ctr
    # return Uglobal[0,:], lamb[j,:,:,:], Uglobal, j-1   # return the u[0] from the global U
## Simulation ###################################
q1,q2 = np.zeros(simu.ite),np.zeros(simu.ite)     #The optimized flows from pumps
q1S,q2S = np.zeros(simu.ite),np.zeros(simu.ite)     #The optimized flows from pumps
h,hS,V = np.zeros(simu.ite), np.zeros(simu.ite), np.zeros(simu.ite+1)    #Tank level and Volume
cum_q1, cum_q2 = np.zeros(simu.ite), np.zeros(simu.ite)
V[0] = tank.h0*tank.area                          #Start Volume
p1,p2 = np.zeros(simu.ite), np.zeros(simu.ite)    #Pressures
Uglobal1 = np.zeros((simu.M,simu.N))
lamb1 = np.zeros((simu.N, simu.M,simu.N))

plot = plotting('Plot1')
plt.figure()
####
javr = 0
cost = np.zeros(simu.N)
for k in range(0,simu.ite): 
    try:
        h[k] = V[k]/tank.area
        hS[k] = h[k]
        
        # To sattisfy extraction per day limits, the horizon starts with 24 hours and is
        # then decreases with one until reset again at 24 hours.
        Mtemp = 24 - (k % 24)
        M = int(Mtemp*60*60 / simu.dt ) 
        
        #Calculating cumulated sum to check sattisfaction of constraints PART 1.
        #Divided into part 1 and 2, to use cumsum1 in optimization.
        
        if k % simu.M == 0:     #Reset cumsum at beginning of each day
            cumsum1 = 0
            cumsum2 = 0
            Uglobal = Uglobal1
            lamb = lamb1
        
        else:
            cumsum1 = cum_q1[k-1]
            cumsum2 = cum_q2[k-1]
            Uglobal = Uglobal[1:,:]
            lamb = lamb[:,1:,:]
    
        # Calculate optimal control action
    
        Q, lamb, Uglobal, j = optiSeparable(M, sups, simu.d[k:k+M], simu.c[k:k+M], h[k], [cumsum1, cumsum2], lamb, Uglobal)
        q1[k] = Q[0]
        q2[k] = Q[1]
        q1S[k] = Q[0]
        q2S[k] = Q[1]
        javr += j
        # print(Q,u)
        p1[k] = sups[0].r * q1[k]**2 + sups[0].Dz
        p2[k] = sups[1].r * q2[k]**2 + sups[1].Dz
        
        dV = q1[k] + q2[k] - simu.d[k]
        V[k+1] = V[k] + dV
        #Done with simulation
        if k % simu.M == 0:     #Keep the 24 hours Uglobal and lamb1, to avoid starting over when time is reset.
            Uglobal1 = Uglobal
            lamb1 = lamb


        #Calculating cumulated sum to check sattisfaction of constraints PART 2.
        cum_q1[k] = cumsum1 + q1[k] 
        cum_q2[k] = cumsum2 + q2[k] 

        #Calculate the commulated cost of solution. (to compare with other solutions)
        for i in range(simu.N):
            cost[i] += simu.c[k] * (1/(simu.dt**2) * sups[i].r * 0.7 * Q[i]**3 + 0.7*Q[i]*(sups[i].Dz -sups[i].p0)) * 3.6 \
             + sups[i].K*Q[i]
        
        print(k, '/', simu.ite, 'ite = ',j)
        # print(k,Q)
        # print(j,Uglobal[0,:])
        # print(Uglobal)
        # print(simu.c[k], simu.d[k])
        plot.updatePlot(k, h[:k], q1[:k],q2[:k],simu.d[:k],cum_q1[:k],cum_q2[:k], p1[:k],p2[:k])
    except KeyboardInterrupt:
        break
print('Average iterations used in MofM: ' ,javr/simu.ite)

print('Commulated cost: ', sum(cost))    