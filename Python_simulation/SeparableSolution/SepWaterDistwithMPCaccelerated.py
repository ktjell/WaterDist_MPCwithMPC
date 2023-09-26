#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:16:23 2023

@author: kst
"""


import numpy as np
import cvxpy as cp
import matplotlib.pylab as plt
import sys
sys.path.append('../Python_simulation')
from Python_simulation.parameters import sups, tank, simu
from Python_simulation.plotting import plotting

################################################
## MPC optimization #########################

def f(x, sup):
    eta = 0.7
    return  ( (1/(simu.dt**2)) * sup.r * eta * x**3 + eta*x*(sup.Dz - sup.p0)) * 3.6 + sup.K*x
    

def opti(sup, i, g, c, h0, lamb, rho, Uglobal, Qextr):
    
    kappa = 2#0.1
    U = cp.Variable((simu.M,simu.N))
    # u = cp.Variable(simu.M)
    A = np.tril(np.ones((simu.M,simu.M)))
    cost = 0
    

    for k in range(simu.M):
        cost += c[k] * f(U[k,i],sup)* 3.6 \
             + sup.K*U[k,i] + cp.power(cp.norm(U[k,i] - U[k-1,i],2),2)    #*3.6 to get from kWh til kWs.
             
    cost += kappa* cp.power(cp.norm(np.ones((1,simu.M)) @ (U @ np.ones((simu.N,1)) - g),2),2)\
          + cp.sum(cp.multiply(lamb , (U-Uglobal))) \
          + rho/2 * cp.power(cp.norm(U-Uglobal,2),2) 
        
          
    constr = [
              U >= np.zeros((simu.M,simu.N)), 
              U[:,i] <= np.ones(simu.M)*sup.Qmax,
              cp.cumsum(U[:,i]) <= np.ones(simu.M)*sup.Vmax - Qextr,
              # cp.sum(U[:,i]) <= sup.Vmax,
              np.ones((simu.M,1))*(h0*tank.area) + A @ (U @ np.ones((simu.N,1)) - g) >= np.ones((simu.M,1))*tank.hmin*tank.area,
              np.ones((simu.M,1))*(h0*tank.area) + A @ (U @ np.ones((simu.N,1)) - g) <= np.ones((simu.M,1))*tank.hmax*tank.area
              ]
    
    problem = cp.Problem(cp.Minimize(cost) , constr)
    problem.solve()#solver = cp.MOSEK, mosek_params = {'MSK_DPAR_OPTIMIZER_MAX_TIME':  10.0})
    # status = problem.status
    # if status=='optimal':
    return U.value[0,i], U.value


def optiSeparable(sups, g, c, h0, lambPrev, UglobalPrev,Qextr):
    ite = 90
    
    lamb = np.zeros((ite+1, simu.N, simu.M, simu.N))
    lamb[0,:,:,:] = lambPrev
    lamb_hat = lambPrev
    LAMB = np.zeros((ite,simu.N))
    Uglobal = np.zeros((ite+1, simu.M, simu.N))
    Uglobal[0,:,:] = UglobalPrev
    Uglobal_hat = UglobalPrev
    
    Utemp = np.zeros((ite, simu.N, simu.M, simu.N))
    # Uglobal = np.zeros((simu.M,simu.N))
    u = np.zeros((simu.N, ite))
    rho = .1
    alphaO = 1
    
    acc = True
    j = 0
    while acc and j < ite:
        print(j)
        #ADMM:
        #STEP 1: minimization wrt local U
        Uavr = np.zeros((simu.M,simu.N))
        for i in range(simu.N):
            
            u[i,j], U = opti(sups[i], i, g, c, h0, lamb_hat[i,:,:], rho, Uglobal_hat, Qextr[:,i])
            Utemp[j,i,:,:] = U
            Uavr += U + 1/rho * lamb_hat[i,:,:]
        #STEP 2: minimization wrt global U
        Uglobal[j+1,:,:] = (1/simu.N) * Uavr     
        
        #STEP 3: update lambda
        for i in range(simu.N):
            lamb[j+1,i,:,:] = lamb[j,i,:,:] + rho*( Utemp[j,i,:,:] - Uglobal[j+1,:,:] )
            LAMB[j,i] = np.linalg.norm(lamb[j+1,i,:,:],2)
            
        #Step 4: nesterov acceleration
        alphaN = (1 + np.sqrt(1+4*alphaO**2)) / 2
        Uglobal_hat = Uglobal[j+1,:,:] + ((alphaO-1)/alphaN ) * (Uglobal[j+1,:,:]-Uglobal[j,:,:]) 
        for i in range(simu.N):
            lamb_hat[i,:,:] = lamb[j+1,i,:,:] + ((alphaO-1)/alphaN ) * (lamb[j+1,i,:,:] - lamb[j,i,:,:])
       
        
        # acc = (np.linalg.norm(Utemp[j,0,:,:] - Utemp[j,1,:,:],2) > 0.01) #Measure the consensus of the U matrix
        acc = (np.linalg.norm(LAMB[j,:] - LAMB[j-1,:], 2) > 0.1)  #>0.9 gives 3 iterations
                                                                  #>0.5 gives 5 ite and still violations 
        j+=1                                                      #>0.3 
        alphaO = alphaN
    for i in range(simu.N):
        plt.plot(LAMB[:j,i])
    
    
    return u[:,j-1], lamb[j,:,:,:], Uglobal[j-1,:,:], j-1 #return the u[0] calculated by the local ctr
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
lamb = np.zeros((simu.N, simu.M,simu.N))
plot = plotting('Plot1')
plt.figure()
####
javr = 0
cost = np.zeros(simu.N)
for k in range(0,simu.ite): 
    try:
        h[k] = V[k]/tank.area
        hS[k] = h[k]
    
        Q, lamb, Uglobal, j = optiSeparable(sups, simu.d[k:k+simu.M], simu.c[k:k+simu.M], h[k], lamb, Uglobal, Qextr)
        q[ k,:] = Q 
        qS[k,:] = Q
        
        #Calculate cummulated consumption for last 23 hours to use at next iteration
        prevq = q[max(k-22,0):k+1, :]
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
        cum_q[k] = np.sum(q[max(k-23,0):k+1,:], axis = 0)


        print(k, '/', simu.ite, 'ite = ',j)

        plot.updatePlot(k+1, h[:k+1], q[:k+1,:],simu.d[:k+1],cum_q[:k+1,:], p[:k+1,:])
    except KeyboardInterrupt:
        break
print('Average iterations used in MofM: ' ,javr/simu.ite)