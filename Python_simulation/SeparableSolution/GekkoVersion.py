#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:56:56 2023

@author: kst
"""


import numpy as np
from gekko import GEKKO
import matplotlib.pylab as plt
import sys
sys.path.append('../Python_simulation')
from Python_simulation.parameters import sups, tank, simu
from Python_simulation.plotting import plotting

################################################
## MPC optimization #########################

def E(x, r, Dz, p0):
    eta = 0.7
    return (simu.dt**2)**(-1) * r * eta * x**3 + eta*x*(Dz - p0) 

def opti(sup, i, g, c, h0, lamb, rho, Uglobal, extr):
    
    kappa = 10
    #Initialize Model
    m = GEKKO(remote = False)

    #initialize variables
    U = m.Array(m.Var,(simu.M, simu.N), lb = 0.0 )
    #set upper bound for each coloumn in U
    for k in range(simu.M):
        U[k,i].UPPER = sup.Qmax

    
    #Variable for the volume
    V = m.Array(m.Var, (simu.M), lb = tank.hmin*tank.area, ub = tank.hmax*tank.area)
    #Set start volume
    m.Equation(V[0] == h0*tank.area)
    
    #CONSTRAINTS
    #The dynamics of the tank
    for k in range(simu.M-1):
        m.Equation(V[k+1] == V[k] + m.sum(U[k,:]) - g[k][0])

    #Require that the extraction of water for the last 24 hours does not excees limit
    m.Equation(U[0,i] <= sup.Vmax - extr[0])

    
    #COST
    cost = 0
    for k in range(simu.M):
        #Cost function
        cost +=  c[k][0]* E(U[k,i],sup.r, sup.Dz, sup.p0)* 3.6 \
             + sup.K*U[k,i] 
        
    #Cost on the fluctuations of U
    for k in range(1, simu.M):
        cost += (U[k,i] - U[k-1,i])**2
        m.Equation(m.sum(U[:k,i]) <= sup.Vmax - extr[k])  #The rest of the extraction constraint.
    
    Udif = U-Uglobal
    Ulamb = lamb * (Udif)
    
    #minimize cap between start volume and volume after 24 hours.
    cost += kappa * (V[0] - V[-1])**2 \
         + m.sum(m.sum(Ulamb))\
         + rho/2 * m.sum(m.sum(Udif * Udif))
    #set cost function and object to minimize 
    m.Minimize(cost)
    #Set global options
    m.options.IMODE = 3 #steady state optimization
    # m.options.DIAGLEVEL = 1 #See timings etc for solver
    #Solve simulation
    m.solve(disp = True)
    
    #Get solution on a numpy array form
    Uv = np.zeros((simu.M,simu.N))
    for k in range(simu.M):
        Uv[k,:] = [U[k,0].value[0], U[k,1].value[0]]
    # print(Uv)
    return Uv


def optiSeparable(sups, g, c, h0, lambPrev, Uglobal,Qextr):
    ite = 100
    
    lamb = np.zeros((ite+1, simu.N, simu.M, simu.N))
    lamb[0,:,:,:] = lambPrev
    LAMB = np.zeros((ite,simu.N))
    
    Utemp = np.zeros((ite, simu.N, simu.M, simu.N))
    # Uglobal = np.zeros((simu.M,simu.N))
    u = np.zeros((simu.N, ite))
    rho = .8

    acc = True
    j = 0
    while acc and j < ite:
        # print(j)
        Uavr = np.zeros((simu.M,simu.N))
        for i in range(simu.N):
            
            U = opti(sups[i], i, g, c, h0, lamb[j,i,:,:], rho, Uglobal, Qextr[:,i])
            u[i,j] = U[0,i]
            Utemp[j,i,:,:] = U
            Uavr += U + 1/rho * lamb[j,i,:,:]
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