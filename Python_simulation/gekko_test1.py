#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:30:34 2023

@author: kst
"""


from gekko import GEKKO
import matplotlib.pylab as plt
import numpy as np
# import sys
# sys.path.append('../Python_simulation')
from parameters import sups, tank, simu
from plotting import plotting



def E(x, r, Dz, p0):
    eta = 0.7
    return (simu.dt**2)**(-1) * r * eta * x**3 + eta*x*(Dz - p0) 

################################################
## MPC optimization #########################
def opti(sups, g, c, h0, extr):
    kappa = 10000
    #Initialize Model
    m = GEKKO(remote = False)
    m.time = range(0,24)
    v = m.Param(value = g)

    #initialize variables
    # U = m.Array(m.Var,(simu.N), lb = 0.0 )
    u1,u2 = (m.MV(lb = 0 , ub = sups[0].Qmax), m.MV(lb = 0, ub=sups[1].Qmax ))
    u = [u1,u2]
    V = m.CV(h0*tank.area)
    V.status = 1
    V.STATUS = 1
    V.FSTATUS = 1
    m.options.CV_TYPE = 1
    m.options.NODES = 2
    V.SPLO = tank.hmin*tank.area
    V.SPHI = tank.hmax*tank.area
    # V.TR_INIT = h0*tank.area
    su = m.Intermediate(m.sum(u))
    
    m.Equation(V.dt() ==  su - v)
    
    u1.STATUS = 1
    u1.DCOST=0.5
    u2.STATUS = 1
    u2.DCOST=0.5
    
    m.Equation(m.sum(u1) <= sups[0].Vmax)
    m.Equation(m.sum(u2) <= sups[1].Vmax)
    constr = []
    cost = 0
    for i in range(simu.N):
        cost +=  c[k][0]* E(u[i],sups[i].r, sups[i].Dz, sups[i].p0)* 3.6 \
                + sups[i].K*u[i] 
                
      
        # constr.extend([
        #             np.cumsum(u[i]) <= sups[i].Vmax - extr[:,i] 
        #             # cp.sum(U[:,i]) <= sups[i].Vmax
        #             ])

    # cost += kappa* np.linalg.norm(np.ones((1,simu.M)) @ (U @ np.ones((simu.N,1)) - g),2)**2
    
    # m.Equations(constr)
    m.Minimize(cost)
    #Set global options
    m.options.IMODE = 6 
    # m.options.DIAGLEVEL = 1
    #Solve simulation
    m.solve(disp = True)
  
    # get additional solution information
    import json
    with open(m.path+'//results.json') as f:
        results = json.load(f)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(m.time,u1.value,'b-',label='u_1')
    plt.plot(m.time,u2.value,'r-',label='u_2')
    plt.legend()
    plt.ylabel('Input')
    plt.subplot(2,1,2)
    plt.plot(m.time,results['v1.tr_hi'],'k-')
    plt.plot(m.time,results['v1.tr_lo'],'k-')
    plt.plot(m.time,V.value,'r--',label='Tank Volume')
    plt.ylabel('Output')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.show()
    
    return u1,u2,V #np.array([U[0,0][0], U[0,1][0]])

       
## Simulation and plotting ###################################
q = np.zeros((simu.ite, simu.N))                  #The optimized flows from pumps
qE = np.zeros((simu.ite, simu.N))
h,hE,V = np.zeros(simu.ite), np.zeros(simu.ite), np.zeros(simu.ite+1)    #Tank level and Volume
cum_q = np.zeros((simu.ite,simu.N))
V[0] = tank.h0*tank.area                          #Start Volume
p = np.zeros((simu.ite,simu.N))    #Pressures
A = np.tril(np.ones((simu.M,simu.M)))
# plot = plotting('Plot1')
cost = np.zeros(simu.N)
Qextr = np.zeros((simu.M, simu.N))

for k in range(0,1):#simu.ite): 
    # try:
    h[k] = V[k]/tank.area   #Level of water in tank: Volume divided by area of tank.
    hE[k] = h[k]
    
    

    u1,u2,V = opti(sups, simu.d[k:k+simu.M], simu.c[k:k+simu.M], h[k], Qextr)
#         q[ k,:] = U        #Delivered water from pump 1 and 2
#         qE[k,:] = U
#         Q = U
#         #Calculate cummulated consumption for last 23 hours to use at next iteration
#         prevq = q[max(k-22,0):k+1, :]
#         prevq_pad = np.pad(prevq, ((simu.M-1 - len(prevq),0),(0,0))) #pad with zeros in front, so array has length M
#         Qextr = np.pad(np.cumsum(prevq_pad[::-1], axis = 0)[::-1],((0,1),(0,0)))

#         for i in range(simu.N):
#             p[k,i] = sups[i].r * q[k,i]**2 + sups[i].Dz   #Calculate the pressure

        
#         dV = sum(q[k,:]) - simu.d[k]      #Change of volume in the tank: the sum of supply minus consumption.
#         V[k+1] = V[k] + dV                  #Volume in tank: volume of last time stem + change in volume
#         ## Done with simulation
        
#         #Calculate the commulated cost of solution. (to compare with other solutions)
#         for i in range(simu.N):
#             cost[i] += simu.c[k] * (1/(simu.dt**2) * sups[i].r * 0.7 * Q[i]**3 + 0.7*Q[i]*(sups[i].Dz -sups[i].p0)) * 3.6 \
#              + sups[i].K*Q[i]
        
        
#         #Calculating moving cumulated sum for the last 24 hours to check sattisfaction of constraints.
#         cum_q[k] = np.sum(q[max(k-23,0):k+1,:], axis = 0)

        
#         #Calculating how good the optimator estimates "what happens"
#         Ve = np.ones((simu.M,1))*(V[k]) + A @ (U @ np.ones((simu.N,1)) - simu.d[k:k+simu.M])
#         he = Ve / tank.area
        

#    ##### TRY TO PLOT MAYBE PREDITION OF EXTRACTION
#         prevq = q[max(k-23,0):k, :] #go to k since index k+1 already is in U.
#         prevq1 = np.pad(prevq, ((simu.M - len(prevq),0),(0,0))) #pad with zeros in front, so array has length M
#         extr = np.vstack((prevq1, U))
#         extr = extr.cumsum(axis = 0)
#         extr = extr[simu.M:,:]-extr[:simu.M,:] #take only the last simu.M values
#         # extr = np.sum(U, axis = 0)
        
#         # cumsum1 = cum_q1[k-1]
#         # cumsum2 = cum_q2[k-1]
#         # if k % simu.M == 0:     #Reset cumsum at beginning of each day
#         #     cumsum1 = 0
#         #     cumsum2 = 0
#         # cum_q1[k] = cumsum1 + q1[k] 
#         # cum_q2[k] = cumsum2 + q2[k] 
#         # print(simu.c[k], simu.d[k])
#         # plot.updatePlot(k+1, h[:k+1], q[:k+1,:],simu.d[:k+1],cum_q[:k+1,:], p[:k+1,:], he, extr)
#         # print(Q)
#         # print(U)
#     except KeyboardInterrupt:
#         break
# print('Commulated cost: ', sum(cost))  
       