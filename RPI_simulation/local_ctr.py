#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:35:27 2023

@author: kst
"""


import numpy as np
from threading import Thread
import tcp_socket as sock
from shamir_real_number import secret_sharing as ss
from ip_config import ipconfigs as ips
from parameters import sups, tank, simu, E
from communication_setup import com_functions
import cvxpy as cp

class loc_ctr(Thread):
    def __init__(self, p_nr, rec_q):
        Thread.__init__(self)
        self.rec_q = rec_q
        self.p_nr = p_nr
        self.com_func = com_functions(p_nr, rec_q)
 
    def distribute_shares(self, name, sec):
        shares = ss.gen_matrix_shares(sec)
        for i, addr in enumerate(ips.cloud_addr):
            sock.TCPclient(*addr, [name + str(self.p_nr) , shares[i]])
        
            
    def reconstruct_secret(self, name):
        return ss.recon_matrix_secret(self.com_func.get_shares(name, len(ips.addr_dict['cloud'])))
    
    ################################################
    ## MPC optimization #########################

    def opti(self, sup, g, c, h0, lamb, rho, Uglobal, Qextr):
        
        kappa = 2#0.1
        U = cp.Variable((simu.M,simu.N))
        A = np.tril(np.ones((simu.M,simu.M)))
        cost = 0
        
        for k in range(simu.M):
            cost += c[k] * E(U[k,self.p_nr],sup.r, sup.Dz, sup.p0)* 3.6 \
                 + sup.K*U[k,self.p_nr] + cp.power(cp.norm(U[k,self.p_nr] - U[k-1,self.p_nr],2),2)    #*3.6 to get from kWh til kWs.
                 
        cost += kappa* cp.power(cp.norm(np.ones((1,simu.M)) @ (U @ np.ones((simu.N,1)) - g),2),2)\
              + cp.sum(cp.multiply(lamb , (U-Uglobal))) \
              + rho/2 * cp.power(cp.norm(U-Uglobal,2),2) 
            
              
        constr = [
                  U >= np.zeros((simu.M,simu.N)), 
                  U[:,self.p_nr] <= np.ones(simu.M)*sup.Qmax,
                  cp.cumsum(U[:,self.p_nr]) <= np.ones(simu.M)*sup.Vmax - Qextr,
                  # cp.sum(U[:,i]) <= sup.Vmax,
                  np.ones((simu.M,1))*(h0*tank.area) + A @ (U @ np.ones((simu.N,1)) - g) >= np.ones((simu.M,1))*tank.hmin*tank.area,
                  np.ones((simu.M,1))*(h0*tank.area) + A @ (U @ np.ones((simu.N,1)) - g) <= np.ones((simu.M,1))*tank.hmax*tank.area
                  ]
        
        problem = cp.Problem(cp.Minimize(cost) , constr)
        problem.solve()#solver = cp.MOSEK, mosek_params = {'MSK_DPAR_OPTIMIZER_MAX_TIME':  10.0})

        return U.value[0,self.p_nr], U.value



    

    def run(self):
        print('Local controller ', self.p_nr+1, ' online')
        
        # ite = 100
        # Qextr = np.zeros((simu.M))
        # Uglobal = np.zeros((simu.M,simu.N))
        
        # lamb = np.zeros((ite+1, simu.M, simu.N))

        # u = np.zeros((ite))
        # rho = .1
        # j = 0
        # u = 0
        for k in range(simu.ite):
            
            #get data
            h = self.com_func.get_data(str(k), 1) #Get tank level (will later be sensor measurement)
            
            U = np.ones((2,2))
            self.distribute_shares(str(k), U)
            
            
            if h >= tank.hmax:
                u = 0
            elif h <= tank.hmin:
                u = 80
            
            # #Reset lambda and use the last lambda from the previous round 
            # lamb_temp = lamb[j-1,:,:]
            # lamb = np.zeros((ite+1, simu.M, simu.N))
            # lamb[0,:,:] = lamb_temp
            # Qextr[k%simu.M] = u    
            # #Set iteration variables
            # acc = True
            # j = 0
            # while acc and j < ite:
            #     #Solve the local opti problem
            #     u, U = self.opti(sups[self.p_nr], simu.d[k:k+simu.M], simu.c[k:k+simu.M], h, lamb[j,:,:], rho, Uglobal, Qextr)
                
            #     #Value to send to cloud to compute sum
            #     to_sum = U + 1/rho * lamb[j,:,:]
            #     self.distribute_shares(str(k), to_sum)
            #     #Get sum of local U's back from cloud and compute Uglobal as the average. 
            #     Usum = self.reconstruct_secret(str(k))
                
            #     Uglobal = (1/simu.N) * Usum
                
            #     #Update local lambda
            #     lamb[j+1,:,:] = lamb[j,:,:] + rho*( U - Uglobal )
            #     #Compute accuracy of lambda
            #     acc = (np.linalg.norm(lamb[j,:,:] - lamb[j-1,:,:], 2) > 0.1)  
            #     #Update j
            #     j+=1  
            
            #Send the computed u to simulator (will later be input to local pump)
            self.com_func.broadcast_data(u, str(k), ips.addr_dict['simulator'])
          
            