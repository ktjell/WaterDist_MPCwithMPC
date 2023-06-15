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
from parameters import sups, tank, simu
from communication_setup import com_functions
import scipy
from scipy.optimize import NonlinearConstraint

## Cost function
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

    def opti(self, sup, i, g, c, h0, lamb, rho, Uglobal, extr, x0):
        
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




    

    def run(self):
        print('Local controller ', self.p_nr+1, ' online')
        
        # ite = 100
        # Qextr = np.zeros((simu.M))
        # Uglobal = np.zeros((simu.M,simu.N))
        
        # lamb = np.zeros((ite+1, simu.M, simu.N))
        # x0 = np.zeros((2*simu.M)) 
        # u = np.zeros((ite))
        # rho = .1
        # j = 0
        # u = 0
        for k in range(simu.ite):
            
            #get data
            h = self.com_func.get_data(str(k), 1) #Get tank level (will later be sensor measurement)
            
            U = np.ones((2,2))*k
            self.distribute_shares(str(k), U)
            Usum = self.reconstruct_secret(str(k))
            print(Usum)
            
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
            #     U, x0 = self.opti(sups[self.p_nr], self.p_nr, simu.d[k:k+simu.M], simu.c[k:k+simu.M], h, lamb[j,:,:], rho, Uglobal, Qextr, x0)
                
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
          
            