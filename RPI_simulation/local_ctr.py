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
from gekko import GEKKO

## Cost function
def E(x, r, Dz, p0):
    eta = 0.7
    return cp.inv_pos(simu.dt**2) * r * eta * cp.power(x,3) + eta*x*(Dz - p0) 


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
          
            