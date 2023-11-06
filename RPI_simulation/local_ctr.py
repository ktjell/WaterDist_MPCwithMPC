#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:30:24 2023

@author: kst
"""

# import opengen as og
import numpy as np
from threading import Thread
import tcp_socket as sock
from shamir_real_number import secret_sharing as ss
from ip_config import ipconfigs as ips
from parameters import sups, tank, simu
from communication_setup import com_functions
from pyModbusTCP.client import ModbusClient
##With python module interface
import sys
sys.path.insert(1, "/home/pi/WaterDist_MPCwithMPC/RPI_simulation/my_optimizers/tank_filler")
import tank_filler
import time



class loc_ctr(Thread):
    def __init__(self, p_nr, rec_q):
        Thread.__init__(self)
        self.rec_q = rec_q
        self.p_nr = p_nr
        self.com_func = com_functions(p_nr, rec_q)
        self.ss = ss()
 
    def distribute_shares(self, name, sec):
        shares = self.ss.gen_matrix_shares(sec)
        for i, addr in enumerate(ips.cloud_addr):
            sock.TCPclient(*addr, [name + str(self.p_nr) , shares[i]])
        
            
    def reconstruct_secret(self, name):
        shares = self.com_func.get_data(name, len(ips.addr_dict['cloud']))
        # print(shares)
        return self.ss.recon_matrix_secret(shares)
    
    ################################################
    ## MPC optimization #########################
    
    # def startSolver(self):
    #     mng = og.tcp.OptimizerTcpManager('my_optimizers/tank_filler')
    #     mng.start()

    #     mng.ping()                 # check if the server is alive
        
    #     self.mng = mng


    def run(self):
        print('Local controller ', self.p_nr+1, ' online')
        # self.startSolver()
        solver = tank_filler.solver()
        print('Solver succesfully started.')
        ip_adr = ips.local_ctr_addr[self.p_nr][0]
        c = ModbusClient(host=ip_adr, port=502, unit_id=1, auto_open=True)
        if c.open():
            print('Modbus client succesfully connected')
        else:
            print('Modbus clint connection failed.')
        
        Qextr = np.zeros((simu.M))
        Uglobal = np.zeros((simu.M,simu.N))
        ite = 20
        lamb = np.zeros((ite+1, simu.M, simu.N))
        u = 0#np.zeros((simu.ite))
        rho = .8
        j = 0
        for k in range(simu.ite):
            
            #get data
            # h = self.com_func.get_data(str(k), 1)[0] #Get tank level (will later be sensor measurement)
            h = 0
            while h == 0:
                h = c.read_holding_registers(k%2, 1)[0]
                
            print('h: ', h)
            h = h/100
            
            #Reset lambda and use the last lambda from the previous round 
            lamb_temp = lamb[j-1,:,:]
            lamb = np.zeros((ite+1, simu.M, simu.N))
            lamb[0,:,:] = lamb_temp
            Qextr[(k%simu.M)-1] = u
            Qex = np.cumsum(Qextr[::-1])[::-1]
            #Set iteration variables
            acc = True
            j = 0
            U = np.zeros((simu.M, simu.N))
            t1 = time.time()
            while acc and j < ite:
                print('\n ADMM iteration: ', j)
                # Solve the local opti problem
                par = simu.c[k:k+simu.M].flatten().tolist()
                par.extend(simu.d[k:k+simu.M].flatten().tolist())
                par.extend(Qex.flatten().tolist())
                par.extend(lamb[j,:,0].flatten().tolist())
                par.extend(lamb[j,:,1].flatten().tolist())
                par.extend(Uglobal[:,0].flatten().tolist())
                par.extend(Uglobal[:,1].flatten().tolist())
                par.append(h)
                par.append(rho)
                # response =  self.mng.call(par)
                print('Optimizing...')
                result = solver.run(p = par)
                print('got result')
                # if response.is_ok():
                #     # Solver returned a solution
                #     solution_data = response.get()
                s = result.solution
                U[:,0] = s[:simu.M]
                U[:,1] = s[simu.M:]
                # else:
                #     print('opti error')
                #     break
                
                

                #Value to send to cloud to compute sum
                to_sum = U + 1/rho * lamb[j,:,:]
                print('sending to cloud')
                self.distribute_shares('0', to_sum)
                #Get sum of local U's back from cloud and compute Uglobal as the average. 
                print('Waiting on cloud...')
                Usum = self.reconstruct_secret('0')
                print('recieved from cloud')
                Uglobal = (1/simu.N) * Usum
                
                #Update local lambda
                lamb[j+1,:,:] = lamb[j,:,:] + rho*( U - Uglobal )
                #Compute accuracy of lambda
                norm = np.linalg.norm(lamb[j,:,:] - lamb[j-1,:,:], 2) 
                print('Accuracy of lambda: ', norm)
                #Update j
                j+=1  
                u = U[0,self.p_nr]
                
            t2 = time.time()
            #Send the computed u to simulator (will later be input to local pump)
            print('### ', t2-t1 ,'seconds on ', ite, ' ADMM iterations = one ctr input.')
            
            # c.write_multiple_registers(10, u)
            self.com_func.broadcast_data(u, str(k), ips.addr_dict['simulator'])
        # self.distribute_shares('0', 'Stop')          
            