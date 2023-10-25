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
from parameters import p_sts, model
from shamir_real_number import secret_sharing as ss
from ip_config import ipconfigs as ips
from communication_setup import com_functions, ModBusCom
from pyModbusTCP.client import ModbusClient
##With python module interface
import sys
sys.path.insert(1, "/home/pi/WaterDist_MPCwithMPC/LAB_implementation/my_optimizers/tank_filler")
import tank_filler
import time

class PID_ctr(Thread):
    def __init__(self,p_nr, q):
        Thread.__init__(self)
        MB = ModBusCom()
        self.p_nr = p_nr
        self.q = q
        self.u = 0
        self.Kp = 0.1
        self.Ki = 0.1
        self.c = MB.local_c
        self.zeta = 0
        self.num_running_pumps = 1
        self.on = True
    
    def set_pump_setting(self, new_setting):
        self.c.write_multiple_registers(5, new_setting)
        
    def run(self):
        while self.on: 
            # Get data
            if not self.q.empty():
                self.u = self.q.get()
                
            meas_flow = self.c.read_input_registers(11, 1)[0]
         
            #Pressure control
            err = self.u - meas_flow
            p_in = self.Kp*err + self.Ki*self.zeta
            
            if p_in < 0:
                p_in = 0
                self.zeta = (p_in -self.Kp*err)/self.Ki
            
            elif p_in > 100:
                p_in = 100
                self.zeta = (p_in -self.Kp*err)/self.Ki
            
            self.zeta += err
    
            #Pump cut in/out
            if p_in > p_sts[self.p_nr].stepup_speed:
                self.num_running_pumps += 1
            elif p_in < p_sts[self.p_nr].stepdown_speed:
                self.num_running_pumps -= 1
            self.num_running_pumps = max(self.num_running_pumps, 1)
            self.num_running_pumps = min(self.num_running_pumps, p_sts[self.pnr].num_of_pumps)
                
            #Adjust pump settings
            new_pump_setting = np.zeros(p_sts[self.p_nr].num_of_pumps)
            new_pump_setting[:self.num_running_pumps] = int(p_in)
            self.set_pump_setting(list(new_pump_setting))
        
        #Test ended
        self.set_pump_setting([0]*p_sts[self.pnr].num_of_pumps)  #Turn off all pumps
        print('Pumps 1,2 and 3 turned off')
        print('PID controller off')



class MPC_ctr():
    def __init__(self, p_nr, rec_q):
        self.rec_q = rec_q
        self.p_nr = p_nr
        self.com_func = com_functions(p_nr, rec_q)
        self.ss = ss()
        self.k = 0 #Iterations number
        self.ite = 20 #Number of ADMM-iterations
        self.init()
        
 
    def init(self):
        print('Local controller for pump ', self.p_nr+1, ' online')        
        # Prepare solver for optimization
        self.solver = tank_filler.solver()
        #Prepare variables that are reused
        self.Qextr = np.zeros((model.M))
        self.Qex = np.cumsum(self.Qextr[::-1])[::-1]
        self.Uglobal = np.zeros((model.M,model.N))
        self.lamb = np.zeros((model.M, model.N))
        self.rho = .8
        
        
    def distribute_shares(self, name, sec):
        shares = self.ss.gen_matrix_shares(sec)
        for i, addr in enumerate(ips.cloud_addr):
            sock.TCPclient(*addr, [name + str(self.p_nr) , shares[i]])
        
    def reconstruct_secret(self, name):
        shares = self.com_func.get_data(name, len(ips.addr_dict['cloud']))
        # print(shares)
        return self.ss.recon_matrix_secret(shares)
    
    def MPC(self, h):
        #Set iteration variables
        acc = True
        j = 0
        U = np.zeros((model.M, model.N))
        t1 = time.time()
        while acc and j < self.ite:
            print('\n ADMM iteration: ', j)
            # Solve the local opti problem
            # price, demand, extr, lamb, Uglobal, h0, rho
            par = model.el_price[self.k:self.k+model.M].flatten().tolist()
            par.extend(model.consum_profile[self.k:self.k+model.M].flatten().tolist())
            par.extend(self.Qex.flatten().tolist())
            par.extend(self.lamb[:,0].flatten().tolist())
            par.extend(self.lamb[:,1].flatten().tolist())
            par.extend(self.Uglobal[:,0].flatten().tolist())
            par.extend(self.Uglobal[:,1].flatten().tolist())
            par.append(h)
            par.append(self.rho)

            print('Optimizing...')
            result = self.solver.run(p = par)
            print('got result')
  
            s = result.solution
            U[:,0] = s[:model.M]
            U[:,1] = s[model.M:]

            #Value to send to cloud to compute sum
            to_sum = U + 1/self.rho * self.lamb
            print('sending to cloud')
            self.distribute_shares('0', to_sum)
            #Get sum of local U's back from cloud and compute Uglobal as the average. 
            print('Waiting on cloud...')
            Usum = self.reconstruct_secret('0')
            print('recieved from cloud')
            Uglobal = (1/model.N) * Usum
            
            #Update local lambda
            lamb_temp = self.lamb
            self.lamb = lamb_temp + self.rho*( U - Uglobal )
            #Compute accuracy of lambda
            norm = np.linalg.norm(self.lamb - lamb_temp, 2) 
            print('Lambda "error": ', norm)
            #Update j
            j+=1  
            u = U[0,self.p_nr]
            
            self.Qextr[(self.k%model.M)] = u
            self.Qex = np.cumsum(self.Qextr[::-1])[::-1]
            
        self.k += 1
            
        t2 = time.time()

        print('### ', t2-t1 ,'seconds on ', self.ite, ' ADMM iterations: ', u)
        
        return U[0,:]
      
class OnOff_ctr(Thread):
    def __init__(self):
        Thread.__init__(self)
        MB = ModBusCom()
        self.c = MB.local_c
        self.on = True
    
    def set_pump_setting(self, new_setting):
        self.c.write_multiple_registers(8, new_setting)
        
    def run(self):
        while self.on: 
            # Get data            
            meas_dp = self.c.read_input_registers(4, 1)[0]
                     
            #Adjust pump setting
            if meas_dp < 80:
                new_pump_setting = 100
            
            elif meas_dp > 450:
                new_pump_setting = 0

            self.set_pump_setting([new_pump_setting]*2)
        
        #Test ended
        self.set_pump_setting([0]*2)  #Turn off all pumps
        print('Pumps 4 and 5 turned off')
        print('On-off controller off')


            