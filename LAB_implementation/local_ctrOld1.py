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
from communication_setup import com_functions
from pyModbusTCP.client import ModbusClient
##With python module interface
import sys
sys.path.insert(1, "/home/pi/WaterDist_MPCwithMPC/LAB_implementation/my_optimizers/tank_filler")
import tank_filler
import time

class PID_ctr(Thread):
    def __init__(self,p_nr, q, start_flow, Kp, Ki):
        Thread.__init__(self)
        self.p_nr = p_nr
        self.q = q
        self.u = start_flow
        self.Kp = Kp
        self.Ki = Ki
        self.c = ModbusClient(host = 'localhost', port = 502, unit_id = 16, auto_open = True) #For CCU communication
        self.zeta = 0
        self.num_running_pumps = 1
    def run(self):
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
        
        self.c.write_multiple_registers(5, list(new_pump_setting))



class MPC_ctr(Thread):
    def __init__(self, p_nr, rec_q, output_q):
        Thread.__init__(self)
        self.rec_q = rec_q
        self.output_q = output_q
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

    def run(self):
        print('Local controller for pump ', self.p_nr+1, ' online')
        # self.startSolver()
        solver = tank_filler.solver()
        print('Solver succesfully started.')
        c_tank = ModbusClient(host=ips.addr_dict['tank'][0][0], port=503, unit_id=15, auto_open=True)
        if c_tank.open():
            print('Modbus client succesfully connected')
        else:
            print('Modbus client connection failed.')
        
        Qextr = np.zeros((model.M))
        Uglobal = np.zeros((model.M,model.N))
        ite = 20
        lamb = np.zeros((ite+1, model.M, model.N))
        u = 0 
        rho = .8
        
        for k in range(simu.ite):
    
            #get data
            h = c_tank.read_input_registers(7, 1)[0]
            h = h/100   
            print('h: ', h)
            j = 0
            
            
            #Reset lambda and use the last lambda from the previous round 
            lamb_temp = lamb[j-1,:,:]
            lamb = np.zeros((ite+1, model.M, simu.N))
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
    
                print('Optimizing...')
                result = solver.run(p = par)
                print('got result')
      
                s = result.solution
                U[:,0] = s[:simu.M]
                U[:,1] = s[simu.M:]
    
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
                print('Lambda "error": ', norm)
                #Update j
                j+=1  
                u = U[0,self.p_nr]
                
            t2 = time.time()
    
            print('### ', t2-t1 ,'seconds on ', ite, ' ADMM iterations: ', u)
            
            self.output_q.put(u)
       
            