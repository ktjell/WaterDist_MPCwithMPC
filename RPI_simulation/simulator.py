#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:54:19 2023

@author: kst
"""

import numpy as np
from threading import Thread
from ip_config import ipconfigs as ips
from parameters import tank, simu
from communication_setup import com_functions
from plotting import plotting

class simulator(Thread):
    def __init__(self, rec_q, p_nr):
        Thread.__init__(self)
        self.recv = {}
        self.p_nr = p_nr
        self.rec_q = rec_q
        self.com_func = com_functions(p_nr, rec_q)
   
    def run(self):
        print('Simulator online')
        
        plot = plotting('Plot1')
        q = np.zeros((simu.ite, simu.N))                  #The optimized flows from pumps
        h,V = np.zeros(simu.ite), np.zeros(simu.ite+1)    #Tank level and Volume
        V[0] = tank.h0*tank.area                          #Start Volume

        for k in range(simu.ite):
            #Level of water in tank: Volume divided by area of tank:
            h[k] = V[k]/tank.area  
            #send to local controllers
            self.com_func.broadcast_data(h[k],str(k), ips.addr_dict['local_ctr'])   
            print('sent data to ctr')
            #Delivered water from pump 1 and 2:
            q[k,:] = np.array(self.com_func.get_data(str(k), len(ips.addr_dict['local_ctr']))).reshape((1,2)) 
            print('recieved data from ctr')
            #Change of volume in the tank: the sum of supply minus consumption.
            dV = sum(q[k,:]) - simu.d[k]      
            #Volume in tank: volume of last time step + change in volume
            V[k+1] = V[k] + dV                  
            ## Done with simulation
            plot.updatePlot(k+1, h[:k+1], q[:k+1,:],simu.d[:k+1])
            
 
            
            
 
