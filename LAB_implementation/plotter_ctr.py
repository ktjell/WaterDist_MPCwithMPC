#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:43:05 2023

@author: kst
"""

import numpy as np
from threading import Thread
from parameters import tank, model
from plotting import plotting
import time

class simulator(Thread):
    def __init__(self, rec_q):
        Thread.__init__(self)
        self.rec_q = rec_q
        self.on = True
        
   
    def run(self):
        plot = plotting('Plot1') 
        q = np.zeros((model.ite, model.N))                  #The optimized flows from pumps
        h,V = np.zeros(model.ite), np.zeros(model.ite+1)    #Tank level and Volume
        V[0] = tank.h0*tank.area                          #Start Volume
        k = 0
        while self.on:
            #Level of water in tank: Volume divided by area of tank:
            h[k] = V[k]/tank.area  
            #Delivered water from pump 1 and 2:

            while self.rec_q.empty():
                time.sleep(1)
            q[k,:] = self.rec_q.get()
            #Change of volume in the tank: the sum of supply minus consumption.
            dV = sum(q[k,:]) - model.consumption_profile[k]      
            #Volume in tank: volume of last time step + change in volume
            V[k+1] = V[k] + dV                  
            ## Done with simulation
            plot.updatePlot(k+1, h[:k+1], q[:k+1,:],model.consumption_profile[:k+1])
            k+=1
            
 
            