#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:14:49 2023

@author: kst
"""
import matplotlib.pylab as plt
import numpy as np
from parameters import simu, tank, sups

class plotting:
    def __init__(self, name):
        name = name
        
        plt.ion()
        T = np.mod(simu.TIME, simu.TIME[simu.M])
        steps = np.where(T[:simu.ite] == [0, 6*60, 12*60, 18*60])[0]
        ticks = np.tile(np.array(['00', '06', '12', '18']),int(len(steps)/4))
        
        self.f, self.ax = plt.subplots(4, sharex=True, figsize = (15,9))
        
        self.ax[0].set_xticks(simu.TIME[steps].flatten(),ticks)
        self.ax[3].set_xlabel('Time')
        self.ax[0].set_ylabel('Tank level')
        self.line_h, = self.ax[0].plot([1], [1], label='Level in tank')
        self.ax[0].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*tank.hmin, 'r', linestyle = 'dashed')
        self.ax[0].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*tank.hmax, 'r', linestyle = 'dashed')
        
        self.line_q1, = self.ax[1].plot([1], [1], label = 'q1')
        self.line_q2, = self.ax[1].plot([1], [1], label = 'q2')
        self.ax[1].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*sups[0].Qmax, 'r', linestyle = 'dashed')
        self.ax[1].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*sups[1].Qmax, 'r', linestyle = 'dashed')
        self.line_d, = self.ax[1].plot([1], [1], linewidth= 0.5,  label = 'Demand')
        self.ax[1].set_ylabel('q1, q2, demand')
        
        # ax[2].plot(TIME[:ite],c[:ite], label = 'price')
        self.ax[2].set_ylabel('Extraction per day')
        self.line_Extq1, = self.ax[2].plot([1], [1],)
        self.ax[2].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*sups[0].Vmax, 'b', linestyle = 'dashed')
        self.line_Extq2, = self.ax[2].plot([1], [1],)
        self.ax[2].plot(simu.TIME[:simu.ite],np.ones(simu.ite)*sups[1].Vmax, 'b', linestyle = 'dashed')
        
        self.line_p1, = self.ax[3].plot([1], [1],)
        self.line_p2, = self.ax[3].plot([1], [1],)
        self.ax[3].set_ylabel('Pressure')
        # f.xticks(steps, ticks)#, rotation='vertical')
        self.f.tight_layout()
        self.f.canvas.draw()
        self.f.show()
        plt.show(block=False)
        
    

        
    def updatePlot(self, k, h, q1,q2,d,cum_q1,cum_q2, p1,p2):

        t = simu.TIME[:k]
        ## Update plot!
        self.line_h.set_data(t, h)
        self.line_q1.set_data(t, q1)
        self.line_q2.set_data(t, q2)
        self.line_d.set_data(t, d)
        self.line_Extq1.set_data(t,cum_q1)
        self.line_Extq2.set_data(t,cum_q2)
        self.line_p1.set_data(t, p1)
        self.line_p2.set_data(t, p2)
        for a in self.ax:
            a.relim() 
            a.autoscale_view(True,True,True) 
        self.f.canvas.draw()
        plt.pause(0.0005)

    ####