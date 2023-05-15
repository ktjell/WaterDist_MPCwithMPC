#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:14:49 2023

@author: kst
"""
import matplotlib.pylab as plt
import matplotlib.dates as md
import numpy as np
from parameters import sups, tank, simu
# from parameters import simu, tank, sups

class plotting:
    def __init__(self, name):
        name = name
        
        # plt.ion()
        # T = np.mod(simu.TIME[:simu.ite], simu.TIME[simu.M])
        # steps = np.where(T[:simu.ite] == [0, 6*60, 12*60, 18*60])[0]
        # ticks = np.tile(np.array(['00', '06', '12', '18']),int(len(steps)/4))
        # print(simu.ite, len(steps), len(ticks))
        
        self.f, self.ax = plt.subplots(2, sharex=True)
        plt.xticks(rotation=45)
        loc = md.AutoDateLocator(interval_multiples=True) # this locator puts ticks at regular intervals
        self.ax[0].xaxis.set_major_formatter(md.ConciseDateFormatter(loc)) #md.DateFormatter('%H:%M'))
        self.ax[0].xaxis.set_major_locator(loc)
        minloc = md.AutoDateLocator(minticks=2, maxticks=5)
        self.ax[0].xaxis.set_minor_locator(minloc)
        for ax in self.ax:
            ax.grid(axis = 'x')

        
        # self.ax[0].set_xticks(simu.TIME[steps].flatten(),ticks)
        self.ax[1].set_xlabel('Time')
        self.ax[0].set_ylabel('Tank level')
        self.line_h, = self.ax[0].plot([1], [1], linewidth  = 3, label='Level in tank') #Line for current volume in tank
        self.lineh1, = self.ax[0].plot([1], [1], 'r', linestyle = 'dashed') #Line for min level in tank
        self.lineh2, = self.ax[0].plot([1], [1], 'r', linestyle = 'dashed') #Line for max level in tank
        
        self.line_q1, = self.ax[1].plot([1], [1], label = 'q1') #Line for current delivered water from pump1
        self.line_q2, = self.ax[1].plot([1], [1], label = 'q2') #Line for current delivered water from pump2
        self.lineq1,  = self.ax[1].plot([1], [1], 'r', linestyle = 'dashed')  #Line for max pump1 performance
        self.lineq2,  = self.ax[1].plot([1], [1], 'r', linestyle = 'dashed')  #Line for max pump2 performance
        self.line_d,  = self.ax[1].plot([1], [1], linewidth= 0.5,  label = 'Demand') #Line for demand of city
        self.ax[1].set_ylabel('q1, q2, demand')
        
        self.f.tight_layout()
        self.f.canvas.draw()
        self.f.show()
        plt.show(block=False)
        
    

        
    def updatePlot(self, k, h, q,d ):

        t = simu.TIMEformat[:k]
        ## Update plot!
        #Requirements:
        self.lineh1.set_data(t, np.ones(k)*tank.hmin)
        self.lineh2.set_data(t, np.ones(k)*tank.hmax)
        self.lineq1.set_data(t, np.ones(k)*sups[0].Qmax)
        self.lineq2.set_data(t, np.ones(k)*sups[1].Qmax,)

        
        self.line_h.set_data(t, h)
        self.line_q1.set_data(t, q[:,0])
        self.line_q2.set_data(t, q[:,1])
        self.line_d.set_data(t, d)

        
        for a in self.ax:
            a.relim() 
            a.autoscale_view(True,True,True) 
        self.f.canvas.draw()
        plt.pause(0.0005)

    ####