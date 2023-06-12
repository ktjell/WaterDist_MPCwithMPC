#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:03:26 2023

@author: kst
"""

import scipy.io
# import cvxpy as cp
# import pandas as pd
import numpy as np



## Parameters ####################

class Networksupply:
    def __init__(self, name):
        name = name
        
class Tank:
    def __init__(self, name):
        name = name

class Simulation:
    def __init__(self, name):
        name = name

Qdesign = 40

tank = Tank('Tank1')
tank.area = 240              #[m^2] tank area
tank.hmin = 2.4              #[m] min level in the tank
tank.hmax = 3.2              #[m] max level in the tank
tank.h0 = 3                  #[m] start level of tank

sup1 = Networksupply('NetworkSupply 1')
sup1.Dz = 30                 #[m] elevation of the tank
sup1.r = 8/(Qdesign**2)      #[m/(m^3h)] network resistance
sup1.Vmax = 12*60            #[m**3/day] Extraction permit #24*60
sup1.Qmax = Qdesign          #[m^3/h] Maximum flow that the pumps can deliver
sup1.num_of_pumps = 2        #number of flow steps 
sup1.K = 5                   #[kr/m^3] Cost of producing water 
sup1.p0 = 0                  # presssure "before" pumping stations

sup2 = Networksupply('NetworkSupply 2')
sup2.Dz = 40                 #[m] elevation of the tank
sup2.r = 8/(Qdesign**2)      #[m/(m^3h)] network resistance
sup2.Vmax = 12*60            #[m**3/day] Extraction permit
sup2.Qmax = Qdesign          #[m^3/h] Maximum flow that the pumps can deliver
sup2.num_of_pumps = 3        #number of flow steps  
sup2.K = 5                   #[kr/m^3] Cost of producing water
sup2.p0 = 0                  # presssure "before" pumping stations

sups = [sup1, sup2]



simu = Simulation('Simu1')
sample_hourly = 1 # Sample every sample_hourly hour
simu.dt = sample_hourly*60*60       #[sec] sample time
simu.simTime = 7*24*3600 #[sec] Sim time 
## Demand and electricity prices ####################
mat1 = scipy.io.loadmat('data/UserConsumption.mat')  # Simulated demand
mat2 = scipy.io.loadmat('data/Elspotprice3mdr.mat')  # Actual electricity prices from https://www.energidataservice.dk/tso-electricity/elspotprices
simu.d = mat1['q_u1'][::4][::sample_hourly] #User consumption every 15 min (so addapt to other dt when necessary)
simu.TIME = mat1['time'][::4][::sample_hourly]
#Convert time to np-time format:
start = np.datetime64('2023-01-01T00:00') #Chose some starting point
simu.TIMEformat = start + simu.TIME.astype('timedelta64[m]')
c0 = mat2['price'] / 1000 #Electricity price hour for hour per kWh
simu.c = c0[::sample_hourly]#np.repeat(c0, 4) #Electricity prices every hour (so addapt to other dt when necessary)


# Cost function
# def E(x, r, Dz, p0):
#     eta = 0.7
#     return cp.inv_pos(simu.dt**2) * r * eta * cp.power(x,3) + eta*x*(Dz - p0) 


simu.M = int(24*60*60 / simu.dt )      #24 hours in seconds divided into M steps by dt
simu.N = 2
MaxIte = min( (len(simu.d)), len(simu.c) ) - 1 
simu.ite = 5*simu.M           #Should be multiple of M for nice plotting :)
if simu.ite > MaxIte:
    print('There is only enough data for ', MaxIte, 'iterations.')
    simu.ite = MaxIte





