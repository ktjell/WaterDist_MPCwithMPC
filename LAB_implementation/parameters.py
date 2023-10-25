#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:03:26 2023

@author: kst
"""

import numpy as np
import scipy.io


## Parameters ####################

class Tank:
    def __init__(self, name):
        name = name

class PumpSt:
    def __init__(self, name):
        name = name

class Model:
    def __init__(self, name):
        name = name

#### Parameters for the resevoir
tank = Tank('Tank1')
tank.area = 0.283 #240             #[m^2] tank area
tank.height = 0.7                  #[m] tank height
tank.hmin = 0.1   #2.4             #[m] min level in the tank
tank.hmax = 0.6   #3.2             #[m] max level in the tank
tank.h0 = 0.12
### Parameters for the pumping stations
Qdesign = 40    
## Pump station 1  
pump_st1 = PumpSt('Pump1')
pump_st1.Dz = 30                 #[m] elevation of the tank
pump_st1.r = 8/(Qdesign**2)      #[m/(m^3h)] network resistance
pump_st1.Vmax = 12*60            #[m**3/day] Extraction permit #24*60
pump_st1.Qmax = Qdesign          #[m^3/h] Maximum flow that the pumps can deliver
pump_st1.K = 5                   #[kr/m^3] Cost of producing water 
pump_st1.p0 = 0                  # presssure "before" pumping stations
pump_st1.num_of_pumps = 3        #number of flow steps 
pump_st1.stepup_speed = 95
pump_st1.stepdown_speed = 80

pump_st2 = PumpSt('Pump2')
pump_st2.Dz = 40                 #[m] elevation of the tank
pump_st2.r = 8/(Qdesign**2)      #[m/(m^3h)] network resistance
pump_st2.Vmax = 12*60            #[m**3/day] Extraction permit
pump_st2.Qmax = Qdesign          #[m^3/h] Maximum flow that the pumps can deliver
pump_st2.K = 5                   #[kr/m^3] Cost of producing water
pump_st2.p0 = 0                  # presssure "before" pumping stations
pump_st1.num_of_pumps = 3        #number of flow steps 
pump_st1.stepup_speed = 95
pump_st1.stepdown_speed = 80

p_sts = [pump_st1, pump_st2]

### Consumption profile
def consum_profile(t):  
    day_length = 7200 #seconds = 2 hours
    day_start = 3600*17
    t_mult = 24*60*60/day_length    #24 hours in seconds 
    t_day = t * t_mult + day_start  #Time on the day in the scenario
    t_day = t_day % (24*60*60)       #Time of virtual day in seconds
    t_day_h = t_day/3600           #Time of virtual day in hours

    
    a0, a1, a2, b1, b2, w = [1, -0.155, -0.217, 0.044, -0.005,0.261]
    demandMult = a0 + a1*np.cos(t_day_h*w) + b1*np.sin(t_day_h*w)\
        + a2*np.cos(2*t_day_h*w) + b2*np.sin(2*t_day_h*w);
    
    v = np.array([0.32, 0.25]).reshape(1, 2)  #There is technically two consumers
    
    try:
        demandMult = demandMult.reshape(len(t), 1)
        dc = np.dot(demandMult, v)
    except:
        dc = demandMult * v
    
    return np.sum(dc,axis = 1).flatten() #We care about the summed consumption

### Parameters for the model and the eksperiment
model = Model('Model1')
sample_hourly = 1 # Sample every sample_hourly hour
model.dt = sample_hourly*60*60       #[sec] sample time
# model.simTime = 7*24*3600 #[sec] Sim time 
## Demand and electricity prices ####################
num_sim_days = 1+1 #Plus one to always be able to predict the next 24 hours (is only used in model.sampling_times)
model.sampling_times = np.linspace(0, 2*60*60 * num_sim_days, 24*num_sim_days) # We run "24 hours" as 2 hours in the lab. So sample evenly 24 times in the 7200 seconds
model.consum_profile = consum_profile(model.sampling_times)
mat2 = scipy.io.loadmat('data/Elspotprice3mdr.mat')  # Actual electricity prices from https://www.energidataservice.dk/tso-electricity/elspotprices
c0 = np.array(mat2.get('price')) / 1000 #Electricity price hour for hour per kWh
model.el_price = c0[::sample_hourly]#np.repeat(c0, 4) #Electricity prices every hour (so addapt to other dt when necessary)
## Number of stations and length of experiment
model.M = int(24*60*60 / model.dt )      #1 update per hour: 24 hours in seconds divided into M steps by dt
model.N = 2     # Numper of pump stations
MaxIte = len(model.el_price) - 1 
model.ite = 1*model.M          
if model.ite > MaxIte:
    print('There is only enough data for ', MaxIte, 'iterations.')
    model.ite = MaxIte





