#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:26:22 2023

@author: kst
"""

from local_ctr import PID_ctr, MPC_ctr, OnOff_ctr
from ip_config import ips
from plotter_ctr import simulator
from communication_setup import setup_com, ModBusCom   
from parameters import model, tank
import queue as que
from pyModbusTCP.client import ModbusClient
import time

##########  Setup communication ###############
## Prepare to receive data:
p_nr, rec_q = setup_com('local_ctr', ['cloud'], 'ip addr show eth1')

#Prepare Modbusclient for reading tank level
MB = ModBusCom()
c_tank = MB.ext_Modbus(ips.addr_dict['tank'][0][0])
# ModbusClient(host=ips.addr_dict['tank'][0][0], port=503, unit_id=15, auto_open=True)
if c_tank.open():
    print('Modbus client succesfully connected')
else:
    print('Modbus client connection failed.')


##########  Initialization
starttime = time.monotonic()  #starttime is here, after 5min the first update should appear
mpc = MPC_ctr(p_nr, rec_q)    #MPC controller
q_pid = que.Queue()
pid = PID_ctr(p_nr, q_pid)    #PID controller
q_sim = que.Queue()     
onOff = OnOff_ctr()           #On-off controller to suppli local pump tanks with water from consumer
# sim = simulator(q_sim)        #Simulator used for local plotting


## Use MPCctr to find start flow
h = c_tank.read_input_registers(7, 1)[0] /1000  #get level in tank from mm to m
print('Start level: ', h)
start_flow = mpc.MPC(h)
print('Start flow', start_flow[p_nr])

##########  Start the controllers
#Start on-off control for the "extra" pumps that fills the tank to supply the "real" pumps.
onOff.start()

#Run the local PID controller
pid.u = start_flow[p_nr]
pid.Kp = 0.1
pid.Ki = 0.1
pid.start()

#Run simulator for plotting
# sim.start()
upd_time = 1 #Update MPC control every upd_time minute
# for i in range(model.ite):
for i in range(4):
    print('OUTER ITERATION:', i)
    h = c_tank.read_input_registers(7, 1)[0] /1000  #get level in tank from mm to m
    print('Level in tank: ', h)
    new_flow = mpc.MPC(h)
    time.sleep(60*upd_time - ((time.monotonic() - starttime) % (60*upd_time)))
    q_pid.put(new_flow[p_nr])
    t = (time.monotonic() - starttime)/60
    print('New flow calculated at %.2f' %t, new_flow[p_nr])
    # q_sim.put(new_flow)

pid.on = False
onOff.on = False
# sim.on = False
print('Waiting on threads to finish.')
pid.join()
onOff.join()
# sim.join()
print('Control stopped')
    