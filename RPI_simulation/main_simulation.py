#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:28:58 2023

@author: kst
"""

from communication_setup import setup_com
from simulator import simulator

#Setup communication
## Prepare to receive data:
p_nr, rec_q = setup_com('simulator', ['local_ctr'], 'ip addr show eth1')


#Initialize and run the simulator
sim = simulator(rec_q, p_nr)  
sim.start()
