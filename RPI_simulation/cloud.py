#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:30:06 2023

@author: kst
"""

from threading import Thread
from ip_confic import ipconfigs as ips
from parameters import simu
from communication_setup import com_functions

class cloud_server(Thread):
    def __init__(self, rec_q, p_nr):
        Thread.__init__(self)
        self.p_nr = p_nr
        self.rec_q = rec_q
        self.com_func = com_functions(p_nr, rec_q)
        
    
    def run(self):
        print('Cloud ', self.p_nr+1, ' online')
  
        for i in range(simu.ite):
            shares = self.com_func.get_data(str(i))
        
            #Perform computation
            compute_share = sum(shares)
            
            self.com_func.broadcast_data(compute_share, self.p_nr, ips.addr_dict['local_ctr'])


        