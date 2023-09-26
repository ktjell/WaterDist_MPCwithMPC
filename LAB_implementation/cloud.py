#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:30:06 2023

@author: kst
"""

from threading import Thread
from ip_config import ipconfigs as ips
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
        T = True
        while T:
            shares = self.com_func.get_data('0', 2) #2 is number of local controllers that sends shares to cloud
            print(shares)
            # if shares[0] == 'Stop':
            #     T = False
            #Perform computation
            compute_share = sum(shares)
            
            self.com_func.broadcast_data(compute_share, '0', ips.addr_dict['local_ctr'])


        