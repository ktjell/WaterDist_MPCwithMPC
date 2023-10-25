#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:33:34 2023

@author: kst
"""

class ipconfigs:
    port = 62
    
    local_ctr_addr = [
                      ['192.168.100.41', 62], #party 0: Local ctr 1
                      ['192.168.100.43', 62]  #party 1: Local ctr 2
                      ]
    
    tank_addr = [
                     ['192.168.100.31', 503]   #Tank
                     ]
    
    cloud_addr = [
                  ['192.168.100.1', 62], #party 0: Cloud 1
                  ['192.168.100.2', 62], #party 1: Cloud 2
                  # ['192.168.100.6', 62]  #-
                  ]
    addr_dict = {'local_ctr':local_ctr_addr, 'tank':tank_addr, 'cloud':cloud_addr}
    
ips = ipconfigs()
