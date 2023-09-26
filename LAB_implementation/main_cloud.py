#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:27:37 2023

@author: kst
"""
from communication_setup import setup_com
from cloud import cloud_server

#Setup communication
## Prepare to receive data:
p_nr, rec_q = setup_com('cloud', ['local_ctr'])


#Initialize and run the cloud
cloud = cloud_server(rec_q, p_nr)  
cloud.start()


