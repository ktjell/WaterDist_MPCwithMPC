#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:26:22 2023

@author: kst
"""

from local_ctr import loc_ctr
from communication_setup import setup_com    

#Setup communication
## Prepare to receive data:
pnr, rec_q = setup_com('local_ctr', ['cloud', 'tank'])


#Initialize and run the local controller
ctr = loc_ctr(pnr, rec_q)
ctr.start()


