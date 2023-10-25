#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:13:36 2023

@author: kst
"""

from pyModbusTCP.client import ModbusClient

c = ModbusClient(host = 'localhost', port = 503, unit_id = 15, auto_open = True)
if c.open():
    print('Modbus client succesfully connected')
else:
    print('Modbus client connection failed.')
    
on = 100*100
    
c.write_multiple_registers(3, [on])

c.close()