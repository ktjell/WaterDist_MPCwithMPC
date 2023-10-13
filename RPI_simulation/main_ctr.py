#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:26:22 2023

@author: kst
"""

from local_ctr import loc_ctr
from communication_setup import setup_com
import argparse
import logging
from pyModbusTCP.server import ModbusServer
from ip_config import ipconfigs as ips
from threading import Thread


class modbus(Thread):
    def __init__(self, p_nr):
        Thread.__init__(self)
        self.rec_q = rec_q
        self.p_nr = p_nr
        
        
    def run(self):
        ip_adr = ips.local_ctr_addr[self.p_nr][0]
        # init logging
        logging.basicConfig()
        # parse args
        parser = argparse.ArgumentParser()
        parser.add_argument('-H', '--host', type=str, default=ip_adr, help='Host (default: localhost)')
        parser.add_argument('-p', '--port', type=int, default=502, help='TCP port (default: 502)')
        parser.add_argument('-d', '--debug', action='store_true', help='set debug mode')
        args = parser.parse_args()
        # logging setup
        if args.debug:
            logging.getLogger('pyModbusTCP.server').setLevel(logging.DEBUG)
        # start modbus server
        server = ModbusServer(host=args.host, port=args.port)
        server.start()
        

#Setup communication
## Prepare to receive data:
pnr, rec_q = setup_com('local_ctr', ['cloud', 'simulator'], 'ip addr show eth1')


#Initialize and run the local controller
ctr = loc_ctr(pnr, rec_q)
ctr.start()

modb = modbus(pnr)
modb.start()
