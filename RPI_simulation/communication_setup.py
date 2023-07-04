#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:44:13 2023

@author: kst
"""
import socket
from threading import Thread
import tcp_socket as sock
import time
import queue as que
import os
from ip_config import ipconfigs as ips

def setup_com(name, connect_to):
    #Setup communication line
    class TCP_Thread (Thread):
       stop = False  
       def __init__(self, threadID, name, server_info,q):
          Thread.__init__(self)
          self.q = q
          self.threadID = threadID
          self.name = name
          self.server_info = server_info  # (Tcp_ip adress, Tcp_port)
          self.Rx_packet = [] # tuple [[client_ip, client_port], [Rx_data[n]]]
    
       def run(self):
    #      print("Starting " + self.name)      
          #Create TCP socket
          tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
          tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
          tcpsock.bind(tuple(self.server_info))
          #Communication loop - Wait->Receive->Put to queue
          while not self.stop:
             Rx_packet = sock.TCPserver(tcpsock)
    #         print("Client info:",Rx_packet[0])
    #         print("Data recv:",Rx_packet[1])
             if not self.q.full():
                self.q.put(Rx_packet)
          print("Exiting " + self.name)
    
    
    #Get party number according to ip adress and ip_config file.
    ipv4 = os.popen('ip addr show eth0').read().split("inet ")[1].split("/")[0]
    pnr = ips.addr_dict[name].index([ipv4, ips.port])
    receive_queue = que.Queue()  #Rec data queue
    
    #Initialization and start tcp server
    server_info = ips.addr_dict[name][pnr]#(TCP_IP adress, TCP_PORT)
    TCP_com = TCP_Thread(1, "Communication Thread", server_info, receive_queue)
    TCP_com.start()
    
    # Establish connection
    connections = []
    for name in connect_to:
        connections.extend(ips.addr_dict[name])
    for addr in connections:
        while True:
            try:
                sock.TCPclient(*addr, ['flag', 1])
                break
            except:
                time.sleep(0.5)
                continue
    
    return pnr, receive_queue

class com_functions:
    def __init__(self, p_nr,  rec_q):
        self.rec_q = rec_q
        self.rec_dict = {}
        self.p_nr = p_nr
    
    def readQueue(self):
        while not self.rec_q.empty():
            b = self.rec_q.get()[1]
            self.rec_dict[b[0]] = b[1]
            
    
    def get_data(self, name, n):
        res = []
        for i in range(n):
            while name+str(i) not in self.rec_dict:
                self.readQueue()
                print(self.rec_dict.keys())
            a = self.rec_dict[name+str(i)]
            del self.rec_dict[name+str(i)]
            res.append(a)
        return res
    
    def broadcast_data(self, data, name, receivers):
        for addr in receivers:
            sock.TCPclient(*addr, [name + str(self.p_nr), data])