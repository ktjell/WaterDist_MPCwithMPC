#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:12:01 2023

@author: kst
"""

import numpy as np
import random
from ip_config import ipconfigs as ips

class secret_sharing:
    def __init__(self, t=0.5, mu = 0,sigma = 10):
        self.n = len(ips.cloud_addr)
        self.t = int(np.ceil(t*self.n))
        self.mu = mu
        self.sigma = sigma
        self.p = np.linspace(1, 2,self.n)

    def gen_shares(self,secret,var = 0):
        if var == 0:
            var = self.sigma
        coefficients = np.empty(self.t+1)
        shares = np.empty(self.n)
        y = np.random.normal(self.mu, var, self.t)
        
        sub = random.sample(range(self.n), self.t)
        x_es = np.empty(self.t+1)
        x_es[0] = 0
        for i in range(self.t):
            x_es[i+1] = self.p[sub[i]]
        
        for i in range(self.n):
            shares[i] = 0
            for j in range(self.t+1):
                temp = 1
                for k in range(self.t+1):
                    if j!=k:
                        temp = temp*(self.p[i]-x_es[k])/(x_es[j]-x_es[k])
                if j == 0:
                    shares[i] = shares[i]+secret*temp
                else:
                    shares[i] = shares[i]+y[j-1]*temp
                    
        return np.array(shares)
       
    def gen_matrix_shares(self, sec):
        n,m = np.shape(sec)
        shares = np.zeros((self.n, n,m))
        for i in range(n):
            for j in range(m):
                tshares = self.gen_shares(sec[i,j])
                for k in range(self.n):
                    shares[k,i,j] = tshares[k]
        return shares
    
    def recon_secret(self,shares):
        L_p = np.empty(len(self.p))
        for i in range(len(self.p)):
            L_prod = 1
            for j in range(0,len(self.p)):
                if self.p[i]!=self.p[j]:
                    L_prod *= (-self.p[j])/(self.p[i]-self.p[j])
            L_p[i] = L_prod
        sums = np.dot(shares,L_p)
        return sums
    
    def recon_matrix_secret(self, shares):
        n,m = np.shape(shares[0])
        res = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                s = []
                for k in range(len(shares)):
                    s.append(shares[k,i,j])
                res[i,j] = self.recon_secret(s)
        return res
                
        
ss = secret_sharing()

m = np.zeros((5,5)) 

shares = ss.gen_matrix_shares(m)

matrix = ss.recon_matrix_secret(shares)

print(matrix)               
    
    
    
    
    