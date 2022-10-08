#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:09:38 2020

Modified on Sat Mar 12 12:59:28 2022
Modified from code at https://github.com/kathendrickson/DistrAsynchGD

@author: kat.hendrickson, dhustigschultz
"""

import numpy as np

class commClassHBF():
    def __init__(self, commRate):
        self.commRate = commRate

    def comm(self, optInputs, X1agents, X2agents):
        numAgents = optInputs.numAgents
        
        B=np.random.rand(numAgents, numAgents)
        X1_new = np.copy(X1agents)
        X2_new = np.copy(X2agents)
        
        for i in range(numAgents):
            a=optInputs.xBlocks[i]  #lower boundary of block (included)
            b=optInputs.xBlocks[i+1]#upper boundary of block (not included)
            for j in range(numAgents):
                if i != j:
                    if B[i,j] <= self.commRate:
                        B[i,j] = 1
                        X1_new[a:b,j] = np.copy(X1agents[a:b,i])
                        X2_new[a:b,j] = np.copy(X2agents[a:b,i])
                
        return X1_new, X2_new
