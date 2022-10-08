#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 20:09:38 2020

@author: kat.hendrickson
"""

import numpy as np

class commClass():
    def __init__(self, commRate):
        self.commRate = commRate

    def comm(self, optInputs, Xagents):
        numAgents = optInputs.numAgents
        
        B=np.random.rand(numAgents, numAgents)
        X_new = np.copy(Xagents)
        
        for i in range(numAgents):
            a=optInputs.xBlocks[i]  #lower boundary of block (included)
            b=optInputs.xBlocks[i+1]#upper boundary of block (not included)
            for j in range(numAgents):
                if i != j:
                    if B[i,j] <= self.commRate:
                        B[i,j] = 1
                        X_new[a:b,j] = np.copy(Xagents[a:b,i])
                
        return X_new
