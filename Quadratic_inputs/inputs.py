#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:40:26 2020

@author: kat.hendrickson
"""

import numpy as np

class inputsClass():
# Problem Parameters
    def __init__(self):

        someConstantHere = 5 ## You can set any common parameters here, and add an input to the class initialization if desired.
        
    def gradient(self,optInputs,x,agent):
        a=optInputs.xBlocks[agent]  #lower boundary of block (included)
        b=optInputs.xBlocks[agent+1]#upper boundary of block (not included)
        gradient=[]
        for i in range(a,b):
            xi=x[i]
            sumterm=0
            for j in range(optInputs.n):     
                if j != (i) :
                    sumterm=sumterm + 2*(xi - x[j])
            gradient= np.append(gradient, 2*(3/10)*(xi) + (1/200)*sumterm)
        return gradient             #Needs to return vector gradient for each agent.
        
    def projection(self,x):
        for i in range(np.size(x)):
            if x[i] > 10.0:
                x[i] = 10.0
            if x[i] < 1:
                x[i] = 1
        x_hat = x
        return x_hat
