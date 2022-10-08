#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:16:47 2020
Modified on Wed Mar 16 13:34:40 2022
Modified from code at https://github.com/kathendrickson/DACOA/blob/main/Network_inputs/cvxpySol_network.py

@author: kat.hendrickson, dhustigschultz
"""
import cvxpy as cp
import numpy as np

def findActual(n):
    x=cp.Variable(n)
    
    constraints = [x >= 1, x <= 10]
    
    OrigF = 0
    
    for i in range(n):
        xi = x[i]
        sumterm = 0
        for j in range(n):
            if j != (i):
                sumterm = sumterm + (xi - x[j])**2
        OrigF = OrigF + (3/10)*xi**2 + (1/200)*sumterm
        
    obj = cp.Minimize(OrigF)
    
    prob = cp.Problem(obj,constraints)
    prob.solve()
    
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    
    return x.value

