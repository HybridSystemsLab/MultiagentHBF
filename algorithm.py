
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:13:42 2020

@author: kat.hendrickson
"""

import numpy as np
import scipy.linalg as la

class DAGD():
    def __init__(self, gamma, n, inputClass, commClass, updateProb=1):
        """Initialize DAGD class. 
        Inputs:   
            * gamma: primal stepsize   
            * n: dimension of entire primal variable  
            * inputClass: input class that contains the gradient and projection functions  
            * commClass: input class that contains the comm function
            * updateProb: optional input that specifies the probability a primal agent performs an update at any given time k. Default value is 1. 
        """
        self.gamma=gamma
        self.n = n
        self.updateProb = updateProb
        
        #Default Values:
        self.xActual=np.zeros(n)
        """Value used to compute primal error; Set with `DAGD.setActual` method."""
        self.flagActual=0   #used to determine whether to store error data later.
        self.xBlocks=np.arange(n+1)    
        """ Array that defines the dimensions of the decision variable blocks. Set with `DAGD.setBlocks` method."""
        self.tolerance = 10 ** -8   #updated with stopIf function
        self.maxIter = 10 ** 5      #updated with stopIf function
        self.maxIterBool=1     #updated with stopIf function
        self.xInit = np.zeros(n)
        """ Initial decision variable that is used at the start of optimization.
        The default value is a zero vector. Set with `DAGD.setInit` method. """
        self.inputClass = inputClass
        self.commClass = commClass

    def setActual(self,xActual):
        """If known, true values for decision variable (`DAGD.xActual`) may be set
        and used for error calculations. If set, the value `DAGD.xError` will be calculated.  
            This error is the L2 norm of the
            distance between the true value and the iterate's value for the decision
            variable."""
        self.flagActual=1
        if np.size(xActual) == self.n:
            self.xActual=np.copy(xActual)
        else: 
            self.flagActual=0
            print("Error: Dimension mismatch between xActual and previously defined n.")
        
    def setBlocks(self,xBlocks):
        """ Defines the non-scalar decision variable blocks. xBlocks is an array containing the first index for each agent block. 
        For example, with two agents, Agent 1's block always starts at 0 but Agent 2's block may start at entry 4. You'd then have the array xBlock = np.array([0,4])."""
        self.xBlocks = np.copy(xBlocks)
        self.xBlocks = np.append(self.xBlocks,[self.n])  #used to know the end of the last block.
    
    def setInit(self,xInit):
        """Sets the initial values for the decision variable `DAGD.xInit`.  
        If this method is not used, zero vectors are used as a default."""
        if np.size(xInit) == self.n:
            self.xInit=np.copy(xInit)
        else:
            print("Error: Dimension mismatch between xInit and previously defined n.")
        
        
    def stopIf(self,tolerance,maxIter,maxIterBool=1):
        """Sets optimization stopping parameters by setting the following DAGD method values:
            * `DAGD.tolerance` : tolerance for distance between iterations. When this tolerance is reached, the optimization algorithm stops.  
            * `DAGD.maxIter` : max number of iterations to run. When this number of iterations is reached, the optimization algorithm stops.  
            * `DAGD.maxIterBool` : optional boolean input that determines whether the optimization code always stops when `DAGD.maxIter` is reached (1) 
            or whether it ignores  `DAGD.maxIter` and instead runs until  `DAGD.tolerance` is reached (0). Default value is (1)."""
        self.tolerance = tolerance
        self.maxIter = maxIter
        self.maxIterBool = maxIterBool    #1 = stop based when maxIter reached

    def run(self):

        # Initialize Primal Variables
        numAgents = np.size(self.xBlocks)-1  #number of agents
        self.numAgents = numAgents
        Xagents = np.outer(self.xInit,np.ones(self.numAgents))  #initialize agent matrix
    

        convdiff=[self.tolerance + 100]        #initialize convergence distance measure
        xError=[la.norm(self.xInit- self.xActual,2)]
        #print("Printing initial xError")
        #print(xError)
        gradRow = np.zeros(self.n)
        gradMatrix= np.array(gradRow)
        xVector = np.copy(self.xInit)
        xValues = [np.zeros(self.n)]
        
        
        # Convergence Parameters
        k=0
        while convdiff[k] > self.tolerance:
            
            if (k % 500 == 0):
                print(k, "iterations...")
            
            if (self.maxIterBool == 1 and k >= self.maxIter):
                break
            
            prevIter = np.copy(xVector)   #used to determine convdiff
            # Update Decision Variables
            for p in range(numAgents):
                x = np.copy(Xagents[:,p])
                a=self.xBlocks[p]  #lower boundary of block (included)
                b=self.xBlocks[p+1] #upper boundary of block (not included)
                updateDraw = np.random.rand(1)
                if updateDraw <= self.updateProb:
                    pGradient = self.inputClass.gradient(self,x,p)
                else:
                    pGradient = np.zeros(b-a)
                gradRow[a:b]=pGradient
                pUpdate = x[a:b] - self.gamma*pGradient
                Xagents[a:b,p] = self.inputClass.projection(pUpdate)
                xVector[a:b] = np.copy(Xagents[a:b,p])
                
            gradMatrix =np.vstack((gradMatrix,gradRow))
           
            
            # Communicate Updates
            Xagents = self.commClass.comm(self, Xagents)
            
    
            # Calculate Iteration Distance and Errors
            k=k+1       # used to count number of iterations (may also be used to limit runs)
            newIter = np.copy(xVector)
            xValues.append(newIter)
            iterNorm = la.norm(prevIter - newIter)  #L2 norm of the diff between prev and current iterations
            convdiff.append(iterNorm)
            xError.append(la.norm(xVector - self.xActual,2))
        
        if self.flagActual == 1:
            self.xError = xError
        
        self.iterNorm = convdiff
        self.numIter = k
        self.xFinal=xVector
        self.gradMatrix = gradMatrix
        self.xValues = xValues
        return self.xFinal
    

