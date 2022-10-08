
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:13:42 2020
Modified on Sat Mar 12 1:20:23 2022
Modified from code at https://github.com/kathendrickson/DistrAsynchGD

@author: kat.hendrickson, dhustigschultz
"""

import numpy as np
import scipy.linalg as la

class DAHBF():
    def __init__(self, alpha, beta, n, inputClass, commClass, updateProb=1):
        """Initialize DAHBF class. 
        Inputs:   
            * alpha: primal stepsize  
            * beta: momentum coefficient            
            * n: dimension of entire primal variable  
            * inputClass: input class that contains the gradient and projection functions  
            * commClass: input class that contains the comm function
            * updateProb: optional input that specifies the probability a primal agent performs an update at any given time k. Default value is 1. 
        """
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.updateProb = updateProb
        
        #Default Values:
        self.x1Actual=np.zeros(n)
        """Value used to compute primal error; Set with `DAHBF.setActual` method."""
        self.x2Actual=np.zeros(n)
        """Value used to compute primal error of history; Set with `DAHBF.setActual` method."""
        self.flagActual=0   #used to determine whether to store error data later.
        self.xBlocks=np.arange(n+1)    
        """ Array that defines the dimensions of the decision variable blocks. Set with `DAHBF.setBlocks` method."""
        self.tolerance = 10 ** -8   #updated with stopIf function
        self.maxIter = 10 ** 5      #updated with stopIf function
        self.maxIterBool=1     #updated with stopIf function
        self.x1Init = np.zeros(n)
        """ Initial decision variable that is used at the start of optimization.
        The default value is a zero vector. Set with `DAHBF.setInit` method. """
        self.x2Init = np.zeros(n)
        """ Initial history variable that is used at the start of optimization.
        The default value is a zero vector. Set with `DAHBF.setInit` method. """
        self.inputClass = inputClass
        self.commClass = commClass

    def setActual(self,x1Actual, x2Actual):
        """If known, true values for decision variable (`DAHBF.x1Actual`) and history variable ('DAHBF.x2Actual') may be set
        and used for error calculations. If set, the values `DAHBF.x1Error` and `DAHBF.x2Error` will be calculated.  
            This error is the L2 norm of the
            distance between the true value and the iterate's value for the decision
            variable."""
        self.flagActual=1
        if np.size(x1Actual) == self.n:
            self.x1Actual=np.copy(x1Actual)
            self.x2Actual=np.copy(x2Actual)
        else: 
            self.flagActual=0
            print("Error: Dimension mismatch between x1Actual, x2Actual and previously defined n.")
        
    def setBlocks(self,xBlocks):
        """ Defines the non-scalar decision variable blocks. xBlocks is an array containing the first index for each agent block. 
        For example, with two agents, Agent 1's block always starts at 0 but Agent 2's block may start at entry 4. You'd then have the array xBlock = np.array([0,4])."""
        self.xBlocks = np.copy(xBlocks)
        self.xBlocks = np.append(self.xBlocks,[self.n])  #used to know the end of the last block.
    
    def setInit(self,x1Init,x2Init):
        """Sets the initial values for the decision variable `DAGD.x1Init` and history variable `DAHBF.x2Init`.  
        If this method is not used, zero vectors are used as a default."""
        if np.size(x1Init) == self.n:
            self.x1Init=np.copy(x1Init)
            self.x2Init=np.copy(x2Init)
        else:
            print("Error: Dimension mismatch between x1Init, x2Init and previously defined n.")
        
        
    def stopIf(self,tolerance,maxIter,maxIterBool=1):
        """Sets optimization stopping parameters by setting the following HBF method values:
            * `DAHBF.tolerance` : tolerance for distance between iterations. When this tolerance is reached, the optimization algorithm stops.  
            * `DAHBF.maxIter` : max number of iterations to run. When this number of iterations is reached, the optimization algorithm stops.  
            * `DAHBF.maxIterBool` : optional boolean input that determines whether the optimization code always stops when `DAHBF.maxIter` is reached (1) 
            or whether it ignores  `DAGD.maxIter` and instead runs until  `DAHBF.tolerance` is reached (0). Default value is (1)."""
        self.tolerance = tolerance
        self.maxIter = maxIter
        self.maxIterBool = maxIterBool    #1 = stop based when maxIter reached

    def run(self):

        # Initialize Primal Variables
        numAgents = np.size(self.xBlocks)-1  #number of agents
        self.numAgents = numAgents
        X1agents = np.outer(self.x1Init,np.ones(self.numAgents))  #initialize agent matrix
        X2agents = np.outer(self.x2Init,np.ones(self.numAgents))  #initialize agent history matrix
    

        convdiff = [self.tolerance + 100]        #initialize convergence distance measure
        x1Error = [la.norm(self.x1Init - self.x1Actual,2)]
        #print("Printing initial x1Error")
        #print(x1Error)
        x2Error = [la.norm(self.x2Init - self.x2Actual,2)]
        gradRow = np.zeros(self.n)          # For debugging purposes, to keep track of gradient during intermediate steps
        gradMatrix = np.array(gradRow)
        gradWRow = np.zeros(self.n)         # For debugging purposes, to keep track of gradient during each iteration of the algorithm
        gradWMatrix = np.array(gradWRow)    
        x1Vector = np.copy(self.x1Init)
        x2Vector = np.copy(self.x2Init)
        x1Values = [np.zeros(self.n)]       # For debugging purposes, to keep track of values during each iteration
        x2Values = [np.zeros(self.n)]
        
        
        # Convergence Parameters
        k=0
        while convdiff[k] > self.tolerance:
            
            if (k % 500 == 0):
                print(k, "iterations...")
            
            if (self.maxIterBool == 1 and k >= self.maxIter):
                break
            
            prevIter1 = np.copy(x1Vector)   # used to determine convdiff
            # prevIter2 = np.copy(x2Vector)   # for keeping track of x2Values, for debugging purposes (might remove later)
            # Update Decision Variables
            for p in range(numAgents):
                x1 = np.copy(X1agents[:,p])
                x2 = np.copy(X2agents[:,p])
                w1 = np.copy(X1agents[:,p]) # used for intermediate step of double update
                w2 = np.copy(X2agents[:,p])
                a=self.xBlocks[p]           #lower boundary of block (included)
                b=self.xBlocks[p+1]         #upper boundary of block (not included)
                updateDraw = np.random.rand(1)
                if updateDraw <= self.updateProb: # Change agent i's value
                    pGradient = self.inputClass.gradient(self,x1,p)
                    gradRow[a:b] = pGradient
                    pwUpdate = x1[a:b] - self.alpha*pGradient + self.beta*(x1[a:b] - x2[a:b])  # First update to x1, of the double-update step 
                    w1[a:b] = self.inputClass.projection(pwUpdate)                        # projecting update 1 of x1
                    pwGradient = self.inputClass.gradient(self,w1,p)
                    gradWRow[a:b] = pwGradient
                    w2[a:b] = np.copy(x1[a:b])
                    X2agents[a:b,p] = np.copy(x1[a:b])  # first update to x2, of the double-update step
                    pUpdate = w1[a:b] - self.alpha*pwGradient + self.beta*(w1[a:b] - w2[a:b]) # Second update to x1, of the double-update step
                    X1agents[a:b,p] = self.inputClass.projection(pUpdate)                 # projecting update 2 of x1
                    X2agents[a:b,p] = np.copy(w1[a:b])                                    # Second update to x2, of the double-update step
                else: # No change to agent i's value
                    pGradient = np.zeros(b-a)  # But we keep track of that fact by recording zeros for this row of gradMatrix and gradWMatrix
                    gradWRow[a:b] = pGradient
                    gradRow[a:b] = pGradient                               
                x1Vector[a:b] = np.copy(X1agents[a:b,p])
                x2Vector[a:b] = np.copy(X2agents[a:b,p])     
            gradMatrix = np.vstack((gradMatrix,gradRow))    # Keeping track of intermediate gradient values, for debugging
            gradWMatrix = np.vstack((gradWMatrix,gradWRow)) # Keeping track of the gradient values, for debugging
            
            # Communicate Updates
            [X1agents, X2agents] = self.commClass.comm(self, X1agents, X2agents)
            
            
    
            # Calculate Iteration Distance and Errors
            k=k+1       # used to count number of iterations (may also be used to limit runs)
            newIter1 = np.copy(x1Vector)
            newIter2 = np.copy(x2Vector)
            x1Values.append(newIter1)
            x2Values.append(newIter2)
            iterNorm = la.norm(prevIter1 - newIter1)  #L2 norm of the diff between prev and current iterations
            convdiff.append(iterNorm)
            x1Error.append(la.norm(x1Vector - self.x1Actual,2)) # Only keeping track of error in x1, but could potentially keep track of error in x2 for debugging or plotting...
        
        if self.flagActual == 1:
            self.x1Error = x1Error
        
        self.iterNorm = convdiff
        self.numIter = k
        self.x1Final=x1Vector
        self.x2Final=x2Vector
        self.gradMatrix = gradMatrix
        self.gradWMatrix = gradWMatrix
        self.x1Values = x1Values
        self.x2Values = x2Values
        return self.x1Final, self.x2Final # returning the final of both, and might only plot x1Final.
    

