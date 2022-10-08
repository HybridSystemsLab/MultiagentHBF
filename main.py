import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from algorithm import DAGD
from algorithmHBF import DAHBF
from Quadratic_inputs.inputs import inputsClass
from communicate import commClass
from communicateHBF import commClassHBF

#-----------------------------------------------------------------------------
#       Plot Formatting
#-----------------------------------------------------------------------------

#import matplotlib #uncomment to reset plot styles
#matplotlib.rc_file_defaults() #uncomment to reset plot styles
plt.set_loglevel("error")
plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams['pdf.fonttype'] = 42
sns.set_context("notebook", rc={"font.size":16,
                                "axes.titlesize":18,
                                "axes.labelsize":16,
                                "figure.figsize": 6.4,
                                "savefig.dpi":600,     
                                "savefig.format": "eps",
                                "savefig.bbox": "tight",
                                "savefig.pad_inches": 0.1
                                })
sns.despine()
light_blue = "#a6cee3"
dark_blue = "#1f78b4"
light_green = "#b2df8a"
dark_green = "#33a02c"
pink = "#fb9a99"
red = "#e31a1c"
light_orange = "#fdbf6f"
dark_orange = "#ff7f00"
light_purple = "#cab2d6"
dark_purple = "#6a3d9a"

#-----------------------------------------------------------------------------
#       Use CVXPY to Find "Actual" Solution
#-----------------------------------------------------------------------------
# This is not necessary to run DAGD or DAHBF but is used to plot distance from the
# solution, if desired.

import Quadratic_inputs.cvxpySol #inputs file for CVXPY
xActual = Quadratic_inputs.cvxpySol.findActual(10)


#-----------------------------------------------------------------------------
#       Algorithm Parameters
#-----------------------------------------------------------------------------

n = 10        # dimension of decision vector, x       
gamma = .3    # stepsize for GD
alpha = .3    # stepsize for HBF
beta = .075   # momentum parameter for HBF

InitCond = 10*np.ones(n)

# Create Inputs Class
inputs = inputsClass()

# Create Communication Class with Comm Rate of 0.50 (agents communicate ~50% of the time)
comm50 = commClass(.5)
commHBF50 = commClassHBF(.5) 

#-----------------------------------------------------------------------------
#       Scalar Blocks with Agent Updates Occuring Every Timestep, 
#       Non Ill-Conditioned Objective Function
#-----------------------------------------------------------------------------

## Scalar Blocks
print("Running with Scalar Blocks...")

# Create DAGD and DAHBF classes with inputs defined above
scalarBlocks = DAGD(gamma, n, inputs, comm50)
scalarBlocksHBF = DAHBF(alpha, beta, n, inputs, commHBF50)

# Optional: Set the "actual" decision variable values to compute error later
#           If not set, error will not be calculated.
scalarBlocks.setActual(xActual)
scalarBlocksHBF.setActual(xActual,xActual)

# Optional: Set the initial decision variable values
#           If not set, zero vectors will be used.
scalarBlocks.setInit(InitCond)
scalarBlocksHBF.setInit(InitCond,InitCond)

# Optional: Set stopping parameters, stopIf(tol, maxIter, maxIterBool=1), where
#               tol = tolerance for distance between iterations, 
#               maxIter = max number of iterations to run
#               maxIterBool = whether to stop at the maxIter (1) or continue 
#                               running until tol is reached (0). 
#           If not set, tol = 10**-8, maxIter = 10 ** 5, maxIterBool=1.
scalarBlocks.stopIf(10 ** -8,10**6)
scalarBlocksHBF.stopIf(10 ** -8,10**6)

# Run DAGD and DAHBF for scalar blocks
scalarBlocks.run()
scalarBlocksHBF.run()

print("Number of iterations for GD: ")
print(scalarBlocks.numIter+1)
print(scalarBlocks.xError[-1])
print("Number of iterations for Double HBF: ")
print(scalarBlocksHBF.numIter+1)
print(scalarBlocksHBF.x1Error[-1])

#----------------------------------------
## Figure Plotting
plt.semilogy(np.arange(0,scalarBlocks.numIter+1), scalarBlocks.iterNorm[0:], color= dark_blue, label="GD")
plt.semilogy(np.arange(0,scalarBlocksHBF.numIter+1), scalarBlocksHBF.iterNorm[0:], color= red, label="D-HBF")
plt.ylabel("$|| z_1(k) - z_1(k-1)||$")
plt.xlabel("Time, k")
plt.ylim(10 ** -8, 10 ** 3)
plt.title("Convergence, GD vs. Double HBF")
plt.legend()
plt.savefig('Convergence.eps')
plt.show()

plt.semilogy(np.arange(0,scalarBlocks.numIter+1), scalarBlocks.xError[0:], color= dark_blue, label="Gradient Descent")
plt.semilogy(np.arange(0,scalarBlocksHBF.numIter+1), scalarBlocksHBF.x1Error[0:], color= red, label="Double HBF")
plt.ylabel('$|| z_1 - x^*||$')
plt.xlabel("Time, k")
plt.ylim(10 ** -8, 10 ** 2)
plt.title("Error, GD vs. Double HBF")
plt.legend()
plt.savefig('Error.eps')
plt.show()

#-----------------------------------------------------------------------------
#       Scalar Blocks with 100% Agent Update Rate, 
#       Comm Rate 100 %
#-----------------------------------------------------------------------------

# Machine epsilon for python, used as the tolerance for the stopping parameters in stopIf, below.
MachineEps = 2.3*(10 ** -16)

# Used for the stopping parameters in stopIf, below, to stop when tolerance is reached.
BoolIter = 0

# Create Communication Class with Comm Rate of 1 (agents communicate ~100% of the time)
comm100 = commClass(1)
commHBF100 = commClassHBF(1) 

## Scalar Blocks
print("Running with Scalar Blocks... 100% Update Rate, 100% Comm Rate")

# Create DAGD and DAHBF classes with inputs defined above
scalarBlocks100 = DAGD(gamma, n, inputs, comm100)
scalarBlocksHBF100 = DAHBF(alpha, beta, n, inputs, commHBF100)

# Optional: Set the "actual" decision variable values to compute error later
#           If not set, error will not be calculated.
scalarBlocks100.setActual(xActual)
scalarBlocksHBF100.setActual(xActual,xActual)

# Optional: Set the initial decision variable values
#           If not set, zero vectors will be used.
scalarBlocks100.setInit(InitCond)
scalarBlocksHBF100.setInit(InitCond,InitCond)

# Optional: Set stopping parameters, stopIf(tol, maxIter, maxIterBool=1), where
#               tol = tolerance for distance between iterations, 
#               maxIter = max number of iterations to run
#               maxIterBool = whether to stop at the maxIter (1) or continue 
#                               running until tol is reached (0). 
#           If not set, tol = 10**-8, maxIter = 10 ** 5, maxIterBool=1.
scalarBlocks100.stopIf(MachineEps,10**6,BoolIter)
scalarBlocksHBF100.stopIf(MachineEps,10**6,BoolIter)

# Run DAGD and DAHBF for scalar blocks
scalarBlocks100.run()
scalarBlocksHBF100.run()

#-----------------------------------------------------------------------------
#       Scalar Blocks with 75% Agent Update Rate, 
#       Comm Rate 100 %
#-----------------------------------------------------------------------------

## Scalar Blocks
print("Running with Scalar Blocks... 75% Update Rate, 100% Comm Rate")

# Create DAGD and DAHBF classes with inputs defined above
scalarBlocks75 = DAGD(gamma, n, inputs, comm100, .75)
scalarBlocksHBF75 = DAHBF(alpha, beta, n, inputs, commHBF100, .75)

# Optional: Set the "actual" decision variable values to compute error later
#           If not set, error will not be calculated.
scalarBlocks75.setActual(xActual)
scalarBlocksHBF75.setActual(xActual,xActual)

# Optional: Set the initial decision variable values
#           If not set, zero vectors will be used.
scalarBlocks75.setInit(InitCond)
scalarBlocksHBF75.setInit(InitCond,InitCond)

# Optional: Set stopping parameters, stopIf(tol, maxIter, maxIterBool=1), where
#               tol = tolerance for distance between iterations, 
#               maxIter = max number of iterations to run
#               maxIterBool = whether to stop at the maxIter (1) or continue 
#                               running until tol is reached (0). 
#           If not set, tol = 10**-8, maxIter = 10 ** 5, maxIterBool=1.
scalarBlocks75.stopIf(MachineEps,10**6,BoolIter)
scalarBlocksHBF75.stopIf(MachineEps,10**6,BoolIter)

# Run DAGD and DAHBF for scalar blocks
scalarBlocks75.run()
scalarBlocksHBF75.run()

#-----------------------------------------------------------------------------
#       Scalar Blocks with 65% Agent Update Rate, 
#       Comm Rate 100 %
#-----------------------------------------------------------------------------

## Scalar Blocks
print("Running with Scalar Blocks... 65% Update Rate, 100% Comm Rate")

# Create DAGD and DAHBF classes with inputs defined above
scalarBlocks65 = DAGD(gamma, n, inputs, comm100, .65)
scalarBlocksHBF65 = DAHBF(alpha, beta, n, inputs, commHBF100, .65)

# Optional: Set the "actual" decision variable values to compute error later
#           If not set, error will not be calculated.
scalarBlocks65.setActual(xActual)
scalarBlocksHBF65.setActual(xActual,xActual)

# Optional: Set the initial decision variable values
#           If not set, zero vectors will be used.
scalarBlocks65.setInit(InitCond)
scalarBlocksHBF65.setInit(InitCond,InitCond)

# Optional: Set stopping parameters, stopIf(tol, maxIter, maxIterBool=1), where
#               tol = tolerance for distance between iterations, 
#               maxIter = max number of iterations to run
#               maxIterBool = whether to stop at the maxIter (1) or continue 
#                               running until tol is reached (0). 
#           If not set, tol = 10**-8, maxIter = 10 ** 5, maxIterBool=1.
scalarBlocks65.stopIf(MachineEps,10**6,BoolIter)
scalarBlocksHBF65.stopIf(MachineEps,10**6,BoolIter)

# Run DAGD and DAHBF for scalar blocks
scalarBlocks65.run()
scalarBlocksHBF65.run()

#-----------------------------------------------------------------------------
#       Scalar Blocks with 50% Agent Update Rate, 
#       Comm Rate 100 %
#-----------------------------------------------------------------------------

## Scalar Blocks
print("Running with Scalar Blocks... 50% Update Rate, 100% Comm Rate")

# Create DAGD and DAHBF classes with inputs defined above
scalarBlocks50 = DAGD(gamma, n, inputs, comm100, .5)
scalarBlocksHBF50 = DAHBF(alpha, beta, n, inputs, commHBF100, .5)

# Optional: Set the "actual" decision variable values to compute error later
#           If not set, error will not be calculated.
scalarBlocks50.setActual(xActual)
scalarBlocksHBF50.setActual(xActual,xActual)

# Optional: Set the initial decision variable values
#           If not set, zero vectors will be used.
scalarBlocks50.setInit(InitCond)
scalarBlocksHBF50.setInit(InitCond,InitCond)

# Optional: Set stopping parameters, stopIf(tol, maxIter, maxIterBool=1), where
#               tol = tolerance for distance between iterations, 
#               maxIter = max number of iterations to run
#               maxIterBool = whether to stop at the maxIter (1) or continue 
#                               running until tol is reached (0). 
#           If not set, tol = 10**-8, maxIter = 10 ** 5, maxIterBool=1.
scalarBlocks50.stopIf(MachineEps,10**6,BoolIter)
scalarBlocksHBF50.stopIf(MachineEps,10**6,BoolIter)

# Run DAGD and DAHBF for scalar blocks
scalarBlocks50.run()
scalarBlocksHBF50.run()

#-----------------------------------------------------------------------------
#       Scalar Blocks with 25% Agent Update Rate, 
#       Comm Rate 100 %
#-----------------------------------------------------------------------------

## Scalar Blocks
print("Running with Scalar Blocks... 25% Update Rate, 100% Comm Rate")

# Create DAGD and DAHBF classes with inputs defined above
scalarBlocks25 = DAGD(gamma, n, inputs, comm100, .25)
scalarBlocksHBF25 = DAHBF(alpha, beta, n, inputs, commHBF100, .25)

# Optional: Set the "actual" decision variable values to compute error later
#           If not set, error will not be calculated.
scalarBlocks25.setActual(xActual)
scalarBlocksHBF25.setActual(xActual,xActual)

# Optional: Set the initial decision variable values
#           If not set, zero vectors will be used.
scalarBlocks25.setInit(InitCond)
scalarBlocksHBF25.setInit(InitCond,InitCond)

# Optional: Set stopping parameters, stopIf(tol, maxIter, maxIterBool=1), where
#               tol = tolerance for distance between iterations, 
#               maxIter = max number of iterations to run
#               maxIterBool = whether to stop at the maxIter (1) or continue 
#                               running until tol is reached (0). 
#           If not set, tol = 10**-8, maxIter = 10 ** 5, maxIterBool=1.
scalarBlocks25.stopIf(MachineEps,10**6,BoolIter)
scalarBlocksHBF25.stopIf(MachineEps,10**6,BoolIter)

# Run DAGD and DAHBF for scalar blocks
scalarBlocks25.run()
scalarBlocksHBF25.run()

#----------------------------------------
## Figure Plotting

## GD Convergence Between Iterations
plt.semilogy(np.arange(0,scalarBlocks100.numIter+1), scalarBlocks100.iterNorm[0:], color= dark_blue, label="100% Comp Rate")
plt.semilogy(np.arange(0,scalarBlocks75.numIter+1), scalarBlocks75.iterNorm[0:], color= red, linestyle= "dotted", label="75% Comp Rate")
plt.semilogy(np.arange(0,scalarBlocks65.numIter+1), scalarBlocks65.iterNorm[0:], color= dark_green, linestyle= "dashdot", label="65% Comp Rate")
plt.semilogy(np.arange(0,scalarBlocks50.numIter+1), scalarBlocks50.iterNorm[0:], color= dark_purple, linestyle= "dashed", label="50% Comp Rate")
plt.ylabel("$|| z_1(k) - z_1(k-1)||$")
plt.xlabel("Time, k")
plt.ylim(10 ** -8, 10 ** 3)
plt.xlim(-5,40)
plt.title("Convergence Between Iterations, GD")
plt.legend()
plt.savefig('ConvergenceGD.eps')
plt.show()

## HBF Convergence Between Iterations
plt.semilogy(np.arange(0,scalarBlocksHBF100.numIter+1), scalarBlocksHBF100.iterNorm[0:], color= dark_blue, label="100% Comp Rate")
plt.semilogy(np.arange(0,scalarBlocksHBF75.numIter+1), scalarBlocksHBF75.iterNorm[0:], color= red, linestyle= "dotted", label="75% Comp Rate")
plt.semilogy(np.arange(0,scalarBlocksHBF65.numIter+1), scalarBlocksHBF65.iterNorm[0:], color= dark_green, linestyle= "dashdot", label="65% Comp Rate")
plt.semilogy(np.arange(0,scalarBlocksHBF50.numIter+1), scalarBlocksHBF50.iterNorm[0:], color= dark_purple, linestyle= "dashed", label="50% Comp Rate")
plt.ylabel("$|| z_1(k) - z_1(k-1)||$")
plt.xlabel("Time, k")
plt.ylim(10 ** -8, 10 ** 3)
plt.xlim(-5,40)
plt.title("Convergence Between Iterations, HBF")
plt.legend()
plt.savefig('ConvergenceHBF.eps')
plt.show()


## GD Error
plt.semilogy(np.arange(0,scalarBlocks100.numIter+1), scalarBlocks100.xError[0:], color= dark_blue, label="100% Comp Rate")
plt.semilogy(np.arange(0,scalarBlocks75.numIter+1), scalarBlocks75.xError[0:], color= red, linestyle= "dotted", label="75% Comp Rate")
plt.semilogy(np.arange(0,scalarBlocks65.numIter+1), scalarBlocks65.xError[0:], dark_green, linestyle= "dashdot", label="65% Comp Rate")
plt.semilogy(np.arange(0,scalarBlocks50.numIter+1), scalarBlocks50.xError[0:], color= dark_purple, linestyle= "dashed", label="50% Comp Rate")
plt.ylabel('$|| z_1 - x^*||$')
plt.xlabel("Time, k")
plt.ylim(10 ** -8, 10 ** 2)
plt.xlim(-5,40)
plt.title("Error, GD")
plt.legend()
plt.savefig('ErrorGD.eps')
plt.show()

## HBF Error
plt.semilogy(np.arange(0,scalarBlocksHBF100.numIter+1), scalarBlocksHBF100.x1Error[0:], color= dark_blue, label="100% Comp Rate")
plt.semilogy(np.arange(0,scalarBlocksHBF75.numIter+1), scalarBlocksHBF75.x1Error[0:], color= red, linestyle= "dotted", label="75% Comp Rate")
plt.semilogy(np.arange(0,scalarBlocksHBF65.numIter+1), scalarBlocksHBF65.x1Error[0:], color= dark_green, linestyle= "dashdot", label="65% Comp Rate")
plt.semilogy(np.arange(0,scalarBlocksHBF50.numIter+1), scalarBlocksHBF50.x1Error[0:], color= dark_purple, linestyle= "dashed", label="50% Comp Rate")
plt.ylabel('$|| z_1 - x^*||$')
plt.xlabel("Time, k")
plt.ylim(10 ** -8, 10 ** 2)
plt.xlim(-5,40)
plt.title("Error, HBF")
plt.legend()
plt.savefig('ErrorHBF.eps')
plt.show()