#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from gprn import covFunction

def newCov(nodes, weight, newPars, q):
    """
    To fix the update of the nodes and weight until something smarter and
    better comes to mind
    
    Parameters
    ----------
    nodes: array
        Old nodes
    weight: array
        Old weights
    newPars: array
        New hyperparameters values
    
    Returns
    -------
    nodes: array
        Updated nodes
    weight: array
        Updated weights
    """
    parsUsed = 0
    for qq in range(q):
            howBig = nodes[qq].params_size - 1
            parsToUse = newPars[parsUsed:parsUsed+howBig]
            nodes[qq] = checkKernel(nodes[qq], parsToUse, node=True)
            parsUsed += howBig
    howBig = weight[0].params_size - 1 +1
    parsToUse = newPars[parsUsed:parsUsed+howBig]
    weight[0] = checkKernel(weight[0], parsToUse, node=False)
    return nodes, weight

def checkKernel(originalCov,newPars, node=True):
    """
    new_kernel() updates the parameters of the covFuntions as the 
    optimizations advances
    
    Parameters
    ----------
    originalCov: kernel
        Original kernel in use
    newPars: array
        New hyperparameters if you prefer using that denomination
    node: bool
        True if its a node False if its a weight
        
    Returns
    -------
    kernel
        Kernel with updated hyperparameters
    """
    #nodes have a amplitude of 1
    if node:
        if isinstance(originalCov, covFunction.WhiteNoise):
            return covFunction.WhiteNoise(newPars[0])
        elif isinstance(originalCov, covFunction.SquaredExponential):
            return covFunction.SquaredExponential(1, newPars[0], newPars[1])
        elif isinstance(originalCov, covFunction.Periodic):
            return covFunction.Periodic(1, newPars[0], newPars[1], newPars[2])
        elif isinstance(originalCov, covFunction.QuasiPeriodic):
            return covFunction.QuasiPeriodic(1, newPars[0], newPars[1], 
                                             newPars[2], newPars[3])
        elif isinstance(originalCov, covFunction.RationalQuadratic):
            return covFunction.RationalQuadratic(1, newPars[0], newPars[1], 
                                                 newPars[2])
        elif isinstance(originalCov, covFunction.RQP):
            return covFunction.RQP(1, newPars[0], newPars[1], newPars[2], 
                                   newPars[3], newPars[4])
        elif isinstance(originalCov, covFunction.Cosine):
            return covFunction.Cosine(1, newPars[0], newPars[1])
        elif isinstance(originalCov, covFunction.Laplacian):
            return covFunction.Laplacian(1, newPars[0], newPars[1])
        elif isinstance(originalCov, covFunction.Exponential):
            return covFunction.Exponential(1, newPars[0], newPars[1])
        elif isinstance(originalCov, covFunction.Matern32):
            return covFunction.Matern32(1, newPars[0], newPars[1])
        elif isinstance(originalCov, covFunction.Matern52):
            return covFunction.Matern52(1, newPars[0], newPars[1])
        elif isinstance(originalCov, covFunction.Linear):
            return covFunction.Linear(1, newPars[0], newPars[1])
        elif isinstance(originalCov, covFunction.GammaExp):
            return covFunction.GammaExp(1, newPars[0], newPars[1], newPars[2])
        elif isinstance(originalCov, covFunction.Polynomial):
            return covFunction.Polynomial(1, newPars[0], newPars[1], newPars[2])
        else:
            print('Something went wrong while updating the covariance functions!')
            return 0
    #weights don't have a white noise term
    if isinstance(originalCov, covFunction.WhiteNoise):
        return covFunction.WhiteNoise(newPars[0])
    elif isinstance(originalCov, covFunction.SquaredExponential):
        return covFunction.SquaredExponential(newPars[0], newPars[1], newPars[2])
    elif isinstance(originalCov, covFunction.Periodic):
        return covFunction.Periodic(newPars[0], newPars[1], newPars[2], 0)
    elif isinstance(originalCov, covFunction.QuasiPeriodic):
        return covFunction.QuasiPeriodic(newPars[0], newPars[1], newPars[2], 
                                         newPars[3], 0)
    elif isinstance(originalCov, covFunction.RationalQuadratic):
        return covFunction.RationalQuadratic(newPars[0], newPars[1], 
                                             newPars[2], 0)
    elif isinstance(originalCov, covFunction.RQP):
        return covFunction.RQP(newPars[0], newPars[1], newPars[2], 
                               newPars[3], newPars[4], 0)
    elif isinstance(originalCov, covFunction.Cosine):
        return covFunction.Cosine(newPars[0], newPars[1], 0)
    elif isinstance(originalCov, covFunction.Laplacian):
        return covFunction.Laplacian(newPars[0], newPars[1], 0)
    elif isinstance(originalCov, covFunction.Exponential):
        return covFunction.Exponential(newPars[0], newPars[1], 0)
    elif isinstance(originalCov, covFunction.Matern32):
        return covFunction.Matern32(newPars[0], newPars[1], 0)
    elif isinstance(originalCov, covFunction.Matern52):
        return covFunction.Matern52(newPars[0], newPars[1], 0)
    elif isinstance(originalCov, covFunction.Linear):
        return covFunction.Linear(newPars[0], newPars[1], 0)
    elif isinstance(originalCov, covFunction.GammaExp):
        return covFunction.GammaExp(newPars[0], newPars[1], newPars[2], 0)
    elif isinstance(originalCov, covFunction.Polynomial):
        return covFunction.Polynomial(newPars[0], newPars[1], newPars[2], 0)
    else:
        print('Something went wrong while updating the covariance functions!')
        return 0