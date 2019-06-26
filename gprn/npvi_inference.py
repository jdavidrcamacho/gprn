#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
from scipy.linalg import inv, cholesky, cho_factor, cho_solve, LinAlgError
from scipy.stats import multivariate_normal
from copy import copy

from gprn.covFunction import Linear as covL
from gprn.covFunction import Polynomial as covP
from gprn.covFunction import WhiteNoise as covWN

class npvi(object):
    """ 
        Class to perform nonparametric variational inference for GPRNs. 
        See Nguyen & Bonilla (2013) for more information.
    """ 
    def  __init__(self, GPRN):
        """
            To make npvi a daughter class of GPRN
        """
        self.GPRN = GPRN

    def EvidenceLowerBound(self, nodes, weight, means, jitters, time, 
                            k = 2, iterations = 100, 
                            prints = False, plots = False):
        """
            Returns the Evidence Lower bound, eq.10 in Nguyen & Bonilla (2013)
            Parameters:
                nodes = array of node functions 
                weight = weight function
                means = array with the mean functions
                jitters = jitters array
                time = time array
                k = mixture of k isotropic Gaussian distributions
                iterations = number of iterations
                prints = True to print ELB value at each iteration
                plots = True to plot ELB evolution 
            Returns:
                sum_ELB = Evidence lower bound
                muF = array with the new means for each node
                muW = array with the new means for each weight
        """ 
        #Initial variational mean
        D = self.GPRN.time.size * self.GPRN.q *(self.GPRN.p+1)
        muF = np.random.randn(D, k) #muF[:, k]
        sigmaF = np.array([])
        for i in range(k):
            sigmaF = np.append(sigmaF, np.var(muF[:,i]))
            
        iterNumber = 0
        ELB = [0]
        if plots:
            ELJ, ENT = [0], [0]
        while iterNumber < iterations:
            muF, sigmaF = self._updateMu(k)
            

            
        return 0


    def _updateMu(self, k):
        """ 
            Update of the mean parameters and variance of the mixture components.
            This doesn't make much sense in my head but I'll keep it until I
        find a better idea to update the variational means
        """
        #variational parameters
        D = self.GPRN.time.size * self.GPRN.q *(self.GPRN.p+1)
        mu = np.random.randn(D, k) #muF[:, k]
        sigma = np.array([])
        muF = np.array([])
        muW = np.array([])
        for i in range(k):
            sigma = np.append(sigma, np.var(mu[:,i]))
            meme, mumu = self.GPRN._fhat_and_w(mu)
            muF = np.append(muF, meme)
            muW = np.append(muW, mumu)

        return mu, sigma


    def _expectedLogJoint(self, nodes, weights, mu, sigma):
        """
            Calculates the expection of the log joint wrt q(f,w) in nonparametric 
        variational inference, corresponds to eq.33 in Nguyen & Bonilla (2013)
        appendix
            Parameters:
                nodes = array of node functions 
                weight = weight function
                sigma_f = array with the covariance for each node
                mu_f = array with the means for each node
                sigma_w = array with the covariance for each weight
                mu_w = array with the means for each weight
            Returns:
                expected log prior
        """
        Kf = np.array([self.GPRN._kernelMatrix(i, self.GPRN.time) for i in nodes])
        Kw = np.array([self.GPRN._kernelMatrix(j, self.GPRN.time) for j in weights]) 

        #we have q nodes -> j in the paper, p output -> i in the paper, 
        #and k distributions -> k in the paper
        first_term = 0
#        for 
        
        
        
        
    def _npvi_expectedLogLike(self, nodes, weight, means, jitters, muF):
        return 0



