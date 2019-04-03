#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import inv, cho_factor, cho_solve, LinAlgError
from scipy.stats import multivariate_normal
from copy import copy

from gprn.nodeFunction import Linear as nodeL
from gprn.nodeFunction import Polynomial as nodeP
from gprn.weightFunction import Linear as weightL
from gprn.weightFunction import Polynomial as weightP

from gprn import nodeFunction, weightFunction


class MFI(object):
    """ 
        Class to perform Mean-Field Inference for GPRNs. 
        See Nguyen & Bonilla (2013) for more information.
        Parameters:
            nodes = latent noide functions f(x), called f hat in the article
            weight = latent weight funtion w(x)
            means = array of means functions being used, set it to None if a 
                    model doesn't use it
            jitters = jitter value of each dataset
            time = time
            *args = the data (or components), it needs be given in order of
                data1, data1_error, data2, data2_error, etc...
    """ 
    def  __init__(self, nodes, weight, means, jitters, time, *args):
        #node functions; f(x) in Wilson et al. (2012)
        self.nodes = np.array(nodes)
        #weight function; w(x) in Wilson et al. (2012)
        self.weight = weight
        #mean functions
        self.means = np.array(means)
        #jitters
        self.jitters = np.array(jitters)
        #time
        self.time = time 
        #the data, it should be given as data1, data1_error, data2, ...
        self.args = args 
        
        #number of nodes being used; q in Wilson et al. (2012)
        self.q = len(self.nodes)
        #number of outputs y(x); p in Wilson et al. (2012)
        self.p = int(len(self.args)/2)
        #total number of weights, we will have q*p weights in total
        self.qp =  self.q * self.p
        #number of observations, N in Wilson et al. (2012)
        self.N = self.time.size
        
        #to organize the data we now join everything
        self.tt = np.tile(time, self.p) #"extended" time because why not?
        ys = [] 
        yerrs = []
        for i,j  in enumerate(args):
            if i%2 == 0:
                ys.append(j)
            else:
                yerrs.append(j)
        self.y = np.array(ys).reshape(self.p, self.N) #matrix p*N of outputs
        self.yerr = np.array(yerrs).reshape(self.p, self.N) #matrix p*N of errors

        #check if the input was correct
        assert self.means.size == self.p, \
        'The numbers of means should be equal to the number of components'
        assert (i+1)/2 == self.p, \
        'Given data and number of components dont match'

##### mean functions definition (for now its better to make them as None)
    @property
    def mean_pars_size(self):
        return self._mean_pars_size

    @mean_pars_size.getter
    def mean_pars_size(self):
        self._mean_pars_size = 0
        for m in self.means:
            if m is None: self._mean_pars_size += 0
            else: self._mean_pars_size += m._parsize
        return self._mean_pars_size

    @property
    def mean_pars(self):
        return self._mean_pars

    @mean_pars.setter
    def mean_pars(self, pars):
        pars = list(pars)
        assert len(pars) == self.mean_pars_size
        self._mean_pars = copy(pars)
        for i, m in enumerate(self.means):
            if m is None: 
                continue
            j = 0
            for j in range(m._parsize):
                m.pars[j] = pars.pop(0)

    def _mean(self, means, time=None):
        """
            Returns the values of the mean functions
        """
        if time is None:
            N = self.time.size
            m = np.zeros_like(self.tt)
            for i, meanfun in enumerate(means):
                if meanfun is None:
                    continue
                else:
                    m[i*N : (i+1)*N] = meanfun(self.time)
        else:
            N = time.size
            ttt = np.tile(time, self.p)
            m = np.zeros_like(ttt)
            for i, meanfun in enumerate(means):
                if meanfun is None:
                    continue
                else:
                    m[i*N : (i+1)*N] = meanfun(time)
        return m

    def _kernel_matrix(self, kernel, time = None):
        """
            Returns the covariance matrix created by evaluating a given kernel 
        at inputs time.
        """
        #if time is None we use the initial time of the class MFI
        r = time[:, None] - time[None, :] if time else self.time[:, None] - self.time[None, :]
        
        #to deal with the non-stationary kernels problem
        if isinstance(kernel, (nodeL, nodeP, weightL, weightP)):
            K = kernel(None, time[:, None], time[None, :])
        else:
            K = kernel(r)
        return K

    def _kernel_pars(self, kernel):
        """
            Returns the hyperparameters of a given kernel
        """
        return kernel.pars

    def _weights_matrix(self, weight):
        """
            Returns a q*p matrix of weights
        """
        weights = [] #we will need a q*p matrix of weights
        for i in range(self.qp):
            weights.append(weight)
        weight_matrix = np.array(weights).reshape((self.p, self.q))
        return weight_matrix

    def _CB_matrix(self, nodes, weight, time):
        """
            Creates the matrix CB (eq. 5 from Wilson et al. 2012), that will be 
        an N*q*(p+1) X N*q*(p+1) block diagonal matrix
            Parameters:
                nodes = array of node functions 
                weight = weight function
                time = array containing the time
            Returns:
                CB = matrix CB
        """
        CB_size = self.N * self.q * (self.p + 1)
        CB = np.zeros((N*q*(p+1), N*q*(p+1))) #initial empty matrix
        
        position = 0 #we start filling CB at position (0,0)
        #first we enter the nodes
        for i in range(self.q):
            node_CovMatrix = self._kernel_matrix(nodes[i], time)
            CB[position:position+N, position:position+N] = node_CovMatrix
            position += N
        weight_CovMatrix = self._kernel_matrix(weight, time)
        #then we enter the weights
        for i in range(self.qp):
            CB[position:position+N, position:position+N] = weight_CovMatrix
            position += N
        return CB

    def _sample_CB(self, nodes, weight, time):
        """ 
            Returns samples from the matrix CB
            Parameters:
                nodes = array of node functions 
                weight = weight function
                time = array containing the time
            Returns:
                Samples of CB
        """

        mean = np.zeros(self.N*self.q*(self.p+1))
        cov = self.CB(nodes, weight, time)
        norm = multivariate_normal(mean, cov, allow_singular=True)
        return norm.rvs()

    def _update_nodes(self, *params):
        

##### Entropy
    def entropy(self, nodes, weight):
        Q = self.q #number of nodes
        p = self.p # number of outputs
        weights_matrix = self._weights_matrix(weight) #matrix of weights
        
        ent_sum = 0 #starts at zero then we sum everything
        for i in range(Q):
            L1 = cho_factor(self._kernel_matrix(nodes[i]), 
                            overwrite_a=True, lower=False)
            ent_sum += np.sum(np.log(np.diag(L1[0])))
            for j in range(p):
                L2 = cho_factor(self._kernel_matrix(weights_matrix[j, i]), 
                            overwrite_a=True, lower=False)
                ent_sum += np.sum(np.log(np.diag(L2[0])))
        return ent_sum

##### Expected log prior
    def expectedLogPrior(self, nodes, weight, mu_f, sigma_f, mu_w, sigma_w):
        Q = self.q #number of nodes
        p = self.p # number of outputs
        N = self.time.size

        #calculation of the first term of eq.10 of Nguyen & Bonilla (2013)
        first_term = 0
        for i in range(Q):
            first_term += 1
        
        return first_term

##### Expected log-likelihood
    def expectedLogLike(self, *params):
        """
            Calculates the expected log-likelihood, eq.14 in Nguyen & Bonilla (2013)
        """
        N = self.time.size #datasets size (y(x)s size)
        P = self.p #number of y(x)
        
        #-0.5 * N * P * log(2*pi / jitter**2)
        jitters = self.jitters
        first_term = []
        for i,j in enumerate(jitters):
            first_term.append(-0.5 * N * P * np.log(2*np.pi * j**2))
        Y = np.array_split(self.y, self.p) #P-dimensional vector 