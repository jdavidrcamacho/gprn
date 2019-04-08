#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import inv, cholesky, cho_factor, cho_solve, LinAlgError, lapack
from scipy.stats import multivariate_normal
from copy import copy

from gprn.nodeFunction import Linear as nodeL
from gprn.nodeFunction import Polynomial as nodeP
from gprn.weightFunction import Linear as weightL
from gprn.weightFunction import Polynomial as weightP

from gprn import nodeFunction, weightFunction


class GPRN_inference(object):
    """ 
        Class to perform variational inference for GPRNs. 
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
        #r = time[:, None] - time[None, :] if time!=None else self.time[:, None] - self.time[None, :]
        r = self.time[:, None] - self.time[None, :]
        
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
        CB = np.zeros((CB_size, CB_size)) #initial empty matrix
        
        position = 0 #we start filling CB at position (0,0)
        #first we enter the nodes
        for i in range(self.q):
            node_CovMatrix = self._kernel_matrix(nodes[i], time)
            CB[position:position+self.N, position:position+self.N] = node_CovMatrix
            position += self.N
        weight_CovMatrix = self._kernel_matrix(weight, time)
        #then we enter the weights
        for i in range(self.qp):
            CB[position:position+self.N, position:position+self.N] = weight_CovMatrix
            position += self.N
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
        cov = self._CB_matrix(nodes, weight, time)
        norm = multivariate_normal(mean, cov, allow_singular=True)
        return norm.rvs()

    def _u_to_fhatw(self, nodes, weight, time):
        """
            Returns the samples of CB that corresponds to the nodes f and
        weights W.
            Parameters:
                nodes = array of node functions 
                weight = weight function
                time = array containing the time
            Returns:
                f = array with the samples of the nodes
                W = array with the samples of the weights
        """
        u = self._sample_CB(nodes, weight, time)
        
        f = u[:self.q * self.N].reshape((self.q, 1, self.N))
        W = u[self.q * self.N:].reshape((self.p, self.q, self.N))
        return f, W

    def _cholPLUSnugget(self, matrix, maximum=1e10):
        """
            NOT USED SO FAR
        """
        try:
            L =  cho_factor(matrix, overwrite_a=True, lower=False)
        except LinAlgError:
            nugget = np.diag(matrix).mean() * 1e-5 #nugget to add to the diagonal
            n = 1 #number of tries
            while n <= maximum:
                print ('n= ',n, 'nugget= ', nugget)
                try:
                    L =  cho_factor(matrix + nugget, overwrite_a=True, lower=False)
                except LinAlgError:
                    nugget *= 10
                finally:
                    n += 1
            raise LinAlgError("Still not positive definite, even with nugget.")
        return L

    def _update_SIGMAandMU(self, nodes, weight, jitters, time,
                           muF, varF, muW , varW):
        """
            Efficient closed-form updates fot variational parameters. This
        corresponds to eqs. 16, 17, 18, and 19 of Nguyen & Bonilla (2013) 
            Parameters:
                nodes = array of node functions 
                weight = weight function
                jitters = jitters array
                time = array containing the time
                muF = array with the initial means for each node
                varF = array with the initial variance for each node
                muW = array with the initial means for each weight
                varW = array with the initial variance for each weight
            Returns:
                sigma_f = array with the covariance for each node
                mu_f = array with the means for each node
                sigma_w = array with the covariance for each weight
                mu_w = array with the means for each weight
        """

        Kf = np.array([self._kernel_matrix(i, time) for i in nodes])
        invKf = []
        for i in range(self.q):
            invKf.append(inv(Kf[i]))
        invKf = np.array(invKf)
        Kw = self._kernel_matrix(weight, time) #this means equal weights for all nodes
        invKw = inv(Kw)
        
        #we have Q nodes -> j in the paper; we have P y(x)s -> i in the paper
        
        #creation of Sigma_fj
        sigma_f = []
        for j in range(self.q):
            sum_muWmuWVarW = np.zeros((self.N, self.N))
            for i in range(self.p):
                sum_muWmuWVarW += np.diag(muW[i][j][:] * muW[i][j][:] + varW[i][j][:])
            sum_muWmuWVarW = sum_muWmuWVarW / jitters[j]**2
            sigma_f.append(inv(invKf[j] + sum_muWmuWVarW))
        sigma_f = np.array(sigma_f)
        #creation of mu_fj
        mu_f = []
        for j in range(self.q):
            sum_YminusSum = np.zeros(self.N)
            for i in range(self.p):
                sum_muWmuF = np.zeros(self.N)
                for k in range(self.q):
                    if k != i:
                        sum_muWmuF += np.array(muW[i][k]) * muF[k].reshape(self.N)
                sum_YminusSum += self.y[i] - sum_muWmuF
                sum_YminusSum = sum_YminusSum * muW[i][j]
            mu_f.append(np.dot(sigma_f[j], sum_YminusSum) / jitters[j]**2)
        mu_f = np.array(mu_f)
        #creation of Sigma_wij
        sigma_w = []
        for j in range(self.q):
            sum_muFmuFVarF = np.diag( muF[j] * muF[j] + varF[j])
            sum_muFmuFVarF = sum_muFmuFVarF / jitters[j]**2
            for i in range(self.p):
                sigma_w.append(inv(invKw + sum_muWmuWVarW))
        sigma_w = np.array(sigma_w)
        #creation of mu_wij
        mu_w = []
        for j in range(self.q):
            sum_YminusSum = np.zeros(self.N)
            for i in range(self.p):
                sum_muFmuW = np.zeros(self.N)
                for k in range(self.q):
                    if k != i:
                        sum_muFmuW += muF[k].reshape(self.N) * np.array(muW[i][k])
                sum_YminusSum += self.y[i] - sum_muFmuW
                sum_YminusSum = sum_YminusSum * muF[j]
                mu_w.append(np.dot(sigma_f[j], sum_YminusSum.T) / jitters[j]**2)
        mu_w = np.array(mu_w).reshape(self.q * self.p, self.N)
        return sigma_f, mu_f, sigma_w, mu_w

##### Entropy
    def _mfi_entropy(self, sigma_f, sigma_w):
        """
            Calculates the entropy in mean-field inference, corresponds to 
        eq.14 in Nguyen & Bonilla (2013)
            Parameters:
                sigma_f = array with the covariance for each node
                sigma_w = array with the covariance for each weight
            Returns:
                ent_sum = final entropy
        """
        Q = self.q #number of nodes
        p = self.p #number of outputs
        
        ent_sum = 0 #starts at zero then we sum everything
        for i in range(Q):
            try:
                L1 = cho_factor(sigma_f[i], overwrite_a=True, lower=False)
            except LinAlgError:
                nugget  = np.diag(1e-5 + np.zeros(self.N)) 
                #adding a nugget to the diagonal of sigma
                L1 = cho_factor(sigma_f[i] + nugget, overwrite_a=True, lower=False)
            ent_sum += np.sum(np.log(np.diag(L1[0])))
            for j in range(p):
                try:
                    L2 = cho_factor(sigma_w[j], overwrite_a=True, lower=False)
                except LinAlgError:
                    nugget  = np.diag(1e-5 + np.zeros(self.N))
                    #adding a nugget to the diagonal of sigma
                    L2 = cho_factor(sigma_w[j] + nugget, overwrite_a=True, lower=False)
                ent_sum += np.sum(np.log(np.diag(L2[0])))
        return ent_sum

##### Expected log prior
    def _mfi_expectedLogPrior(self, nodes, weight, sigma_f, mu_f, sigma_w, mu_w):
        """
            Calculates the expection of the log prior wrt q(f,w) in mean-field 
        inference, corresponds to eq.15 in Nguyen & Bonilla (2013)
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
        Kf = np.array([self._kernel_matrix(i, self.time) for i in nodes])
        invKf = []
        for i in range(self.q):
            invKf.append(inv(Kf[i]))
        invKf = np.array(invKf) 
        Kw = self._kernel_matrix(weight, self.time) #this means equal weights for all nodes
        invKw = inv(Kw)
        
        #we have Q nodes -> j in the paper; we have P y(x)s -> i in the paper
        
        #calculation of the first term of eq.15 of Nguyen & Bonilla (2013)
        first_term = 0
        for j in range(self.q):
            L1 = cho_factor(Kf[j], overwrite_a=True, lower=False)
            logKf = 2 * np.sum(np.log(np.diag(L1[0])))
            muKmu = np.dot(np.dot(mu_f[j].reshape(self.N).T, invKf[j]), 
                           mu_f[j].reshape(self.N))
            trace = np.trace(invKf[j] * sigma_f[j])
            first_term += logKf + muKmu + trace
        first_term = -0.5 * first_term
        
        #calculation of the second term of eq.15 of Nguyen & Bonilla (2013)
        second_term = 0
        L2 = cho_factor(Kw, overwrite_a=True, lower=False)
        logKf = 2 * np.sum(np.log(np.diag(L2[0])))
        for j in range(self.q):
            for i in range(self.p):
                muKmu = np.dot(np.dot(mu_w[i,j,:], invKw), mu_w[i,j,:].T)
                trace = np.trace(invKw * sigma_w[j])
                second_term += logKf + muKmu + trace
        second_term = -0.5 * second_term
        return first_term + second_term

##### Expected log-likelihood
    def _mfi_expectedLogLike(self, nodes, weight, jitters, sigma_f, mu_f, sigma_w, mu_w):
        """
            Calculates the expected log-likelihood in mean-field inference, 
        corresponds to eq.14 in Nguyen & Bonilla (2013)
            Parameters:
                nodes = array of node functions 
                weight = weight function
                jitters = jitters array
                sigma_f = array with the covariance for each node
                mu_f = array with the means for each node
                sigma_w = array with the covariance for each weight
                mu_w = array with the means for each weight
            Returns:
                expected log-likelihood
        """
        #not sure about this j but I'll keep it for now
        j = np.array(jitters).mean() 
        
        #-0.5 * N * P * log(2*pi / jitter**2)
        first_term = -0.5 * self.N * self.p * np.log(2*np.pi * j**2)
            
        Y = self.y #P dimensional vector
        muw = mu_w.reshape(self.p, self.q, self.N) #PxQ dimensional vector
        muf = mu_f.reshape(self.q, self.N) #Q dimensional vector
        second_term = 0
        for i in range(self.N):
            YOmegaMu = np.array(Y[:,i] - np.dot(muw[:,:,i], muf[:,i]))
            second_term += np.dot(YOmegaMu.T, YOmegaMu)
        
        third_term = 0
        for j in range(self.q):
            diag_sigmaf = np.diag(sigma_f[j][:])
            mu_f_j = mu_f[j]
            for i in range(self.p):
                mu_w_ij = mu_w[i][:]
                diag_sigmaw = np.diag(sigma_w[i][:])
                third_term += np.dot(diag_sigmaf, mu_w_ij[j]*mu_w_ij[j]) \
                                + np.dot(diag_sigmaw, 
                                         mu_f_j.reshape(self.N)*mu_f_j.reshape(self.N))
        return first_term -0.5*second_term/j - 0.5*third_term/j

##### Evidence Lower Bound
    def MFI_EvidenceLowerBound(self, nodes, weight, jitters, time,
                               muF, varF, muW, varW):
        """
            Returns the Evidence Lower bound, eq.10 in Nguyen & Bonilla (2013)
            Parameters:
                nodes = array of node functions 
                weight = weight function
                jitters = jitters array
                time = time array
                muF = array with the initial means for each node
                varF = array with the initial variance for each node
                muW = array with the initial means for each weight
                varW = array with the initial variance for each weight
            Returns:
                sum_ELB = Evidence lower bound
                muF = array with the new mean for each node
                varF = array with the new variance for each node
                muW = array with the new mean for each weight
                varW = array with the new variance for each weight
        """ 
        #Variational parameters
        sigmaF, muF, sigmaW, muW = self._update_SIGMAandMU(nodes, weight, 
                                                           jitters, time,
                                                           muF, varF, muW, varW)
        
        muF = muF.reshape(self.q, 1, self.N) #new mean for the nodes
        varF =  []
        for i in range(self.q):
            varF.append(np.diag(sigmaF[1]))
        varF = np.array(varF).reshape(self.q, 1, self.N) #new variance for the nodes
        muW = muW.reshape(self.p, self.q, self.N) #new mean for the weights
        varW =  []
        for i in range(self.q * self.p):
            varW.append(np.diag(sigmaW[1]))
        varW = np.array(varW).reshape(self.p, self.q, self.N) #new variance for the weights
        
        #Entropy
        Entropy = self._mfi_entropy(sigmaF, sigmaW)
        #Expected log prior
        ExpLogPrior = self._mfi_expectedLogPrior(nodes, weight, 
                                            sigmaF, muF,  sigmaW, muW)
        #Expected log-likelihood
        ExpLogLike = self._mfi_expectedLogLike(nodes, weight, jitters, 
                                          sigmaF, muF, sigmaW, muW)
        #Evidence Lower Bound
        sum_ELB = ExpLogLike + ExpLogPrior + Entropy
        return sum_ELB, muF, varF, muW, varW

###### Optimization
#    def Optimization(*params):
#        