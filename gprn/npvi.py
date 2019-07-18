#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
from scipy.linalg import inv, cholesky, cho_factor, cho_solve, LinAlgError, norm
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from copy import copy

from gprn.covFunction import Linear as covL
from gprn.covFunction import Polynomial as covP
from gprn.covFunction import WhiteNoise as covWN

class inference(object):
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
            k = mixture of k isotropic gaussian distributions
            *args = the data (or components), it needs be given in order of
                data1, data1_error, data2, data2_error, etc...
    """ 
    def  __init__(self, nodes, weight, means, jitters, time, k, *args):
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
        #mixture of k isotropic gaussian distributions
        self.k  = k
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


##### mean functions definition ################################################
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
        for _, m in enumerate(self.means):
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
            tt = np.tile(time, self.p)
            m = np.zeros_like(tt)
            for i, meanfun in enumerate(means):
                if meanfun is None:
                    continue
                else:
                    m[i*N : (i+1)*N] = meanfun(time)
        return m


##### To create matrices and samples ###########################################
    def _kernelMatrix(self, kernel, time = None):
        """
            Returns the covariance matrix created by evaluating a given kernel 
        at inputs time.
        """
        r = time[:, None] - time[None, :]
        
        #to deal with the non-stationary kernels problem
        if isinstance(kernel, (covL, covP)):
            K = kernel(None, time[:, None], time[None, :])
        else:
            K = kernel(r) + 1e-5*np.diag(np.diag(np.ones_like(r)))
        return K

    def _predictKernelMatrix(self, kernel, time):
        """
            To be used in predict_gp()
        """
        if isinstance(kernel, (covL, covP)):
            K = kernel(None, time, self.time[None, :])
        if isinstance(kernel, covWN):
            K = 0*np.ones_like(self.time) 
        else:
            if time.size == 1:
                r = time - self.time[None, :]
            else:
                r = time[:,None] - self.time[None,:]
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
        for _ in range(self.qp):
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
        CB_size = time.size * self.q * (self.p + 1)
        CB = np.zeros((CB_size, CB_size)) #initial empty matrix
        
        pos = 0 #we start filling CB at position (0,0)
        #first we enter the nodes
        for i in range(self.q):
            node_CovMatrix = self._kernelMatrix(nodes[i], time)
            CB[pos:pos+time.size, pos:pos+time.size] = node_CovMatrix
            pos += time.size
        weight_CovMatrix = self._kernelMatrix(weight, time)
        #then we enter the weights
        for i in range(self.qp):
            CB[pos:pos+time.size, pos:pos+time.size] = weight_CovMatrix
            pos += time.size
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
        mean = np.zeros(time.size*self.q*(self.p+1))
        cov = self._CB_matrix(nodes, weight, time)
        normal = multivariate_normal(mean, cov, allow_singular=True)
        return normal.rvs()

    def _fhat_and_w(self, u):
        """
            Given a list, divides it in the corresponding nodes (f hat) and
        weights (w) parts.
            Parameters:
                u = array
            Returns:
                f = array with the samples of the nodes
                w = array with the samples of the weights
        """
        f = u[:self.q * self.N].reshape((1, self.q, self.N))
        w = u[self.q * self.N:].reshape((self.p, self.q, self.N))
        return f, w

    def u_to_fhatW(self, nodes, weight, time):
        """
            Returns the samples of CB that corresponds to the nodes f hat and
        weights w.
            Parameters:
                nodes = array of node functions 
                weight = weight function
                time = array containing the time
            Returns:
                fhat = array with the samples of the nodes
                w = array with the samples of the weights
        """
        u = self._sample_CB(nodes, weight, time)
        fhat = u[:self.q * time.size].reshape((1, self.q, time.size))
        w = u[self.q * time.size:].reshape((self.p, self.q, time.size))
        return fhat, w

    def get_y(self, n, w, time, means = None):
        # obscure way to do it
        y = np.einsum('ij...,jk...->ik...', w, n).reshape(self.p, time.size)
        y = (y + self._mean(means, time)) if means else time
        return y

    def _cholNugget(self, matrix, maximum=10):
        """
            Returns the cholesky decomposition to a given matrix, if this matrix
        is not positive definite, a nugget is added to its diagonal.
            Parameters:
                matrix = matrix to decompose
                maximum = number of times a nugget is added.
            Returns:
                L = matrix containing the Cholesky factor
                nugget = nugget added to the diagonal
        """
        nugget = 0 #our nugget starts as zero
        try:
            nugget += np.abs(np.diag(matrix).mean()) * 1e-5
            L = cholesky(matrix).T
            return L, nugget
        except LinAlgError:
            print('NUGGETS ADDED TO DIAGONAL!')
            n = 0 #number of tries
            while n < maximum:
                print ('n:', n+1, ', nugget:', nugget)
                try:
                    L = cholesky(matrix + nugget*np.identity(matrix.shape[0])).T
                    return L, nugget
                except LinAlgError:
                    nugget *= 10.0
                finally:
                    n += 1
            raise LinAlgError("Still not positive definite, even with nugget.")


##### Nonparametric Variational Inference functions ############################
    def _updadeMean(self, nodes, weight, means, jitters, muF, muW):
        mu = np.hstack((muF.flatten(), muW.flatten()))
        print(mu[0:5])
        res = minimize(self._ELBO_updadeMean, x0 = mu, 
                       args = (nodes, weight, means, jitters), method='COBYLA', 
                       options={'disp': True, 'maxiter': 20})
        mu  = res.x
        
        muF = mu[0 : self.k*self.q*self.N].reshape(self.k, 1, self.q, self.N)
        muW = mu[self.k*self.q*self.N :].reshape(self.k, self.p, self.q, self.N)
        sigma = []
        for i in range(self.k):
            sigma = np.append(sigma, np.var(np.hstack((muF[i,:,:,:].flatten(), muW[i,:,:,:].flatten()))))
        return muF, muW , sigma
        
        
    def _ELBO_updadeMean(self, mu, nodes, weight, means, jitters):
        muF = mu[0 : self.k*self.q*self.N].reshape(self.k, 1, self.q, self.N)
        muW = mu[self.k*self.q*self.N :].reshape(self.k, self.p, self.q, self.N)
        sigma = []
        for i in range(self.k):
            sigma = np.append(sigma, np.var(np.hstack((muF[i,:,:,:].flatten(), muW[i,:,:,:].flatten()))))
        ExpLogJoint = self._expectedLogJoint(nodes, weight, means, jitters, 
                                             muF, muW, sigma)
        Entropy = self._entropy(muF, muW, sigma)
        return -(ExpLogJoint + Entropy)
        
        
    def _expectedLogJoint(self, nodes, weights, means, jitters, muF, muW, sigma):
        """
            Calculates the expection of the log prior wrt q(f,w) in nonparametric 
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
        new_y = np.concatenate(self.y) - self._mean(means, self.time)
        new_y = np.array(np.array_split(new_y, self.p)) #Px1 dimensional vector
        
        Kf = np.array([self._kernelMatrix(i, self.time) for i in nodes])
        Kf_inv = np.array([inv(i) for i in Kf ])
        logKf = np.array([np.sum(np.log(np.diag(self._cholNugget(i)[0]))) \
                          for i in Kf])
        Kw = np.array([self._kernelMatrix(j, self.time) for j in weights]) 
        Kw_inv = np.array([inv(j) for j in Kw ])
        logKw = np.array([np.sum(np.log(np.diag(self._cholNugget(j)[0]))) \
                          for j in Kw])
        
        sigma_y = 0
        for i in range(self.p):
            sigma_y += jitters[i]**2 + (np.sum(self.yerr[i,:])/self.N)**2
            
        first_term = 0
        for ki in range(self.k):
            for j in range(self.q):
                first_term += np.float(logKf[j])
                first_term += muF[ki,:,j,:] @(Kf_inv[j] + \
                                 self.p *sigma[ki]**2 *np.identity(self.N) /sigma_y) @muF[ki,:,j,:].T
                first_term += sigma[ki]**2 * np.trace(Kf_inv[j])
        first_term = -0.5 * np.float(first_term) / self.k
        
        second_term = 0
        for ki in range(self.k):
            for i in range(self.p):
                for j in range(self.q):
                    second_term += np.float(logKw)
                    second_term += muW[ki, i, j, :] @(np.squeeze(Kw_inv) + \
                                 sigma[ki]**2 * np.identity(self.N) / sigma_y) @muW[ki, i, j, :].T
                    #print(second_term)
                    second_term += sigma[ki]**2 * np.trace(np.squeeze(Kw_inv))
                    #print(second_term, '\n')
        second_term = -0.5 * np.float(second_term) / self.k
        
        third_term = 0
        for ki in range(self.k):
            for i in range(self.p):
                for n in range(self.N):
                    YOmegaMu = np.array(new_y[i,n].T - muW[ki,i,:,n] @ muF[ki,:,:,n].T)
                    third_term += np.dot(YOmegaMu.T, YOmegaMu)
        third_term = -0.5 * np.float(third_term) / (self.k*sigma_y)
        
        fourth_term = 0
        log_sigma_y = 0
        for i in range(self.p):
            log_sigma_y += np.log(jitters[i]**2 + (np.sum(self.yerr[i,:])/self.N)**2)
        for ki in range(self.k):
            fourth_term += sigma[ki]**4 * self.q /sigma_y + self.k*log_sigma_y
        fourth_term = -0.5 * np.float(fourth_term) / self.k
        #print(first_term, second_term, third_term, fourth_term)
        return first_term + second_term + third_term + fourth_term
        
        
    def _entropy(self, muF, muW, sigma):
        Sig_nj =[]
        for ki in range(self.k):
            Sig_nj.append([])
        Sig_nj = np.array(Sig_nj)
        
        for j in range(self.q):
            Sig_nj = np.hstack((Sig_nj, np.squeeze(muF[:, :, j, :])))
            #Sig_nj = np.hstack((Sig_nj, muF[:, :, j, :].reshape(self.k, self.N)))
        for i in range(self.p):
            for j in range(self.q):
                Sig_nj = np.hstack((Sig_nj, muW[:, i, j, :].reshape(self.k, self.N)))
        
        sig_nj = np.diag((sigma[0]**2 + sigma[0]**2) * np.identity(Sig_nj.shape[0]))
        logP = []
        for ki in range(self.k):
            logP.append(-0.5 * np.divide(Sig_nj[ki,:], sig_nj[ki]) \
                        -0.5 * self.p * np.log(sig_nj[ki]))
        logP = np.array(logP)
        a = np.zeros((1, self.k))
        
        for ki in range(self.k):
            max_val = max(logP[ki,:])
            ls = max_val + np.log(np.sum(np.exp(logP[:, ki] - max_val)))
            a[0,ki] = -np.log(self.k) + ls
            
        beta = np.ones((self.k,1)) / self.k
        Entropy_result = np.float(a @ beta)
        return Entropy_result
    
    
    def EvidenceLowerBound(self, nodes, weight, means, jitters, time, 
                           iterations = 100, prints = False, plots = False):
        """
            Returns the Evidence Lower bound, eq.10 in Nguyen & Bonilla (2013)
            Parameters:
                nodes = array of node functions 
                weight = weight function
                means = array with the mean functions
                jitters = jitters array
                time = time array
                iterations = number of iterations
                prints = True to print ELB value at each iteration
                plots = True to plot ELB evolution 
            Returns:
                sum_ELB = Evidence lower bound
                muF = array with the new means for each node
                muW = array with the new means for each weight
        """ 
        #initial variational parameters
        D = self.time.size * self.q *(self.p+1)
        mu = np.random.randn(D, self.k) #muF[:, k]
        sigma, muF, muW = [], [], []
        for ki in range(self.k):
            sigma = np.append(sigma, np.var(mu[:, ki]))
            #sigma.append(1)
            meme, mumu = self._fhat_and_w(mu[:, ki])
            muF.append(meme)
            muW.append(mumu)
        muF = np.array(muF)
        muW = np.array(muW)
        sigma = np.array(sigma)
        
        iterNumber = 0
        ELB = [0]
        if plots:
            ELJ, ENT = [0], [0]
        while iterNumber < iterations:
            muF, muW, sigma = self._updadeMean(nodes, weight, means, jitters, 
                                               muF, muW)
            #Expected log-likelihood
            ExpLogJoint = self._expectedLogJoint(nodes, weight, means, jitters, 
                                               muF, muW, sigma)
            Entropy = self._entropy(muF, muW, sigma)
            sum_ELB = ExpLogJoint/self.k + Entropy
            if plots: 
                ELJ.append(ExpLogJoint/self.k)
                ENT.append(Entropy)
                ELB.append(sum_ELB)
            if prints:
                self._prints(sum_ELB, ExpLogJoint/self.k, Entropy)
            #Stoping criteria
            criteria = np.abs(np.mean(ELB[-5:]) - sum_ELB)
            if criteria < 1e-5 and criteria != 0 :
                if prints:
                    print('\nELB converged to ' +str(sum_ELB) \
                          + '; algorithm stopped at iteration ' \
                          +str(iterNumber) +'\n')
                if plots:
                    self._plots(ELB[1:], ELJ[1:-1], ENT[1:-1])
                return sum_ELB, muF, muW
            iterNumber += 1
        if plots:
            self._plots(ELB[1:], ELJ[1:-1], ENT[1:-1])
        return sum_ELB, muF, muW


    def Prediction(self, nodes, weights, means, jitters, tstar, muF, muW):
        """
            Prediction for mean-field inference
            Parameters:
                nodes = array of node functions 
                weight = weight function
                means = array with the mean functions
                jitters = jitters array
                tstar = predictions time
                muF = array with the initial means for each node
                varF = array with the initial variance for each node
                muW = array with the initial means for each weight
            Returns:
                ystar = predicted means
        """
        Kf = np.array([self._kernelMatrix(i, self.time) for i in nodes])
        invKf = np.array([inv(i) for i in Kf])
        Kw = np.array([self._kernelMatrix(j, self.time) for j in weights])
        invKw = np.array([inv(j) for j in Kw])

        #mean
        ystar = []
        for n in range(tstar.size):
            Kfstar = np.array([self._predictKernelMatrix(i1, tstar[n]) for i1 in nodes])
            Kwstar = np.array([self._predictKernelMatrix(i2, tstar[n]) for i2 in weights])
            Ksum = 0
            for ki in range(self.k):
                Efstar, Ewstar = 0, 0
                for j in range(self.q):
                    Efstar += Kfstar[j] @(invKf[j] @muF[ki,:,j,:].T) 
                    for i in range(self.p):
                        Ewstar += Kwstar[0] @(invKw[0] @muW[ki,i,j,:].T)
                Ksum += Ewstar @ Efstar
            ystar.append(Ksum / self.k)
        ystar = np.array(ystar).reshape(tstar.size) #final mean
        ystar += self._mean(means, tstar) #adding the mean function


    def _plots(self, ELB, ELJ, ENT):
        """
            Plots the evolution of the evidence lower bound, expected log joint, 
        and entropy
        """
        plt.figure()
        ax1 = plt.subplot(311)
        plt.plot(ELB, '-')
        plt.ylabel('Evidence lower bound')
        plt.subplot(312, sharex=ax1)
        plt.plot(ELJ, '-')
        plt.ylabel('Expected log joint')
        plt.subplot(313, sharex=ax1)
        plt.plot(ENT, '-')
        plt.ylabel('Entropy')
        plt.xlabel('iteration')
        plt.show()
        return 0


    def _prints(self, sum_ELB, ExpLogJoint, Entropy):
        """
            Prints the evidence lower bound, expected log joint, and entropy
        """
        print('ELB: ' + str(sum_ELB))
        print(' logjoint: ' + str(ExpLogJoint) + \
              ' \n entropy: ' + str(Entropy) + ' \n')
        return 0
