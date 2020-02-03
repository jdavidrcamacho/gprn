#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt

from scipy.linalg import cholesky, LinAlgError
from scipy.optimize import minimize

from gprn.covFunction import Linear as covL
from gprn.covFunction import Polynomial as covP
from gprn.covUpdate import newCov
from gprn.utils import run_mcmc


class inference(object):
    """ 
    Class to perform mean field variational inference for GPRNs. 
    See Nguyen & Bonilla (2013) for more information.
    
    Parameters
    ----------
    num_nodes: int
        Number of latent node functions f(x), called f hat in the article
    time: array
        Time coordinates
    *args: arrays
        The actual data (or components), it needs be given in order of data1, 
        data1error, data2, data2error, etc...
    """ 
    def  __init__(self, num_nodes, time, *args):
        #number of node functions; f(x) in Wilson et al. (2012)
        self.num_nodes = num_nodes
        self.q = num_nodes
        #array of the time
        self.time = time 
        #number of observations, N in Wilson et al. (2012)
        self.N = self.time.size
        #the data, it should be given as data1, data1error, data2, ...
        self.args = args 
        
        #number of outputs y(x); p in Wilson et al. (2012)
        self.p = int(len(self.args)/2)
        #total number of weights, we will have q*p weights in total
        self.qp =  self.q * self.p
        
        #to organize the data we now join everything
        self.tt = np.tile(time, self.p) #"extended" time because why not?
        ys = []
        ystd = []
        yerrs = []
        for i,j  in enumerate(args):
            if i%2 == 0:
                ys.append(j)
                ystd.append(np.std(j))
            else:
                yerrs.append(j)
        self.ystd = np.array(ystd).reshape(self.p, 1)
        self.y = np.array(ys).reshape(self.p, self.N) #matrix p*N of outputs
        self.yerr = np.array(yerrs).reshape(self.p, self.N) #matrix p*N of errors
        self.yerr2 = self.yerr**2
        #check if the input was correct
        assert int((i+1)/2) == self.p, \
        'Given data and number of components dont match'
        
        
##### mean functions definition ###############################################
    def _mean(self, means, time=None):
        """
        Returns the values of the mean functions
        
        Parameters
        ----------
        
        Returns
        -------
        m: float
            Value of the mean
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
        at inputs time
        
        Parameters
        ----------
        
        Returns
        -------
        K: array
            Matrix of a covariance function
        """
        r = time[:, None] - time[None, :]
        
        #to deal with the non-stationary kernels problem
        if isinstance(kernel, (covL, covP)):
            K = kernel(None, time[:, None], time[None, :])
        else:
            K = kernel(r) #+ 1e-6*np.diag(np.diag(np.ones_like(r)))
        K[np.abs(K)<1e-15] = 0.
        return K

    def _predictKernelMatrix(self, kernel, time):
        """
        To be used in predict_gp()
        
        Parameters
        ----------
        
        Returns
        -------
        K: array
            Matrix of a covariance function
        """
        if isinstance(kernel, (covL, covP)):
            K = kernel(None, time, self.time[None, :])
        else:
            if time.size == 1:
                r = time - self.time[None, :]
            else:
                r = time[:,None] - self.time[None,:]
            K = kernel(r) 
        return K

    def _kernel_pars(self, kernel):
        """
        Hyperparameters of a given kernel
        
        Parameters
        ----------
        
        Returns
        -------
        kernel.pars: array
            Hyperparameter values
        """
        return kernel.pars

    def _u_to_fhatW(self, u):
        """
        Given an array of values, divides it in the corresponding nodes (f hat)
        and weights (w) parts
        
        Parameters
        ----------
        u: array
        
        Returns
        -------
        f: array
            Samples of the nodes
        w: array
            Samples of the weights
        """
        f = u[:self.q * self.N].reshape((1, self.q, self.N))
        w = u[self.q * self.N:].reshape((self.p, self.q, self.N))
        return f, w

    def _cholNugget(self, matrix, maximum=10):
        """
        Returns the cholesky decomposition to a given matrix, if it is not
        positive definite, a nugget is added to its diagonal.
        
        Parameters
        ----------
        matrix: array
            Matrix to decompose
        maximum: int
            Number of times a nugget is added.
        
        Returns
        -------
        L: array
            Matrix containing the Cholesky factor
        nugget: float
            Nugget added to the diagonal
        """
        nugget = 0 #our nugget starts as zero
        try:
            nugget += np.abs(np.diag(matrix).mean()) * 1e-5
            L = cholesky(matrix, lower=True, overwrite_a=True)
            return L, nugget
        except LinAlgError:
            #print('NUGGET ADDED TO DIAGONAL!')
            n = 0 #number of tries
            while n < maximum:
                #print ('n:', n+1, ', nugget:', nugget)
                try:
                    L = cholesky(matrix + nugget*np.identity(matrix.shape[0]),
                                 lower=True, overwrite_a=True)
                    return L, nugget
                except LinAlgError:
                    nugget *= 10.0
                finally:
                    n += 1
            raise LinAlgError("Not positive definite, even with nugget.")
            
            
            
##### Mean-Field Inference functions ##########################################
    def EvidenceLowerBound(self, node, weight, mean, jitter, 
                           optimalSolution = False):
        """
        Evidence Lower bound, eq.10 in Nguyen & Bonilla (2013)
        
        Parameters
        ----------
        node: array
            Node functions 
        weight: array
            Weight function
        mean: array
            Mean functions
        jitter: array
            Jitter terms
        optimalSolution: bool
            True if GPRN is already opimized
        
        Returns
        -------
        ELBO: float
            Evidence lower bound
        """ 
        D = self.time.size * self.q *(self.p+1)
        np.random.seed(230190)
        mu = np.random.randn(D, 1)
        var = np.random.rand(D, 1)
        #to separate the variational parameters between the nodes and weights
        muF, muW = self._u_to_fhatW(mu.flatten())
        varF, varW = self._u_to_fhatW(var.flatten())

        if optimalSolution is True:
            sigmaF, muF, sigmaW, muW = self._updateSigmaMu(node, weight, mean, 
                                                           jitter, muF, varF, 
                                                           muW, varW)
            #new mean for the nodes
            muF = muF.reshape(1, self.q, self.N)
            varF =  []
            for i in range(self.q):
                varF.append(np.diag(sigmaF[i]))
            #new variance for the nodes
            varF = np.array(varF).reshape(1, self.q, self.N)
            #new mean for the weights
            muW = muW.reshape(self.p, self.q, self.N)
            varW =  []
            for j in range(self.q):
                for i in range(self.p):
                    varW.append(np.diag(sigmaW[j, i, :]))
            #new variance for the weights
            varW = np.array(varW).reshape(self.p, self.q, self.N)
            finalMu = np.concatenate((muF, muW))
            finalVar = np.concatenate((varF, varW))
            return finalMu, finalVar, sigmaF, sigmaW
        #updating mu and var
        sigmaF, muF, sigmaW, muW = self._updateSigmaMu(node, weight, mean, 
                                                       jitter, muF, varF, 
                                                       muW, varW)
        #new mean for the nodes
        muF = muF.reshape(1, self.q, self.N)
        varF =  []
        for i in range(self.q):
            varF.append(np.diag(sigmaF[i]))
        #new variance for the nodes
        varF = np.array(varF).reshape(1, self.q, self.N)
        #new mean for the weights
        muW = muW.reshape(self.p, self.q, self.N)
        varW =  []
        for j in range(self.q):
            for i in range(self.p):
                varW.append(np.diag(sigmaW[j, i, :]))
        #new variance for the weights
        varW = np.array(varW).reshape(self.p, self.q, self.N)
            
        #Entropy
        Entropy = self._entropy(sigmaF, sigmaW)
        #Expected log prior
        ExpLogPrior = self._expectedLogPrior(node, weight, 
                                            sigmaF, muF, sigmaW, muW)
        #Expected log-likelihood
        ExpLogLike = self._expectedLogLike(node, weight, mean, jitter, 
                                           sigmaF, muF, sigmaW, muW)
        #Evidence Lower Bound
        ELBO = (ExpLogLike + ExpLogPrior + Entropy)
        #print(ExpLogPrior,ExpLogLike,Entropy)
        return ELBO
    
    
    def Prediction(self, node, weights, means, tstar, mu):
        """
        Prediction for mean-field inference
        
        Parameters
        ----------
        node: array
            Node functions 
        weight: array
            Weight function
        means: array
            Mean functions
        tstar: array
            Predictions time
        mu: array
            Variational means
            
        Returns
        -------
        final_ystar: array
            Predicted means
        """
        Kf = np.array([self._kernelMatrix(i, self.time) for i in node])
        Lf = np.array([self._cholNugget(i)[0] for i in Kf])
        Kw = np.array([self._kernelMatrix(j, self.time) for j in weights])
        Lw = np.array([self._cholNugget(j)[0] for j in Kw])
        
        #mean functions
        means = self._mean(means, tstar)
        means = np.array_split(means, self.p)
        #means = np.zeros_like(means)

        muF, muW = self._u_to_fhatW(mu.flatten())
        
        ystar = np.zeros((self.p, tstar.size))
        for i in range(tstar.size):
            Kfstar = np.array([self._predictKernelMatrix(i1, tstar[i]) for i1 in node])
            Kwstar = np.array([self._predictKernelMatrix(i2, tstar[i]) for i2 in weights])
            Lwstar = np.linalg.solve(np.squeeze(Lw), np.squeeze(Kwstar).T)
            countF, countW = 1, 1
            Wstar, fstar = np.zeros((self.p, self.q)), np.zeros((self.q, 1))
            for q in range(self.q):
                Lfstar = np.linalg.solve(np.squeeze(Lf[q,:,:]), np.squeeze(Kfstar[q,:]).T)
                fstar[q] = Lfstar @ np.linalg.solve(np.squeeze(Lf[q,:,:]), muF[:,q,:].T)
                countF += self.N
                for p in range(self.p):
                    Wstar[p, q] = Lwstar @ np.linalg.solve(np.squeeze(Lw[0]), muW[p][q].T)
                    countW += self.N
            ystar[:,i] = ystar[:, i] + np.squeeze(Wstar @ fstar)
        final_ystar = []
        for i in range(self.p):
            final_ystar.append(ystar[i] + means[i])
        final_ystar = np.array(final_ystar)
        return final_ystar
    
    def _updateSigmaMu(self, nodes, weight, mean, jitter, muF, varF, muW, varW):
        """
        Efficient closed-form updates fot variational parameters. This
        corresponds to eqs. 16, 17, 18, and 19 of Nguyen & Bonilla (2013)
        
        Parameters
        ----------
        nodes: array
            Node functions 
        weight: array
            Weight function
        mean: array
            Mean functions
        jitter: array
            Jitter terms
        muF: array
            Initial variational mean of each node
        varF: array
            Initial variational variance of each node
        muW: array
            Initial variational mean of each weight
        varW: array
            Initial variational variance of each weight
            
        Returns
        -------
        sigma_f: array
            Updated variational covariance of each node
        mu_f: array
            Updated variational mean of each node
        sigma_w: array
            Updated variational covariance of each weight
        mu_w: array
            Updated variational mean of each weight
        """
        new_y = np.concatenate(self.y) - self._mean(mean)
        new_y = np.array(np.array_split(new_y, self.p))
        jitt2 = np.array(jitter)**2 #jitters
        
        #kernel matrix for the nodes
        Kf = np.array([self._kernelMatrix(i, self.time) for i in nodes])
        #kernel matrix for the weights
        Kw = np.array([self._kernelMatrix(j, self.time) for j in weight]) 
        #we have Q nodes => j in the paper; we have P y(x)s => i in the paper
        if self.q == 1:
            sigma_f, mu_f = [], [] #creation of Sigma_fj and mu_fj
            for j in range(self.q):
                diagFj = np.zeros_like(self.N)
                auxCalc = np.zeros_like(self.N)
                for i in range(self.p):
                    diagFj = diagFj + (muW[i,j,:]*muW[i,j,:]+varW[i,j,:])\
                                                /(jitt2[i] + self.yerr2[i,:])
                    sumNj = np.zeros(self.N)
                    for k in range(self.q):
                        if k != j:
                            muF = muF.T.reshape(1, self.q, self.N )
                            sumNj += muW[i,k,:]*muF[:,k,:].reshape(self.N)
                    auxCalc = auxCalc + ((new_y[i,:]-sumNj)*muW[i,j,:])\
                                        / (jitt2[i] + self.yerr2[i,:])
                CovF0 = np.diag(1 / diagFj) + Kf[j]
                CovF = Kf[j] - Kf[j] @ np.linalg.solve(CovF0, Kf[j])
            sigma_f.append(CovF)
            mu_f.append(CovF @ auxCalc)
            sigma_f = np.array(sigma_f)
            mu_f = np.array(mu_f)
            sigma_w, mu_w = [], [] #creation of Sigma_wij and mu_wij
            for i in range(self.p):
                for j in range(self.q):
                    mu_fj = mu_f[j]
                    var_fj = np.diag(sigma_f[j])
                    Diag_ij = (mu_fj * mu_fj + var_fj)/(jitt2[i] + self.yerr2[i,:])
                    Kw = np.squeeze(Kw)
                    CovWij = np.diag(1 / Diag_ij) + Kw
                    CovWij = Kw - Kw @ np.linalg.solve(CovWij, Kw)
                    sumNj = np.zeros(self.N)
                    for k in range(self.q):
                        if k != j:
                            sumNj += mu_f[k].reshape(self.N)*np.array(muW[i,k,:])
                    auxCalc = ((new_y[i,:]-sumNj)*mu_f[j,:])\
                                        / (jitt2[i] + self.yerr2[i,:])
                    sigma_w.append(CovWij)
                    mu_w.append(CovWij @ auxCalc)
            sigma_w = np.array(sigma_w).reshape(self.q, self.p, self.N, self.N)
            mu_w = np.array(mu_w)
        else:
            muF = np.squeeze(muF)
            sigma_f, mu_f = [], [] #creation of Sigma_fj and mu_fj
            for j in range(self.q):
                diagFj = np.zeros_like(self.N)
                auxCalc = np.zeros_like(self.N)
                for i in range(self.p):
                    diagFj = diagFj + (muW[i,j,:]*muW[i,j,:]+varW[i,j,:])\
                                                /(jitt2[i] + self.yerr2[i,:])
                    sumNj = np.zeros(self.N)
                    for k in range(self.q):
                        if k != j:
                            sumNj += muW[i,k,:]*muF[k,:].reshape(self.N)
                    auxCalc = auxCalc + ((new_y[i,:]-sumNj)*muW[i,j,:]) \
                                        / (jitt2[i] + self.yerr2[i,:])
                CovF = np.diag(1 / diagFj) + Kf[j]
                CovF = Kf[j] - Kf[j] @ np.linalg.solve(CovF, Kf[j])
                sigma_f.append(CovF)
                mu_f.append(CovF @ auxCalc)
                muF = np.array(mu_f)
            sigma_f = np.array(sigma_f)
            mu_f = np.array(mu_f)
            sigma_w, mu_w = [], np.zeros_like(muW) #creation of Sigma_wij and mu_wij
            for j in range(self.q):
                for i in range(self.p):
                    mu_fj = mu_f[j]
                    var_fj = np.diag(sigma_f[j])
                    Diag_ij = (mu_fj*mu_fj+var_fj) /(jitt2[i] + self.yerr2[i,:])
                    Kw = np.squeeze(Kw)
                    #print(Kw)
                    CovWij = np.diag(1 / Diag_ij) + Kw
                    CovWij = Kw - Kw @ np.linalg.solve(CovWij, Kw)
                    #print(CovWij)
                    sumNj = np.zeros(self.N)
                    for k in range(self.q):
                        if k != j:
                            sumNj += mu_f[k].reshape(self.N)*np.array(muW[i,k,:])
                    auxCalc = ((new_y[i,:]-sumNj)*mu_f[j,:]) \
                                        / (jitt2[i] + self.yerr2[i,:])
                    sigma_w.append(CovWij)
                    muW[i,j,:] = CovWij @ auxCalc
            sigma_w = np.array(sigma_w).reshape(self.q, self.p, self.N, self.N)
            mu_w = np.array(muW)
        return sigma_f, mu_f, sigma_w, mu_w
    
    
    def _expectedLogLike(self, nodes, weight, mean, jitter, sigma_f, mu_f,
                         sigma_w, mu_w):
        """
        Calculates the expected log-likelihood in mean-field inference, 
        corresponds to eq.14 in Nguyen & Bonilla (2013)
        
        Parameters
        ----------
        nodes: array
            Node functions 
        weight: array
            Weight function
        jitter: array
            Jitter terms
        sigma_f: array 
            Variational covariance for each node
        mu_f: array
            Variational mean for each node
        sigma_w: array
            Variational covariance for each weight
        mu_w: array
            Variational mean for each weight
            
        Returns
        -------
        logl: float
            Expected log-likelihood value
        """
        new_y = np.concatenate(self.y) - self._mean(mean, self.time)
        new_y = np.array(np.array_split(new_y, self.p)).T #NxP dimensional vector
        jitt = np.exp(np.array(jitter)) #jitters
        jitt2 = np.exp(2*np.array(jitter)) #jitters squared
        ycalc = new_y.T #new_y0.shape = (p,n)
        ycalc1 = new_y.T
        
        #self.yerr2 = 0*self.yerr2
        self.yerr = 0*self.yerr
        
        logl = 0
        for p in range(self.p):
            ycalc[p] = new_y.T[p,:] / (jitt[p] + self.yerr[p,:])
            #ycalc[p] = new_y.T[p,:] / (jitt2[p] + self.yerr2[p,:])
            for n in range(self.N):
                logl += np.log(jitt2[p] + self.yerr2[p,n])
        logl = -0.5 * logl
        
        if self.q == 1:
            Wcalc, Wcalc1 = np.array([]), np.array([])
            for n in range(self.N):
                for p in range(self.p):
                    Wcalc = np.append(Wcalc, mu_w[p,:,n])
                    #Wcalc1 = np.append(Wcalc1, mu_w[p,:,n])
            Fcalc, Fcalc1 = np.array([]), np.array([])
            for n in range(self.N):
                for q in range(self.q):
                    for p in range(self.p):
                        Fcalc = np.append(Fcalc, (mu_f[:, q, n] / (jitt[p] + self.yerr[p,n])))# - jitt[p]*self.yerr[p,n])))
                        #Fcalc = np.append(Fcalc, (mu_f[:, q, n] / (jitt2[p] + self.yerr2[p,n])))
                        #Fcalc1 = np.append(Fcalc1, mu_f[:, q, n])
            Ymean = (Wcalc * Fcalc).reshape(self.N, self.p)
            #Ymean1 = (Wcalc1 * Fcalc1).reshape(self.N, self.p)
            Ydiff = (ycalc - Ymean.T) * (ycalc - Ymean.T)
            #Ydiff = (ycalc - Ymean.T) * (ycalc1 - Ymean1.T)
            logl += -0.5 * np.sum(Ydiff)
            value = 0
            for i in range(self.p):
                for j in range(self.q):
                    value += np.sum((np.diag(sigma_f[j,:,:])*mu_w[i,j,:]*mu_w[i,j,:] +\
                                    np.diag(sigma_w[j,i,:,:])*mu_f[:,j,:]*mu_f[:,j,:] +\
                                    np.diag(sigma_f[j,:,:])*np.diag(sigma_w[j,i,:,:]))\
                                    /(jitt2[p] + self.yerr2[p,:]))
            logl += -0.5 * value 
        else:
            Wcalc = []
            for p in range(self.p):
                Wcalc.append([])
            for n in range(self.N):
                for p in range(self.p):
                    Wcalc[p].append(mu_w[p, :, n])
            Wcalc = np.array(Wcalc).reshape(self.p, self.N * self.q)
            Wcalc1 = Wcalc
            Fcalc, Fcalc1  = [], []#np.array([])
            for p in range(self.p):
                Fcalc.append([])
                Fcalc1.append([])
            for n in range(self.N):
                for q in range(self.q):
                    for p in range(self.p):
                        Fcalc[p] = np.append(Fcalc, (mu_f[:, q, n] / (jitt2[p] + self.yerr2[p,n])))
                        Fcalc1[p] = np.append(Fcalc1, mu_f[:, q, n])
            Ymean = np.sum((Wcalc * Fcalc).reshape(self.N, self.q), axis=1)
            Ymean1 = np.sum((Wcalc1 * Fcalc1).reshape(self.N, self.q), axis=1)
            #Ydiff = (ycalc - Ymean.T) * (ycalc - Ymean.T)
            Ydiff = (ycalc1 - Ymean1.T) * (ycalc - Ymean.T)
            logl += -0.5 * np.sum(Ydiff)

            value = 0
            for i in range(self.p):
                for j in range(self.q):
                    value += np.sum((np.diag(sigma_f[j,:,:])*mu_w[i,j,:]*mu_w[i,j,:] +\
                                    np.diag(sigma_w[j,i,:,:])*mu_f[:,j,:]*mu_f[:,j,:] +\
                                    np.diag(sigma_f[j,:,:])*np.diag(sigma_w[j,i,:,:]))\
                                    /(jitt2[p] + self.yerr2[p,:]))
            logl += -0.5* value
        return logl


    def _expectedLogPrior(self, nodes, weights, sigma_f, mu_f, sigma_w, mu_w):
        """
        Calculates the expection of the log prior wrt q(f,w) in mean-field 
        inference, corresponds to eq.15 in Nguyen & Bonilla (2013)
        
        Parameters
        ----------
            nodes: array
                Node functions 
            weight: array
                Weight function
            sigma_f: array
                Variational covariance for each node
            mu_f: array
                Variational mean for each node
            sigma_w: array
                Variational covariance for each weight
            mu_w: array
                Variational mean for each weight
        
        Returns
        -------
        logp: float
            Expected log prior value
        """
        Kf = np.array([self._kernelMatrix(i, self.time) for i in nodes])
        Kw = np.array([self._kernelMatrix(j, self.time) for j in weights])
        
        #we have Q nodes -> j in the paper; we have P y(x)s -> i in the paper
        first_term = 0 #calculation of the first term of eq.15 of Nguyen & Bonilla (2013)
        second_term = 0 #calculation of the second term of eq.15 of Nguyen & Bonilla (2013)
        Lw = self._cholNugget(Kw[0])[0]
        logKw = np.float(np.sum(np.log(np.diag(Lw))))
        muW = mu_w.reshape(self.q, self.p, self.N)
        
        sumSigmaF = np.zeros_like(sigma_f[0])
        for j in range(self.q):
            Lf = self._cholNugget(Kf[j])[0]
            logKf = np.float(np.sum(np.log(np.diag(Lf))))
            muK =  np.linalg.solve(Lf, mu_f[:,j, :].reshape(self.N))
            muKmu = muK @ muK
            sumSigmaF = sumSigmaF + sigma_f[j]
            trace = np.trace(np.linalg.solve(Kf[j], sumSigmaF))#sigma_f[j]))
            first_term += -self.q*logKf - 0.5*(muKmu + trace)
            for i in range(self.p):
                muK = np.linalg.solve(Lw, muW[j,i]) 
                muKmu = muK @ muK
                trace = np.trace(np.linalg.solve(Kw[0], sigma_w[j, i, :, :]))
                second_term += -self.q*logKw - 0.5*(muKmu + trace)
        logp = first_term + second_term
        return logp


    def _entropy(self, sigma_f, sigma_w):
        """
        Calculates the entropy in mean-field inference, corresponds to eq.14 
        in Nguyen & Bonilla (2013)
        
        Parameters
        ----------
            sigma_f: array
                Variational covariance for each node
            sigma_w: array
                Variational covariance for each weight
        
        Returns
        -------
        entropy: float
            Final entropy value
        """
        entropy = 0 #starts at zero then we sum everything
        for j in range(self.q):
            L1 = self._cholNugget(sigma_f[j])
            entropy += np.sum(np.log(np.diag(L1[0])))
            for i in range(self.p):
                L2 = self._cholNugget(sigma_w[j, i, :, :])
                entropy += np.sum(np.log(np.diag(L2[0])))
        return entropy

