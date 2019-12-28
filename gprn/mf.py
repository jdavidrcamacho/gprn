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
    def EvidenceLowerBound(self, node, weight, mean, jitter, mu = None, 
                           var = None, sigmaF=None, sigmaW=None, opt_step = 1):
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
        mu: array
            Variational means
        var: array
            Variational variances
        opt_step: int
            0 to optimize mu and var
        
        Returns
        -------
        ELBO: float
            Evidence lower bound
        """ 
        D = self.time.size * self.q *(self.p+1)
        if mu is None:
            np.random.seed(100)
            mu = np.random.rand(D, 1)
        if var is None:
            np.random.seed(200)
            var = np.random.rand(D, 1)
        #to separate the variational parameters between the nodes and weights
        muF, muW = self._u_to_fhatW(mu.flatten())
        varF, varW = self._u_to_fhatW(var.flatten())
        
        if sigmaF is None:
            sigmaF = []
            for q in range(self.q):
                sigmaF.append(np.diag(varF[0, q, :]))
#                for p in range(self.p):
#                    print(np.array(sigmaF).shape)
#                    #sigmaF = np.array(sigmaF).reshape(1, self.N, self.N)
        if sigmaW is None:
            sigmaW = []
            for q in range(self.q):
                for p in range(self.p):
                    sigmaW.append(np.diag(varW[p, q, :]))
            sigmaW = np.array(sigmaW).reshape(self.q, self.p, self.N, self.N)
        #updating mu and var
        if opt_step == 0:
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
            new_mu = np.concatenate((muF, muW))
            new_var = np.concatenate((varF, varW))
            return 0, new_mu, new_var, sigmaF, sigmaW
        #Entropy
        Entropy = self._entropy(sigmaF, sigmaW)
        #Expected log prior
        ExpLogPrior = self._expectedLogPrior(node, weight, 
                                            sigmaF, muF, sigmaW, muW)
        #Expected log-likelihood
        ExpLogLike = self._expectedLogLike(node, weight, mean, jitter, 
                                           sigmaF, muF, sigmaW, muW)
        #Evidence Lower Bound
        ELBO = -(ExpLogLike + ExpLogPrior + Entropy)
        print('LogL =', ExpLogLike,'\nLogP =', ExpLogPrior, '\nEnt =', Entropy)
        return ELBO
    
    
    def optimizeGPRN(self, nodes, weight, mean, jitter, iterations = 1000,
                     updateVarParams = True, updateMeanParams = False, 
                     updateJittParams = False, updateHyperParams = False):
        """
        Optimizes the GPRN using scipy
        
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
        iterations: int
            Number of iterations 
        updateVarParams: bool
            True to optimize the variational parameters
        updateMeanParams: bool
            True to optimize the mean functions
        updateJittParams: bool
            True to optimize the jitters
        updateHyperParams: boll
            True to optimize the hyperparameters of the nodes and the weights
        
        Returns
        -------
        hyperparameters: array
            Optimized hyperparameters
        jitter: array
            Optimized jitters
        elboArray: array
            Value of the ELBO per iteration
        mu: array
            Optimized variational means
        var: array
            Optimized variational variance (diagonal of sigma)
        """
        #lets prepare the nodes, weights, means, and jitter
        nodesParams = np.array([])
        #we put the nodes parameters all in one array
        for q in range(self.q):
            nodesParams = np.append(nodesParams, nodes[q].pars[1:])
        #same for the weight
        weightParams = weight[0].pars[:-1]
        #and we finish putting everything in one giant array
        initParams = np.concatenate((nodesParams, weightParams))
        initParams = np.log(np.array(initParams))
        initParams[initParams == -np.inf] = 0
        #same logic for the means
        meanParams = np.array([])
        noneMean = 0
        for p in range(self.p):
            if mean[p] is None:
                noneMean += 1
            else:
                meanParams = np.append(meanParams, mean[p].pars)
        meanParams = np.log(np.array(meanParams))
        meanParams[meanParams == -np.inf] = 0
        #the jitter also become an array in log space
        jittParams = np.log(np.array(jitter))
        jittParams[jittParams == -np.inf] = 0
        
        #initial variational parameters (they start as random)
        D = self.time.size * self.q *(self.p+1)
        np.random.seed(100)
        mu = np.random.rand(D, 1)
        np.random.seed(200)
        var = np.random.rand(D, 1)
        
        elboArray = np.array([]) #To add new elbo values inside
        iterNumber = 0
        while iterNumber < iterations:
            #print('\n*** ITERATION {0} ***'.format(iterNumber+1))
            #Optimize mu and var analytically
            if updateVarParams:
                _, mu, var, sigF, sigW = self.EvidenceLowerBound(nodes, weight, 
                                                                 mean, jittParams, 
                                                                 mu, var, 
                                                                 opt_step=0)
            #1st step - optimize the means
            if updateMeanParams:
                res0 = minimize(fun = self._meanELBO, x0 = meanParams, 
                               args = (nodes, weight, mean, jittParams, mu, sigF, sigW), 
                               method = 'Nelder-Mead',
                               options = {'maxiter': 200, 'adaptive': False})
                meanParams = res0.x #updated jitters array
            #2nd step - optimize the jitters
            if updateJittParams:
                res1 = minimize(fun = self._jittELBO, x0 = jittParams, 
                               args = (nodes, weight, mean, mu, sigF, sigW), 
                               method = 'Nelder-Mead',
                               options = {'maxiter': 200, 'adaptive': False})
                jittParams = res1.x #updated jitters array
            jitter = np.exp(np.array(jittParams)) #updated jitter values
            #3rdstep - optimize nodes, weights, and means
            if updateHyperParams:
                res2 = minimize(fun = self._paramsELBO, x0 = initParams,
                               args = (nodes, weight, mean, mu, var, sigF, sigW), 
                               method = 'Nelder-Mead',
                               options={'maxiter': 200, 'adaptive': False})
                initParams = res2.x
            hyperparameters = np.exp(np.array(initParams))
            nodes, weight = newCov(nodes, weight, hyperparameters, self.q)
            #4th step - ELBO to check stopping criteria
            ELBO  = -self.EvidenceLowerBound(nodes, weight, mean, jittParams, 
                                             mu, var, sigF, sigW, opt_step=1)
            elboArray = np.append(elboArray, ELBO)
            iterNumber += 1
            #Stoping criteria:
            criteria = np.abs(np.mean(elboArray[-2:-1]) - ELBO)
            if criteria < 1e-5 and criteria != 0 :
                print('\nELBO converged to '+ str(round(float(ELBO),5)) \
                      +' at iteration ' + str(iterNumber))
                print('nodes:', nodes)
                print('weights:', weight)
                print('mean:', mean)
                print('jitter:', jitter, '\n')
                return hyperparameters, jitter, elboArray, mu, var
            print('ELBO:', ELBO)
            print('nodes:', nodes)
            print('weights:', weight)
            print('mean:', mean)
            print('jitter:', jitter, '\n')
        return hyperparameters, jitter, elboArray, mu, var
    
    
    def mcmcGPRN(self, nodes, weight, mean, jitter, 
                 elboFunc, priorFunc, iterations = 1000):
        D = self.time.size * self.q *(self.p+1)
        mu = np.random.rand(D, 1)
        var = np.random.rand(D, 1)
        #to separate the variational parameters between the nodes and weights
        muF, muW = self._u_to_fhatW(mu.flatten())
        varF, varW = self._u_to_fhatW(var.flatten())
    
        return 0
    
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
        jitt2 = np.exp(2*np.array(jitter)) #jitters
        
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
    
    
    def _meanELBO(self, params, node, weight, mean, jitter, mu, sigF, sigW):
        """
        Function used to optimeze the mean functions
        
        Parameters
        ----------
        params : array
            Means parameters
        node : array
            Node functions of the GPRN
        weight: array
            Weights of the GPRN
        mean: array
            Mean fucntions of the GPRN
        mu: array
            Means of the variational parameters
        sigF: array
            Covariance of the variational parameters (nodes)
        sigW: array
            Covariance of the variational parameters (weights)
        Returns
        -------
        ExpLogLike: float
            Minus expected log-likelihood
        """
        #to separate the means between the nodes and weights
        muF, muW = self._u_to_fhatW(mu.flatten())
        
        parsUsed = 0
        for p in range(self.p):
            if mean[p] == None:
                pass
            else:
                howBig = len(mean[p].pars) 
                parsToUse = params[parsUsed:parsUsed+howBig]
                mean[p].pars = np.exp(np.array(parsToUse))
                parsUsed += howBig

        #Expected log-likelihood
        ExpLogLike = self._expectedLogLike(node, weight, mean, jitter,
                                           sigF, muF, sigW, muW)
        return -ExpLogLike
    
    
    def _jittELBO(self, jitter, node, weight, mean, mu, sigF, sigW):
        """
        Function used to optimeze the jitter terms
        
        Parameters
        ----------
        jitter : array
            Jitter parameters
        node : array
            Node functions of the GPRN
        weight: array
            Weights of the GPRN
        mean: array
            Mean fucntions of the GPRN
        mu: array
            Means of the variational parameters
        sigF: array
            Covariance of the variational parameters (nodes)
        sigW: array
            Covariance of the variational parameters (weights)
        Returns
        -------
        ExpLogLike: float
            Minus expected log-likelihood
        """
        #to separate the means between the nodes and weights
        muF, muW = self._u_to_fhatW(mu.flatten())
        #Expected log-likelihood
        ExpLogLike = self._expectedLogLike(node, weight, mean, jitter,
                                           sigF, muF, sigW, muW)
        return -ExpLogLike
    
    
    def _paramsELBO(self, params, node, weight, mean, mu, var, sigF, sigW):
        """
        Function used to optimize the hyperparameters of the nodes and the
        weights
        
        Parameters
        ----------
        params : array
            Hyperparameters to optimize
        node : array
            Node functions of the GPRN
        weight: array
            Weights of the GPRN
        mean: array
            Mean fucntions of the GPRN
        mu: array
            Means of the variational parameters
        var: array
            Variance of the variational parameters
        sigF: array
            Covariance of the variational parameters (nodes)
        sigW: array
            Covariance of the variational parameters (weights)
            
        Returns
        -------
        ExpLogPrior: float
            Minus expected log prior
        """
        params = np.exp(np.array(params))
        node, weight = newCov(node, weight, params, self.q)
        
        #to separate the variational parameters between the nodes and weights
        muF, muW = self._u_to_fhatW(mu.flatten())
        varF, varW = self._u_to_fhatW(var.flatten())

        ExpLogPrior = self._expectedLogPrior(node, weight, 
                                             sigF, muF, sigW, muW)
        return -ExpLogPrior
    
    
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
                        Fcalc = np.append(Fcalc, (mu_f[:, q, n] / (jitt2[p] + self.yerr[p,n])))
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
            Fcalc = []#np.array([])
            for p in range(self.p):
                Fcalc.append([])
            for n in range(self.N):
                for q in range(self.q):
                    for p in range(self.p):
                        Fcalc[p] = np.append(Fcalc, (mu_f[:, q, n] / (jitt[p] + self.yerr[p,n])))
            Ymean = np.sum((Wcalc * Fcalc).reshape(self.N, self.q), axis=1)
            Ydiff = (ycalc - Ymean.T) * (ycalc - Ymean.T) 
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


    def _plots(self, ELB, ELL, ELP, ENT):
        """
        Plots the evolution of the evidence lower bound, expected log 
        likelihood, expected log prior, and entropy
        
        Parameters
        ----------
        
        Returns
        -------
        0: int
        """
        plt.figure()
        ax1 = plt.subplot(411)
        plt.plot(ELB, '-')
        plt.ylabel('Evidence lower bound')
        plt.subplot(412, sharex=ax1)
        plt.plot(ELL, '-')
        plt.ylabel('Expected log likelihood')
        plt.subplot(413, sharex=ax1)
        plt.plot(ELP, '-')
        plt.ylabel('Expected log prior')
        plt.subplot(414, sharex=ax1)
        plt.plot(ENT, '-')
        plt.ylabel('Entropy')
        plt.xlabel('iteration')
        plt.show()
        return 0


    def _prints(self, ELBO, ExpLogLike, ExpLogPrior, Entropy):
        """
        Prints the evidence lower bound, expected log likelihood, expected
        log prior, and entropy
        
        Parameters
        ----------
        
        Returns
        -------
        0: int
        """
        print('ELBO: ' + str(float(ELBO)))
        print(' loglike: ' + str(float(ExpLogLike)) + ' \n logprior: ' \
              + str(ExpLogPrior) + ' \n entropy: ' + str(Entropy) + ' \n')
        return 0

