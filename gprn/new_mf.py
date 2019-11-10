#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt

from scipy.linalg import inv, cholesky, LinAlgError
from scipy.optimize import minimize, fmin

from gprn.covFunction import Linear as covL
from gprn.covFunction import Polynomial as covP

class inference(object):
    """ 
        Class to perform mean field variational inference for GPRNs. 
        See Nguyen & Bonilla (2013) for more information.
        Parameters:
            num_nodes = number of latent node functions f(x), called f hat in 
                        the article
            time = array with the x/time coordinates
            *args = the data (or components), it needs be given in order of
                data1, data1error, data2, data2error, etc...
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
            K = kernel(r) + 1e-6*np.diag(np.diag(np.ones_like(r)))
        return K

    def _predictKernelMatrix(self, kernel, time):
        """
            To be used in predict_gp()
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
            Returns the hyperparameters of a given kernel
        """
        return kernel.pars

    def _u_to_fhatW(self, u):
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
            print('NUGGET ADDED TO DIAGONAL!')
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
            raise LinAlgError("Not positive definite, even with nugget.")
            
            
##### Mean-Field Inference functions ##########################################
    def EvidenceLowerBound(self, node, weight, mean, jitter, mu = None, 
                           var = None, opt_step = 1):
        """
            Returns the Evidence Lower bound, eq.10 in Nguyen & Bonilla (2013)
            Parameters:
                node = array of node functions 
                weight = weight function
                mean = array with the mean functions
                jitter = array of jitter terms
                mu = variational means
                var = variational variances
                standardize = True to standardize the data
                opt_step = 1 to optimize mu and var
                         = 2 to optimize jitter
                         = 3 to optimize the hyperparametes 
            Returns:
                ELBO = Evidence lower bound
                new_mu = array with the new variational means
                new_muW = array with the new variational variances
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
        sigmaF, sigmaW = [], []
        for q in range(self.q):
            sigmaF.append(np.diag(varF[0, q, :]))
            for p in range(self.p):
                sigmaW.append(np.diag(varW[p, q, :]))
        sigmaF = np.array(sigmaF).reshape(self.q, self.N, self.N)
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
        #new variational means and variances 
        new_mu = np.concatenate((muF, muW))
        new_var = np.concatenate((varF, varW))
        return ELBO, new_mu, new_var, sigmaF, sigmaW

    def _jittELBO(self, jitter, node, weight, mean, mu, sigF, sigW):
        
        #jitter= np.exp(2*jitter)
        #jitter= jitter
        
        
        #to separate the means between the nodes and weights
        muF, muW = self._u_to_fhatW(mu.flatten())
        #Entropy
        Entropy = self._entropy(sigF, sigW)
        #Expected log prior
        ExpLogPrior = self._expectedLogPrior(node, weight, 
                                            sigF, muF, sigW, muW)
        #Expected log-likelihood
        ExpLogLike = self._expectedLogLike(node, weight, mean, jitter, 
                                           sigF, muF, sigW, muW)
        
        #Evidence Lower Bound
        ELBO = (ExpLogLike + ExpLogPrior + Entropy)
        return -ELBO

    def _paramsELBO(self, params, node, weight, mean, jitter, mu, var, sigF, sigW):
        paramNodes = 0
        for q in range(self.q):
            paramNodes += node[q].params_size
        #number of parameters that come from the nodes
        paramNodes -= self.q 
        #number for parameters that come from the weight
        paramWeight = weight[0].params_size -1
        #parameters array of our nodes and weight
        paramGP = params[0: paramNodes+paramWeight] 
        
        jitter= np.exp(2*jitter)
        
        #Updating the nodes
        paramUsed = 0 
        for q in range(self.q):
            paramArray = np.array([1]) #the amplitude of the node is 1
            paramNumber = paramUsed + node[q].params_size - 1
            node[q].pars = np.concatenate((paramArray, paramGP[paramUsed:paramNumber]))
            paramUsed += paramNumber
        #Updating the weights
        paramToUse = weight[0].params_size-1
        weight[0].pars = np.concatenate((paramGP[paramUsed:paramUsed+paramToUse], np.array([0])))

        #parameters array of our means
        paramMean = params[paramNodes+paramWeight:]
        #Updating the means
        paramUsed, noneMean = 0, 0
        for p in range(self.p):
            if mean[p] is None:
                noneMean += 1
            else:
                paramNumber = paramUsed + mean[p]._parsize
                mean[p].pars = paramMean[paramUsed:paramNumber]
                paramUsed += paramNumber
        
        #to separate the variational parameters between the nodes and weights
        muF, muW = self._u_to_fhatW(mu.flatten())
        varF, varW = self._u_to_fhatW(var.flatten())
        #Now lets calculate the ELBO
        Entropy = self._entropy(sigF, sigW)
        #Expected log prior
        ExpLogPrior = self._expectedLogPrior(node, weight, 
                                            sigF, muF, sigW, muW)
        #Expected log-likelihood
        ExpLogLike = self._expectedLogLike(node, weight, mean, jitter, 
                                           sigF, muF, sigW, muW)
        #Evidence Lower Bound
        ELBO = (ExpLogLike + ExpLogPrior + Entropy)
        return -ELBO

    def Prediction(self, node, weights, means, tstar, muF, muW, 
                   standardize=False):
        """
            Prediction for mean-field inference
            Parameters:
                node = array of node functions 
                weight = weight function
                means = array with the mean functions
                tstar = predictions time
                muF = array with the initial means for each node
                varF = array with the initial variance for each node
                muW = array with the initial means for each weight
            Returns:
                ystar = predicted means
        """
        Kf = np.array([self._kernelMatrix(i, self.time) for i in node])
        Lf = np.array([self._cholNugget(i)[0] for i in Kf])
        Kw = np.array([self._kernelMatrix(j, self.time) for j in weights])
        Lw = np.array([self._cholNugget(j)[0] for j in Kw])
        
        #mean functions
        means = self._mean(means, tstar)
        means = np.array_split(means, self.p)
        
        ystar = np.zeros((self.p, tstar.size))
        for i in range(tstar.size):
            Kf_s = np.array([self._predictKernelMatrix(i1, tstar[i]) for i1 in node])
            Kw_s = np.array([self._predictKernelMatrix(i2, tstar[i]) for i2 in weights])
            alphaLw = inv(np.squeeze(Lw)) @ np.squeeze(Kw_s).T
            idx_f, idx_w = 1, 1
            Wstar, fstar = np.zeros((self.p, self.q)), np.zeros((self.q, 1))
            for q in range(self.q):
                alphaLf = inv(np.squeeze(Lf[q,:,:])) @ np.squeeze(Kf_s[q,:]).T
                fstar[q] = alphaLf@(inv(np.squeeze(Lf[q,:,:])) @ muF[:,q,:].T)
                idx_f += self.N
                for p in range(self.p):
                    Wstar[p, q] = alphaLw.T@(inv(np.squeeze(Lw[0]))@muW[p][q].T)
                    idx_w += self.N
            ystar[:,i] = ystar[:, i] + np.squeeze(Wstar @ fstar)
        combined_ystar = []
        for i in range(self.p):
            if standardize:
                combined_ystar.append(ystar[i]*self.ystd[i] + means[i])
            else:
                combined_ystar.append(ystar[i] + means[i])
        combined_ystar = np.array(combined_ystar)
        return combined_ystar


    def optimizeGPRN(self, nodes, weight, mean, jitter, iterations = 1000):
        """
            Returns the Evidence Lower bound, eq.10 in Nguyen & Bonilla (2013)
            Parameters:
                node = array of node functions 
                weight = weight function
                mean = array with the mean functions
                jitter = array of jitter terms
                iterations = number of iterations 
            Returns:
                initParams = optimized parameters of the node, weight, and mean
                jittParams = optimized jitter value
        """
        #lets prepare the nodes, weights, means, and jitter
        nodesParams = np.array([])
        #we put the nodes parameters all in one array
        for q in range(self.q):
            nodesParams = np.append(nodesParams, nodes[q].pars[1:])
        #same for the weight
        weightParams = weight[0].pars[:-1]
        #same for the means
        meanParams = np.array([])
        
        noneMean = 0
        for p in range(self.p):
            if mean[p] is None:
                noneMean += 1
            else:
                meanParams = np.append(meanParams, mean[p].pars)
        
        #and we finish putting everything in one giant array
        initParams = np.concatenate((nodesParams, weightParams, meanParams))
        
        #the jitter also become an array
        #jittParams = np.log(np.array(jitter))
        jittParams = np.array(jitter)
        
        
        #initial variational parameters (they start as random)
        D = self.time.size * self.q *(self.p+1)
        np.random.seed(100)
        mu = np.random.rand(D, 1)
        np.random.seed(200)
        var = np.random.rand(D, 1)
        
        elboArray = np.array([]) #To add new elbo values inside
        iterNumber = 0
        
        while iterNumber < iterations:
            print('Iteration {0}'.format(iterNumber+1))
            #1st step - optimize mu and var
            _, mu, var, sigF, sigW = self.EvidenceLowerBound(nodes, weight, 
                                                             mean, jitter, 
                                                             mu, var, 
                                                             opt_step=0)
            #print(mu)
            
            #2nd step - optimize the jitters
            jittConsts = [{'type': 'ineq', 'fun': lambda x: x},
                          {'type': 'ineq', 'fun': lambda x: 1-x}]
            res = minimize(fun = self._jittELBO, x0 = jittParams, 
                           args = (nodes, weight, mean, mu, sigF, sigW), 
                           method = 'Nelder-Mead', constraints = jittConsts,
                           options = {'maxiter':10, 'maxfev':10, 'adaptive':False})
            jittParams = res.x #updated jitters array
            jitter = np.array(res.x) #updated jitter values

#           #2nd step - optimize the jitters
#            res = fmin(func = self._jittELBO, x0 = jittParams, 
#                           args = (nodes, weight, mean, mu, sigF, sigW))
#            jittParams = res #updated jitters array
#            jitter = jittParams #updated jitter values
            
            #3rdstep - optimize nodes, weights, and means
            parsConsts = [{'type': 'ineq', 'fun': lambda x: x}]
            res = minimize(fun = self._paramsELBO, x0 = initParams,
                           args = (nodes,weight,mean, jitter,mu,var,sigF,sigW), 
                           method = 'COBYLA', constraints = parsConsts, 
                           options={'maxiter': 250})
            initParams = res.x
            
#            #3rdstep - optimize nodes, weights, and means
#            res = fmin(func = self._paramsELBO, x0 = initParams,
#                           args = (nodes,weight,mean,jitter,mu,var,sigF,sigW))
#            initParams = res
            
            #4th step - ELBO to check stopping criteria
            ELBO  = self.EvidenceLowerBound(nodes, weight, mean, jitter, mu, var, 
                                    opt_step=1)[0]
            elboArray = np.append(elboArray, ELBO)
            iterNumber += 1
            #Stoping criteria
            criteria = np.abs(np.mean(elboArray[-5:]) - ELBO)
            if criteria < 1e-5 and criteria != 0:
                print('\nELBO converged to '+ str(round(float(ELBO),5)) +' at iteration ' + str(iterNumber))
                return initParams, jittParams, elboArray
            print('\t', nodes, '\n \t', weight, '\n \t', mean, '\n \t', jitter)
            print('ELBO:',ELBO, '\n')
        return initParams, jittParams, elboArray


    def _updateSigmaMu(self, nodes, weight, mean, jitter, muF, varF, muW, varW):
        """
            Efficient closed-form updates fot variational parameters. This
        corresponds to eqs. 16, 17, 18, and 19 of Nguyen & Bonilla (2013) 
            Parameters:
                nodes = array of node functions 
                weight = weight function
                mean = array with the mean functions
                jitter = array of jitter terms
                muF = array with the initial means for each node
                varF = array with the initial variance for each node
                muW = array with the initial means for each weight
                varW = array with the initial variance for each weight
                standardize = True to standardize the data
            Returns:
                sigma_f = array with the covariance for each node
                mu_f = array with the means for each node
                sigma_w = array with the covariance for each weight
                mu_w = array with the means for each weight
        """
        new_y = np.concatenate(self.y) - self._mean(mean)
        new_y = np.array(np.array_split(new_y, self.p))
        jitt2 = np.exp(2*np.array(jitter))
        
        #kernel matrix for the nodes
        Kf = np.array([self._kernelMatrix(i, self.time) for i in nodes])
        #kernel matrix for the weights
        Kw = np.array([self._kernelMatrix(j, self.time) for j in weight]) 

        print(muF)
        #we have Q nodes => j in the paper; we have P y(x)s => i in the paper
        if self.q == 1:
            sigma_f, mu_f = [], [] #creation of Sigma_fj and mu_fj
            for j in range(self.q):
                Diag_fj, tmp = 0, 0
                for i in range(self.p):
                    Diag_fj += (muW[i,j,:]*muW[i,j,:]+varW[i,j,:]) \
#                                            / (self.yerr2[i,:] + jitt2[i])
                    Sum_nj = np.zeros(self.N)
                    for k in range(self.q):
                        if k != j:
                            muF = muF.T.reshape(1, self.q, self.N )
                            Sum_nj += muW[i,k,:]*muF[:,k,:].reshape(self.N)
                    tmp += ((new_y[i,:]-Sum_nj)*muW[i,j,:]) \
#                                            / (self.yerr2[i,:] + jitt2[i])
                CovF = np.diag(jitt2 / Diag_fj) + Kf[j]
                CovF = Kf[j] - Kf[j] @ (inv(CovF) @ Kf[j])
                sigma_f.append(CovF)
                mu_f.append(CovF @ tmp)#/ jitt2)
            print(CovF)
            sigma_f = np.array(sigma_f)
            mu_f = np.array(mu_f)
            print(mu_f)
            sigma_w, mu_w = [], [] #creation of Sigma_wij and mu_wij
            for i in range(self.p):
                for j in range(self.q):
                    mu_fj = mu_f[j]
                    var_fj = np.diag(sigma_f[j])
                    Diag_ij = (mu_fj * mu_fj + var_fj) \
                                            / (self.yerr2[i,:] + jitt2[i])
                    Kw = np.squeeze(Kw)
                    CovWij = np.diag(1 / Diag_ij) + Kw
                    CovWij = Kw - Kw @ (inv(CovWij) @ Kw)
                    Sum_nj = 0
                    for k in range(self.q):
                        if k != j:
                            Sum_nj += mu_f[k].reshape(self.N)*np.array(muW[i,k,:])
                    tmp = ((new_y[i,:]-Sum_nj)*mu_f[j,:]) \
                                            / (self.yerr2[i,:] + jitt2[i])
                    sigma_w.append(CovWij)
                    mu_w.append(CovWij @ tmp)# /jitt2)
            sigma_w = np.array(sigma_w).reshape(self.q, self.p, self.N, self.N)
            mu_w = np.array(mu_w)
        else:
            muF = np.squeeze(muF)
            sigma_f, mu_f = [], [] #creation of Sigma_fj and mu_fj
            for j in range(self.q):
                Diag_fj, tmp = 0, 0
                for i in range(self.p):
                    Diag_fj += (muW[j,i,:]*muW[j,i,:]+varW[j,i,:]) \
                                            / (self.yerr2[i,:] + jitt2[i])
                    Sum_nj = np.zeros(self.N)
                    for k in range(self.q):
                        if k != j:
                            Sum_nj += muW[k,i,:]*muF[k,:].reshape(self.N)
                    tmp += ((new_y[i,:]-Sum_nj)*muW[j,i,:]) \
                                            / (self.yerr2[i,:] + jitt2[i])
                CovF = np.diag(1 / Diag_fj) + Kf[j]
                CovF = Kf[j] - Kf[j] @ (inv(CovF) @ Kf[j])
                sigma_f.append(CovF)
                mu_f.append(CovF @ tmp)# /jitt2)
                muF = np.array(mu_f)
            sigma_f = np.array(sigma_f)
            mu_f = np.array(mu_f)
            sigma_w, mu_w = [], [] #creation of Sigma_wij and mu_wij
            for j in range(self.q):
                for i in range(self.p):
                    mu_fj = mu_f[j]
                    var_fj = np.diag(sigma_f[j])
                    Diag_ij = (mu_fj*mu_fj+var_fj) \
                                            / (self.yerr2[i,:] + jitt2[i])
                    Kw = np.squeeze(Kw)
                    CovWij = np.diag(1 / Diag_ij) + Kw
                    CovWij = Kw - Kw @ (inv(CovWij) @ Kw)
                    Sum_nj = 0
                    for k in range(self.q):
                        if k != j:
                            Sum_nj += mu_f[k].reshape(self.N)*np.array(muW[k,i,:])
                    tmp = ((new_y[i,:]-Sum_nj)*mu_f[j,:]) \
                                            / (self.yerr2[i,:] + jitt2[i])
                    sigma_w.append(CovWij)
                    mu_w.append(CovWij @ tmp)# / jitt2)
            sigma_w = np.array(sigma_w).reshape(self.q, self.p, self.N, self.N)
            mu_w = np.array(mu_w)
        return sigma_f, mu_f, sigma_w, mu_w


    def _expectedLogLike(self, nodes, weight, mean, jitter, sigma_f, mu_f,
                         sigma_w, mu_w):
        """
            Calculates the expected log-likelihood in mean-field inference, 
        corresponds to eq.14 in Nguyen & Bonilla (2013)
            Parameters:
                nodes = array of node functions 
                weight = weight function
                jitter = array of jitter terms
                sigma_f = array with the covariance for each node
                mu_f = array with the means for each node
                sigma_w = array with the covariance for each weight
                mu_w = array with the means for each weight
            Returns:
                expected log-likelihood
        """
        new_y = np.concatenate(self.y) - self._mean(mean, self.time)
        #NxP dimensional vector
        new_y = np.array(np.array_split(new_y, self.p)).T
        
        jitt2 = np.array(jitter)
        
        Ydiffyerr = np.zeros_like(self.yerr2)
        for i in range(self.p):
            Ydiffyerr[i,:] = self.yerr2[i,:] + jitt2[i]
            
        if self.q == 1:
            Wblk = np.array([])
            for n in range(self.N):
                for p in range(self.p):
                    Wblk = np.append(Wblk, mu_w[p,:,n])
            Fblk = np.array([])
            for n in range(self.N):
                for q in range(self.q):
                    for p in range(self.p):
                        Fblk = np.append(Fblk, mu_f[:, q, n])
            Ymean = Wblk * Fblk
            Ymean = Ymean.reshape(self.N,self.p)
            Ydiff = ((new_y - Ymean) * (new_y - Ymean)) #/Ydiffyerr.T
            #print('this is ydiff shape', Ydiff.shape, (Ydiff/Ydiffyerr.T).shape)
            logl = -0.5 * np.sum(Ydiff/Ydiffyerr.T)# /jitt2
            
            value = 0
            for i in range(self.p):
                for j in range(self.q):
                    value += np.sum((np.diag(sigma_f[j,:,:])*mu_w[i,j,:]*mu_w[i,j,:] +\
                                    np.diag(sigma_w[j,i,:,:])*mu_f[:,j,:]*mu_f[:,j,:] +\
                                    np.diag(sigma_f[j,:,:])*np.diag(sigma_w[j,i,:,:]))\
                                            / (self.yerr2[i,:] + jitt2[i]))
            logl += -0.5* value# /jitt2
            
        else:
            Wblk = []
            for p in range(self.p):
                Wblk.append([])
            for n in range(self.N):
                for p in range(self.p):
                    Wblk[p].append(mu_w[p, :, n])
            Wblk = np.array(Wblk).reshape(self.p, self.N * self.p)
            Fblk = []
            for p in range(self.p):
                Fblk.append([])
            for n in range(self.N):
                for q in range(self.q):
                    for p in range(self.p):
                        Fblk[q].append(mu_f[:, q, n])
            Fblk = np.array(Fblk).reshape(self.p, self.N * self.p)
            Ymean = np.sum((Wblk * Fblk).T, axis=1)
            Ymean = Ymean.reshape(self.N,self.p)
            Ydiff = ((new_y - Ymean) * (new_y - Ymean)) #/Ydiffyerr.T
            logl = -0.5 * np.sum(Ydiff/Ydiffyerr.T)
            value = 0
            for i in range(self.p):
                for j in range(self.q):
                    value += np.sum((np.diag(sigma_f[j,:,:])*mu_w[i,j,:]*mu_w[i,j,:] +\
                                    np.diag(sigma_w[j,i,:,:])*mu_f[:,j,:]*mu_f[:,j,:] +\
                                    np.diag(sigma_f[j,:,:])*np.diag(sigma_w[j,i,:,:]))\
                                            / (self.yerr2[i,:] + jitt2[i]))
            logl += -0.5* value
        return logl


    def _expectedLogPrior(self, nodes, weights, sigma_f, mu_f, sigma_w, mu_w):
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
        Kf = np.array([self._kernelMatrix(i, self.time) for i in nodes])
        Kw = np.array([self._kernelMatrix(j, self.time) for j in weights]) 
        
        #we have Q nodes -> j in the paper; we have P y(x)s -> i in the paper
        first_term = 0 #calculation of the first term of eq.15 of Nguyen & Bonilla (2013)
        second_term = 0 #calculation of the second term of eq.15 of Nguyen & Bonilla (2013)
        Lw = self._cholNugget(Kw[0])[0]
        Kw_inv = inv(Kw[0])
        #logKw = -self.q * np.sum(np.log(np.diag(L2)))
        logKw = -np.float(np.sum(np.log(np.diag(Lw))))
        mu_w = mu_w.reshape(self.q, self.p, self.N)
        
        for j in range(self.q):
            Lf = self._cholNugget(Kf[j])[0]
            logKf = -np.float(np.sum(np.log(np.diag(Lf))))
            Kf_inv = inv(Kf[j])
            muKmu = (Kf_inv @mu_f[:,j, :].reshape(self.N)) @mu_f[:,j, :].reshape(self.N)
            trace = np.trace(sigma_f[j] @Kf_inv)
            first_term += logKf -0.5*muKmu -0.5*trace
            for i in range(self.p):
                muKmu = (Kw_inv @mu_w[j,i])  @mu_w[j,i].T
                trace = np.trace(sigma_w[j, i, :, :] @Kw_inv)
                second_term += logKw -0.5*muKmu -0.5*trace
        return first_term + second_term


    def _entropy(self, sigma_f, sigma_w):
        """
            Calculates the entropy in mean-field inference, corresponds to 
        eq.14 in Nguyen & Bonilla (2013)
            Parameters:
                sigma_f = array with the covariance for each node
                sigma_w = array with the covariance for each weight
            Returns:
                ent_sum = final entropy
        """
        ent_sum = 0 #starts at zero then we sum everything
        for j in range(self.q):
            L1 = self._cholNugget(sigma_f[j])
            ent_sum += np.sum(np.log(np.diag(L1[0])))
            for i in range(self.p):
                L2 = self._cholNugget(sigma_w[j, i, :, :])
                ent_sum += np.sum(np.log(np.diag(L2[0])))
        return ent_sum


    def _plots(self, ELB, ELL, ELP, ENT):
        """
            Plots the evolution of the evidence lower bound, expected log 
        likelihood, expected log prior, and entropy
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


    def _prints(self, sum_ELB, ExpLogLike, ExpLogPrior, Entropy):
        """
            Prints the evidence lower bound, expected log likelihood, expected
        log prior, and entropy
        """
        print('ELBO: ' + str(float(sum_ELB)))
        print(' loglike: ' + str(float(ExpLogLike)) + ' \n logprior: ' \
              + str(ExpLogPrior) + ' \n entropy: ' + str(Entropy) + ' \n')
        return 0

###############################################################################

    def _OLDupdateSigmaMu(self, nodes, weight, means, jitter, 
                           muF, varF, muW, varW):
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
        new_y = np.concatenate(self.y) - self._mean(means)
        new_y = np.array(np.array_split(new_y, self.p))
#        #To standardize the ys
#        for i in range(self.p):
#            new_y[i,:] = new_y[i,:]/self.ystd[i]
            
#        error_term = np.sqrt(np.sum(np.array(jitters)**2)) / self.p
#        for i in range(self.p):
#            error_term += np.sqrt(np.sum(self.yerr[i,:]**2)) / (self.N)
        error_term = np.sqrt(np.sum(np.array(jitter)**2))
#        error_term = np.array(jitter**2)
        
        #kernel matrix for the nodes
        Kf = np.array([self._kernelMatrix(i, self.time) for i in nodes])
        #kernel matrix for the weights
        Kw = np.array([self._kernelMatrix(j, self.time) for j in weight]) 

        #we have Q nodes => j in the paper; we have P y(x)s => i in the paper
        if self.q == 1:
            sigma_f, mu_f = [], [] #creation of Sigma_fj and mu_fj
            for j in range(self.q):
                Diag_fj, tmp = 0, 0
                for i in range(self.p):
                    Diag_fj += muW[i, j, :] * muW[i, j, :] + varW[i, j, :]
                    Sum_nj = np.zeros(self.N)
                    for k in range(self.q):
                        if k != j:
                            muF = muF.T.reshape(1, self.q, self.N )
                            Sum_nj += muW[i, k, :] * muF[:, k,:].reshape(self.N)
                    tmp += (new_y[i][:] - Sum_nj) * muW[i, j, :]
                CovF = np.diag(error_term / Diag_fj) + Kf[j]
                #print(CovF)
                CovF = Kf[j] - Kf[j] @ (inv(CovF) @ Kf[j])
                sigma_f.append(CovF)
                mu_f.append(CovF @ tmp / error_term)
            sigma_f = np.array(sigma_f)
            mu_f = np.array(mu_f)
            sigma_w, mu_w = [], [] #creation of Sigma_wij and mu_wij
            for i in range(self.p):
                for j in range(self.q):
                    mu_fj = mu_f[j]
                    var_fj = np.diag(sigma_f[j])
                    Diag_ij = mu_fj * mu_fj + var_fj
                    Kw = np.squeeze(Kw)
                    CovWij = np.diag(error_term / Diag_ij) + Kw
                    CovWij = Kw - Kw @ (inv(CovWij) @ Kw)
                    Sum_nj = 0
                    for k in range(self.q):
                        if k != j:
                            Sum_nj += mu_f[k].reshape(self.N) * np.array(muW[i, k, :])
                    tmp = (new_y[i][:] - Sum_nj) * mu_f[j,:]
                    sigma_w.append(CovWij)
                    mu_w.append(CovWij @ tmp / error_term)
            sigma_w = np.array(sigma_w).reshape(self.q, self.p, self.N, self.N)
            mu_w = np.array(mu_w)
        else:
            muF = np.squeeze(muF)
            sigma_f, mu_f = [], [] #creation of Sigma_fj and mu_fj
            for j in range(self.q):
                Diag_fj, tmp = 0, 0
                for i in range(self.p):
                    Diag_fj += muW[j, i, :] * muW[j, i, :] + varW[j, i, :]
                    Sum_nj = np.zeros(self.N)
                    for k in range(self.q):
                        if k != j:
                            Sum_nj += muW[k, i, :] * muF[k,:].reshape(self.N)
                    tmp += (new_y[i][:] - Sum_nj) * muW[j, i, :]
                CovF = np.diag(error_term / Diag_fj) + Kf[j]
                CovF = Kf[j] - Kf[j] @ (inv(CovF) @ Kf[j])
                sigma_f.append(CovF)
                mu_f.append(CovF @ tmp / error_term)
                muF = np.array(mu_f)
            sigma_f = np.array(sigma_f)
            mu_f = np.array(mu_f)
            sigma_w, mu_w = [], [] #creation of Sigma_wij and mu_wij
            for j in range(self.q):
                for i in range(self.p):
                    mu_fj = mu_f[j]
                    var_fj = np.diag(sigma_f[j])
                    Diag_ij = mu_fj * mu_fj + var_fj
                    Kw = np.squeeze(Kw)
                    CovWij = np.diag(error_term / Diag_ij) + Kw
                    CovWij = Kw - Kw @ (inv(CovWij) @ Kw)
                    Sum_nj = 0
                    for k in range(self.q):
                        if k != j:
                            Sum_nj += mu_f[k].reshape(self.N) * np.array(muW[k, i, :])
                    tmp = (new_y[i][:] - Sum_nj) * mu_f[j,:]
                    sigma_w.append(CovWij)
                    mu_w.append(CovWij @ tmp / error_term)
                    muW[j,i,:] = np.array(CovWij @ tmp / error_term)
            sigma_w = np.array(sigma_w).reshape(self.q, self.p, self.N, self.N)
            mu_w = np.array(mu_w)
        return sigma_f, mu_f, sigma_w, mu_w


    def _OLDexpectedLogLike(self, nodes, weight, means, jitter, 
                             sigma_f, mu_f, sigma_w, mu_w):
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
        new_y = np.concatenate(self.y) - self._mean(means, self.time)
        new_y = np.array(np.array_split(new_y, self.p)).T #NxP dimensional vector
#        #To standardize the ys
#        for i in range(self.p):
#            new_y[i,:] = new_y[i,:]/self.ystd[i]
        
#        error_term = np.sqrt(np.sum(np.array(jitters)**2)) / self.p
#        for i in range(self.p):
#            error_term += np.sqrt(np.sum(self.yerr[i,:]**2)) / (self.N)
#        error_term = error_term
        error_term = np.sqrt(np.sum(np.array(jitter)**2))
 #       error_term = np.array(jitter**2)
        
        if self.q == 1:
            Wblk = np.array([])
            for n in range(self.N):
                for p in range(self.p):
                    Wblk = np.append(Wblk, mu_w[p,:,n])
            Fblk = np.array([])
            for n in range(self.N):
                for q in range(self.q):
                    for p in range(self.p):
                        Fblk = np.append(Fblk, mu_f[:, q, n])
            Ymean = Wblk * Fblk
            Ymean = Ymean.reshape(self.N,self.p)
            Ydiff = (new_y - Ymean) * (new_y - Ymean)
            logl = -0.5 * np.sum(Ydiff) / error_term
            
            value = 0
            for i in range(self.p):
                for j in range(self.q):
                    value += np.sum(np.diag(sigma_f[j,:,:]) * mu_w[i,j,:] * mu_w[i,j,:]) +\
                        np.sum(np.diag(sigma_w[j,i,:,:]) * mu_f[:,j,:] * mu_f[:,j,:]) +\
                        np.sum(np.diag(sigma_f[j,:,:]) * np.diag(sigma_w[j,i,:,:]))
            logl += -0.5* value / error_term

        else:
            Wblk = []
            for p in range(self.p):
                Wblk.append([])
            for n in range(self.N):
                for p in range(self.p):
                    Wblk[p].append(mu_w[p, :, n])
            Wblk = np.array(Wblk).reshape(self.p, self.N * self.p)
            Fblk = []
            for p in range(self.p):
                Fblk.append([])
            for n in range(self.N):
                for q in range(self.q):
                    for p in range(self.p):
                        Fblk[q].append(mu_f[:, q, n])
            Fblk = np.array(Fblk).reshape(self.p, self.N * self.p)
            Ymean = np.sum((Wblk * Fblk).T, axis=1)
            Ymean = Ymean.reshape(self.N,self.p)
            Ydiff = (new_y - Ymean) * (new_y - Ymean)
            logl = -0.5 * np.sum(Ydiff) / error_term
            value = 0
            for i in range(self.p):
                for j in range(self.q):
                    value += np.sum(np.diag(sigma_f[j,:,:]) * mu_w[j,i,:] * mu_w[j,i,:]) +\
                        np.sum(np.diag(sigma_w[j,i,:,:]) * mu_f[:,j,:] * mu_f[:,j,:]) +\
                        np.sum(np.diag(sigma_f[j,:,:]) * np.diag(sigma_w[j,i,:,:]))
            logl += -0.5* value / error_term
        return logl


    def _OLDexpectedLogPrior(self, nodes, weights, sigma_f, mu_f, sigma_w, mu_w):
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
        Kf = np.array([self._kernelMatrix(i, self.time) for i in nodes])
        Kw = np.array([self._kernelMatrix(j, self.time) for j in weights]) 
        
        #we have Q nodes -> j in the paper; we have P y(x)s -> i in the paper
        first_term = 0 #calculation of the first term of eq.15 of Nguyen & Bonilla (2013)
        second_term = 0 #calculation of the second term of eq.15 of Nguyen & Bonilla (2013)
        Lw = self._cholNugget(Kw[0])[0]
        Kw_inv = inv(Kw[0])
        #logKw = -self.q * np.sum(np.log(np.diag(L2)))
        logKw = -np.float(np.sum(np.log(np.diag(Lw))))
        mu_w = mu_w.reshape(self.q, self.p, self.N)
        
        for j in range(self.q):
            Lf = self._cholNugget(Kf[j])[0]
            logKf = -np.float(np.sum(np.log(np.diag(Lf))))
            Kf_inv = inv(Kf[j])
            muKmu = (Kf_inv @mu_f[:,j, :].reshape(self.N)) @mu_f[:,j, :].reshape(self.N)
            trace = np.trace(sigma_f[j] @Kf_inv)
            first_term += logKf -0.5*muKmu -0.5*trace
            for i in range(self.p):
                muKmu = (Kw_inv @mu_w[j,i])  @mu_w[j,i].T
                trace = np.trace(sigma_w[j, i, :, :] @Kw_inv)
                second_term += logKw -0.5*muKmu -0.5*trace
        return first_term + second_term


    def _OLDentropy(self, sigma_f, sigma_w):
        """
            Calculates the entropy in mean-field inference, corresponds to 
        eq.14 in Nguyen & Bonilla (2013)
            Parameters:
                sigma_f = array with the covariance for each node
                sigma_w = array with the covariance for each weight
            Returns:
                ent_sum = final entropy
        """
        ent_sum = 0 #starts at zero then we sum everything
        for j in range(self.q):
            L1 = self._cholNugget(sigma_f[j])
            ent_sum += np.sum(np.log(np.diag(L1[0])))
            for i in range(self.p):
                L2 = self._cholNugget(sigma_w[j, i, :, :])
                ent_sum += np.sum(np.log(np.diag(L2[0])))
        return ent_sum