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


##### To create matrices and samples
    def _kernel_matrix(self, kernel, time = None):
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

    def _predict_kernel_matrix(self, kernel, time):
        """
            To be used in predict_gp()
        """
        if isinstance(kernel, (covL, covP)):
            K = kernel(None, time[:, None], self.time[None, :])
        if isinstance(kernel, covWN):
            K = 0*np.ones_like(self.time) 
        else:
            r = time[:, None] - self.time[None, :]
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
        CB_size = time.size * self.q * (self.p + 1)
        CB = np.zeros((CB_size, CB_size)) #initial empty matrix
        
        pos = 0 #we start filling CB at position (0,0)
        #first we enter the nodes
        for i in range(self.q):
            node_CovMatrix = self._kernel_matrix(nodes[i], time)
            CB[pos:pos+time.size, pos:pos+time.size] = node_CovMatrix
            pos += time.size
        weight_CovMatrix = self._kernel_matrix(weight, time)
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
        norm = multivariate_normal(mean, cov, allow_singular=True)
        return norm.rvs()

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
        f = u[:self.q * self.N].reshape((self.q, 1, self.N))
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
        fhat = u[:self.q * time.size].reshape((self.q, 1, time.size))
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
        print('ELB: ' + str(sum_ELB))
        print(' loglike: ' + str(ExpLogLike) + ' \n logprior: ' \
              + str(ExpLogPrior) + ' \n entropy: ' + str(Entropy) + ' \n')
        return 0


##### Mean-Field Inference functions ###########################################
    def _update_SIGMAandMU(self, nodes, weight, means, jitters,  time,
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
        yy = np.concatenate(self.y) - self._mean(means)
        new_y = np.array_split(yy, self.p)
        
        #kernel matrix for the nodes
        Kf = np.array([self._kernel_matrix(i, time) for i in nodes])
        invKf = []
        for i in range(self.q):
            invKf.append(inv(Kf[i]))
        invKf = np.array(invKf) #inverse matrix of Kf
        #kernel matrix for the weights
        Kw = np.array([self._kernel_matrix(j, time) for j in weight]) 
        invKw = []
        for i,j in enumerate(Kw):
            invKw = inv(j)
        invKw = np.array(invKw) #inverse matrix of Kw
        
        #we have Q nodes => j in the paper; we have P y(x)s => i in the paper
        sigma_f = [] #creation of Sigma_fj
        for j in range(self.q):
            muWmuWVarW = np.zeros((self.N, self.N))
            for i in range(self.p):
                muWmuWVarW += np.diag(muW[i][j][:] * muW[i][j][:] + varW[i][j][:])
                error_term = np.sum(jitters[i]**2) + np.sum(self.yerr[i,:]**2)
            sigma_f.append(inv(invKf[j] + muWmuWVarW/error_term))
        sigma_f = np.array(sigma_f)

        mu_f = [] #creation of mu_fj
        for j in range(self.q):
            sum_YminusSum = np.zeros(self.N)
            for i in range(self.p):
                error_term = np.sum(jitters[i]**2) + np.sum(self.yerr[i,:]**2)
                sum_muWmuF = np.zeros(self.N)
                for k in range(self.q):
                    if k != j:
                        sum_muWmuF += np.array(muW[i][j][:]) * muF[j].reshape(self.N)
                    sum_YminusSum += new_y[i][:] - sum_muWmuF
                sum_YminusSum *= muW[i][j][:]
            mu_f.append(np.dot(sigma_f[j], sum_YminusSum/error_term))
        mu_f = np.array(mu_f)
        
        sigma_w = [] #creation of Sigma_wij
        for j in range(self.q):
            muFmuFVarF = np.zeros((self.N, self.N))
            for i in range(self.p):
                error_term = np.sum(jitters[i]**2) + np.sum(self.yerr[i,:]**2)
                muFmuFVarF += np.diag(mu_f[j] * mu_f[j] + np.diag(sigma_f[j]))
                sigma_w.append(inv(invKw + muFmuFVarF/error_term))
        sigma_w = np.array(sigma_w).reshape(self.q, self.p, self.N, self.N)
        
        mu_w = [] #creation of mu_wij
        for j in range(self.q):
            sum_YminusSum = np.zeros(self.N)
            for i in range(self.p):
                sum_muFmuW = np.zeros(self.N)
                for k in range(self.q):
                    if k != j:
                        sum_muFmuW += mu_f[j].reshape(self.N) * np.array(muW[i][j][:])
                    sum_YminusSum += new_y[i][:] - sum_muFmuW
                sum_YminusSum *= mu_f[j].reshape(self.N)
                error = np.sum(jitters[i]**2) + np.sum(self.yerr[i,:]**2)
                mu_w.append(np.dot(sigma_w[j][i], sum_YminusSum/error))
        mu_w = np.array(mu_w)
        return sigma_f, mu_f, sigma_w, mu_w


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
        q = self.q #number of nodes
        p = self.p #number of outputs
        
        ent_sum = 0 #starts at zero then we sum everything
        for i in range(q):
            L1 = self._cholNugget(sigma_f[i])
            ent_sum += np.sum(np.log(np.diag(L1[0])))
            for j in range(p):
                L2 = self._cholNugget(sigma_w[i][j])
                ent_sum += np.sum(np.log(np.diag(L2[0])))
        return ent_sum


    def _mfi_expectedLogPrior(self, nodes, weights, sigma_f, mu_f, sigma_w, mu_w):
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
        Kw = np.array([self._kernel_matrix(j, self.time) for j in weights]) 
        
        #we have Q nodes -> j in the paper; we have P y(x)s -> i in the paper
        first_term = 0 #calculation of the first term of eq.15 of Nguyen & Bonilla (2013)
        second_term = 0 #calculation of the second term of eq.15 of Nguyen & Bonilla (2013)
        Lw = self._cholNugget(Kw[0])[0]
        Kw_inv = inv(Kw[0])
        #logKw = -self.q * np.sum(np.log(np.diag(L2)))
        logKw = -np.sum(np.log(np.diag(Lw)))
        mu_w = mu_w.reshape(self.q, self.p, self.N)
        
        for j in range(self.q):
            Lf = self._cholNugget(Kf[j])[0]
            #logKf = - self.q * np.sum(np.log(np.diag(L1)))
            logKf = -np.sum(np.log(np.diag(Lf)))
            Kf_inv = inv(Kf[j])
            muKmu = (Kf_inv @mu_f[j].reshape(self.N)) @mu_f[j].reshape(self.N)
            trace = np.trace(sigma_f[j] @Kf_inv)
            first_term += logKf -0.5*muKmu -0.5*trace
            for i in range(self.p):
                muKmu = (Kw_inv @mu_w[j,i])  @mu_w[j,i].T
                trace = np.trace(sigma_w[j][i] @Kw_inv)
                second_term += logKw -0.5*muKmu -0.5*trace
        return first_term + second_term


    def _mfi_expectedLogPrior_old(self, nodes, weights, sigma_f, mu_f, sigma_w, mu_w):
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
        Kw = np.array([self._kernel_matrix(j, self.time) for j in weights]) 
        
        #we have Q nodes -> j in the paper; we have P y(x)s -> i in the paper
        first_term = 0 #calculation of the first term of eq.15 of Nguyen & Bonilla (2013)
        second_term = 0 #calculation of the second term of eq.15 of Nguyen & Bonilla (2013)
        L2 = cho_factor(Kw[0], overwrite_a=True, lower=True)
        logKw = - self.q* np.sum(np.log(np.diag(L2[0])))
        mu_w = mu_w.reshape(self.q, self.p, self.N)
        for j in range(self.q):
            L1 = cho_factor(Kf[j], overwrite_a=True, lower=True)
            logKf = - self.q * np.sum(np.log(np.diag(L1[0])))
            muKmu = mu_f[j].reshape(self.N) @ cho_solve(L1, mu_f[j].reshape(self.N))
            trace = np.trace(cho_solve(L1, sigma_f[j]))
            first_term += logKf -0.5*muKmu -0.5*trace
            for i in range(self.p):
                muKmu = mu_w[j,i] @ cho_solve(L2, mu_w[j,i])
                trace = np.trace(cho_solve(L2, sigma_w[j][i]))
                second_term += logKw -0.5*muKmu -0.5*trace
        return first_term + second_term


    def _mfi_expectedLogLike(self, nodes, weight, means, jitters, 
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
        yy = np.concatenate(self.y) - self._mean(means, self.time)
        new_y = np.array(np.array_split(yy, self.p)) #Px1 dimensional vector
        muw = mu_w.reshape(self.p, self.q, self.N) #PxQ dimensional vector
        muf = mu_f.reshape(self.q, self.N) #Qx1 dimensional vector
        
        first_term = 0
        second_term = 0
        third_term = 0
        for i in range(self.p):
            for n in range(self.N):
                error = np.sum(jitters[i]**2) + np.sum(self.yerr[i,n]**2)
                first_term += np.log(error)
                YOmegaMu = np.array(new_y[i,n].T - muw[i,:,n] * muf[:,n])
                second_term += np.dot(YOmegaMu.T, YOmegaMu)/ error
            for j in range(self.q):
                first = np.diag(sigma_f[j][:][:]) * muw[i][j] @ muw[i][j]
                second = np.diag(sigma_w[j][i][:]) * mu_f[j] @ mu_f[j].T
                third = np.diag(sigma_f[j][:][:]) @ np.diag(sigma_w[j][i][:])
                error = np.sum(jitters[i]**2) + np.sum(self.yerr[i,:]**2)
                third_term += (first + second[0][0] + third)/ error
        first_term = -0.5 * first_term
        second_term = -0.5 * second_term
        third_term = -0.5 * third_term
        return first_term + second_term + third_term


    def EvidenceLowerBound_MFI(self, nodes, weight, means, jitters, time, 
                               iterations=100, prints = False, plots = False):
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
        #Initial variational parameters
        D = self.time.size * self.q *(self.p+1);
        mu = np.random.randn(D,1);
        var = np.random.rand(D,1);
        
#        #experiment
#        np.random.seed(100)
#        mu = np.random.rand(D,1);
#        np.random.seed(200)
#        var = np.random.rand(D,1);
        
        muF, muW = self._fhat_and_w(mu)
        varF, varW = self._fhat_and_w(var)

        iterNumber = 0
        ELB = [0]
        if plots:
            ELP, ELL, ENT = [0], [0], [0]
        while iterNumber < iterations:
            sigmaF, muF, sigmaW, muW = self._update_SIGMAandMU(nodes, weight, means,
                                                               jitters, time,
                                                               muF, varF, muW, varW)
            muF = muF.reshape(self.q, 1, self.N) #new mean for the nodes
            varF =  []
            for i in range(self.q):
                varF.append(np.diag(sigmaF[i]))
            varF = np.array(varF).reshape(self.q, 1, self.N) #new variance for the nodes
            muW = muW.reshape(self.p, self.q, self.N) #new mean for the weights
            varW =  []
            for j in range(self.q):
                for i in range(self.p):
                    varW.append(np.diag(sigmaW[j][i]))
            varW = np.array(varW).reshape(self.p, self.q, self.N) #new variance for the weights
            
            #Expected log prior
            ExpLogPrior = self._mfi_expectedLogPrior(nodes, weight, 
                                                sigmaF, muF,  sigmaW, muW)
            #Expected log-likelihood
            ExpLogLike = self._mfi_expectedLogLike(nodes, weight, means, jitters,
                                                   sigmaF, muF, sigmaW, muW)
            #Entropy
            Entropy = self._mfi_entropy(sigmaF, sigmaW)
            if plots:
                ELL.append(ExpLogLike)
                ELP.append(ExpLogPrior)
                ENT.append(Entropy)
            
            #Evidence Lower Bound
            sum_ELB = (ExpLogLike + ExpLogPrior + Entropy)
            if prints:
                self._prints(sum_ELB, ExpLogLike, ExpLogPrior, Entropy)
            #Stoping criteria
            criteria = np.abs(np.mean(ELB[-10:]) - ELB[-1])
            if criteria < 1e-10 and criteria != 0 :
                if prints:
                    print('\nELB converged to ' +str(sum_ELB) \
                          + '; algorithm stopped at iteration ' +str(iterNumber))
                if plots:
                    self._plots(ELB[1:], ELL[1:-1], ELP[1:-1], ENT[1:-1])
                return sum_ELB, muF, muW
            ELB.append(sum_ELB)
            iterNumber += 1
        if plots:
            self._plots(ELB[1:], ELL[1:-1], ELP[1:-1], ENT[1:-1])
        return sum_ELB, muF, muW
        
        
    def Prediction_MFI(self, nodes, weights, means, jitters, tstar, muF, muW):
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
        ystar = np.zeros([self.p, tstar.size])
        
        Kf = np.array([self._kernel_matrix(i, self.time) for i in nodes])
        invKf = []
        for i in range(self.q):
            invKf.append(inv(Kf[i]))
        invKf = np.array(invKf)
        Kw = np.array([self._kernel_matrix(j, self.time) for j in weights])
        invKw = []
        for i, j in enumerate(Kw):
            invKw = inv(j)
        
#        ystar = []
#        for n in range(tstar.size):
        
        #Kfstar and Kwstar are arrays
        Kfstar = np.array([self._predict_kernel_matrix(i, tstar) for i in nodes])
        Kwstar = np.array([self._predict_kernel_matrix(j, tstar) for j in weights])

        Kfstarstar = np.array([self._kernel_matrix(i, tstar) for i in nodes])
        Kwstarstar = np.array([self._kernel_matrix(j, tstar) for j in weights])
        
        wstar = []
        fstar = []
        for q in range(self.q):
#            fstar.append(np.dot(np.dot(Kfstar[q][:][:], invKf[q]), muF[q].T))
            fstar.append((invKf[q] @muF[q].T).T @Kfstar[q][:][:].T)
        for p in range(self.p):
            muW = muW.reshape(self.p, self.N)
#            wstar.append(np.dot(np.dot(Kwstar[0][:][:], invKw), muW[p][:].T))
            wstar.append((invKw @muW[p][:].T) @Kwstar[0][:][:].T)

        #for now this will only work with one dataset, to be fixed the in future
        fstar = np.array(fstar[0][0])#.reshape(self.q, tstar.size)
        wstar = np.array(wstar[0])#.reshape(self.p, tstar.size)

#        print((wstar * fstar).shape)
        ystar = wstar * fstar
#        print(ystar.shape)
#        ystar = np.array(ystar).T.reshape(tstar.size)
#        ystar = np.concatenate(ystar)
#        ystar = (ystar - self._mean(means, tstar)) if means else ystar
        ystar = ystar + self._mean(means, tstar)
#        ystar = np.array_split(ystar, self.p)
        return ystar


##### Nonparametric Variational Inference functions ############################



##### Other functions ##########################################################
def jitChol(A, maxTries=10, warning=True):

    """Do a Cholesky decomposition with jitter.
    Description:
    U = jitChol(A, maxTries, warning) attempts a Cholesky
     decomposition on the given matrix, if matrix isn't positive
     definite the function adds 'jitter' and tries again. Thereafter
     the amount of jitter is multiplied by 10 each time it is added
     again. This is continued for a maximum of 10 times.  The amount of
     jitter added is returned.
     Returns:
      U - the Cholesky decomposition for the matrix.
     Arguments:
      A - the matrix for which the Cholesky decomposition is required.
      maxTries - the maximum number of times that jitter is added before
       giving up (default 10).
      warning - whether to give a warning for adding jitter (default is True)
    See also
    CHOL, PDINV, LOGDET
    Copyright (c) 2005, 2006 Neil D. Lawrence
    
    """
    warning = True
    jitter = 0
    i = 0
    while(True):
        try:
            # Try --- need to check A is positive definite
            if jitter == 0:
                jitter = abs(np.trace(A))/A.shape[0]*1e-6
                LC = cholesky(A, lower=True)
                return LC.T, jitter
            else:
                if warning:
                    # pdb.set_trace()
                    print("Adding jitter of %f in jitChol()." % jitter)
                LC = cholesky(A+jitter*np.eye(A.shape[0]), lower=True)
                return LC.T, jitter
        except LinAlgError:
            # Seems to have been non-positive definite.
            if i<maxTries:
                jitter = jitter*10
            else:
                raise LinAlgError("Matrix non positive definite, jitter of " \
                                  + str(jitter) + " added but failed after " \
                                  + str(i) + " trials.")
        i += 1

def _cholesky(A):
    """
        Source:
        https://rosettacode.org/wiki/Cholesky_decomposition
    """
    L = [[0.0] * len(A) for _ in range(len(A))]
    for i, (Ai, Li) in enumerate(zip(A, L)):
        for j, Lj in enumerate(L[:i+1]):
            s = sum(Li[k] * Lj[k] for k in range(j))
            Li[j] = np.sqrt(Ai[i] - s) if (i == j) else \
                      (1.0 / Lj[j] * (Ai[j] - s))
    return L

def _cholSimple(matrix):
    """
        Returns the cholesky decomposition to a given matrix.
        Parameters:
            matrix = matrix to decompose
        Returns:
            L = Matrix containing the Cholesky factor
    """
    nugget = 0 #because of all the tests
    try:
        L =  cholesky(matrix, lower=True)
        return L, nugget
    except LinAlgError:
        L = cholesky(matrix).T
        return L, nugget