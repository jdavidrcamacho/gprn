#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
from scipy.linalg import inv, cholesky, cho_factor, cho_solve, LinAlgError

from .gprn import GPRN


class mfi(object):
    """ 
        Class to perform mean field variational inference for GPRNs. 
        See Nguyen & Bonilla (2013) for more information.
    """ 
    def  __init__(self, GPRN):
        """
            To make mfi a daughter class of GPRN
        """
        self.GPRN = GPRN


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
        #Initial variational parameters
        D = self.GPRN.time.size * self.GPRN.q *(self.GPRN.p+1)
        mu = np.random.randn(D,1)
        var = np.random.rand(D,1)

        muF, muW = self.GPRN._fhat_and_w(mu)
        varF, varW = self.GPRN._fhat_and_w(var)

        iterNumber = 0
        ELB = [0]
        if plots:
            ELP, ELL, ENT = [0], [0], [0]
        while iterNumber < iterations:
            sigmaF, muF, sigmaW, muW = self._updateSigmaMu(nodes, weight, 
                                                               means, jitters, time,
                                                               muF, varF, muW, varW)
            muF = muF.reshape(self.GPRN.q, 1, self.GPRN.N) #new mean for the nodes
            varF =  []
            for i in range(self.GPRN.q):
                varF.append(np.diag(sigmaF[i]))
            varF = np.array(varF).reshape(self.GPRN.q, 1, self.GPRN.N) #new variance for the nodes
            muW = muW.reshape(self.GPRN.p, self.GPRN.q, self.GPRN.N) #new mean for the weights
            varW =  []
            for j in range(self.GPRN.q):
                for i in range(self.GPRN.p):
                    varW.append(np.diag(sigmaW[j][i]))
            varW = np.array(varW).reshape(self.GPRN.p, self.GPRN.q, self.GPRN.N) #new variance for the weights
            
            #Expected log prior
            ExpLogPrior = self._expectedLogPrior(nodes, weight, 
                                                sigmaF, muF,  sigmaW, muW)
            #Expected log-likelihood
            ExpLogLike = self._expectedLogLike(nodes, weight, means, jitters,
                                                   sigmaF, muF, sigmaW, muW)
            #Entropy
            Entropy = self._entropy(sigmaF, sigmaW)
            if plots:
                ELL.append(ExpLogLike)
                ELP.append(ExpLogPrior)
                ENT.append(Entropy)
            
            #Evidence Lower Bound
            sum_ELB = (ExpLogLike + ExpLogPrior + Entropy)
            ELB.append(sum_ELB)
            if prints:
                self._prints(sum_ELB, ExpLogLike, ExpLogPrior, Entropy)
            #Stoping criteria
            criteria = np.abs(np.mean(ELB[-5:]) - ELB[-1])
            if criteria < 1e-3 and criteria != 0 :
                if prints:
                    print('\nELB converged to ' +str(sum_ELB) \
                          + '; algorithm stopped at iteration ' \
                          +str(iterNumber) +'\n')
                if plots:
                    self._plots(ELB[1:], ELL[1:-1], ELP[1:-1], ENT[1:-1])
                return sum_ELB, muF, muW
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
        Kf = np.array([self.GPRN._kernelMatrix(i, self.GPRN.time) for i in nodes])
        invKf = np.array([inv(i) for i in Kf])
        Kw = np.array([self.GPRN._kernelMatrix(j, self.GPRN.time) for j in weights])
        invKw = np.array([inv(j) for j in Kw])

        #mean
        ystar = []
        for n in range(tstar.size):
            Kfstar = np.array([self.GPRN._predictKernelMatrix(i1, tstar[n]) for i1 in nodes])
            Kwstar = np.array([self.GPRN._predictKernelMatrix(i2, tstar[n]) for i2 in weights])
            Efstar, Ewstar = 0, 0
            for j in range(self.GPRN.q):
                Efstar += Kfstar[j] @(invKf[j] @muF[j].T) 
                for i in range(self.GPRN.p):
                    Ewstar += Kwstar[0] @(invKw[0] @muW[i][j].T)
            ystar.append(Ewstar@ Efstar)
        ystar = np.array(ystar).reshape(tstar.size) #final mean
        ystar += self.GPRN._mean(means, tstar) #adding the mean function

        #standard deviation
        Kfstar = np.array([self.GPRN._predictKernelMatrix(i, tstar) for i in nodes])
        Kwstar = np.array([self.GPRN._predictKernelMatrix(j, tstar) for j in weights])
        Kfstarstar = np.array([self.GPRN._kernelMatrix(i, tstar) for i in nodes])
        Kwstarstar = np.array([self.GPRN._kernelMatrix(j, tstar) for j in weights])
        
        #firstTerm = tstar.size x tstar.size matrix
        firstTermAux1 = (Kwstar[0] @invKw[0].T @muW[0].T).T @(Kwstar[0] @invKw[0] @muW[0].T)
        firstTermAux2 = Kfstarstar - (Kfstar[0] @invKf[0].T @Kfstar[0].T)
        firstTerm = np.array(firstTermAux1 * firstTermAux2).reshape(tstar.size, tstar.size)
        #secondTerm = tstar.size x tstar.size matrix
        secondTermAux1 = Kwstarstar - Kwstar[0] @invKw[0].T @Kwstar[0].T
        secondTermAux2 = firstTermAux2.reshape(tstar.size, tstar.size)
        secondTermAux3 = (Kfstar[0] @invKf[0].T @muF[0].T) @(Kfstar[0] @invKf[0].T @muF[0].T).T
        secondTerm = secondTermAux1[0] @(secondTermAux2 + secondTermAux3)
        
        errors = np.identity(tstar.size) * ((np.sum(jitters)/self.GPRN.p)**2 \
                            + (np.sum(self.GPRN.yerr[0,:])/self.GPRN.N)**2)
        total = firstTerm + secondTerm + errors
        stdstar = np.sqrt(np.diag(total)) #final standard deviation
        return ystar, stdstar

        
    def _updateSigmaMu(self, nodes, weight, means, jitters,  time,
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
        new_y = np.concatenate(self.GPRN.y) - self.GPRN._mean(means)
        new_y = np.array_split(new_y, self.GPRN.p)
        
        #kernel matrix for the nodes
        Kf = np.array([self.GPRN._kernelMatrix(i, time) for i in nodes])
        invKf = []
        for i in range(self.GPRN.q):
            invKf.append(inv(Kf[i]))
        invKf = np.array(invKf) #inverse matrix of Kf
        #kernel matrix for the weights
        Kw = np.array([self.GPRN._kernelMatrix(j, time) for j in weight]) 
        invKw = []
        for i,j in enumerate(Kw):
            invKw = inv(j)
        invKw = np.array(invKw) #inverse matrix of Kw
        
        #we have Q nodes => j in the paper; we have P y(x)s => i in the paper
        sigma_f = [] #creation of Sigma_fj
        for j in range(self.GPRN.q):
            muWmuWVarW = np.zeros((self.GPRN.N, self.GPRN.N))
            for i in range(self.GPRN.p):
                muWmuWVarW += np.diag(muW[i][j][:] * muW[i][j][:] + varW[i][j][:])
                error_term = (np.sum(jitters[i])/self.GPRN.p)**2 \
                            + (np.sum(self.GPRN.yerr[i,:])/self.GPRN.N)**2
            sigma_f.append(inv(invKf[j] + muWmuWVarW/error_term))
        sigma_f = np.array(sigma_f)

        muF = muF.reshape(self.GPRN.q, self.GPRN.N)
        mu_f = [] #creation of mu_fj
        for j in range(self.GPRN.q):
            sum_YminusSum = np.zeros(self.GPRN.N)
            for i in range(self.GPRN.p):
                error_term = (np.sum(jitters[i])/self.GPRN.p)**2 \
                            + (np.sum(self.GPRN.yerr[i,:])/self.GPRN.N)**2
                sum_muWmuF = np.zeros(self.GPRN.N)
                for k in range(self.GPRN.q):
                    if k != j:
                        sum_muWmuF += np.array(muW[i][j][:]) * muF[j].reshape(self.GPRN.N)
                    sum_YminusSum += new_y[i][:] - sum_muWmuF
                sum_YminusSum *= muW[i][j][:]
            mu_f.append(np.dot(sigma_f[j], sum_YminusSum/error_term))
        mu_f = np.array(mu_f)

        sigma_w = [] #creation of Sigma_wij
        for j in range(self.GPRN.q):
            muFmuFVarF = np.zeros((self.GPRN.N, self.GPRN.N))
            for i in range(self.GPRN.p):
                error_term = (np.sum(jitters[i])/self.GPRN.p)**2 \
                            + (np.sum(self.GPRN.yerr[i,:])/self.GPRN.N)**2
                muFmuFVarF += np.diag(mu_f[j] * mu_f[j] + np.diag(sigma_f[j]))
                sigma_w.append(inv(invKw + muFmuFVarF/error_term))
        sigma_w = np.array(sigma_w).reshape(self.GPRN.q, self.GPRN.p, self.GPRN.N, self.GPRN.N)
        
        mu_w = [] #creation of mu_wij
        for j in range(self.GPRN.q):
            sum_YminusSum = np.zeros(self.GPRN.N)
            for i in range(self.GPRN.p):
                sum_muFmuW = np.zeros(self.GPRN.N)
                for k in range(self.GPRN.q):
                    if k != j:
                        sum_muFmuW += mu_f[j].reshape(self.GPRN.N) * np.array(muW[i][j][:])
                    sum_YminusSum += new_y[i][:] - sum_muFmuW
                sum_YminusSum *= mu_f[j].reshape(self.GPRN.N)
                error = (np.sum(jitters[i])/self.GPRN.p)**2 \
                        + (np.sum(self.GPRN.yerr[i,:])/self.GPRN.N)**2
                mu_w.append(np.dot(sigma_w[j][i], sum_YminusSum/error))
        mu_w = np.array(mu_w)
        return sigma_f, mu_f, sigma_w, mu_w


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
        for i in range(self.GPRN.q):
            L1 = self.GPRN._cholNugget(sigma_f[i])
            ent_sum += np.sum(np.log(np.diag(L1[0])))
            for j in range(self.GPRN.p):
                L2 = self.GPRN._cholNugget(sigma_w[i][j])
                ent_sum += np.sum(np.log(np.diag(L2[0])))
        return ent_sum


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
        Kf = np.array([self.GPRN._kernelMatrix(i, self.GPRN.time) for i in nodes])
        Kw = np.array([self.GPRN._kernelMatrix(j, self.GPRN.time) for j in weights]) 
        
        #we have Q nodes -> j in the paper; we have P y(x)s -> i in the paper
        first_term = 0 #calculation of the first term of eq.15 of Nguyen & Bonilla (2013)
        second_term = 0 #calculation of the second term of eq.15 of Nguyen & Bonilla (2013)
        Lw = self.GPRN._cholNugget(Kw[0])[0]
        Kw_inv = inv(Kw[0])
        #logKw = -self.q * np.sum(np.log(np.diag(L2)))
        logKw = -np.float(np.sum(np.log(np.diag(Lw))))
        mu_w = mu_w.reshape(self.GPRN.q, self.GPRN.p, self.GPRN.N)
        
        for j in range(self.GPRN.q):
            Lf = self.GPRN._cholNugget(Kf[j])[0]
            #logKf = - self.q * np.sum(np.log(np.diag(L1)))
            logKf = -np.float(np.sum(np.log(np.diag(Lf))))
            Kf_inv = inv(Kf[j])
            muKmu = (Kf_inv @mu_f[j].reshape(self.GPRN.N)) @mu_f[j].reshape(self.GPRN.N)
            trace = np.trace(sigma_f[j] @Kf_inv)
            first_term += logKf -0.5*muKmu -0.5*trace
            for i in range(self.GPRN.p):
                muKmu = (Kw_inv @mu_w[j,i])  @mu_w[j,i].T
                trace = np.trace(sigma_w[j][i] @Kw_inv)
                second_term += logKw -0.5*muKmu -0.5*trace
        return first_term + second_term


    def _expectedLogLike(self, nodes, weight, means, jitters, 
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
        new_y = np.concatenate(self.GPRN.y) - self.GPRN._mean(means, self.GPRN.time)
        new_y = np.array(np.array_split(new_y, self.GPRN.p)) #Px1 dimensional vector
        muw = mu_w.reshape(self.GPRN.p, self.GPRN.q, self.GPRN.N) #PxQ dimensional vector
        muf = mu_f.reshape(self.GPRN.q, self.GPRN.N) #Qx1 dimensional vector
        
        first_term = 0
        second_term = 0
        third_term = 0
        for i in range(self.GPRN.p):
            for n in range(self.GPRN.N):
                error = np.sum(jitters[i]**2) + np.sum(self.GPRN.yerr[i,n]**2)
                first_term += np.log(error)
                YOmegaMu = np.array(new_y[i,n].T - muw[i,:,n] * muf[:,n])
                second_term += np.dot(YOmegaMu.T, YOmegaMu)/ error
            for j in range(self.GPRN.q):
                first = np.diag(sigma_f[j][:][:]) * muw[i][j] @ muw[i][j]
                second = np.diag(sigma_w[j][i][:]) * mu_f[j] @ mu_f[j].T
                third = np.diag(sigma_f[j][:][:]) @ np.diag(sigma_w[j][i][:])
                error = (np.sum(jitters[i])/self.GPRN.p)**2 \
                            + (np.sum(self.GPRN.yerr[i,:])/self.GPRN.N)**2
                third_term += (first + second[0][0] + third)/ error
        first_term = -0.5 * first_term
        second_term = -0.5 * second_term
        third_term = -0.5 * third_term
        return first_term + second_term + third_term


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