#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
from scipy.linalg import inv, cho_factor, cho_solve, LinAlgError
from scipy.stats import multivariate_normal
from copy import copy

from gprn.nodeFunction import Linear as nodeL
from gprn.nodeFunction import Polynomial as nodeP
from gprn.weightFunction import Linear as weightL
from gprn.weightFunction import Polynomial as weightP
from gprn.weightFunction import WhiteNoise as WN

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
        #if time is None we use the initial time of the class MFI
        #r = time[:, None] - time[None, :] if time!=None else self.time[:, None] - self.time[None, :]
        r = time[:, None] - time[None, :]
        
        #to deal with the non-stationary kernels problem
        if isinstance(kernel, (nodeL, nodeP, weightL, weightP)):
            K = kernel(None, time[:, None], time[None, :])
        else:
            K = kernel(r)
        return K

    def _predict_kernel_matrix(self, kernel, time):
        """
            To be used in predict_gp()
        """
        size = [time]
        if isinstance(kernel, (nodeL, nodeP, weightL, weightP)):
            K = kernel(None, time[:, None], self.time[None, :])
        if isinstance(kernel, WN):
#            zeros = np.zeros([time.size, self.time.size])
#            nonzeros = np.vstack(10*np.diag(np.ones(time.size)), np.zeros([time.size, self.time.size-time.size]))
            K = 10*np.ones_like(self.time) #+ np.zeros([time.size, self.time.size]) 
#            K = zeros + nonzeros
#            print(K.shape)
        else:
            if len(size) == 1:
                r = time - self.time[None,:]
            else:
                r = time[:, None] - self.time[None, :]
            K = kernel(r)
        #print(kernel)
#        print(K)
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
        
        position = 0 #we start filling CB at position (0,0)
        #first we enter the nodes
        for i in range(self.q):
            node_CovMatrix = self._kernel_matrix(nodes[i], time)
            CB[position:position+time.size, position:position+time.size] = node_CovMatrix
            position += time.size
        weight_CovMatrix = self._kernel_matrix(weight, time)
        #then we enter the weights
        for i in range(self.qp):
            CB[position:position+time.size, position:position+time.size] = weight_CovMatrix
            position += time.size
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

    #def _u_to_fhatw(self, nodes, weight, time):
    def _u_to_fhatw(self, u):
        """
            Given a list, divides it in the correspondinf nodes f and
        weights W parts.
            Parameters:
                u = array
            Returns:
                f = array with the samples of the nodes
                W = array with the samples of the weights
        """
        f = u[:self.q * self.N].reshape((self.q, 1, self.N))
        W = u[self.q * self.N:].reshape((self.p, self.q, self.N))
        return f, W


    def u_to_fW(self, nodes, weight, time):
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


    def get_y(self, n, w, time):
        # obscure way to do it
        y = np.einsum('ij...,jk...->ik...', w, n).reshape(self.p, time.size)
        return y

    def _cholNugget(self, matrix, maximum=10):
        """
            Returns the cholesky decomposition to a given matrix, if this matrix
        is not positive definite, a nugget is added to its diagonal.
            Parameters:
                matrix = matrix to decompose
                maximum = number of times a nugget is added.
            Returns:
                L = Matrix containing the Cholesky factor
        """
        try:
            L =  cho_factor(matrix, overwrite_a=True, lower=False)
        except LinAlgError:
            nugget = np.abs(np.diag(matrix).mean()) * 1e-5 #nugget to add to the diagonal
            n = 1 #number of tries
            while n <= maximum:
                print ('n:',n, ', nugget:', nugget)
                try:
                    L =  cho_factor(matrix + nugget, overwrite_a=True, lower=False)
                except LinAlgError:
                    nugget *= 10
                finally:
                    n += 1
            raise LinAlgError("Still not positive definite, even with nugget.")
        return L[0]

    def _cholShift(self, matrix, maximum=10):
        """
            Returns the cholesky decomposition to a given matrix, if this matrix
        is not positive definite, we shift all the eigenvalues up by the 
        positive scalar to avoid a ill-conditioned matrix
            Parameters:
                matrix = matrix to decompose
                maximum = number of times a shift is added.
            Returns:
                L = matrix containing the Cholesky factor
        """
        try:
            L =  cho_factor(matrix, overwrite_a=True, lower=False)
        except LinAlgError:
            shift = 1e-3 #shift to add to the diagonal
            n = 1 #number of tries
            while n <= maximum:
                print ('n:', n, ', shift:', shift)
                try:
                    L =  cho_factor(matrix + shift*np.identity(self.time.size), 
                                    overwrite_a=True, lower=False)
                except LinAlgError:
                    shift *= 10
                finally:
                    n += 1
            raise LinAlgError("Still not positive definite, even with a shift in eigenvalues.")
        return L[0]

##### Mean-Field Inference functions
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
        yy = np.concatenate(self.y)
        yy = yy - self._mean(means) if means else yy
        new_y = np.array_split(yy, self.p)
        new_y = self.y
        
        #the nodes can be different
        Kf = np.array([self._kernel_matrix(i, time) for i in nodes])
        invKf = []
        for i in range(self.q):
            invKf.append(inv(Kf[i]))
        invKf = np.array(invKf)
        #but we will have equal weights for all nodes
        Kw = np.array([self._kernel_matrix(j, time) for j in weight]) 
        invKw = []
        for i,j in enumerate(Kw):
            invKw = inv(j)
        
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
##                        print('sum muWmuF', sum_muWmuF)
                    sum_YminusSum += new_y[i][:] - sum_muWmuF
                sum_YminusSum *= muW[i][j][:]
            mu_f.append(np.dot(sigma_f[j], sum_YminusSum)/error_term)
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
                mu_w.append(np.dot(sigma_w[j][i], sum_YminusSum)/error)
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
        import matplotlib.pylab as plt

        q = self.q #number of nodes
        p = self.p #number of outputs
        
        ent_sum = 0 #starts at zero then we sum everything
        for i in range(q):
#            print(sigma_f[i])
#            plt.imshow(sigma_f[i])
            L1 = self._cholNugget(sigma_f[i])
            ent_sum += np.sum(np.log(np.diag(L1)))
            for j in range(p):
                L2 = self._cholNugget(sigma_w[i][j])
                ent_sum += np.sum(np.log(np.diag(L2)))
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
        #this means we will have equal weights for all nodes
        Kw = np.array([self._kernel_matrix(j, self.time) for j in weights]) 
        
        #we have Q nodes -> j in the paper; we have P y(x)s -> i in the paper
        first_term = 0 #calculation of the first term of eq.15 of Nguyen & Bonilla (2013)
        second_term = 0 #calculation of the second term of eq.15 of Nguyen & Bonilla (2013)
        L2 = cho_factor(Kw[0], overwrite_a=True, lower=False)
        logKw = - self.q* np.sum(np.log(np.diag(L2[0])))
        mu_w = mu_w.reshape(self.q, self.p, self.N)
        for j in range(self.q):
            L1 = cho_factor(Kf[j], overwrite_a=True, lower=False)
            logKf = - self.q * np.sum(np.log(np.diag(L1[0])))
            muKmu = np.dot(mu_f[j].reshape(self.N), cho_solve(L1, mu_f[j].reshape(self.N)))
            trace = np.trace(cho_solve(L1, sigma_f[j]))
            first_term += logKf -0.5*muKmu -0.5*trace
            for i in range(self.p):
                muKmu = np.dot(mu_w[j,i], cho_solve(L2, mu_w[j,i]))
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
        yy = np.concatenate(self.y)
        yy = yy - self._mean(means, self.time) if means else yy
        new_y = np.array_split(yy, self.p) #Px1 dimensional vector
        new_y = np.array(new_y)
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
                first = np.dot(muw[i][j], np.dot(sigma_f[j][:][:], muw[i,j,:].T))
                second = np.dot(mu_f[j], np.dot(sigma_w[j][i][:], mu_f[j].T))
                third = np.trace(np.dot(sigma_f[j][:][:], sigma_w[j][i][:]))
                error = np.sum(jitters[i]**2) + np.sum(self.yerr[i,:]**2)
                third_term += (first + second[0][0] + third) / error
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
                muF = array with the initial means for each node
                varF = array with the initial variance for each node
                muW = array with the initial means for each weight
                varW = array with the initial variance for each weight
                iternum = number of iterations
                prints = True to print ELB value at each iteration
                plots = True to plot ELB evolution 
            Returns:
                sum_ELB = Evidence lower bound
                muF = array with the new means for each node
                varF = array with the new variance for each node
                muW = array with the new means for each weight
                varW = array with the new variance for each weight
        """ 
        #Initial variational parameters
        D = self.time.size * self.q *(self.p+1);
        mu = np.random.randn(D,1);
        var = np.random.rand(D,1);
        #experiment
        np.random.seed(100)
        mu = np.random.rand(D,1);
        np.random.seed(200)
        var = np.random.rand(D,1);
        
        muF, muW = self._u_to_fhatw(mu)
        varF, varW = self._u_to_fhatw(var)
#        print('sigma y', jitters[0]**2)
        iterNumber = 0
        ELB = [0]
        if plots:
            ENT, ELP, ELL = [0], [0], [0]
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
            
            #Entropy
            Entropy = self._mfi_entropy(sigmaF, sigmaW)
            #Expected log prior
            ExpLogPrior = self._mfi_expectedLogPrior(nodes, weight, 
                                                sigmaF, muF,  sigmaW, muW)
            #Expected log-likelihood
            ExpLogLike = self._mfi_expectedLogLike(nodes, weight, means, jitters,
                                                   sigmaF, muF, sigmaW, muW)
            if plots:
                ENT.append(Entropy)
                ELL.append(ExpLogLike)
                ELP.append(ExpLogPrior)
            
            #Evidence Lower Bound
            sum_ELB = (ExpLogLike + ExpLogPrior + Entropy)
            if prints:
                print('ELB: {0}'.format(sum_ELB))
                print(' loglike: {0} \n logprior: {1} \n entropy {2} \n'.format(ExpLogLike, 
                                                                          ExpLogPrior, Entropy))
#            if np.abs(sum_ELB - ELB[-1]) < 1e-5:
            criteria = np.abs(np.mean(ELB[-10:]) - ELB[-1])
            if criteria < 1e-5 and criteria != 0 :
                if prints:
                    print('\nELB converged to {0}; algorithm stopped at iteration {1}'.format(sum_ELB,iterNumber))
                if plots:
                    ax1 = plt.subplot(411)
                    plt.plot(ELB[1:], '-')
                    plt.ylabel('Evidence lower bound')
                    plt.subplot(412, sharex=ax1)
                    plt.plot(ELL[1:-1], '-')
                    plt.ylabel('Expected log likelihood')
                    plt.subplot(413, sharex=ax1)
                    plt.plot(ELP[1:-1], '-')
                    plt.ylabel('Expected log prior')
                    plt.subplot(414, sharex=ax1)
                    plt.plot(ENT[1:-1], '-')
                    plt.ylabel('Entropy')
                    plt.xlabel('iteration')
                return sum_ELB, muF, muW
            ELB.append(sum_ELB)
            iterNumber += 1
        if plots:
            ax1 = plt.subplot(411)
            plt.plot(ELB[1:], '-')
            plt.ylabel('Evidence lower bound')
            plt.subplot(412, sharex=ax1)
            plt.plot(ELL[1:-1], '-')
            plt.ylabel('Expected log likelihood')
            plt.subplot(413, sharex=ax1)
            plt.plot(ELP[1:-1], '-')
            plt.ylabel('Expected log prior')
            plt.subplot(414, sharex=ax1)
            plt.plot(ENT[1:-1], '-')
            plt.ylabel('Entropy')
            plt.xlabel('iteration')
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
        
#        print(Kfstar.shape, Kwstar.shape)
#        print(muF.shape, muW.shape)
        Kf = np.array([self._kernel_matrix(i, self.time) for i in nodes])
        invKf = []
        for i in range(self.q):
            invKf.append(inv(Kf[i]))
        invKf = np.array(invKf)
        Kw = np.array([self._kernel_matrix(j, self.time) for j in weights])
        invKw = []
        for i, j in enumerate(Kw):
            invKw = inv(j)
        
        ystar = []
        for n in range(tstar.size):
            Kfstar = np.array([self._predict_kernel_matrix(i, tstar[n]) for i in nodes])
            Kwstar = np.array([self._predict_kernel_matrix(j, tstar[n]) for j in weights])
            wstar = [] #np.zeros([self.p, self.q]) #PxQ matrix
            fstar = [] #np.zeros([self.q, 1]) #Qx1 matrix
            for q in range(self.q):
#                print(Kfstar[q][:][:].shape, invKf[q].shape, muF[q].shape)
                fstar.append(np.dot(np.dot(Kfstar[q][:][:], invKf[q]), muF[q].T))
                for p in range(self.p):
#                    print(Kwstar[0][:][:].shape, invKw.shape, muW.shape)
                    muW = muW.reshape(self.p, self.N)
                    wstar.append(np.dot(np.dot(Kwstar[0][:][:], invKw), muW[p][:].T))
                    
            fstar = np.array(fstar[0][0])#.reshape(self.q, tstar.size)
            wstar = np.array(wstar)#.reshape(self.p, tstar.size)
            ystar.append(np.dot(wstar, fstar.T))
        ystar = np.array(ystar).T.reshape(self.p, tstar.size)

        ystar = np.concatenate(ystar)
        ystar = ystar + self._mean(means, tstar) if means else ystar
        ystar = np.array_split(ystar, self.p)

        return ystar
