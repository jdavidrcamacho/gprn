#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from copy import copy

from gprn.nodeFunction import Linear as nodeL
from gprn.nodeFunction import Polynomial as nodeP
from gprn.weightFunction import Linear as weightL
from gprn.weightFunction import Polynomial as weightP

class complexGP(object):
    """ 
        Class to create our Gaussian process regression network. See Wilson et
    al. (2012) for more information on this framework.
        Parameters:
            nodes = latent noide functions f(x), f hat in the article
            weight = latent weight funtion w(x), as far as I understood
                            this is the same kernels for all nodes except for 
                            the "amplitude" that varies from node to node
            weight_values = array with the weight w11, w12, etc... size needs to 
                        be equal to the number of nodes times the number of 
                        components (self.q * self.p)
            means = array of means functions being used, set it to None if a 
                    model doesn't use it
            jitters = jitter value of each dataset
            time = time
            *args = the data (or components), it needs be given in order of
                data1, data1_error, data2, data2_error, etc...
    """ 
    def  __init__(self, nodes, weight, weight_values, means, jitters, time, 
                  *args):
        #node functions; f(x) in Wilson et al. (2012)
        self.nodes = np.array(nodes)
        #number of nodes being used; q in Wilson et al. (2012)
        self.q = len(self.nodes)
        #weight function; w(x) in Wilson et al. (2012)
        self.weight = weight
        #amplitudes of the weight function
        self.weight_values = np.array(weight_values)
        #mean functions
        self.means = np.array(means)
        #jitters
        self.jitters = np.array(jitters)
        #time
        self.time = time 
        #the data, it should be given as data1, data1_error, data2, ...
        self.args = args 
        #number of components of y(x); p in Wilson et al. (2012)
        self.p = int(len(self.args)/2)
        #total number of weights we will have
        self.qp =  self.q * self.p

        #to organize the data we now join everything
        self.tt = np.tile(time, self.p) #"extended" time
        self.y = [] 
        self.yerr = []
        for i,j  in enumerate(args):
            if i%2 == 0:
                self.y.append(j)
            else:
                self.yerr.append(j**2)
        self.y = np.array(self.y) #"extended" measurements
        self.yerr = np.array(self.yerr) #"extended" errors
        #check if the input was correct
        assert self.means.size == self.p, \
        'The numbers of means should be equal to the number of components'
        assert (i+1)/2 == self.p, \
        'Given data and number of components dont match'

    def _kernel_matrix(self, kernel, time = None):
        """
            Returns the covariance matrix created by evaluating a given kernel 
        at inputs time.
        """
        #if time is None we use the initial time of our complexGP
        r = time[:, None] - time[None, :] if time.any() else self.time[:, None] - self.time[None, :]
        #to deal with the non-stationary kernels problem
        if isinstance(kernel, (nodeL, nodeP, weightL, weightP)):
            K = kernel(None, time[:, None], time[None, :])
        else:
            K = kernel(r)
        return K

    def _predict_kernel_matrix(self, kernel, tstar):
        """
            To be used in predict_gp()
        """
        if isinstance(kernel, (nodeL, nodeP, weightL, weightP)):
            K = kernel(None, tstar[:, None], self.time[None, :])
        else:
            r = tstar[:, None] - self.time[None, :]
            K = kernel(r)
        return K

    def _kernel_pars(self, kernel):
        """
            Returns a given kernel hyperparameters
        """
        return kernel.pars


##### mean functions
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


##### marginal likelihood functions
    def _covariance_matrix(self, nodes, weight, weight_values, time, position_p):
        """ 
            Creates the smaller matrices that will be used in a big final matrix
            Parameters:
                node = the node functions f(x) (f hat)
                weight = the weight funtion w(x)
                weight_values = array with the weights of w11, w12, etc... 
                time = time 
                position_p = position necessary to use the correct node
                                and weight
            Return:
                k_ii = block matrix in position ii
        """
        #block matrix starts empty
        k_ii = np.zeros((time.size, time.size))
        for i in range(1,self.q + 1):
            #hyperparameteres of the kernel of a given position
            nodePars = self._kernel_pars(nodes[i - 1])
            #all weight function will have the same parameters
            weightPars = weight.pars
            #except for the amplitude
            weightPars[0] =  weight_values[i-1 + self.q*(position_p-1)]
            #node and weight functions kernel
            if isinstance(weight, (nodeL, nodeP, weightL, weightP)):
                w_xa = type(self.weight)(*weightPars)(None, time[:,None], np.zeros(time.size))
                f_hat = self._kernel_matrix(type(self.nodes[i - 1])(*nodePars),time)
                w_xw = type(self.weight)(*weightPars)(None, np.zeros(time.size), time[None,:])
            else:
                w_xa = type(self.weight)(*weightPars)(time[:,None])
                f_hat = self._kernel_matrix(type(self.nodes[i - 1])(*nodePars),time)
                w_xw = type(self.weight)(*weightPars)(time[None,:])
            #now we add all the necessary stuff; eq. 4 of Wilson et al. (2012)
            k_ii = k_ii + (w_xa * f_hat * w_xw)
        
        return k_ii

    def compute_matrix(self, nodes, weight, weight_values, time, 
                       nugget = False, shift = False):
        """
            Creates the big covariance matrix K that will be used in the 
        log marginal likelihood calculation
            Parameters:
                nodes = the latent noide functions f(x) (f hat)
                weight = the latent weight function w(x)
                weight_values = array with the weights of w11, w12, etc... 
                time = time  
                nugget = True if K is not positive definite, False otherwise
                shift = True if K is not positive definite, False otherwise
            Returns:
                K = final covariance matrix 
        """
        #columns and lines size of the "final matrix"
        K_size = self.time.size * self.p
        #initially our "final matrix" K is empty
        K_start = np.zeros((K_size, K_size))
        #now we calculate the block matrices to be added to K
        for i in range(1, self.p+1):
            k = self._covariance_matrix(nodes, weight, weight_values, self.time, 
                         position_p = i)
            K_start[(i-1)*self.time.size : (i)*self.time.size, 
                        (i-1)*self.time.size : (i)*self.time.size] = k
        #addition of the measurement errors
        diag = np.concatenate(self.yerr) * np.identity(self.time.size * self.p)
        K = K_start + diag
        #more "weight" to the diagonal to avoid a ill-conditioned matrix
        if nugget:
            nugget_value = 0.01
            K = (1 - nugget_value)*K + nugget_value*np.diag(np.diag(K))
        #shifting all the eigenvalues up by the positive scalar to avoid a ill-conditioned matrix
        if shift:
            shift = 0.01
            K = K + shift*np.identity(self.time.size * self.p)
        return K

    def log_likelihood(self, nodes, weight, weight_values, means):
        """ 
            Calculates the marginal log likelihood. This version creates a big
        covariance matrix K made of block matrices of each dataset and then
        calculates just one log-likelihood.
            See Rasmussen & Williams (2006), page 113.
            Parameters:
                nodes = the node functions f(x) (f hat)
                weight = the weight funtion w(x)
                weight_values = array with the weights of w11, w12, etc... 
                means = mean function being used
            Returns:
                log_like  = Marginal log likelihood
        """
        #calculation of the covariance matrix
        K = self.compute_matrix(nodes, weight, weight_values, self.time)
        
        #calculation of the means
        yy = np.concatenate(self.y)
        means = self.means
        yy = yy - self._mean(means)
        #log marginal likelihood calculation
        try:
            L1 = cho_factor(K, overwrite_a=True, lower=False)
            log_like = - 0.5*np.dot(yy.T, cho_solve(L1, yy)) \
                       - np.sum(np.log(np.diag(L1[0]))) \
                       - 0.5*yy.size*np.log(2*np.pi)
        except LinAlgError:
            return -np.inf
        return log_like


    def new_log_like(self, nodes, weight, weight_values, means, jitters):
        """ 
            Calculates the marginal log likelihood for a GPRN. The main 
        difference is that it sums the log likelihoods of each dataset instead 
        of making a big covariance matrix K to calculate it.
            See Rasmussen & Williams (2006), page 113.
            Parameters:
                nodes = the node functions f(x) (f hat)
                weight = the weight funtion w(x)
                weight_values = array with the weights of w11, w12, etc... 
                means = mean function being used
                jitters = jitter value of each dataset
            Returns:
                log_like  = Marginal log likelihood
        """
        #means
        yy = np.concatenate(self.y)
        yy = yy - self._mean(means) if means else yy
        new_y = np.array_split(yy, self.p)
        yy_err = np.concatenate(self.yerr)
        new_yyerr = np.array_split(yy_err, self.p)


        log_like = 0 #"initial" likelihood starts at zero to then add things
        #calculation of each log-likelihood
        for i in range(1, self.p+1):
            k_ii = np.zeros((self.time.size, self.time.size))
            for j in range(1,self.q + 1):
                #hyperparameteres of the kernel of a given position
                nodePars = self._kernel_pars(nodes[j - 1])
                #all weight function will have the same parameters
                weightPars = weight.pars
                #except for the amplitude
                weightPars[0] =  weight_values[j-1 + self.q*(i-1)]
                #node and weight functions kernel
                if isinstance(weight, (nodeL, nodeP, weightL, weightP)):
                    w_xa = type(self.weight)(*weightPars)(None, self.time[:,None], np.zeros(self.time.size))
                    f_hat = self._kernel_matrix(type(self.nodes[i - 1])(*nodePars), self.time)
                    w_xw = type(self.weight)(*weightPars)(None, np.zeros(self.time.size), self.time[None,:])
                else:
                    w_xa = type(self.weight)(*weightPars)(self.time[:,None])
                    f_hat = self._kernel_matrix(type(self.nodes[j - 1])(*nodePars), self.time)
                    w_xw = type(self.weight)(*weightPars)(self.time[None,:])
                #now we add all the necessary stuff; eq. 4 of Wilson et al. (2012)
                k_ii = k_ii + (w_xa * f_hat * w_xw)
            #k_ii = k_ii + diag(error) + diag(jitter)
            k_ii += (new_yyerr[i - 1]) * np.identity(self.time.size) \
                    + (self.jitters[i - 1]) * np.identity(self.time.size)
            #log marginal likelihood calculation
            try:
                L1 = cho_factor(k_ii, overwrite_a=True, lower=False)
                log_like += - 0.5*np.dot(new_y[i - 1].T, cho_solve(L1, new_y[i - 1])) \
                           - np.sum(np.log(np.diag(L1[0]))) \
                           - 0.5*new_y[i - 1].size*np.log(2*np.pi)
            except LinAlgError:
                return -np.inf
        return log_like


##### GP prediction funtions
    def predict_gp(self, nodes = None, weight = None, weight_values = None,
                   means = None, jitters= None, time = None, dataset = 1):
        """ 
            NOTE: NOT WORKING PROPERLY
            Conditional predictive distribution of the Gaussian process
            Parameters:
                time = values where the predictive distribution will be calculated
                nodes = the node functions f(x) (f hat)
                weight = the weight function w(x)
                weight_values = array with the weights of w11, w12, etc...
                means = list of means being used
                jitters = jitter of each dataset
                dataset = 1,2,3,... accordingly to the data we are using, 
                        1 represents the first y(x), 2 the second y(x), etc...
            Returns:
                y_mean = mean vector
                y_std = standard deviation vector
                y_cov = covariance matrix
        """
        print('Working with dataset {0}'.format(dataset))
        #Nodes
        nodes = nodes if nodes else self.nodes
        #Weights
        weight  = weight if weight else self.weight
        #Weight values
        weight_values = weight_values if weight_values else self.weight_values
        #means
        yy = np.concatenate(self.y)
        yy = yy - self._mean(means) if means else yy
        #Jitters
        jitters = jitters if jitters else self.jitters
        #Time
        time = time if time.any() else self.time

        new_y = np.array_split(yy, self.p)
        yy_err = np.concatenate(self.yerr)
        new_yerr = np.array_split(yy_err, self.p)

        #cov = k + diag(error) + diag(jitter)
        cov = self._covariance_matrix(nodes, weight, weight_values, 
                                      self.time, dataset)
        cov += (new_yerr[dataset - 1]) * np.identity(self.time.size) \
                    + (self.jitters[dataset - 1]) * np.identity(self.time.size)
        L1 = cho_factor(cov)
        sol = cho_solve(L1, new_y[dataset - 1])
        tshape = time[:, None] - self.time[None, :]

        k_ii = np.zeros((tshape.shape[0],tshape.shape[1]))
        for i in range(1,self.q + 1):
            #hyperparameteres of the kernel of a given position
            nodePars = self._kernel_pars(nodes[i - 1])
            #all weight function will have the same parameters
            weightPars = self._kernel_pars(weight)
            #except for the amplitude
            weightPars[0] =  weight_values[i-1 + self.q*(dataset - 1)]
            #node and weight functions kernel
            if isinstance(weight, (nodeL, nodeP, weightL, weightP)):
                w_xa = type(self.weight)(*weightPars)(None, time[:,None], 0)
                f_hat = self._predict_kernel_matrix(type(self.nodes[i - 1])(*nodePars),time)
                w_xw = type(self.weight)(*weightPars)(None, 0, self.time[None,:])
            else:
                w_xa = type(self.weight)(*weightPars)(time[:,None])
                f_hat = self._predict_kernel_matrix(type(self.nodes[i - 1])(*nodePars), time)
                w_xw = type(self.weight)(*weightPars)(self.time[None,:])
            #now we add all the necessary stuff; eq. 4 of Wilson et al. (2012)
            k_ii = k_ii + (w_xa * f_hat * w_xw)


        Kstar = k_ii
        Kstarstar = self._covariance_matrix(nodes, weight, weight_values, time, 
                                            dataset)

        new_mean = np.array_split(self._mean(means, time), self.p)
        y_mean = np.dot(Kstar, sol) + new_mean[dataset-1]#mean

        kstarT_k_kstar = []
        for i, e in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        y_var = np.diag(y_cov) #variance
        y_std = np.sqrt(y_var) #standard deviation
        return y_mean, y_std, y_cov

