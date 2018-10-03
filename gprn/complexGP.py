#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from copy import copy

import matplotlib.pyplot as plt
from gprn.weightFunction import Linear

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
            time = time
            *args = the data, it should be given as data1, data1_error, etc...
    """ 
    def  __init__(self, nodes, weight, weight_values, means, time, *args):
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
        #if time is None we use the time of our complexGP
        if time is None:
            r = self.time[:, None] - self.time[None, :]
        #if we define a new time we will use that time
        else:
            r = time[:, None] - time[None, :]
        if isinstance(kernel, Linear):
            K = kernel(None, time[:, None], time[None, :])
        else:
            K = kernel(r)
        return K

    def _predict_kernel_matrix(self, kernel, tstar):
        """
            To be used in predict_gp()
        """
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

    def _mean(self, means):
        """
            Returns the values of the mean functions
        """
        N = self.time.size
        m = np.zeros_like(self.tt)
        for i, meanfun in enumerate(means):
            if meanfun is None:
                continue
            else:
                m[i*N : (i+1)*N] = meanfun(self.time)
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
            weightPars = self._kernel_pars(weight)
            #except for the amplitude
            weightPars[0] =  weight_values[i-1 + self.q*(position_p-1)]
            #node and weight functions kernel
            w_xa = type(self.weight)(*weightPars)(time[:,None])
            f_hat = self._kernel_matrix(type(self.nodes[i - 1])(*nodePars),time)
            w_xw = type(self.weight)(*weightPars)(time[None,:])
            #now we add all the necessary stuff; eq. 4 of Wilson et al. (2012)
            k_ii = k_ii + (w_xa * f_hat * w_xw)
#            print(weightPars[0], self.weight,'*',self.nodes[i - 1],'*',self.weight)
#        print()
        return k_ii

    def compute_matrix(self, nodes, weight,weight_values, time, 
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
#        plt.imshow(K)
        return K

    def log_likelihood(self, nodes, weight, weight_values, means):
        """ 
            Calculates the marginal log likelihood.
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


##### GP prediction funtions
    def predict_gp(self, nodes = None, weight = None, weight_values = None,
                   means = None, time = None, dataset = 1):
        """ 
            NOTE: NOT WORKING PROPERLY
            Conditional predictive distribution of the Gaussian process
            Parameters:
                time = values where the predictive distribution will be calculated
                nodes = the node functions f(x) (f hat)
                weight = the weight function w(x)
                weight_values = array with the weights of w11, w12, etc...
                means = list of means being used 
                dataset = 1,2,3,... accordingly to the data we are using, 
                        1 represents the first y(x), 2 the second y(x), etc...
            Returns:
                y_mean = mean vector
                y_std = standard deviation vector
                y_cov = covariance matrix
        """
        print('Working with dataset {0}'.format(dataset))
        #Nodes
        if nodes:
            #To use a new node function
            pass
        else:
            #To use the one we defined at start
            nodes = self.nodes
        #Weights
        if weight:
            #To use a new weight function
            pass
        else:
            #To use the one we defined at start
            weight = self.weight
        #Weight values
        if weight:
            #To use new weight values
            pass
        else:
            #To use the one we defined at start
            weight_values = self.weight_values
        #Means
        if means:
            #To use the new defined mean
            yy = np.concatenate(self.y)
            yy =  yy - self._mean(means)
        else:
            #to not use a mean
            yy = np.concatenate(self.y)
        #Time
        if time.any():
            #To use a new time
            pass
        else:
            #to use the one defined earlier
            time = self.time

        new_y = np.array_split(yy, self.p)
        cov = self._covariance_matrix(nodes, weight, weight_values, 
                                      self.time, dataset)
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
            w_xa = type(self.weight)(*weightPars)(time[:,None])
            f_hat = self._predict_kernel_matrix(type(self.nodes[i - 1])(*nodePars), time)
            w_xw = type(self.weight)(*weightPars)(self.time[None,:])
            #now we add all the necessary stuff; eq. 4 of Wilson et al. (2012)
            k_ii = k_ii + (w_xa * f_hat * w_xw)

        Kstar = k_ii
        Kstarstar = self._covariance_matrix(nodes, weight, weight_values, time, 
                                            dataset)

        y_mean = np.dot(Kstar, sol) #mean
        kstarT_k_kstar = []
        for i, e in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        y_var = np.diag(y_cov) #variance
        y_std = np.sqrt(y_var) #standard deviation
        return y_mean, y_std, y_cov

