#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError

class simpleGP(object):
    """ 
        Class to create our simple Gaussian process.
        Parameters:
            node = node function being used
            weight = weight function being used
            amplitude = amplitude of the kernel
            means = mean function being used, None if model doesn't use it
            time = time array
            y = measurements array
            yerr = measurements errors array
    """
    def __init__(self, node, weight, mean, time, y, yerr = None):
        self.node = node            #node function
        self.weight = weight        #weight function
        self.mean = mean            #mean function
        self.time = time            #time
        self.y = y                  #measurements
        if yerr is None:
            self.yerr = 1e-12 * np.identity(self.t.size)
        else:
            self.yerr = yerr        #measurements errors


    def _kernel_pars(self, kernel):
        """
            Returns a kernel parameters
        """
        return kernel.pars


    def _kernel_matrix(self, kernel, time = None):
        """
            Returns the covariance matrix created by evaluating a given kernel 
        at inputs time.
        """
        #if time is None we use the time of our simpleGP
        if time is None:
            r = self.time[:, None] - self.time[None, :]
        #if we define a new time we will use that time
        else:
            r = time[:, None] - time[None, :]
        K = kernel(r)
        return K


    def _predict_kernel_matrix(self, kernel, time, tstar):
        """
            To be used in predict_gp()
        """
        r = time[:, None] - self.time[None, :]
        K = kernel(r)
        return K


    def _mean_function(self, mean):
        """
            Returns the value of the mean function
        """
        m = np.zeros_like(self.time)
        if mean is None:
            m = m               #not really necessary
        else:
            m = mean(self.time)
        return m


    def _covariance_matrix(self, node, weight, weight_value, time):
        """ 
            Creates the covariance matrix that will be used for the
        compute_matrix() function
            Parameters:
                node = the latent noide functions f(x) (f hat)
                weight = the latent weight funtion w(x)
                weight_value = weight value, basically the amplitude of the 
                                weight function
                time = time 
            Return:
                A_ii = final covariance matrix
        """
        #our matrix starts empty
        A_ii = np.zeros((time.size, time.size))
        #hyperparameteres of the node function
        nodePars = self._kernel_pars(node)
        #hyperparameteres of the weight function
        weightPars = self._kernel_pars(weight)

        #node and weight functions kernel
        f = self._kernel_matrix(type(self.node)(*nodePars),time)
        w = self._kernel_matrix(type(self.weight)(*weightPars), time)
        #now we add all the necessary elements to a_ii and b_ii
        a_ii = weight_value * (w * f)
        #now we fill our matrix
        A_ii = A_ii + a_ii
        return A_ii


    def compute_matrix(self, node, weight, weight_value, time, 
                       yerr = True, nugget = False, shift = False):
        """
            Creates the big covariance matrix K that will be used in the 
        log marginal likelihood calculation
            Parameters:
                node = the latent noide functions f(x) (f hat)
                weight = the latent weight funtion w(x)
                weight_value = weight value, basically the amplitude of the 
                                weight function
                time = time  
                yerr = True if measurements dataset has errors, False otherwise
                nugget = True if K is not positive definite, False otherwise
                shift = True if K is not positive definite, False otherwise
            Returns:
                K = final covariance matrix 
        """
        #Our K starts empty
        K = np.zeros((time.size, time.size))
        #Then we calculate the covariance matrix
        k = self._covariance_matrix(node, weight, weight_value, self.time)
        
        #addition of the measurement errors
        diag = self.yerr * np.identity(self.time.size)
        K = k + diag

        #shifting all the eigenvalues up by the positive scalar
        if shift:
            shift = 0.01
            K = K + shift * np.identity(self.time.size)
        #To give more "weight" to the diagonal
        if nugget:
            nugget_value = 0.01
            K = (1 - nugget_value)*K + nugget_value*np.diag(np.diag(K))
        return K


    def log_likelihood(self, node, weight, weight_value, mean):
        """ Calculates the marginal log likelihood. 
            Parameters:
                node = the latent noide functions f(x) (f hat)
                weight = the latent weight funtion w(x)
                weight_value = weight value, basically the amplitude of the 
                                weight function
                mean = mean function being used
            Returns:
                log_like  = Marginal log likelihood
        """
        #calculate covariance matrix with kernel parameters a
        K = self.compute_matrix(node, weight, weight_value, self.time)

        #calculate mean and residuals
        if mean:
            #in case we defined a new mean function
            mean = mean
            y = self.y - mean(self.time)
        else:
            mean = self.mean
            if mean is None:
                #in case we defined it to be a zero mean GP
                y = self.y 
            else:
                #in case we defined a mean at the start of our simpleGP
                y = self.y - self.mean(self.time)

        #log marginal likelihood calculation
        try:
            L1 = cho_factor(K, overwrite_a=True, lower=False)
            log_like = - 0.5*np.dot(y.T, cho_solve(L1, y)) \
                       - np.sum(np.log(np.diag(L1[0]))) \
                       - 0.5*y.size*np.log(2*np.pi)
        except LinAlgError:
            return -np.inf
        return log_like


    def predict_gp(self, node = None, weight = None, weight_value = None, 
                   mean = None, time = None):
        """ Conditional predictive distribution of the Gaussian process
            Parameters:
                node = the latent noide functions f(x) (f hat)
                weight = the latent weight funtion w(x)
                weight_value = weight value, basically the amplitude of the 
                                weight function
                mean = mean function being used
                time = time  
        Returns:
            mean vector, covariance matrix, standard deviation vector
        """
        if node:
            #To use a new node funtion
            node = node
        else:
            #To use the one we defined earlier 
            node = self.node
        if weight:
            #To use a new weight funtion
            weight = weight
        else:
            #To use the one we defined earlier 
            weight = self.weight
        #defining the amplitude of the weight function
        if weight_value:
            weight_value = weight_value
        else:
            weight_value = 1
        #calculate mean and residuals
        if mean:
            mean = mean
            r = self.y - mean(self.time)
        else:
            mean = self.mean
            if mean is None:
                #In case we defined it to be a zero mean GP
                r = self.y 
            else:
                #In case we defined a mean at the start of our baseGP
                r = self.y - mean(self.time)

        
        cov = self._covariance_matrix(node, weight, weight_value, self.time)
        L1 = cho_factor(cov)
        sol = cho_solve(L1, r)

        #K star calculation
        fstar = self._predict_kernel_matrix(node, time, self.time)
        wstar = self._predict_kernel_matrix(weight, time, self.time)
        Kstar = weight_value * (wstar * fstar)

        #Kstarstar
        fstarstar = self._kernel_matrix(node, time)
        wstarstar = self._kernel_matrix(weight, time)
        Kstarstar =  weight_value * (wstarstar * fstarstar)

        y_mean = np.dot(Kstar, sol) #mean
        kstarT_k_kstar = []
        for i, e in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        y_var = np.diag(y_cov) #variance
        y_std = np.sqrt(y_var) #standard deviation
        return y_mean, y_std, y_cov
