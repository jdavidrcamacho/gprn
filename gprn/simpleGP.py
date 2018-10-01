#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from gprn.weightFunction import Linear

class simpleGP(object):
    """ 
        Class to create our simple Gaussian process "branch".
        Parameters:
            node = node function being used
            weight = weight function being used
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


##### marginal likelihood functions
    def _covariance_matrix(self, node, weight, time):
        """ 
            Creates the covariance matrix that will be used in the
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
        if type(self.weight) == Linear:
            w = self.weight(time)
        else:
            w = self._kernel_matrix(type(self.weight)(*weightPars), time)
        #now we add all the necessary elements to A_ii
        a_ii = w * f
        #now we fill our matrix
        A_ii = A_ii + a_ii
        return A_ii

    def compute_matrix(self, node, weight, time, 
                       yerr = True, nugget = False, shift = False):
        """
            Creates the big covariance matrix K that will be used in the 
        log marginal likelihood calculation
            Parameters:
                node = the latent noide functions f(x) (f hat)
                weight = the latent weight funtion w(x)
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
        k = self._covariance_matrix(node, weight, self.time)
        #addition of the measurement errors
        diag = self.yerr * np.identity(self.time.size)
        K = k + diag
        #more "weight" to the diagonal to avoid a ill-conditioned matrix
        if nugget:
            nugget_value = 0.01 #might be too big
            K = (1 - nugget_value)*K + nugget_value*np.diag(np.diag(K))
        #shifting all the eigenvalues up by the positive scalar to avoid a ill-conditioned matrix
        if shift:
            shift = 0.01 #might be too big
            K = K + shift * np.identity(self.time.size)
        return K

    def log_likelihood(self, node, weight, mean):
        """ 
            Calculates the marginal log likelihood.
        See Rasmussen & Williams (2006), page 113.
            Parameters:
                node = the latent noide functions f(x) (f hat)
                weight = the latent weight funtion w(x)
                mean = mean function being used
            Returns:
                log_like  = Marginal log likelihood
        """
        #calculates the  covariance matrix
        K = self.compute_matrix(node, weight, self.time)
        #calculation of the mean
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
                #in case we defined a mean at the start of our complexGP
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


##### marginal likelihood gradient functions
    def _compute_matrix_derivative(self, kernel_to_derive, kernel, 
                                   nugget = False):
        """ 
            Creates the covariance matrices of dK/dOmega, the derivatives of the
        kernels.
            Parameters:
                kernel_to_derive = node/weight function derivatives we want to
                                    use this time
                kernel = remaining node/weight function
            Return:
                k = final covariance matrix
        """
        #our matrix starts empty
        A = np.zeros((self.time.size, self.time.size))
        #measurement errors, should I add the errors in the derivatives???
        diag = self.yerr * np.identity(self.time.size)
        #node and weight functions in use
        a1 = self._kernel_matrix(kernel_to_derive,self.time)
        a2 = self._kernel_matrix(kernel, self.time)
        #final matrix
        A = A + a1 * a2 + diag
        #to avoid a ill-conditioned matrix
        if nugget:
            nugget_value = 0.01
            A = (1 - nugget_value) * A + nugget_value * np.diag(np.diag(A))
        return A

    def _log_like_grad(self, kernel_to_derive, kernel, 
                       node, weight, mean, nugget = False):
        """ 
            Calculates the gradient of the marginal log likelihood for a given
        kernel derivative. 
        See Rasmussen & Williams (2006), page 114.
            Parameters:
                kernel_to_derive = node/weight function derivatives we want 
                                    using this time
                kernel = remaining node/weight function
                node = the latent noide functions f(x) (f hat)
                weight = the latent weight funtion w(x)
                mean = mean function being used
            Returns:
                log_like  = Marginal log likelihood
        """
        #calculates the  covariance matrix of K and its inverse Kinv
        K = self.compute_matrix(node, weight, self.time)
        Kinv = np.linalg.inv(K)
        #calculates the  covariance matrix of dK/dOmega
        dK = self._compute_matrix_derivative(kernel_to_derive, kernel, nugget)
        #d(log marginal likelihood)/dOmega calculation
        try:
            alpha = np.dot(Kinv, self.y) #gives an array
            A = np.einsum('i,j',alpha, alpha) - Kinv #= alpha @ alpha.T - Kinv
            log_like_grad = 0.5 * np.einsum('ij,ij', A, dK) #= trace(a @ dK)
        except LinAlgError:
            return -np.inf
        return log_like_grad

    def log_likelihood_gradient(self, node, weight, mean, 
                                nugget = False):
        """ 
            Returns the marginal log likelihood gradients for a given 
        gprn "branch". 
            Parameters:
                node = the latent noide functions f(x) (f hat)
                weight = the latent weight funtion w(x)
                mean = mean function being used
            Returns:
                grads  = array of gradients
        """
        #First we derive the node
        parameters = node.pars #kernel parameters to use
        k = type(node).__subclasses__() #derivatives list
        node_array = [] #its a list and not an array but thats ok
        for _, j in enumerate(k):
            derivative = j(*parameters)
            loglike = self._log_like_grad(derivative, weight, 
                                          node, weight, mean, nugget)
            node_array.append(loglike)
        #Then we derive the weight
        parameters = weight.pars #kernel parameters to use
        k = type(weight).__subclasses__() #derivatives list
        weight_array = []
        for _, j in enumerate(k):
            derivative = j(*parameters)
            loglike = self._log_like_grad(derivative, node, 
                                          node, weight, mean, nugget)
            weight_array.append(loglike)
        #To finalize we merge both list into an array
        grads = np.array(node_array + weight_array)
        return grads


##### GP prediction funtions
    def predict_gp(self, node = None, weight = None, 
                   mean = None, time = None):
        """ 
            Conditional predictive distribution of the Gaussian process
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
        #calculation of the mean
        if mean:
            #To use a new mean
            mean = mean
            r = self.y - mean(self.time)
        else:
            mean = self.mean
            if mean is None:
                #In case we defined it to be a zero mean GP
                r = self.y 
            else:
                #In case we defined a mean at the start of our GP class
                r = self.y - mean(self.time)

        cov = self._covariance_matrix(node, weight, self.time)
        L1 = cho_factor(cov)
        sol = cho_solve(L1, r)
        #Kstar calculation
        fstar = self._predict_kernel_matrix(node, time, self.time)
        wstar = self._predict_kernel_matrix(weight, time, self.time)
        Kstar = wstar * fstar
        #Kstarstar
        fstarstar = self._kernel_matrix(node, time)
        wstarstar = self._kernel_matrix(weight, time)
        Kstarstar =  wstarstar * fstarstar
        #final calculations
        y_mean = np.dot(Kstar, sol) #mean
        kstarT_k_kstar = []
        for i, e in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        y_var = np.diag(y_cov) #variance
        y_std = np.sqrt(y_var) #standard deviation
        return y_mean, y_std, y_cov


### END
