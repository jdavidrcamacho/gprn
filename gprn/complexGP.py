#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import inv, cho_factor, cho_solve, LinAlgError
from scipy.stats import multivariate_normal
from copy import copy

from gprn.nodeFunction import Linear as nodeL
from gprn.nodeFunction import Polynomial as nodeP
from gprn.weightFunction import Linear as weightL
from gprn.weightFunction import Polynomial as weightP

from gprn import nodeFunction, weightFunction
#class kernelfix(nodeFunction, weightFunction):
#    """
#        Definition the kernels that will be used. To simplify my life all the
#    kernels defined are the sum of kernel + white noise
#    """
#    def __init__(self, *args):
#        """
#            Puts all kernel arguments in an array pars.
#        """
#        self.pars = np.array(args, dtype=float)
#
#    def __call__(self, r):
#        """
#            r = t - t' 
#        """
#        raise NotImplementedError
#
#    def __repr__(self):
#        """
#            Representation of each kernel instance
#        """
#        return "{0}({1})".format(self.__class__.__name__,
#                                 ", ".join(map(str, self.pars)))
#
#    def __add__(self, b):
#        return Sum(self, b)
#    def __radd__(self, b):
#        return self.__add__(b)
#
#    def __mul__(self, b):
#        return Multiplication(self, b)
#    def __rmul__(self, b):
#        return self.__mul__(b)


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

    def _predict_kernel_matrix(self, kernel, time):
        """
            To be used in predict_gp()
        """
        if isinstance(kernel, (nodeL, nodeP, weightL, weightP)):
            K = kernel(None, time[:, None], self.time[None, :])
        else:
            r = time[:, None] - self.time[None, :]
            K = kernel(r)
        #print(kernel)
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
    def _covariance_matrix(self, nodes, weight, weight_values, time, 
                           position_p, add_errors = False):
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
        #measurement errors
        yy_err = np.concatenate(self.yerr)
        new_yyerr = np.array_split(yy_err, self.p)
        
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
            #w = type(self.weight)(*weightPars)
            #f = type(self.nodes[i - 1])(*nodePars)
            #wf = self._kernel_matrix(w*f, time)
            w = self._kernel_matrix(type(self.weight)(*weightPars), time)
            f_hat = self._kernel_matrix(type(self.nodes[i - 1])(*nodePars),time)
            #now we add all the necessary stuff; eq. 4 of Wilson et al. (2012)
            k_ii += (w * f_hat)
        #adding measurement errors to our covariance matrix
        if add_errors:
            k_ii +=  (new_yyerr[position_p - 1]**2) * np.identity(time.size)

        return k_ii

    def compute_matrix(self, nodes, weight, weight_values,time, 
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
                                        position_p = i, add_errors = False)
            K_start[(i-1)*self.time.size : (i)*self.time.size, 
                        (i-1)*self.time.size : (i)*self.time.size] = k
        #addition of the measurement errors
        diag = np.concatenate(self.yerr) * np.identity(self.time.size * self.p)
        K = K_start + diag
        #more "weight" to the diagonal to avoid a ill-conditioned matrix
#        if nugget:
#            nugget_value = 0.01
#            K = (1 - nugget_value)*K + nugget_value*np.diag(np.diag(K))
#        #shifting all the eigenvalues up by the positive scalar to avoid a ill-conditioned matrix
#        if shift:
#            shift = 0.01
#            K = K + shift*np.identity(self.time.size * self.p)
        return K

    def old_log_like(self, nodes, weight, weight_values, means, jitters):
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
                jitters = jitter value of each dataset
            Returns:
                log_like  = Marginal log likelihood
        """
        #calculation of the covariance matrix
        K = self.compute_matrix(nodes, weight, weight_values, jitters, self.time)
        jitt = [] #jitters
        for i in  range(1, self.p+1):
            jitt.append((jitters[i - 1])**2 * np.ones_like(self.time))
        jitt = np.array(jitt)
        jitt = np.concatenate(jitt)
        K += jitt * np.diag(np.diag(K))
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
#                weightPars[0] =  weight_values[j-1 + self.q*(i-1)]
                #node and weight functions kernel
                w = self._kernel_matrix(type(self.weight)(*weightPars), self.time)
                f_hat = self._kernel_matrix(type(self.nodes[j - 1])(*nodePars), self.time)
#                ### cov[E_ii(x),E_ii(x')]
#                kw = self._kernel_matrix(type(self.weight)(*weightPars) * type(self.weight)(*weightPars), self.time)
#                #kw2 = np.linalg.matrix_power(kw,2)
#                kf = self._kernel_matrix(type(self.nodes[j - 1])(*nodePars) * type(self.nodes[j - 1])(*nodePars), self.time)
#                #kf2 = np.linalg.matrix_power(kf,2)
#                kw2,kf2 = kw,kf
#                cov_ii = 2 * self.q**2 * kw2 * kf2 + 2 * self.q * (kw2 + kf2)
#                cov_ii = cov_ii * np.identity(cov_ii.shape[0])
#                cov_ij = 0.5 * (3*self.q + self.q**2) * kw2 * kf2 + self.q*kw2
#                np.fill_diagonal(cov_ij, 0)
#                cov = cov_ii + cov_ij
#                #now we add all the necessary stuff; eq. 4 of Wilson et al. (2012)
#                k_ii = k_ii + cov
                k_ii = k_ii + (w * f_hat)
            #k_ii = k_ii + diag(error) + diag(jitter)
            k_ii += (new_yyerr[i - 1]**2) * np.identity(self.time.size) \
                    + (jitters[i - 1]**2) * np.identity(self.time.size)
            #log marginal likelihood calculation
            try:
                L1 = cho_factor(k_ii, overwrite_a=True, lower=False)
                log_like += - 0.5*np.dot(new_y[i - 1].T, cho_solve(L1, new_y[i - 1])) \
                           - np.sum(np.log(np.diag(L1[0]))) \
                           - 0.5*new_y[i - 1].size*np.log(2*np.pi)
            except LinAlgError:
                return -np.inf
        return log_like


    def CB(self, nodes, weight, time):
        """
            Creates the matrix CB (eq. 5 from Wilson et al. 2012), that will be 
        an N*q*(p+1) X N*q*(p+1) block diagonal matrix
        """
        p = int(self.p) #number of components
        q = int(self.q) #number of nodes
        w = int(p * q) #number of weights
        N = int(time.size) #N

        CB = np.zeros((N*q*(p+1), N*q*(p+1))) #initial empty matrix
        position = 0 #we start filling CB at position (0,0)
        for i in range(q):
            node_matrix = self._kernel_matrix(nodes[i], time)
            CB[position:position+N, position:position+N] = node_matrix
            position += N
        weight_matrix = self._kernel_matrix(weight, time)
        for i in range(w):
            CB[position:position+N, position:position+N] = weight_matrix
            position += N
        return CB


    def inv_CB(self, nodes, weight, time):
        """
            Creates the inverse of the matrix CB, that will be another
        N*q*(p+1) X N*q*(p+1) block diagonal matrix
        """
        p = int(self.p) #number of components
        q = int(self.q) #number of nodes
        w = int(p * q) #number of weights
        N = int(time.size) #N
        
        inv_CB = np.zeros((N*q*(p+1), N*q*(p+1))) #initial empty matrix
        position = 0 #we start filling inv_CB at position (0,0)
        for i in range(q):
            node_matrix = self._kernel_matrix(nodes[i], time)
            inv_node_matrix = inv(node_matrix)
            inv_CB[position:position+N, position:position+N] = inv_node_matrix
            position += N
        weight_matrix = self._kernel_matrix(weight, time)
        inv_weight_matrix = inv(weight_matrix)
        for i in range(w):
            inv_CB[position:position+N, position:position+N] = inv_weight_matrix
            position += N
        return inv_CB


    def sample_CB(self, nodes, weight, time):
        """ 
            Returns samples from the matrix CB
            Parameters:
                kernel = covariance funtion
                time = time array
            Returns:
                Sample of CB
        """
        p = int(self.p) #number of components
        q = int(self.q) #number of nodes
        N = int(time.size) #N
        
        mean = np.zeros(N*q*(p+1))
        cov = self.CB(nodes, weight, time)
        norm = multivariate_normal(mean, cov, allow_singular=True)
        return norm.rvs()


    def other_log(self, nodes, weight, means, jitters):
        """ 
            Calculates the marginal log likelihood for a GPRN.
            Parameters:
                nodes = the node functions f(x) (f hat)
                weight = the weight funtion w(x)
                means = mean function being used
                jitters = jitter value of each dataset
            Returns:
                log_like  = Marginal log likelihood
        """
        p = int(self.p) #number of components
        q = int(self.q) #number of nodes
        w = int(p * q) #number of weights
        N = int(self.y.size / p) #N
        
        ys = self.y.T #our components as columns
        x = self.time #our xi as a column
        cov = np.diag(jitters)**2  #our jitter matrix

        #means
        yy = np.concatenate(self.y)
        yy = yy - self._mean(means) if means else yy
        new_y = np.array_split(yy, self.p)
        yy_err = np.concatenate(self.yerr)
        new_yyerr = np.array_split(yy_err, self.p)
        
        #samples from matrix CB
        cb = self.sample_CB(nodes, weight, self.time) 

        wf = []
        for i in range(p):
            sample=[]
            for j in range(q):
                hadamard = cb[j*N : j*N + N] * cb[(j + (1+ i)*q)*N : (j + (1+ i)*q)*N + N]
                sample.append(hadamard)
            wf.append([np.prod(x) for x in np.array(sample).T])
        wf = np.array(wf).T

        p = 0
        for i in range(N):
            #print(ys[i], wf[i])
            p += multivariate_normal(wf[i], cov).logpdf(ys[i])

        return p


##### GP prediction funtions
    def prediction(self, nodes = None, weight = None, weight_values = None,
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
                                      self.time, dataset, add_errors = False)
        cov += (new_yerr[dataset - 1]**2) * np.identity(self.time.size) \
                    + (self.jitters[dataset - 1]**2) * np.identity(self.time.size)
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
            w = self._predict_kernel_matrix(type(self.weight)(*weightPars), time)
            f_hat = self._predict_kernel_matrix(type(self.nodes[i - 1])(*nodePars), time)
#            kw = self._predict_kernel_matrix(type(self.weight)(*weightPars), time)
#            kw2 = np.linalg.matrix_power(kw,2)
#            kf = self._predict_kernel_matrix(type(self.nodes[i - 1])(*nodePars), time)
#            kf2 = np.linalg.matrix_power(kf,2)

##            cov_ii = 2 * self.q**2 * kw2 * kf2 + 2 * self.q * (kw2 + kf2)
##            cov_ii = cov_ii * np.identity(cov_ii.shape[0])
#            cov_ij = 0.5 * (3*self.q + self.q**2) * kw2 * kf2 + self.q*kw2
##            np.fill_diagonal(cov_ij, 0)
##            cov_final = cov_ij

            #now we add all the necessary stuff; eq. 4 of Wilson et al. (2012)
#            k_ii = k_ii +cov_ij
            k_ii = k_ii + (w * f_hat)

        Kstar = k_ii
        Kstarstar = self._covariance_matrix(nodes, weight, weight_values, time, 
                                            dataset, add_errors = False)
        Kstarstar += (jitters[dataset - 1]**2) * np.identity(time.size)

        new_mean = np.array_split(self._mean(means, time), self.p)
        y_mean = np.dot(Kstar, sol) + new_mean[dataset-1]#mean

        kstarT_k_kstar = []
        for i, e in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        y_var = np.diag(y_cov) #variance
        y_std = np.sqrt(y_var) #standard deviation
        return y_mean, y_std, y_cov
        
        
##### END
