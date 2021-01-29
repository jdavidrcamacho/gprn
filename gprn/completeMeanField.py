import numpy as np
from scipy.linalg import cholesky, LinAlgError
from scipy.stats import multivariate_normal
from gprn.covFunction import Linear as covL
from gprn.covFunction import Polynomial as covP
np.random.seed(23011990)

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
            K = kernel(r) + 1e-6*np.diag(np.diag(np.ones_like(r)))
        K[np.abs(K)<1e-12] = 0.
        return K
    
    
    def _predictKMatrix(self, kernel, time):
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
            
            
    def _CBMatrix(self, nodes, weight):
        """
        Creates the matrix CB (eq. 5 from Wilson et al. 2012), that will be 
        an N*q*(p+1) X N*q*(p+1) block diagonal matrix

        Parameters
        ----------
            nodes = array of node functions 
            weight = weight function

        Returns
        -------
            CB = matrix CB
        """
        time = self.time
        CB_size = time.size * self.q * (self.p + 1)
        CB = np.zeros((CB_size, CB_size)) #initial empty matrix
        pos = 0 #we start filling CB at position (0,0)
        #first we enter the nodes
        for i in range(self.q):
            node_CovMatrix = self._kernelMatrix(nodes[i], time)
            CB[pos:pos+time.size, pos:pos+time.size] = node_CovMatrix
            pos += time.size
        weight_CovMatrix = self._kernelMatrix(weight[0], time)
        #then we enter the weights
        for i in range(self.qp):
            CB[pos:pos+time.size, pos:pos+time.size] = weight_CovMatrix
            pos += time.size
        return CB
    
    
    def _sampleCB(self, nodes, weight):
        """ 
        Returns samples from the matrix CB
        
        Parameters
        ----------
            nodes = array of node functions 
            weight = weight function
            
        Returns
        -------
            Samples of CB
        """
        time = self.time
        mean = np.zeros(time.size*self.q*(self.p+1))
        cov = self._CBMatrix(nodes, weight)
        norm = multivariate_normal(mean, cov, allow_singular=True).rvs()
        return norm
            
            
    def sampleIt(self, latentFunc, time=None):
        """
        Returns samples from the kernel
        
        Parameters
        ----------
        latentFunc: func
            Covariance function
        time: array
            Time array
        
        Returns
        -------
        norm: array
            Sample of K 
        """
        if time is None:
            time = self.time
        
        mean = np.zeros_like(time)
        K = np.array([self._kernelMatrix(i, time) for i in latentFunc])
        norm = multivariate_normal(mean, np.squeeze(K), allow_singular=True).rvs()
        return norm
    
    
##### Mean-Field Inference functions ##########################################
    def ELBOcalc(self, nodes, weight, mean, jitter, iterations = 10000,
                     mu = None, var = None):
        """
        Function to use in the the sampling of the GPRN
        
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
        mu: array
            Variational means
        var: array
            Variational variances
            
        Returns
        -------
        ELBO: array
            Value of the ELBO per iteration
        mu: array
            Optimized variational means
        var: array
            Optimized variational variance (diagonal of sigma)
        """
        np.random.seed(23011990)
        #initial variational parameters (they start as random)
        D = self.time.size * self.q *(self.p+1)
        if mu is None and var is None:
            mu = np.random.randn(D, 1)
            var = np.random.rand(D, 1)
        varF, varW = self._u_to_fhatW(var.flatten())
        sigF, sigW = [], []
        for q in range(self.q):
            sigF.append(np.diag(varF[0, q, :]))
            for p in range(self.p):
                sigW.append(np.diag(varW[p, q, :]))
        elboArray = np.array([-1e15]) #To add new elbo values inside
        iterNumber = 0
        while iterNumber < iterations:
            #Ou calcular media de 10 elbos ou 
            
            #Optimize mu and var analytically
            ELBO, mu, var, sigF, sigW= self.ELBOaux(nodes, weight, mean, jitter, 
                                                       mu, var, sigF, sigW)
            elboArray = np.append(elboArray, ELBO)
            iterNumber += 1
            #Stoping criteria:
            if iterNumber > 5:
                means = np.mean(elboArray[-5:])
                criteria = np.abs(np.std(elboArray[-5:]) / means)
                if criteria < 1e-2 and criteria !=0:
                    return ELBO, mu, var
        print('Max iterations reached')
        return ELBO, mu, var
    
    
    def ELBOaux(self, node, weight, mean, jitter, mu, var, sigmaF, sigmaW):
        """
        Evidence Lower bound to use in ELBOcalc()
        
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
            
        Returns
        -------
        ELBO: float
            Evidence lower bound
        new_mu: array
            New variational means
        new_var: array
            New variational variances
        """ 
        #to separate the variational parameters between the nodes and weights
        muF, muW = self._u_to_fhatW(mu.flatten())
        varF, varW = self._u_to_fhatW(var.flatten())
        sigmaF, muF, sigmaW, muW = self._updateSigMu(node, weight, mean, 
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
        return ELBO, new_mu, new_var, sigmaF, sigmaW
    
    
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
        Kw = Kw.reshape(self.p, self.q, self.N, self.N)
        Lw = Lw.reshape(self.p, self.q, self.N, self.N)
        #mean functions
        means = self._mean(means, tstar)
        means = np.array_split(means, self.p)
        muF, muW = self._u_to_fhatW(mu.flatten())
        ystar = np.zeros((self.p, tstar.size))
        for i in range(tstar.size):
            Kfstar = np.array([self._predictKMatrix(i1, tstar[i]) for i1 in node])
            Kwstar = np.array([self._predictKMatrix(i2, tstar[i]) for i2 in weights])
            Kwstar = Kwstar.reshape(self.p, self.q, self.N)
            #Lwstar = np.linalg.solve(np.squeeze(Lw), np.squeeze(Kwstar).T)
            countF, countW = 1, 1
            Wstar, fstar = np.zeros((self.p, self.q)), np.zeros((self.q, 1))
            for q in range(self.q):
                Lfstar = np.linalg.solve(np.squeeze(Lf[q,:,:]), 
                                         np.squeeze(Kfstar[q,:]).T)
                fstar[q] = Lfstar @ np.linalg.solve(np.squeeze(Lf[q,:,:]), 
                                                     muF[:,q,:].T)
                countF += self.N
                for p in range(self.p):
                    Wstar[p,q] = np.linalg.solve(Lw[p,q,:,:],Kwstar[p,q,:].T)\
                                     @ np.linalg.solve(Lw[p,q,:,:], muW[p,q].T)
                    countW += self.N
            ystar[:,i] = ystar[:, i] + np.squeeze(Wstar @ fstar)
        final_ystar = []
        for i in range(self.p):
            final_ystar.append(ystar[i] + means[i])
        final_ystar = np.array(final_ystar)
        return final_ystar
    
    
    def _updateSigMu(self, nodes, weight, mean, jitter, muF, varF, muW, varW):
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
        jitt2 = np.array(jitter)**2 #jitters
        #kernel matrix for the nodes
        Kf = np.array([self._kernelMatrix(i, self.time) for i in nodes])
        #kernel matrix for the weights
        Kw = np.array([self._kernelMatrix(j, self.time) for j in weight]) 
        Kw = Kw.reshape(self.p, self.q, self.N, self.N)
        #we have Q nodes => j in the paper; we have P y(x)s => i in the paper
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
                Kw_ij = Kw[i,j,:,:]
                CovWij = np.diag(1 / Diag_ij) + Kw_ij
                CovWij = Kw_ij - Kw_ij @ np.linalg.solve(CovWij, Kw_ij)
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
        jitt2 = np.array(jitter)**2 #jitters squared
        ycalc = new_y.T #new_y0.shape = (p,n)
        logl = 0
        for p in range(self.p):
            for n in range(self.N):
                logl += np.log(jitt2[p] + self.yerr2[p,n])
        logl = -0.5 * logl
        #print('1st:',logl)
        
        Wcalc, Fcalc, bottom = [], [], []
        for p in range(self.p):
            Wcalc.append([])
            Fcalc.append([])
            bottom.append([])
        for n in range(self.N):
            for q in range(self.q):
                for p in range(self.p):
                    ycalc[p,n] = new_y.T[p,n]
                    Wcalc[p].append(mu_w[p, :, n])
                    Fcalc[p] = np.append(Fcalc[p],mu_f[0, q, n].T)
                    bottom[p].append(jitt2[p]+self.yerr2[p,n])
        Wcalc = np.array(Wcalc)#.reshape(self.p, self.N * self.q)
        Fcalc = np.array(Fcalc)
        Ymean = (Fcalc @ Wcalc).reshape(-1)
        ycalc = ycalc.reshape(-1)
        bottom = np.array(bottom).reshape(-1)
        Ydiff = ((ycalc - Ymean).T * (ycalc - Ymean))/bottom
        #print('2nd:', np.sum(Ydiff))
        logl += -0.5 * np.sum(Ydiff)
        
        value = 0
        for i in range(self.p):
            for j in range(self.q):
                value += np.sum((np.diag(sigma_f[j,:,:])*mu_w[i,j,:]*mu_w[i,j,:] +\
                                np.diag(sigma_w[j,i,:,:])*mu_f[:,j,:]*mu_f[:,j,:] +\
                                np.diag(sigma_f[j,:,:])*np.diag(sigma_w[j,i,:,:]))\
                                /(jitt2[i]+self.yerr2[i,:]))
        #print('3rd:', value)
        logl += -0.5* value
        return logl
        
        
#         new_y = np.concatenate(self.y) - self._mean(mean, self.time)
#         new_y = np.array(np.array_split(new_y, self.p)).T #NxP dimensional vector
#         jitt = np.array(jitter) #jitters
#         jitt2 = np.array(jitter)**2 #jitters squared
#         ycalc = new_y.T #new_y0.shape = (p,n)
#         logl = 0
#         for p in range(self.p):
#             ycalc[p] = new_y.T[p,:] / (jitt[p]+self.yerr[p,:])
#             for n in range(self.N):
#                 logl += np.log(jitt2[p] + self.yerr2[p,n])
#         logl = -0.5 * logl
#         #print(logl)
        
#         Wcalc, Fcalc = [], []
#         for p in range(self.p):
#             Wcalc.append([])
#             Fcalc.append([])
#         for n in range(self.N):
#             for q in range(self.q):
#                 for p in range(self.p):
#                     Wcalc[p].append(mu_w[p, :, n])
#                     Fcalc[p] = np.append(Fcalc[p], 
#                          (mu_f[:, q, n] / (jitt2[p]+self.yerr2[p,n])))
#         Wcalc = np.array(Wcalc).reshape(self.p, self.N * self.q)
#         Fcalc = np.array(Fcalc)
#         Ymean = np.sum((Wcalc * Fcalc).T, axis=1).reshape(self.N, self.q)
#         print(ycalc.shape)
#         Ydiff = (ycalc - Ymean.T).T @ (ycalc - Ymean.T)
#         #print(-0.5 * np.sum(Ydiff))
#         logl += -0.5 * np.sum(Ydiff)
        
# #        Wcalc = []
# #        for p in range(self.p):
# #            Wcalc.append([])
# #        for n in range(self.N):
# #            for p in range(self.p):
# #                Wcalc[p].append(mu_w[p, :, n])
# #        Wcalc = np.array(Wcalc).reshape(self.p, self.N * self.q)
# #        Fcalc = []#np.array([])
# #        for p in range(self.p):
# #            Fcalc.append([])
# #            #Fcalc1.append([])
# #        for n in range(self.N):
# #            for q in range(self.q):
# #                for p in range(self.p):
# #                    Fcalc[p] = np.append(Fcalc[p], 
# #                         (mu_f[:, q, n] / (jitt[p] + self.yerr[p,n])))
# #        Wcalc, Fcalc = np.array(Wcalc), np.array(Fcalc)
# #        Ymean = np.sum((Wcalc * Fcalc).T, axis=1).reshape(self.N, self.q)
# #        Ydiff = (ycalc - Ymean.T) * (ycalc - Ymean.T)
# #        logl += -0.5 * np.sum(Ydiff)
        
#         value = 0
#         for i in range(self.p):
#             for j in range(self.q):
#                 value += np.sum((np.diag(sigma_f[j,:,:])*mu_w[i,j,:]*mu_w[i,j,:] +\
#                                 np.diag(sigma_w[j,i,:,:])*mu_f[:,j,:]*mu_f[:,j,:] +\
#                                 np.diag(sigma_f[j,:,:])*np.diag(sigma_w[j,i,:,:]))\
#                                 /(jitt2[p]+self.yerr2[p,:]))
        
#         #print(value)
#         logl += -0.5* value
#         return logl/self.qp


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
        Kw0 = np.array([self._kernelMatrix(j, self.time) for j in weights])
        Kw = Kw0.reshape(self.q, self.p, self.N, self.N)
        #we have Q nodes -> j in the paper; we have P y(x)s -> i in the paper
        first_term = 0 #calculation of the first term of eq.15 of Nguyen & Bonilla (2013)
        second_term = 0 #calculation of the second term of eq.15 of Nguyen & Bonilla (2013)
        Lw = np.array([self._cholNugget(j)[0] for j in Kw0])
        Lw = Lw.reshape(self.q, self.p, self.N, self.N)
        muW = mu_w.reshape(self.q, self.p, self.N)
        sumSigmaF = np.zeros_like(sigma_f[0])
        for j in range(self.q):
            Lf = self._cholNugget(Kf[j])[0]
            logKf = np.float(np.sum(np.log(np.diag(Lf))))
            muK =  np.linalg.solve(Lf, mu_f[:,j, :].reshape(self.N))
            muKmu = muK @ muK
            sumSigmaF = sumSigmaF + sigma_f[j]
            trace = np.trace(np.linalg.solve(Kf[j], sumSigmaF))#sigma_f[j]))
            first_term += -logKf - 0.5*(muKmu + trace)
            for i in range(self.p):
                muK = np.linalg.solve(Lw[j,i,:,:], muW[j,i]) 
                muKmu = muK @ muK
                trace = np.trace(np.linalg.solve(Kw[j,i,:,:], sigma_w[j,i,:,:]))
                second_term += -np.float(np.sum(np.log(np.diag(Lw[j,i,:,:])))) - 0.5*(muKmu + trace)
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
        return entropy + self.qp*(1+np.log(2*np.pi))


    def _scipyEntropy(self, latentFunc, time=None):
        """
        Returns entropy from the kernel
        
        Parameters
        ----------
        latentFunc: func
            Covariance functions
        time: array
            Time array
        
        Returns
        -------
        norm: array
            Sample of K 
        """
        if time is None:
            time = self.time
        mean = np.zeros_like(time)
        K = np.array([self._kernelMatrix(i, time) for i in [latentFunc]])
        ent = multivariate_normal(mean, np.squeeze(K), allow_singular=True).entropy()
        return ent
        
