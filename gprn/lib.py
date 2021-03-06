"""
Librabry of random functions
"""
import random
import math
import numpy as np
import scipy.linalg
import scipy.stats


def log_sum(log_summands):
    """ log sum operation """
    a = np.inf
    x = log_summands.copy()
    while a == np.inf or a == -np.inf or np.isnan(a):
        a = x[0] + np.log(1 + np.sum(np.exp(x[1:] - x[0])))
        random.shuffle(x)
    return a


def multivariate_normal(r, c, method='cholesky'):
    """
    Computes multivariate normal density for "residuals" vector r and
    covariance c.

    :param array r:
        1-D array of k dimensions.

    :param array c:
        2-D array or matrix of (k x k).

    :param string method:
        Method used to compute multivariate density.
        Possible values are:
            * "cholesky": uses the Cholesky decomposition of the covariance c,
              implemented in scipy.linalg.cho_factor and scipy.linalg.cho_solve.
            * "solve": uses the numpy.linalg functions solve() and slogdet().

    :return array: multivariate density at vector position r.
    """
    # Compute normalization factor used for all methods.
    kk = len(r) * math.log(2*math.pi)
    if method == 'cholesky':
        # Use Cholesky decomposition of covariance.
        cho, lower = scipy.linalg.cho_factor(c)
        alpha = scipy.linalg.cho_solve((cho, lower), r)
        return -0.5 * (kk + np.dot(r, alpha) + 2 * np.sum(np.log(np.diag(cho))))
    if method == 'solve':
        # Use slogdet and solve
        (_, d) = np.linalg.slogdet(c)
        alpha = np.linalg.solve(c, r)
        return -0.5 * (kk + np.dot(r, alpha) + d)


class MultivariateGaussian(scipy.stats.rv_continuous):
    """ Multivatiate Gaussian distribution"""
    def __init__(self, mu, cov):
        super(MultivariateGaussian, self).__init__(mu, cov)
        self.mu = mu
        self.covariance = cov + 1e-10
        self.dimensions = len(cov)

    # CHANGE THIS TO COMPUTE ON MULTI DIMENSIONAL x
    def pdf(self, x, method='cholesky'):
        if 1 < len(x.shape) < 3:
            # Input array is multi-dimensional
            # Check that input array is well aligned with covariance.
            if x.T.shape[0] != len(self.covariance):
                raise ValueError('Input array not aligned with covariance. '
                                 'It must have dimensions (n x k), where k is '
                                 'the dimension of the multivariate Gaussian.')
            # If ok, create array to contain results
            mvg = np.zeros(len(x))
            for s, rr in enumerate(x):
                mvg[s] = multivariate_normal(rr - self.mu, self.covariance,
                                             method)
            return mvg
        if len(x.shape) == 1:
            return multivariate_normal(x - self.mu, self.covariance, method)
        raise ValueError('Input array must be 1- or 2-D.')

    def rvs(self, nsamples):
        return np.random.multivariate_normal(self.mu, self.covariance, nsamples)
