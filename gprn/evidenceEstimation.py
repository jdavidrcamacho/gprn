#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
import scipy.stats
from math import sqrt, log

### Original functions taken from https://github.com/exord/bayev

def compute_perrakis_estimate(marginal_sample, lnlikefunc, lnpriorfunc, 
                              nsamples=1000, lnlikeargs=(), lnpriorargs=(),
                              densityestimation='histogram', **kwargs):
    """
    Computes the Perrakis estimate of the bayesian evidence.
    The estimation is based on n marginal posterior samples
    (indexed by s, with s = 0, ..., n-1).
    :param array marginal_sample:
        A sample from the parameter marginal posterior distribution.
        Dimensions are (n x k), where k is the number of parameters.
    :param callable lnlikefunc:
        Function to compute ln(likelihood) on the marginal samples.
    :param callable lnpriorfunc:
        Function to compute ln(prior density) on the marginal samples.
    :param nsamples:
        Number of samples to produce.
    :param tuple lnlikeargs:
        Extra arguments passed to the likelihood function.
    :param tuple lnpriorargs:
        Extra arguments passed to the lnprior function.
    :param str densityestimation:
        The method used to estimate the marginal posterior density of each
        model parameter ("normal", "kde", or "histogram").
    Other parameters
    ----------------
    :param kwargs:
        Additional arguments passed to estimate_density function.
    :return:
    References
    ----------
    Perrakis et al. (2014; arXiv:1311.0674)
    """
    marginal_sample = make_marginal_samples(marginal_sample, nsamples)
    if not isinstance(marginal_sample, np.ndarray):
        marginal_sample = np.array(marginal_sample)
    number_parameters = marginal_sample.shape[1]
    #Estimate marginal posterior density for each parameter.
    marginal_posterior_density = np.zeros(marginal_sample.shape)
    for parameter_index in range(number_parameters):
        #Extract samples for this parameter.
        x = marginal_sample[:, parameter_index]
        #Estimate density with method "densityestimation".
        marginal_posterior_density[:, parameter_index] = \
            estimate_density(x, method=densityestimation, **kwargs)
    #Compute produt of marginal posterior densities for all parameters
    prod_marginal_densities = marginal_posterior_density.prod(axis=1)
    #Compute lnprior and likelihood in marginal sample.
    log_prior = lnpriorfunc(marginal_sample, *lnpriorargs)
    log_likelihood = lnlikefunc(marginal_sample, *lnlikeargs)
    #Mask values with zero likelihood (a problem in lnlike)
    cond = log_likelihood != 0
    log_summands = (log_likelihood[cond] + log_prior[cond] -
                    np.log(prod_marginal_densities[cond]))
    perr = log_sum(log_summands) - log(len(log_summands))
    return perr


def estimate_density(x, method='histogram', **kwargs):
    """
    Estimate probability density based on a sample. Return value of density at
    sample points.
    :param array_like x: sample.
    :param str method:
        Method used for the estimation. 'histogram' estimates the density based
        on a normalised histogram of nbins bins; 'kde' uses a 1D non-parametric
        gaussian kernel; 'normal approximates the distribution by a normal
        distribution.
    Additional parameters
    :param int nbins:
        Number of bins used in "histogram method".
    :return: density estimation at the sample points.
    """
    nbins = kwargs.pop('nbins', 100)
    if method == 'normal':
        #Approximate each parameter distribution by a normal.
        return scipy.stats.norm.pdf(x, loc=x.mean(), scale=sqrt(x.var()))
    elif method == 'kde':
        #Approximate each parameter distribution using a gaussian kernel estimation
        return scipy.stats.gaussian_kde(x)(x)
    elif method == 'histogram':
        #Approximate each parameter distribution based on the histogram
        density, bin_edges = np.histogram(x, nbins, density=True)
        #Find to which bin each element corresponds
        density_indexes = np.searchsorted(bin_edges, x, side='left')
        #Correct to avoid index zero from being assiged to last element
        density_indexes = np.where(density_indexes > 0, density_indexes,
                                   density_indexes + 1)
        return density[density_indexes - 1]


def make_marginal_samples(joint_samples, nsamples=None):
    """
    Reshuffles samples from joint distribution of k parameters to obtain samples
    from the _marginal_ distribution of each parameter.
    :param np.array joint_samples:
        Samples from the parameter joint distribution. Dimensions are (n x k),
        where k is the number of parameters.
    :param nsamples:
        Number of samples to produce. If 0, use number of joint samples.
    :type nsamples:
        int or None
    """
    if nsamples > len(joint_samples) or nsamples is None:
        nsamples = len(joint_samples)
    marginal_samples = joint_samples[-nsamples:, :].copy()
    number_parameters = marginal_samples.shape[-1]
    # Reshuffle joint posterior samples to obtain _marginal_ posterior samples
    for parameter_index in range(number_parameters):
        random.shuffle(marginal_samples[:, parameter_index])
    return marginal_samples


def log_sum(log_summands):
    a = np.inf
    x = log_summands.copy()
    while a == np.inf or a == -np.inf or np.isnan(a):
        a = x[0] + np.log(1 + np.sum(np.exp(x[1:] - x[0])))
        random.shuffle(x)
    return a


def compute_harmonicmean(lnlike_post, posterior_sample=None, lnlikefunc=None,
                         lnlikeargs=(), **kwargs):
    """
    Computes the harmonic mean estimate of the marginal likelihood.
    The estimation is based on n posterior samples
    (indexed by s, with s = 0, ..., n-1), but can be done directly if the
    log(likelihood) in this sample is passed.
    :param array lnlike_post:
        log(likelihood) computed over a posterior sample. 1-D array of length n.
        If an emply array is given, then compute from posterior sample.
    :param array posterior_sample:
        A sample from the parameter posterior distribution.
        Dimensions are (n x k), where k is the number of parameters. If None
        the computation is done using the log(likelihood) obtained from the
        posterior sample.
    :param callable lnlikefunc:
        Function to compute ln(likelihood) on the marginal samples.
    :param tuple lnlikeargs:
        Extra arguments passed to the likelihood function.
    Other parameters
    ----------------
    :param int size:
        Size of sample to use for computation. If none is given, use size of
        given array or posterior sample.
    References
    ----------
    Kass & Raftery (1995), JASA vol. 90, N. 430, pp. 773-795
    """
    if len(lnlike_post) == 0 and posterior_sample is not None:
        samplesize = kwargs.pop('size', len(posterior_sample))
        if samplesize < len(posterior_sample):
            posterior_subsample = np.random.choice(posterior_sample,
                                                      size=samplesize,
                                                      replace=False)
        else:
            posterior_subsample = posterior_sample.copy()
        #Compute log likelihood in posterior sample.
        log_likelihood = lnlikefunc(posterior_subsample, *lnlikeargs)
    elif len(lnlike_post) > 0:
        samplesize = kwargs.pop('size', len(lnlike_post))
        log_likelihood = np.random.choice(lnlike_post, size=samplesize,
                                             replace=False)
    hme = -log_sum(-log_likelihood) + log(len(log_likelihood))
    return hme


def run_hme_mc(log_likelihood, nmc, samplesize):
    hme = np.zeros(nmc)
    for i in range(nmc):
        hme[i] = compute_harmonicmean(log_likelihood, size=samplesize)
    return hme
