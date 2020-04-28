#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
from math import log, pi
import time
import warnings
import multiprocessing as mp
from multiprocessing.queues import Empty

import gprn.perrakis as perr
import emcee


### Original functions taken from
### https://github.com/exord/MCMC


def emcee_flatten(sampler, bi=None, chainindexes=None):
    """
    sampler can be an emcee.Sampler instance or an iterable. If the latter,
    first element must be chain array of shape [nwalkers, nsteps, dim],
    second element must be lnprobability array [nwalkers, nsteps],
    third element is acceptance rate (nwalkers,)
    fouth element is the sampler.args attribute of the emcee.Sampler isntance.
    lnprior function]

    chainindexes must be boolean
    """
    if bi is None:
        bi = 0
    else:
        bi = int(bi)
    if isinstance(sampler, emcee.Sampler):
        nwalkers, nsteps, dim = sampler.chain.shape
        chain = sampler.chain
    elif np.iterable(sampler):
        nwalkers, nsteps, dim = sampler[0].shape
        chain = sampler[0]
    else:
        raise TypeError('Unknown type for sampler')
    if chainindexes is None:
        chainind = np.array([True] * nwalkers)
    else:
        chainind = np.array(chainindexes)
        assert len(chainind) == chain.shape[0]
    fc = chain[chainind, bi:, :].reshape(sum(chainind) * (nsteps - bi), dim)
    # Shuffle once to loose correlations (bad idea, as this screws map)
    # np.random.shuffle(fc)
    return fc


def read_samplers(samplerfiles, rootdir):
    f = open(samplerfiles)
    samplerlist = f.readlines()
    samplers = []
    for sam in samplerlist:
        print('Reading sampler from file {}'.format(sam.rstrip()))
        f = open(os.path.join(rootdir, sam.rstrip()))
        samplers.append(pickle.load(f))
    return samplers


def emcee_perrakis(sampler, nsamples=5000, bi=0, cind=None):
    """
    Compute the Perrakis estimate of ln(Z) for a given sampler.

    See docstring in emcee_flatten for format of sampler
    """
    # Flatten chain first
    fc = emcee_flatten(sampler, bi=bi, chainindexes=cind)

    # Get functions and arguments
    lnlikefunc, lnpriorfunc, lnlikeargs, lnpriorargs = get_func_args(sampler)

    # Change this ugly thing!
    def lnl(x, *args):
        y = np.empty(len(x))
        for i, xx in enumerate(x):
            y[i] = lnlikefunc(xx, *args)
        return y

    def lnp(x, *args):
        y = np.empty(len(x))
        for i, xx in enumerate(x):
            y[i] = lnpriorfunc(xx, *args)
        return y
    # Construct marginal samples
    marginal = perr.make_marginal_samples(fc, nsamples)
    # Compute perrakis
    ln_z = perr.compute_perrakis_estimate(marginal, lnl, lnp,
                                          lnlikeargs, lnpriorargs)
    # Correct for missing term in likelihood
    datadict = get_datadict(sampler)
    nobs = 0
    for inst in datadict:
        nobs += len(datadict[inst]['data'])
    # print('{} datapoints.'.format(nobs))
    ln_z += -0.5 * nobs * log(2 * pi)
    return ln_z, ln_z / log(10)


def get_func_args(sampler):
    """
    Read functions and arguments from sampler to compute lnprior and lnlike

    :param sampler:
    :return:
    """
    if isinstance(sampler, emcee.Sampler):
        # Get functions and parameters
        lnlikefunc = sampler.args[1]
        lnpriorfunc = sampler.args[2]
        lnlikeargs = [sampler.args[0], ]
        lnpriorargs = [sampler.args[0], ]
        lnlikeargs.extend(sampler.kwargs['lnlikeargs'])
        lnpriorargs.extend(sampler.kwargs['lnpriorargs'])
    elif np.iterable(sampler):
        # Check which generation of sampler are we using.
        if hasattr(sampler[-1], '__module__'):
            # Sampler from Model instance
            lnlikefunc = sampler[-1].lnlike
            lnpriorfunc = sampler[-1].lnprior
            lnlikeargs = ()
            lnpriorargs = ()
        else:
            # Sampler using functions.
            # Get functions and parameters
            lnlikefunc = sampler[-2][1]
            lnpriorfunc = sampler[-2][2]
            lnlikeargs = [sampler[-2][0], ]
            lnpriorargs = [sampler[-2][0], ]
            lnlikeargs.extend(sampler[-1]['lnlikeargs'])
            lnpriorargs.extend(sampler[-1]['lnpriorargs'])
    else:
        raise TypeError('Unknown type for sampler')
    return lnlikefunc, lnpriorfunc, lnlikeargs, lnpriorargs


def get_datadict(sampler):
    if isinstance(sampler, emcee.Sampler):
        return sampler.kwargs['lnlikeargs'][1]
    elif np.iterable(sampler):
        try:
            return sampler[-1]['lnlikeargs'][1]
        except TypeError:
            return sampler[-1].data
    else:
        raise TypeError('Unknown type for sampler')


def emcee_multi_perrakis(sampler, nsamples=5000, bi=0, thin=1, nrepetitions=1,
                         cind=None, ncpu=None, datacorrect=False,
                         outputfile='./perrakis_out.txt'):
    """
    Compute the Perrakis estimate of ln(Z) for a given sampler
    repeateadly using multicore

    WRITE DOC
    """
    # Flatten chain first
    fc = emcee_flatten(sampler, bi=bi, chainindexes=cind)[::thin]
    # Get functions and arguments
    lnlikefunc, lnpriorfunc, lnlikeargs, lnpriorargs = get_func_args(sampler)
    # Change this ugly thing! Used to vectorize lnlikefunc and lnpriorfunc
    def lnl(x, *args):
        y = np.empty(len(x))
        for ii, xx in enumerate(x):
            y[ii] = lnlikefunc(xx, *args)
        return y
    
    def lnp(x, *args):
        y = np.empty(len(x))
        for ii, xx in enumerate(x):
            y[ii] = lnpriorfunc(xx, *args)
        return y

    lnz = multi_cpu_perrakis(fc, lnl, lnp, lnlikeargs, lnpriorargs, nsamples,
                             nrepetitions, ncpu=ncpu)
    if datacorrect:
        # Correct for missing term in likelihood
        datadict = get_datadict(sampler)
        nobs = 0
        for inst in datadict:
            nobs += len(datadict[inst]['data'])
        lnz += -0.5 * nobs * log(2 * pi)
    # Write to file
    f = open(outputfile, 'a+')
    for ll in lnz:
        f.write('{:.6f}\t{:.6f}\n'.format(ll, ll / log(10)))
    f.close()
    return lnz, lnz / log(10)


def single_perrakis(fc, nsamples, lnl, lnp, lnlargs, lnpargs, nruns,
                    output_queue):
    # Prepare output array
    lnz = np.empty(nruns)
    #
    np.random.seed()
    for i in range(nruns):
        if i % 10 == 0:
            print(i)
        # Construct marginal samples
        marginal = perr.make_marginal_samples(fc, nsamples)
        # Compute perrakis
        lnz[i] = perr.compute_perrakis_estimate(marginal, lnl, lnp,
                                                lnlargs, lnpargs)
    if output_queue is None:
        return lnz
    else:
        output_queue.put(lnz)
    return


def multi_cpu_perrakis(fc, lnl, lnp, lnlikeargs, lnpriorargs,
                       nsamples, nrepetitions=1, ncpu=None, ):
    # Prepare multiprocessing
    if ncpu is None:
        ncpu = mp.cpu_count()
    # Check if number of requested repetitions below ncpu
    ncpu = min(ncpu, nrepetitions)
    # Instantiate output queue
    q = mp.Queue()
    print('Running {} repetitions on {} CPU(s).'.format(nrepetitions, ncpu))
    if ncpu == 1:
        # Do not use multiprocessing
        lnz = single_perrakis(fc, nsamples, lnl, lnp, lnlikeargs, lnpriorargs,
                              nrepetitions, None)
    else:
        # Number of repetitions per process
        nrep_proc = int(nrepetitions / ncpu)
        # List of jobs to run
        jobs = []
        for i in range(ncpu):
            p = mp.Process(target=single_perrakis, args=[fc, nsamples, lnl,
                                                         lnp, lnlikeargs,
                                                         lnpriorargs,
                                                         nrep_proc, q])
            jobs.append(p)
            p.start()
            time.sleep(1)
        # Wait until all jobs are done
        for p in jobs:
            p.join()
        # Recover output from jobs
        try:
            print(q.empty())
            lnz = np.concatenate([q.get(block=False) for p in jobs])
        except Empty:
            warnings.warn('At least one of the jobs failed to produce output.')
        try:
            len(lnz)
        except UnboundLocalError:
            raise UnboundLocalError('Critical error! No job produced any '
                                    'output. Aborting!')
    return lnz

