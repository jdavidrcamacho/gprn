#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

from gprn.simpleGP import simpleGP
from gprn import weightFunction, nodeFunction

from matplotlib.ticker import MaxNLocator
from scipy import stats
import emcee


##### Data #####
phase, rv = np.loadtxt("data/SOAP_2spots.rdb",
                                  skiprows=2, unpack=True, 
                                  usecols=(0, 2))
t = 25.05 * phase
rv = 1000 * rv
rms_rv = np.sqrt((1./rv.size * np.sum(rv**2)))
rverr = np.random.uniform(0.1, 0.5, 100) * rms_rv * np.ones(rv.size)

plt.figure()
plt.errorbar(t, rv, rverr, fmt = '.')
plt.close()

##### Our GP #####
node = nodeFunction.QuasiPeriodic(10, 10, 1, 0.1)
weight = weightFunction.Constant(1)

gpOBJ = simpleGP(node = node, weight = weight, mean = None, 
                 time = t, y = rv, yerr = rverr)

##### Log marginal likelihood #####
log_like = gpOBJ.log_likelihood(node, weight, mean = None)
print(log_like)


##### Plots #####
#mu11, std11, cov11 = gpOBJ.predict_gp(weight_value = 1, time = np.linspace(t.min(), t.max(), 500))
#
#plt.figure()
#plt.plot(np.linspace(t.min(), t.max(), 500), mu11, "k--", alpha=1, lw=1.5)
#plt.fill_between(np.linspace(t.min(), t.max(), 500), 
#                 mu11+std11, mu11-std11, color="grey", alpha=0.5)
#plt.plot(t,rv,"b.")
#plt.ylabel("RVs")


##### Seting priors #####
#weight function
#weight_const = stats.uniform(1, 100 -1) #from exp(-10) to 100
weight_const = 1
#node function
node_le = stats.uniform(100, 1000 -100) #from exp(-10) to 1
node_p = stats.uniform(15, 50 -15) #from 15 to 35
node_lp = stats.uniform(np.exp(-10), 1 -np.exp(-10)) #from exp(-10) to exp(10)
#node_lp = stats.halfcauchy(0,1)
#node_wn = stats.uniform(np.exp(-10), 0.1 -np.exp(-10)) #from exp(-10) to exp(10)
node_wn = stats.halfcauchy(0,1)


def from_prior():
    wn = node_wn.rvs()
    #to truncate the wn between 0 and 0.1
    while wn > 1:
        wn = node_wn.rvs()
    #to truncate the lp between 0 and 1
#    lp = node_lp.rvs()
#    while lp > 1:
#        lp = node_lp.rvs()

    return np.array([weight_const, node_le.rvs(), node_p.rvs(), 
                     node_lp.rvs(), wn])
#    return np.array([weight_const.rvs(), node_le.rvs(), node_p.rvs(), 
#                     node_lp.rvs(), node_wn.rvs(), weight_amp.rvs()])


##### MCMC properties #####
runs, burns = 50, 50 #Defining runs and burn-ins

#Probabilistic model
def logprob(p):
    if any([
            p[1] < np.log(100), p[1] > np.log(1000), 
            p[2] < np.log(15), p[2] > np.log(50), 
            p[3] < -10, p[3] > np.log(1), 
            p[4] < -100, p[4] > np.log(1)]):
        return -np.inf
    else:
        logprior = 0.0
        #new kernel
        new_weight = weightFunction.Constant(np.exp(p[0]))
        #new constants
        new_node = nodeFunction.QuasiPeriodic( np.exp(p[1]), np.exp(p[2]), 
                                              np.exp(p[3]), np.exp(p[4]))

        #print(gpOBJ.log_likelihood(new_node, new_weight, new_weight_value, mean = None))
        #print(np.exp(p))
        return logprior + gpOBJ.log_likelihood(new_node, new_weight, mean = None)

#Setingt up the sampler
nwalkers, ndim = 2*5, 5
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, threads= 4)

#Initialize the walkers
p0=[np.log(from_prior()) for i in range(nwalkers)]

print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, burns)
print("Running production chain")
sampler.run_mcmc(p0, runs);


##### MCMC analysis #####
burnin = burns
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
samples = np.exp(samples)

#median and quantiles
w1, l1, p1, l2, wn1 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('Constant = {0[0]} +{0[1]} -{0[2]}'.format(w1))
print()
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()


#plotting the results
print('graphics')
fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 9))
axes[0].plot(np.exp(sampler.chain[:, burns:, 0]).T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].set_ylabel("$Constant$")
axes[1].plot(np.exp(sampler.chain[:, burns:, 1]).T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].set_ylabel("$APeriodic length scale$")
axes[2].plot(np.exp(sampler.chain[:, burns:, 2]).T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].set_ylabel("$Period$")
axes[3].plot(np.exp(sampler.chain[:, burns:, 3]).T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].set_ylabel("$Periodic length scale$")
axes[4].plot(np.exp(sampler.chain[:, burns:, 4]).T, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].set_ylabel("$WN$")
axes[4].set_xlabel("step number")
fig.tight_layout(h_pad=0.0)
plt.show()

plt.figure()
for i in range(sampler.lnprobability.shape[0]):
    plt.plot(sampler.lnprobability[i, :])