#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

from gprn.simpleGP import simpleGP
from gprn import weightFunction, nodeFunction

from scipy import stats
import emcee


##### Data #####
phase, rv = np.loadtxt("data/SOAP_2spots.rdb",
                                  skiprows=2, unpack=True, 
                                  usecols=(0, 2))
phase = np.concatenate((phase,1+phase, 2+phase))
rv = np.concatenate((rv,rv, rv))

t = 25.05 * phase 
rv = 1000 * rv * np.linspace(1, 0.2, t.size)
rms_rv = np.sqrt((1./rv.size * np.sum(rv**2)))
rverr = np.random.uniform(0.1, 0.2, t.size) * rms_rv * np.ones(rv.size)

plt.figure()
plt.errorbar(t, rv, rverr, fmt = '.')
plt.close()


##### Our GP #####
node = nodeFunction.QuasiPeriodic(10, 10, 1.1, 0.1)
weight = weightFunction.Exponential(10, 1.1)

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
#node function
node_le = stats.uniform(100, 1000 -100) #from exp(-10) to 1
node_p = stats.uniform(15, 30 -15) #from 15 to 35
node_lp = stats.uniform(0.1, 2 -0.1) #from exp(-10) to exp(10)
#node_lp = stats.halfcauchy(0,1)
#node_wn = stats.uniform(np.exp(-10), 0.1 -np.exp(-10)) #from exp(-10) to exp(10)
node_wn = stats.halfcauchy(0, 1)
#weight function
weight_const = stats.uniform(1, 100 -1) #from exp(-10) to 100
weight_ell = stats.uniform(1, 1000-1)

def from_prior():
    return np.array([node_le.rvs(), node_p.rvs(), node_lp.rvs(), node_wn.rvs(),
                     weight_const.rvs(), weight_ell.rvs()])


##### MCMC properties #####
runs, burns = 10000, 10000 #Defining runs and burn-ins

#Probabilistic model
def logprob(p):
    if any([p[0] < np.log(100), p[0] > np.log(1000), 
            p[1] < np.log(15), p[1] > np.log(30), 
            p[2] < np.log(0.1), p[2] > np.log(2), 
            
            p[4] < np.log(1), p[4] > np.log(100),
            p[5] < np.log(1), p[5] > np.log(1000),]):
        return -np.inf
    else:
        logprior = 0.0
        #new kernels
        new_weight = weightFunction.SquaredExponential(np.exp(p[4]), np.exp(p[5]))
        new_node = nodeFunction.QuasiPeriodic( np.exp(p[1]), np.exp(p[2]), 
                                              np.exp(p[3]), np.exp(p[4]))

        #print(gpOBJ.log_likelihood(new_node, new_weight, new_weight_value, mean = None))
        #print(np.exp(p))
        return logprior + gpOBJ.log_likelihood(new_node, new_weight, mean = None)

#Setingt up the sampler
nwalkers, ndim = 2*6, 6
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
l1, p1, l2, wn1,w1, w2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print('weight = {0[0]} +{0[1]} -{0[2]}'.format(w1))
print('weight_l = {0[0]} +{0[1]} -{0[2]}'.format(w2))
print()

##plotting the results
#print('graphics')
#fig, axes = plt.subplots(6, 1, sharex=True, figsize=(8, 9))
#axes[0].plot(np.exp(sampler.chain[:, burns:, 0]).T, color="k", alpha=0.4)
#axes[0].yaxis.set_major_locator(MaxNLocator(5))
#axes[0].set_ylabel("$Constant$")
#axes[1].plot(np.exp(sampler.chain[:, burns:, 1]).T, color="k", alpha=0.4)
#axes[1].yaxis.set_major_locator(MaxNLocator(5))
#axes[1].set_ylabel("$weight length scale$")
#
#axes[2].plot(np.exp(sampler.chain[:, burns:, 2]).T, color="k", alpha=0.4)
#axes[2].yaxis.set_major_locator(MaxNLocator(5))
#axes[2].set_ylabel("$APeriodic length scale$")
#axes[3].plot(np.exp(sampler.chain[:, burns:, 3]).T, color="k", alpha=0.4)
#axes[3].yaxis.set_major_locator(MaxNLocator(5))
#axes[3].set_ylabel("$Period$")
#axes[4].plot(np.exp(sampler.chain[:, burns:, 4]).T, color="k", alpha=0.4)
#axes[4].yaxis.set_major_locator(MaxNLocator(5))
#axes[4].set_ylabel("$Periodic length scale$")
#axes[5].plot(np.exp(sampler.chain[:, burns:, 5]).T, color="k", alpha=0.4)
#axes[5].yaxis.set_major_locator(MaxNLocator(5))
#axes[5].set_ylabel("$WN$")
#axes[5].set_xlabel("step number")
#fig.tight_layout(h_pad=0.0)
#plt.show()

plt.figure()
for i in range(sampler.lnprobability.shape[0]):
    plt.plot(sampler.lnprobability[i, :])


##### likelihood calculations #####
likes=[]
for i in range(samples[:,0].size):
    new_weight = weightFunction.SquaredExponential(samples[i,4], samples[i,5])
    new_node = nodeFunction.QuasiPeriodic(samples[i,0], samples[i,1], 
                                          samples[i,2], samples[i,3])
    likes.append(gpOBJ.log_likelihood(new_node, new_weight, mean = None))

#plt.figure()
#plt.hist(likes, bins = 15, label='likelihood')

datafinal = np.vstack([samples.T,np.array(likes).T]).T
np.save('test_sspots_ExpQP.npy', datafinal)


##### checking the likelihood that matters to us #####
samples = datafinal
values = np.where(samples[:,-1] > -1000)
#values = np.where(samples[:,-1] < -300)
likelihoods = samples[values,-1].T
plt.figure()
plt.hist(likelihoods)
plt.title("Likelihoood")
plt.xlabel("Value")
plt.ylabel("Samples")

samples = samples[values,:]
samples = samples.reshape(-1, 7)

l1, p1, l2, wn1, w1, w2,likes = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('FINAL RESULT')
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print('weight = {0[0]} +{0[1]} -{0[2]}'.format(w1))
print('weight_l = {0[0]} +{0[1]} -{0[2]}'.format(w2))
print()
print('likelihood = {0[0]} +{0[1]} -{0[2]}'.format(likes))
print()

final_node = nodeFunction.QuasiPeriodic(l1[0], p1[0], l2[0], wn1[0])
final_weight = weightFunction.SquaredExponential(w1[0],w2[0])
mu22, std22, cov22 = gpOBJ.predict_gp(node = final_node, weight= final_weight, 
                                      time = np.linspace(t.min(), t.max(), 500))
plt.figure()
plt.plot(np.linspace(t.min(), t.max(), 500), mu22, "k--", alpha=1, lw=1.5)
plt.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu22+std22, mu22-std22, color="grey", alpha=0.5)
plt.plot(t,rv,"b.")
plt.ylabel("RVs")