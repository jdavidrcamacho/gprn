#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(2102018)
import matplotlib.pyplot as plt
plt.close('all')

from gprn.complexGP import complexGP
from gprn import weightFunction, nodeFunction



##### Data #####
phase, rv = np.loadtxt("data/1spot_25points.rdb",
                                  skiprows=2, unpack=True, 
                                  usecols=(0, 2))


t = 25.05 * phase 
rv1 = 1000 * rv * np.linspace(1, 0.2, t.size)
rms_rv1 = np.sqrt(1./rv1.size * np.sum(rv1**2))
rverr1 = np.random.uniform(0.1, 0.2, t.size) * rms_rv1 * np.ones(rv1.size)

rv2 = rv1/2
rms_rv2 = np.sqrt(1./rv2.size * np.sum(rv2**2))
rverr2 = 2 * np.random.uniform(0.1, 0.2, t.size) * rms_rv2 * np.ones(rv2.size)

#plt.figure()
#plt.errorbar(t, rv1, rverr1, fmt = '.')
#plt.errorbar(t, rv2, rverr2, fmt = '.')

nodes = [nodeFunction.QuasiPeriodic(1, 10, 1, 0.1)]
weight = weightFunction.Cosine(1,1)
weight_values = [10, 5]
means= [None, None]

GPobj = complexGP(nodes, weight, weight_values, means, t, 
                  rv1, rverr1, rv2, rverr2)
loglike = GPobj.log_likelihood(nodes, weight, weight_values, means)
print(loglike)


##### Seting priors #####
from scipy import stats
import emcee

#node function
node_le = stats.uniform(10, np.exp(10) -10) #from exp(-10) to 1
node_p = stats.uniform(15, 35-15) #from 15 to 35
node_lp = stats.uniform(np.exp(-10), 1 -np.exp(-10)) #from exp(-10) to exp(10)
node_wn = stats.uniform(np.exp(-10), 1 -np.exp(-10))
#weight function
weight_1w = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10)) #from exp(-10) to 100
weight_1p = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))
weight_2w = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10)) #from exp(-10) to 100
weight_2p = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))

def from_prior():
    return np.array([node_le.rvs(), node_p.rvs(), node_lp.rvs(), node_wn.rvs(),
                     weight_1w.rvs(), weight_1p.rvs(),
                     weight_2w.rvs(), weight_2p.rvs()])

    
##### MCMC properties #####
runs, burns = 10000, 10000 #Defining runs and burn-ins

#Probabilistic model
def logprob(p):
    if any([p[0] < np.log(10), p[0] > 10, 
            p[1] < np.log(15), p[1] > np.log(35), 
            p[2] < -10, p[2] > np.log(1), 
            p[3] < -10, p[3] > np.log(1), 
            
            p[4] < -10, p[4] > 10,
            p[5] < -10, p[5] > 10,
            p[6] < -10, p[6] > 10,
            p[7] < -10, p[7] > 10]):
        return -np.inf
    else:
        logprior = 0.0
        #new nodes
        new_node = [nodeFunction.QuasiPeriodic( np.exp(p[0]), np.exp(p[1]), 
                                              np.exp(p[2]), np.exp(p[3]))]
        new_weights = [np.exp(p[4]), np.exp(p[5]),np.exp(p[6]), np.exp(p[7])]
        return logprior + GPobj.log_likelihood(new_node, weight, new_weights, means)

#Seting up the sampler
nwalkers, ndim = 2*8, 8
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
l1, p1, l2, wn1, w1w, w1p, w2w, w2p = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print('weight 1 amp = {0[0]} +{0[1]} -{0[2]}'.format(w1w))
print('weight 1 per = {0[0]} +{0[1]} -{0[2]}'.format(w1p))
print('weight 2 amp = {0[0]} +{0[1]} -{0[2]}'.format(w2w))
print('weight 2 per = {0[0]} +{0[1]} -{0[2]}'.format(w2p))
print()

#plt.figure()
#for i in range(sampler.lnprobability.shape[0]):
#    plt.plot(sampler.lnprobability[i, :])


##### likelihood calculations #####
likes=[]
for i in range(samples[:,0].size):
    new_node = [nodeFunction.QuasiPeriodic(samples[i,0], samples[i,1], 
                                          samples[i,2], samples[i,3])]
    new_weights = [samples[i,4], samples[i,5], samples[i,6], samples[i,7]]
    likes.append(GPobj.log_likelihood(new_node, weight, new_weights, means))
#plt.figure()
#plt.hist(likes, bins = 15, label='likelihood')

datafinal = np.vstack([samples.T,np.array(likes).T]).T
np.save('test_1spot25points_1node2datasets_RVsRVs.npy', datafinal)


##### checking the likelihood that matters to us #####
samples = datafinal
values = np.where(samples[:,-1] > -5000)
#values = np.where(samples[:,-1] < -300)
likelihoods = samples[values,-1].T
#plt.figure()
#plt.hist(likelihoods)
#plt.title("Likelihoood")
#plt.xlabel("Value")
#plt.ylabel("Samples")

samples = samples[values,:]
samples = samples.reshape(-1,9)
l1, p1, l2, wn1, w1w, w1p, w2w, w2p, likes = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('FINAL RESULT')
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print('weight 1 amp = {0[0]} +{0[1]} -{0[2]}'.format(w1w))
print('weight 1 per = {0[0]} +{0[1]} -{0[2]}'.format(w1p))
print('weight 2 amp = {0[0]} +{0[1]} -{0[2]}'.format(w2w))
print('weight 2 per = {0[0]} +{0[1]} -{0[2]}'.format(w2p))
print()


#final result
nodes = [nodeFunction.QuasiPeriodic(l1[0], p1[0], l2[0], wn1[0])]
weight = weightFunction.Cosine(1,1)
weight_values = [w1w[0], w1p[0], w2w[0], w2p[0]]


##### final plots #####
mu11, std11, cov11 = GPobj.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = None,
                                      time = np.linspace(t.min(), t.max(), 500),
                                      dataset = 1)
plt.figure()
plt.plot(np.linspace(t.min(), t.max(), 500), mu11, "k--", alpha=1, lw=1.5)
plt.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu11+std11, mu11-std11, color="grey", alpha=0.5)
plt.errorbar(t,rv1, rverr1, fmt = "b.")
plt.ylabel("RVs")

mu22, std22, cov22 = GPobj.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = None,
                                      time = np.linspace(t.min(), t.max(), 500),
                                      dataset = 2)
plt.figure()
plt.plot(np.linspace(t.min(), t.max(), 500), mu22, "k--", alpha=1, lw=1.5)
plt.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu22+std22, mu22-std22, color="grey", alpha=0.5)
plt.errorbar(t,rv2, rverr2, fmt = "b.")
plt.ylabel("RVs")

