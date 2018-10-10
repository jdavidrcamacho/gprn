#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(2102018)
import matplotlib.pyplot as plt
plt.close('all')

from gprn.complexGP import complexGP
from gprn import weightFunction, nodeFunction, meanFunction

##### Data #####
time, rv, rverr = np.loadtxt("corot7.txt", skiprows=2, unpack=True, 
                            usecols=(0, 1, 2))

plt.figure()
plt.errorbar(time, rv, rverr, fmt = '.')


##### GP object #####
nodes = [nodeFunction.QuasiPeriodic(1, 10, 1, 0.1)]
weight = weightFunction.Constant(0)
weight_values = [10]
means= [meanFunction.Keplerian(1, 10, 0.5, 0, 0) + meanFunction.Keplerian(0.5, 7, 0.5, 0, 0)]

GPobj = complexGP(nodes, weight, weight_values, means, time, rv, rverr)
loglike = GPobj.log_likelihood(nodes, weight, weight_values, means)
print(loglike)


##### Seting priors #####
from scipy import stats

#node function
node_le = stats.uniform(1, np.exp(10) -1) 
node_p = stats.uniform(10, 40-10) 
node_lp = stats.uniform(np.exp(-10), 2 -np.exp(-10)) 
node_wn = stats.uniform(np.exp(-10), 1 -np.exp(-10))

#weight function
weight_1 = stats.uniform(1, 20 -1)

#mean function
#p, k, e, w, t0
mean_p1 = stats.uniform(np.exp(-10), 2 -np.exp(-10))
mean_k1 = stats.uniform(np.exp(-10), 5 -np.exp(-10))
mean_e1 = stats.uniform(np.exp(-10), 0.9 -np.exp(-10))
mean_w1 = stats.uniform(np.exp(-10), 2*np.pi -np.exp(-10))
mean_t1 = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))

mean_p2 = stats.uniform(np.exp(-10), 5 -np.exp(-10))
mean_k2 = stats.uniform(np.exp(-10), 10 -np.exp(-10))
mean_e2 = stats.uniform(np.exp(-10), 0.9 -np.exp(-10))
mean_w2 = stats.uniform(np.exp(-10), 2*np.pi -np.exp(-10))
mean_t2 = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))

def from_prior():
    return np.array([node_le.rvs(), node_p.rvs(), node_lp.rvs(), node_wn.rvs(),
                     weight_1.rvs(),
                     mean_p1.rvs(), mean_k1.rvs(), mean_e1.rvs(), mean_w1.rvs(), mean_t1.rvs(),
                     mean_p2.rvs(), mean_k2.rvs(), mean_e2.rvs(), mean_w2.rvs(), mean_t2.rvs()])

    
##### MCMC properties #####
import emcee
runs, burns = 5000, 5000 #Defining runs and burn-ins

#Probabilistic model
def logprob(p):
    if any([p[0] < np.log(1), p[0] > 10, 
            p[1] < np.log(10), p[1] > np.log(40), 
            p[2] < -10, p[2] > np.log(2), 
            p[3] < -10, p[3] > np.log(1), 
            
            p[4] < np.log(1), p[4] > np.log(20),
            
            p[5] < -10, p[5] > np.log(2),
            p[6] < -10, p[6] > np.log(5),
            p[7] < -10, p[7] > np.log(0.9),
            p[8] < -10, p[8] > np.log(2*np.pi),
            p[9] < -10, p[9] > 10,
            
            p[10] < -10, p[10] > np.log(5),
            p[11] < -10, p[11] > np.log(10),
            p[12] < -10, p[12] > np.log(0.9),
            p[13] < -10, p[13] > np.log(2*np.pi),
            p[14] < -10, p[14] > 10 ]):
        return -np.inf
    else:
        logprior = 0.0
        new_node = [nodeFunction.QuasiPeriodic(np.exp(p[0]), np.exp(p[1]), 
                                               np.exp(p[2]), np.exp(p[3]))]
        new_weight_values = [np.exp(p[4])]
        new_mean = [meanFunction.Keplerian(np.exp(p[5]), np.exp(p[6]), np.exp(p[7]), np.exp(p[8]), np.exp(p[9])) \
                    + meanFunction.Keplerian(np.exp(p[10]), np.exp(p[11]), np.exp(p[12]), np.exp(p[13]), np.exp(p[14]))]
        
        return logprior + GPobj.log_likelihood(new_node, weight, 
                                               new_weight_values, new_mean)

#Seting up the sampler
nwalkers, ndim = 2*15, 15
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
l1,p1,l2,wn1, w1, k11,k12,k13,k14,k15, k21,k22,k23,k24,k25 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print('weight = {0[0]} +{0[1]} -{0[2]}'.format(w1))
print()
print('period 1 = {0[0]} +{0[1]} -{0[2]}'.format(k11))
print('k 1 = {0[0]} +{0[1]} -{0[2]}'.format(k12))
print('e 1 = {0[0]} +{0[1]} -{0[2]}'.format(k13))
print('w 1 = {0[0]} +{0[1]} -{0[2]}'.format(k14))
print('T0 1 = {0[0]} +{0[1]} -{0[2]}'.format(k15))
print()
print('period 2 = {0[0]} +{0[1]} -{0[2]}'.format(k21))
print('k 2 = {0[0]} +{0[1]} -{0[2]}'.format(k22))
print('e 2 = {0[0]} +{0[1]} -{0[2]}'.format(k23))
print('w 2 = {0[0]} +{0[1]} -{0[2]}'.format(k24))
print('T0 2 = {0[0]} +{0[1]} -{0[2]}'.format(k25))
print()

plt.figure()
for i in range(sampler.lnprobability.shape[0]):
    plt.plot(sampler.lnprobability[i, :])


##### likelihood calculations #####
likes=[]
for i in range(samples[:,0].size):
    new_node = [nodeFunction.QuasiPeriodic(samples[i,0], samples[i,1], 
                                           samples[i,2], samples[i,3])]
    new_weight = [samples[i,4]]
    new_means = [meanFunction.Keplerian(samples[i,5], samples[i,6], samples[i,7], samples[i,8], samples[i,9]) \
                    + meanFunction.Keplerian(samples[i,10], samples[i,11], samples[i,12], samples[i,13], samples[i,14])]
    likes.append(GPobj.log_likelihood(new_node, weight, new_weight, new_means))
#plt.figure()
#plt.hist(likes, bins = 15, label='likelihood')

datafinal = np.vstack([samples.T,np.array(likes).T]).T
np.save('test_corot7_justRVs.npy', datafinal)


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
samples = samples.reshape(-1, 16)

#nem median and quantiles
l1,p1,l2,wn1, w1, k11,k12,k13,k14,k15, k21,k22,k23,k24,k25, likes = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('FINAL SOLUTION')
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print('weight = {0[0]} +{0[1]} -{0[2]}'.format(w1))
print()
print('period 1 = {0[0]} +{0[1]} -{0[2]}'.format(k11))
print('k 1 = {0[0]} +{0[1]} -{0[2]}'.format(k12))
print('e 1 = {0[0]} +{0[1]} -{0[2]}'.format(k13))
print('w 1 = {0[0]} +{0[1]} -{0[2]}'.format(k14))
print('T0 1 = {0[0]} +{0[1]} -{0[2]}'.format(k15))
print()
print('period 2 = {0[0]} +{0[1]} -{0[2]}'.format(k21))
print('k 2 = {0[0]} +{0[1]} -{0[2]}'.format(k22))
print('e 2 = {0[0]} +{0[1]} -{0[2]}'.format(k23))
print('w 2 = {0[0]} +{0[1]} -{0[2]}'.format(k24))
print('T0 2 = {0[0]} +{0[1]} -{0[2]}'.format(k25))
print()