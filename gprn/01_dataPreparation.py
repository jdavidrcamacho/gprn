#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#np.random.seed(22102018)
import matplotlib.pyplot as plt
plt.close('all')


##### SAMPLING #####
from tedi import process, kernels

#QP(amp, ell_e, P, ell_p. wn)
kernel = kernels.QuasiPeriodic(10, 50, 31, 1, 0.8)
mean = None
time = np.linspace(1, 100, 50)
y = np.ones_like(time)
yerr = 0.1*y

GPobj = process.GP(kernel, mean, time, y, yerr)
sample = GPobj.sample(kernel, time)

#plt.figure()
#plt.plot(time, sample)


##### GPRN construction
from gprn.complexGP import complexGP
from gprn import weightFunction, nodeFunction, meanFunction

time = time
y = sample
rms_y = np.sqrt((1./y.size * np.sum(y**2)))
yerr = 0.2 * rms_y * np.ones(y.size)

#plt.figure()
#plt.errorbar(time, y, yerr, fmt='.')

# GP object 
nodes = [nodeFunction.QuasiPeriodic(50, 31, 1, 0.1)]
weight = weightFunction.Constant(1)
weight_values = [10]
means = [None]

GPobj = complexGP(nodes, weight, weight_values, means, time, y, yerr)
loglike = GPobj.log_likelihood(nodes, weight, weight_values, means)
print(loglike)


################################################################################
i=1
if i == 0:
    print()
    raise SystemExit()
##### Seting priors #####
from scipy import stats

#node function
node_le = stats.uniform(1, 500 -1) 
node_p = stats.uniform(10, 50-10) 
node_lp = stats.uniform(0.1, 2 -0.1) 
node_wn = stats.uniform(0.1, 1.5 -0.1)

#weight function
weight_1 = stats.uniform(1, 50 -1)

def from_prior():
    return np.array([node_le.rvs(), node_p.rvs(), node_lp.rvs(), node_wn.rvs(),
                     weight_1.rvs()])

##### MCMC properties #####
import emcee
runs, burns = 50000, 50000 #Defining runs and burn-ins

#Probabilistic model
def logprob(p):
    if any([p[0] < np.log(1), p[0] > np.log(500), 
            p[1] < np.log(10), p[1] > np.log(50), 
            p[2] < np.log(0.1), p[2] > np.log(2), 
            p[3] < np.log(0.1), p[3] > np.log(1.5), 
            p[4] < np.log(1), p[4] > np.log(50)]):
        return -np.inf
    else:
        logprior = 0.0
        new_node = [nodeFunction.QuasiPeriodic(np.exp(p[0]), np.exp(p[1]), 
                                               np.exp(p[2]), np.exp(p[3]))]
        new_weight_values = [np.exp(p[4])]
        new_mean = mean
        return logprior + GPobj.log_likelihood(new_node, weight, 
                                               new_weight_values, new_mean)

#Seting up the sampler
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

import corner
fig = corner.corner(samples, 
                    labels=["eta 2", "eta 3", "eta 4", "s", "eta 1"],
                    show_titles=True)
#fig.savefig("triangle.png")

#median and quantiles
l1,p1,l2,wn1, w1 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('eta 1 (amp) = {0[0]} +{0[1]} -{0[2]}'.format(w1))
print('eta 2 (ell_e) = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('eta 3 (P) = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('eta 4 (ell_p) = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('s (wn) = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print()

plt.figure()
for i in range(sampler.lnprobability.shape[0]):
    plt.plot(sampler.lnprobability[i, :])


################################################################################
i=1
if i == 0:
    print()
    raise SystemExit()


##### likelihood calculations #####
likes=[]
for i in range(samples[:,0].size):
    new_node = [nodeFunction.QuasiPeriodic(samples[i,0], samples[i,1], 
                                           samples[i,2], samples[i,3])]
    new_weight = [samples[i,4]]
    new_means = mean
    likes.append(GPobj.log_likelihood(new_node, weight, new_weight, new_means))
#plt.figure()
#plt.hist(likes, bins = 15, label='likelihood')

datafinal = np.vstack([samples.T,np.array(likes).T]).T
np.save('sampling_test1.npy', datafinal)


##### checking the likelihood that matters to us #####
samples = datafinal
values = np.where(samples[:,-1] > -500)
#values = np.where(samples[:,-1] < -300)
likelihoods = samples[values,-1].T
#plt.figure()
#plt.hist(likelihoods)
#plt.title("Likelihoood")
#plt.xlabel("Value")
#plt.ylabel("Samples")

samples = samples[values,:]
samples = samples.reshape(-1, 6)

#nem median and quantiles
l1,p1,l2,wn1, w1, likes = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('FINAL SOLUTION')
print('eta 1 (amp) = {0[0]} +{0[1]} -{0[2]}'.format(w1))
print('eta 2 (ell_e) = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('eta 3 (P) = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('eta 4 (ell_p) = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('s (wn) = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print()



#final result
nodes = [nodeFunction.QuasiPeriodic(l1[0], p1[0], l2[0], wn1[0])]
weight = weightFunction.Constant(0)
weight_values = [w1[0]]
means = None
loglike = GPobj.log_likelihood(nodes, weight, weight_values, means)
print(loglike)


##### final plots #####
mu11, std11, cov11 = GPobj.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = None,
                                      time = np.linspace(time.min(), time.max(), 5000),
                                      dataset = 1)


f, (ax1) = plt.subplots(1, sharex=True)
ax1.set_title(' ')
ax1.fill_between(np.linspace(time.min(), time.max(), 5000), 
                 mu11+std11, mu11-std11, color="grey", alpha=0.5)
ax1.plot(np.linspace(time.min(), time.max(), 5000), mu11, "k--", alpha=1, lw=1.5)
ax1.errorbar(time, y, yerr, fmt = "b.")
ax1.set_ylabel("RVs")
plt.show()