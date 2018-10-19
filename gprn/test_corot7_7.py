#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(2102018)
import matplotlib.pyplot as plt
plt.close('all')

from gprn.simpleGP import simpleGP
from gprn import weightFunction, nodeFunction, meanFunction

##### Data .rdb file #####
time, rv, rverr = np.loadtxt("corot7.txt", skiprows=2, unpack=True, usecols=(0, 1, 2))

#removinhg 'planets'
from tedi import astro
_, p1 = astro.keplerian(P = 0.85359165, K = 3.42, e = 0.12, w = 105*np.pi/180, 
                        T = 4398.21, t = time)
_, p2 = astro.keplerian(P = 3.70, K = 6.01, e = 0.12, w = 140*np.pi/180, 
                        T = 5953.3, t=time)
rv = rv - p1 -p2


##### GP object #####
node = nodeFunction.QuasiPeriodic(3.28, 22.21, 0.93, 0.88)
weight = weightFunction.Constant(9.31)
mean = meanFunction.Keplerian(P = 0.85359165, K = 3.42, e = 0.12, w = 105*np.pi/180, T0 = 4398.21) \
                    + meanFunction.Keplerian(P = 3.70, K = 6.01, e = 0.12, w = 140*np.pi/180, T0 = 5953.3)

GPobj = simpleGP(node = node, weight = weight, mean = mean, time = time, 
                  y = rv, yerr = rverr)
loglike = GPobj.log_likelihood(node, weight, mean = mean)
print(loglike)



################################################################################
i=1
if i == 0:
    print()
    raise SystemExit()
##### Seting priors #####
from scipy import stats

#node function
node_le = stats.uniform(1, 100 -1) 
node_p = stats.uniform(10, 40-10) 
node_lp = stats.uniform(0.1, 10 -0.1) 
node_wn = stats.uniform(np.exp(-10), 5 -np.exp(-10))
#node_wn = stats.cauchy(loc=0, scale=1)

#weight function
weight_1 = stats.uniform(0.1, 50 -0.1)

def from_prior():
#    wn = node_wn.rvs()
#    while wn <= 0:
#        wn = node_wn.rvs()

    return np.array([node_le.rvs(), node_p.rvs(), node_lp.rvs(), node_wn.rvs(),
                     weight_1.rvs()])

##### MCMC properties #####
import emcee
runs, burns = 50000, 50000 #Defining runs and burn-ins

#Probabilistic model
def logprob(p):
    if any([p[0] < np.log(1), p[0] > np.log(100), 
            p[1] < np.log(10), p[1] > np.log(40), 
            p[2] < np.log(0.1), p[2] > np.log(10), 
            p[3] < -10, p[3] > np.log(5),
            p[4] < np.log(0.1), p[4] > np.log(50)]):
        return -np.inf
    else:
        logprior = 0.0
        new_node = nodeFunction.QuasiPeriodic(np.exp(p[0]), np.exp(p[1]), 
                                               np.exp(p[2]), np.exp(p[3]))
        new_weight = weightFunction.Constant(np.exp(p[4]))
        return logprior + GPobj.log_likelihood(new_node, new_weight, mean = mean)

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
                    labels=["aper l-scale", "period", "per l-scale", "wn", "amp"],
                    show_titles=True)
#fig.savefig("triangle.png")


#median and quantiles
l1,p1,l2,wn1, w1 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
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

plt.figure()
for i in range(sampler.lnprobability.shape[0]):
    plt.plot(sampler.lnprobability[i, :])


i=0
if i == 0:
    print()
    raise SystemExit()


##### likelihood calculations #####
likes=[]
for i in range(samples[:,0].size):
    new_node = [nodeFunction.QuasiPeriodic(samples[i,0], samples[i,1], 
                                           samples[i,2], samples[i,3])]
    new_weight = [samples[i,4]]
    new_means = [meanFunction.Keplerian(P = 0.85359165, K = 3.42, e = 0.12, w = 105*np.pi/180, T0 = 4398.21) \
                    + meanFunction.Keplerian(P = 3.70, K = 6.01, e = 0.12, w = 140*np.pi/180, T0 = 5953.3)]
    likes.append(GPobj.log_likelihood(new_node, weight, new_weight, new_means))
#plt.figure()
#plt.hist(likes, bins = 15, label='likelihood')

datafinal = np.vstack([samples.T,np.array(likes).T]).T
np.save('test_corot7_5.npy', datafinal)


##### checking the likelihood that matters to us #####
samples = datafinal
values = np.where(samples[:,-1] > -230)
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
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print('weight = {0[0]} +{0[1]} -{0[2]}'.format(w1))
print()


#final result
nodes = [nodeFunction.QuasiPeriodic(l1[0], p1[0], l2[0], wn1[0])]
weight = weightFunction.Constant(0)
weight_values = [w1[0]]
means = [meanFunction.Keplerian(P = 0.85359165, K = 3.42, e = 0.12, w = 105*np.pi/180, T0 = 4398.21) \
                    + meanFunction.Keplerian(P = 3.70, K = 6.01, e = 0.12, w = 140*np.pi/180, T0 = 5953.3)]
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
ax1.errorbar(time,rv, rverr, fmt = "b.")
ax1.set_ylabel("RVs")
plt.show()