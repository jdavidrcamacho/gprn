#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(2102018)
import matplotlib.pyplot as plt
plt.close('all')

from gprn.complexGP import complexGP
from gprn import weightFunction, nodeFunction


##### Data #####
phase, rv, bis = np.loadtxt("data/1spot_100points.rdb", skiprows=2, unpack=True, 
                            usecols=(0, 2, 3))

t = 25.05 * phase 
rv = rv * 1000
rms_rv = np.sqrt(1./rv.size * np.sum(rv**2))
rverr = np.random.uniform(0.1, 0.2, t.size) * rms_rv * np.ones(rv.size)
bis = bis * 1000
rms_bis = np.sqrt(1./bis.size * np.sum(bis**2))
biserr = np.random.uniform(0.1, 0.2, t.size) * rms_bis * np.ones(bis.size)

#plt.figure()
#plt.errorbar(t, rv, rverr, fmt = '.')
#plt.errorbar(t, bis, biserr, fmt = '.')

nodes = [nodeFunction.QuasiPeriodic(1, 10, 1, 0.1)]
weight = weightFunction.Constant(0)
weight_values = [10, 5]
means= [None, None]

GPobj = complexGP(nodes, weight, weight_values, means, t, 
                  rv, rverr, bis, biserr)
loglike = GPobj.log_likelihood(nodes, weight, weight_values, means)
print(loglike)


##### Seting priors #####
from scipy import stats

#node function
node_le = stats.uniform(10, np.exp(10) -10) #from exp(-10) to 1
node_p = stats.uniform(15, 35-15) #from 15 to 35
node_lp = stats.uniform(np.exp(-10), 1 -np.exp(-10)) #from exp(-10) to exp(10)
node_wn = stats.uniform(np.exp(-10), 1 -np.exp(-10))

#weight function
weight_1 = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10)) #from exp(-10) to 100
weight_2 = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))


def from_prior():
    return np.array([node_le.rvs(), node_p.rvs(), node_lp.rvs(), node_wn.rvs(),
                     weight_1.rvs(), weight_2.rvs()])


##### MCMC properties #####
import emcee
runs, burns = 10000, 10000 #Defining runs and burn-ins

#Probabilistic model
def logprob(p):
    if any([p[0] < np.log(10), p[0] > 10, 
            p[1] < np.log(15), p[1] > np.log(35), 
            p[2] < -10, p[2] > np.log(1), 
            p[3] < -10, p[3] > np.log(1), 
            
            p[4] < -10, p[4] > 10,
            p[5] < -10, p[5] > 10]):
        return -np.inf
    else:
        logprior = 0.0
        new_node = [nodeFunction.QuasiPeriodic(np.exp(p[0]), np.exp(p[1]), 
                                               np.exp(p[2]), np.exp(p[3]))]
        new_weights = [np.exp(p[4]), np.exp(p[5])]
        return logprior + GPobj.log_likelihood(new_node, weight, new_weights, means)

#Seting up the sampler
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
l1,p1,l2,wn1, w1,w2 = map(lambda v: (v[1], v[2]-v[1], 
                                                         v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print('weight 1= {0[0]} +{0[1]} -{0[2]}'.format(w1))
print('weight 2= {0[0]} +{0[1]} -{0[2]}'.format(w2))
print()

#plt.figure()
#for i in range(sampler.lnprobability.shape[0]):
#    plt.plot(sampler.lnprobability[i, :])


##### likelihood calculations #####
likes=[]
for i in range(samples[:,0].size):
    new_node = [nodeFunction.QuasiPeriodic(samples[i,0], samples[i,1], 
                                           samples[i,2], samples[i,3])]
    new_weights = [samples[i,4], samples[i,5]]
    likes.append(GPobj.log_likelihood(new_node, weight, new_weights, means))
#plt.figure()
#plt.hist(likes, bins = 15, label='likelihood')

datafinal = np.vstack([samples.T,np.array(likes).T]).T
np.save('test_1spot100points_2nodes2datasets_RVsBIS.npy', datafinal)


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
samples = samples.reshape(-1, 7)

#median and quantiles
l1,p1,l2,wn1, w1,w2, likes = map(lambda v: (v[1], v[2]-v[1], 
                                                         v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('FINAL RESULT')
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print('weight 1= {0[0]} +{0[1]} -{0[2]}'.format(w1))
print('weight 2= {0[0]} +{0[1]} -{0[2]}'.format(w2))
print()
print('likelihood = {0[0]} +{0[1]} -{0[2]}'.format(likes))
print()

#final result
nodes = [nodeFunction.QuasiPeriodic(l1[0], p1[0], l2[0], wn1[0])]
weight = weightFunction.Constant(0)
weight_values = [w1[0], w2[0]]
loglike = GPobj.log_likelihood(nodes, weight, weight_values, means)
print(loglike)


##### final plots #####
mu11, std11, cov11 = GPobj.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = None,
                                      time = np.linspace(t.min(), t.max(), 500),
                                      dataset = 1)
mu22, std22, cov22 = GPobj.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = None,
                                      time = np.linspace(t.min(), t.max(), 500),
                                      dataset = 2)


f, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.set_title(' ')
ax1.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu11+std11, mu11-std11, color="grey", alpha=0.5)
ax1.plot(np.linspace(t.min(), t.max(), 500), mu11, "k--", alpha=1, lw=1.5)
ax1.errorbar(t,rv, rverr, fmt = "b.")
ax1.set_ylabel("RVs")

ax2.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu22+std22, mu22-std22, color="grey", alpha=0.5)
ax2.plot(np.linspace(t.min(), t.max(), 500), mu22, "k--", alpha=1, lw=1.5)
ax2.errorbar(t, bis, biserr, fmt = "b.")
ax2.set_ylabel("BIS")

plt.show()
