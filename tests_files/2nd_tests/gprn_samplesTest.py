#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(2102018)
import matplotlib.pyplot as plt
plt.close('all')

from gprn.complexGP import complexGP
from gprn import weightFunction, nodeFunction, meanFunction

##### Data .rdb file #####
time, rv, rverr, fwhm, bis, rhk, rhkerr = np.loadtxt("sampled_data.rdb", 
                                                     skiprows=1, unpack=True, 
                            usecols=(0, 1, 2, 3, 4, 5, 6))

#remaning errors
fwhmerr = 2.35 * rverr
biserr = 2.0* rverr
##data plots
#f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
#ax1.set_title('RVs, fwhm, BIS and Rhk')
#ax1.errorbar(time,rv, rverr, fmt = "b.")
#ax1.set_ylabel("RVs")
#ax2.errorbar(time,fwhm, fwhmerr, fmt = "r.")
#ax2.set_ylabel("fwhm")
#ax3.errorbar(time,bis, biserr, fmt = "g.")
#ax3.set_ylabel("BIS")
#ax4.errorbar(time,rhk, rhkerr, fmt = "y.")
#ax4.set_ylabel("Rhk")
#plt.show()


##### GP object #####
nodes = [nodeFunction.QuasiPeriodic(1, 1, 1, 1)]
weight = weightFunction.Constant(1)
weight_values = [1, 1, 1, 1]
means = [meanFunction.Constant(1), None, None, None]
jitters =[0, 0, 0, 0]
 
GPobj = complexGP(nodes, weight, weight_values, means, jitters, time, 
                  rv, rverr, fwhm, fwhmerr, bis, biserr, rhk, rhkerr)
#loglike = GPobj.log_likelihood(nodes, weight, weight_values, means)
#print(loglike)

otherloglike = GPobj.new_log_like(nodes, weight, weight_values, means, jitters)
print(otherloglike)

##### Setting priors #####
from scipy import stats

#node function
node_le = stats.uniform(1, 10 -1) 
node_p = stats.uniform(5, 20-5) 
node_lp = stats.uniform(np.exp(-10), 1 -np.exp(-10)) 
node_wn = stats.uniform(np.exp(-10), np.exp(-9) -np.exp(-10))

#weight function
weight_1 = stats.uniform(np.exp(-10), 50 -np.exp(-10))

#mean function
mean_c = stats.uniform(1, 100 -1)

#jitter
jitt = stats.uniform(np.exp(-10), 50 -np.exp(-10))

def from_prior():
    return np.array([node_le.rvs(), node_p.rvs(), node_lp.rvs(), node_wn.rvs(),
                     weight_1.rvs(), weight_1.rvs(), weight_1.rvs(), weight_1.rvs(),
                     mean_c.rvs(), 
                     jitt.rvs(), jitt.rvs(), jitt.rvs(), jitt.rvs()])

    
##### MCMC properties #####
import emcee
runs, burns = 5000, 5000 #Defining runs and burn-ins

#Probabilistic model
def logprob(p):
    if any([p[0] < np.log(1), p[0] > np.log(10), 
            p[1] < np.log(5), p[1] > np.log(20), 
            p[2] < -10, p[2] > np.log(1), 
            p[3] < -10, p[3] > -9, 
            
            p[4] < -10, p[4] > np.log(50),
            p[5] < -10, p[5] > np.log(50),
            p[6] < -10, p[6] > np.log(50),
            p[7] < -10, p[7] > np.log(50),

            p[8] < np.log(1), p[8] > np.log(100),
            
            p[9] < -10, p[9] > np.log(50),
            p[10] < -10, p[10] > np.log(50),
            p[11] < -10, p[11] > np.log(50),
            p[12] < -10, p[12] > np.log(50)]):
        return -np.inf
    else:
        logprior = 0.0
        new_node = [nodeFunction.QuasiPeriodic(np.exp(p[0]), np.exp(p[1]), 
                                               np.exp(p[2]), np.exp(p[3]))]
        new_weight_values = [np.exp(p[4]), np.exp(p[5]), np.exp(p[6]), np.exp(p[7])]
        new_mean = [meanFunction.Constant(np.exp(p[8])), None, None, None]
        new_jitt = [np.exp(p[9]), np.exp(p[10]), np.exp(p[11]), np.exp(p[12])]
        return logprior + GPobj.new_log_like(new_node, weight, 
                                             new_weight_values, new_mean, 
                                             new_jitt)

#Seting up the sampler
nwalkers, ndim = 2*13, 13
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
                    labels=["eta2", "eta3", "eta4", "s", 
                            "w1", "w2", "w3", "w4", "offset", 
                            "jitter 1", "jitter 2", "jitter 3", "jitter 4"],
                    show_titles=True)


fig, axes = plt.subplots(13, figsize=(10, 7), sharex=True)
labels=["eta2", "eta3", "eta4", "s", 
                            "w1", "w2", "w3", "w4", "offset", 
                            "jitter 1", "jitter 2", "jitter 3", "jitter 4"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(np.exp(sampler.chain[:, :, i]).T, "k", alpha=0.3)
    ax.set_xlim(runs, runs+burns)
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number");




#median and quantiles
l1,p1,l2,wn1, w1, w2,w3,w4, c1, j1, j2, j3, j4 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print('weight 1 = {0[0]} +{0[1]} -{0[2]}'.format(w1))
print('weight 2 = {0[0]} +{0[1]} -{0[2]}'.format(w2))
print('weight 3 = {0[0]} +{0[1]} -{0[2]}'.format(w3))
print('weight 4 = {0[0]} +{0[1]} -{0[2]}'.format(w4))
print()
print('offset = {0[0]} +{0[1]} -{0[2]}'.format(c1))
print()
print('jitter 1 = {0[0]} +{0[1]} -{0[2]}'.format(j1))
print('jitter 2 = {0[0]} +{0[1]} -{0[2]}'.format(j2))
print('jitter 3 = {0[0]} +{0[1]} -{0[2]}'.format(j3))
print('jitter 4 = {0[0]} +{0[1]} -{0[2]}'.format(j4))
print()

#plt.figure()
#for i in range(sampler.lnprobability.shape[0]):
#    plt.plot(sampler.lnprobability[i, :])
#
#
###### likelihood calculations #####
#likes=[]
#for i in range(samples[:,0].size):
#    new_node = [nodeFunction.QuasiPeriodic(samples[i,0], samples[i,1], 
#                                           samples[i,2], samples[i,3])]
#    new_weight = [samples[i,4], samples[i,5], samples[i,6], samples[i,7]]
#    new_means = [meanFunction.Constant(samples[i,8]), None, None, None]
#    new_jitt = [samples[i,9], samples[i,10], samples[i,11], samples[i,12]]
#    likes.append(GPobj.new_log_like(new_node, weight, new_weight, new_means,
#                                    new_jitt))
##plt.figure()
##plt.hist(likes, bins = 15, label='likelihood')
#
#datafinal = np.vstack([samples.T,np.array(likes).T]).T
#np.save('samples_results.npy', datafinal)
#
#
###### checking the likelihood that matters to us #####
#samples = datafinal
#values = np.where(samples[:,-1] > -1000)
##values = np.where(samples[:,-1] < -300)
#likelihoods = samples[values,-1].T
#
#samples = samples[values,:]
#samples = samples.reshape(-1, 14)
#
##median and quantiles
#l1,p1,l2,wn1, w1, w2,w3,w4, c1, j1, j2, j3, j4, logl = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))
#
##printing results
#print('FINAL SOLUTION')
#print()
#print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
#print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
#print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
#print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
#print()
#print('weight 1 = {0[0]} +{0[1]} -{0[2]}'.format(w1))
#print('weight 2 = {0[0]} +{0[1]} -{0[2]}'.format(w2))
#print('weight 3 = {0[0]} +{0[1]} -{0[2]}'.format(w3))
#print('weight 4 = {0[0]} +{0[1]} -{0[2]}'.format(w4))
#print()
#print('offset = {0[0]} +{0[1]} -{0[2]}'.format(c1))
#print()
#print('jitter 1 = {0[0]} +{0[1]} -{0[2]}'.format(j1))
#print('jitter 2 = {0[0]} +{0[1]} -{0[2]}'.format(j2))
#print('jitter 3 = {0[0]} +{0[1]} -{0[2]}'.format(j3))
#print('jitter 4 = {0[0]} +{0[1]} -{0[2]}'.format(j4))
#print()
#
###final result
##nodes = [nodeFunction.QuasiPeriodic(l1[0], p1[0], l2[0], wn1[0])]
##weight = weightFunction.Constant(0)
##weight_values = [w1[0]]
##means = [meanFunction.Keplerian(k11[0], k12[0], k13[0], k14[0], k15[0]) \
##                    + meanFunction.Keplerian(k21[0], k22[0], k23[0], k24[0], k25[0])]
##loglike = GPobj.log_likelihood(nodes, weight, weight_values, means)
##print(loglike)
##
##
####### final plots #####
##mu11, std11, cov11 = GPobj.predict_gp(nodes = nodes, weight = weight, 
##                                      weight_values = weight_values, means = None,
##                                      time = np.linspace(time.min(), time.max(), 5000),
##                                      dataset = 1)
##
##
##f, (ax1) = plt.subplots(1, sharex=True)
##ax1.set_title(' ')
##ax1.fill_between(np.linspace(time.min(), time.max(), 5000), 
##                 mu11+std11, mu11-std11, color="grey", alpha=0.5)
##ax1.plot(np.linspace(time.min(), time.max(), 5000), mu11, "k--", alpha=1, lw=1.5)
##ax1.errorbar(time,rv, rverr, fmt = "b.")
##ax1.set_ylabel("RVs")
##plt.show()