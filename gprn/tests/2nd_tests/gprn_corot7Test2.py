#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#np.random.seed(2102018)
import matplotlib.pyplot as plt
plt.close('all')

from gprn.complexGP import complexGP
from gprn import weightFunction, nodeFunction, meanFunction

###### Data .rdb file #####
time,rv,rverr,fwhm,fwhmerr,bis,biserr,rhk,rhkerr = np.loadtxt("corot7_clean.rdb", 
                                                              skiprows=102, unpack=True, 
                                                              usecols=(0,1,2,3,4,5,6,7,8))

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
means = [meanFunction.Keplerian(1,1,0.5,1,1) + meanFunction.Keplerian(1,1,0.5,1,1) \
                                                     + meanFunction.Constant(1),
         meanFunction.Constant(1), 
         meanFunction.Constant(1), 
         meanFunction.Constant(1)]
jitters =[0, 0, 0, 0]
 
GPobj = complexGP(nodes, weight, weight_values, means, jitters, time, 
                  rv, rverr, fwhm, fwhmerr, bis, biserr, rhk, rhkerr)
#loglike = GPobj.log_likelihood(nodes, weight, weight_values, means)
#print(loglike)

otherloglike = GPobj.new_log_like(nodes, weight, weight_values, means, jitters)
print(otherloglike)

##### Setting priors #####
from scipy import stats
def loguniform(low=0, high=1, size=None):
    return np.exp(stats.uniform(low, high -low).rvs())

#node function
#node_le = stats.uniform(np.exp(-10), 16 -np.exp(-10)) 
#node_le = loguniform(np.exp(-10), 16)
node_p = stats.uniform(10, 40- 10) 
node_lp = stats.uniform(np.exp(-10), 1 -np.exp(-10)) 
node_wn = stats.uniform(np.exp(-20), np.exp(-10) -np.exp(-20))

#weight function
weight_1 = stats.uniform(np.exp(-10), 50 -np.exp(-10))

#mean function
mean_Kp = stats.uniform(np.exp(-10), 10 -np.exp(-10))
mean_Kk = stats.uniform(1, 10 -1)
mean_Ke = stats.uniform(np.exp(-20), 1 -np.exp(-20))
mean_Kw = stats.uniform(np.exp(-20), 2*np.pi -1)
mean_Kphi = stats.uniform(np.exp(-20), 2*np.pi -np.exp(-20))

mean_c1 = stats.uniform(10, 50 -10)
mean_c2 = stats.uniform(1, 10 -1)
mean_c3 = stats.uniform(np.exp(-10), 10 -np.exp(-10))
mean_c4 = stats.uniform(2, 8 -2)


#jitter
jitt1 = stats.uniform(np.exp(-2), 2 -np.exp(-2))
jitt2 = stats.uniform(np.exp(-2), 2 -np.exp(-2))
jitt3 = stats.uniform(np.exp(-2), 2 -np.exp(-2))
jitt4= stats.uniform(np.exp(-2), 2 -np.exp(-2))

def from_prior():
    eta2 = loguniform(np.exp(-10), 20)
    if eta2 > 20:
        eta2 = loguniform(np.exp(-10), 20)
    else:
        eta2 = eta2
    return np.array([eta2, node_p.rvs(), node_lp.rvs(), node_wn.rvs(),
                     weight_1.rvs(), weight_1.rvs(), weight_1.rvs(), weight_1.rvs(),
                     mean_Kp.rvs(), mean_Kk.rvs(), mean_Ke.rvs(), mean_Kw.rvs(), mean_Kphi.rvs(),
                     mean_Kp.rvs(), mean_Kk.rvs(), mean_Ke.rvs(), mean_Kw.rvs(), mean_Kphi.rvs(),
                     mean_c1.rvs(), mean_c2.rvs(), mean_c3.rvs(), mean_c4.rvs(),
                     jitt1.rvs(), jitt2.rvs(), jitt3.rvs(), jitt4.rvs()])

##### MCMC properties #####
import emcee
runs, burns = 50000, 50000 #Defining runs and burn-ins

#Probabilistic model
def logprob(p):
    if any([ 
            p[1] < np.log(10), p[1] > np.log(40.0), 
            p[2] < -10, p[2] > np.log(1), 
            p[3] < -20, p[3] > -10, 
            
            p[4] < -10, p[4] > np.log(50),
            p[5] < -10, p[5] > np.log(50),
            p[6] < -10, p[6] > np.log(50),
            p[7] < -10, p[7] > np.log(50),
            
            p[8] < -10, p[8] > np.log(10),
            p[9] < np.log(1), p[9] > np.log(10),
            p[10] < -20, p[10] > np.log(1),
            p[11] < -20, p[11] > np.log(2*np.pi),
            p[12] < -20, p[12] > np.log(2*np.pi),

            p[13] < -10, p[13] > np.log(10),
            p[14] < np.log(1), p[14] > np.log(10),
            p[15] < -20, p[15] > np.log(1),
            p[16] < -20, p[16] > np.log(2*np.pi),
            p[17] < -20, p[17] > np.log(2*np.pi),

            p[18] < np.log(10), p[18] > np.log(50),
            p[19] < np.log(1), p[19] > np.log(10),
            p[20] < -10, p[20] > np.log(10),
            p[21] < np.log(2), p[21] > np.log(8),

            p[22] < -20, p[22] > np.log(2),
            p[23] < -20, p[23] > np.log(2),
            p[24] < -20, p[24] > np.log(2),
            p[25] < -20, p[25] > np.log(2)]):
        return -np.inf
    else:
        logprior = 0.0
        new_node = [nodeFunction.QuasiPeriodic(np.exp(p[0]), np.exp(p[1]), 
                                               np.exp(p[2]), np.exp(p[3]))]
    
        new_weight_values = [np.exp(p[4]), np.exp(p[5]), 
                             np.exp(p[6]), np.exp(p[7])]
        
        new_mean = [meanFunction.Keplerian(np.exp(p[8]),
                                          np.exp(p[9]),
                                          np.exp(p[10]),
                                          np.exp(p[11]),
                                          np.exp(p[12])) \
                + meanFunction.Keplerian(np.exp(p[13]),
                                          np.exp(p[14]),
                                          np.exp(p[15]),
                                          np.exp(p[16]),
                                          np.exp(p[17])) \
                + meanFunction.Constant(np.exp(p[18])), 
                    meanFunction.Constant(np.exp(p[19])),
                    meanFunction.Constant(np.exp(p[20])),
                    meanFunction.Constant(np.exp(p[21]))]
        
        new_jitt = [np.exp(p[22]), np.exp(p[23]), np.exp(p[24]), np.exp(p[25])]
        return logprior + GPobj.new_log_like(new_node, 
                                             weight, new_weight_values, 
                                             new_mean, new_jitt)

#Seting up the sampler
nwalkers, ndim = 2*26, 26
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
samples[:,0] = np.log(samples[:,0])

#median and quantiles
l1,p1,l2,wn1, w1,w2,w3,w4, \
kp1,kk1,ke1,kw1,kphi1, kp2,kk2,ke2,kw2,kphi2, \
c1, c2, c3, c4, j1, j2, j3, j4 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
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
print('keplerian 1 period= {0[0]} +{0[1]} -{0[2]}'.format(kp1))
print('keplerian 1 K = {0[0]} +{0[1]} -{0[2]}'.format(kk1))
print('keplerian 1 e = {0[0]} +{0[1]} -{0[2]}'.format(ke1))
print('keplerian 1 w = {0[0]} +{0[1]} -{0[2]}'.format(kw1))
print('keplerian 1 phi  = {0[0]} +{0[1]} -{0[2]}'.format(kphi1))
print('+')
print('keplerian 2 period= {0[0]} +{0[1]} -{0[2]}'.format(kp2))
print('keplerian 2 K = {0[0]} +{0[1]} -{0[2]}'.format(kk2))
print('keplerian 2 e = {0[0]} +{0[1]} -{0[2]}'.format(ke2))
print('keplerian 2 w = {0[0]} +{0[1]} -{0[2]}'.format(kw2))
print('keplerian 2 phi  = {0[0]} +{0[1]} -{0[2]}'.format(kphi2))
print('+')
print('keplerians offset = {0[0]} +{0[1]} -{0[2]}'.format(c1))
print()
print('offset 2 = {0[0]} +{0[1]} -{0[2]}'.format(c2))
print('offset 3 = {0[0]} +{0[1]} -{0[2]}'.format(c3))
print('offset 4 = {0[0]} +{0[1]} -{0[2]}'.format(c4))
print()
print('jitter 1 = {0[0]} +{0[1]} -{0[2]}'.format(j1))
print('jitter 2 = {0[0]} +{0[1]} -{0[2]}'.format(j2))
print('jitter 3 = {0[0]} +{0[1]} -{0[2]}'.format(j3))
print('jitter 4 = {0[0]} +{0[1]} -{0[2]}'.format(j4))
print()

plt.figure()
for i in range(sampler.lnprobability.shape[0]):
    plt.plot(sampler.lnprobability[i, :])



##### final results #####
nodes = [nodeFunction.QuasiPeriodic(l1[0], p1[0], l2[0], wn1[0])]

weight = weightFunction.Constant(0)
weight_values = [w1[0], w2[0], w3[0], w4[0]]

means = [meanFunction.Keplerian(kp1[0],kk1[0],ke1[0],kw1[0],kphi1[0]) \
            + meanFunction.Keplerian(kp2[0],kk2[0],ke2[0],kw2[0],kphi2[0]) \
            + meanFunction.Constant(c1[0]), 
         meanFunction.Constant(c2[0]), meanFunction.Constant(c3[0]), meanFunction.Constant(c4[0])]

jitters = [j1[0], j2[0], j3[0], j4[0]]

loglike = GPobj.new_log_like(nodes, weight, weight_values, means, jitters)
print(loglike)

##### final plots #####
mu11, std11, cov11 = GPobj.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min(), time.max(), 500),
                                      dataset = 1)
mu22, std22, cov22 = GPobj.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min(), time.max(), 500),
                                      dataset = 2)
mu33, std33, cov33 = GPobj.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min(), time.max(), 500),
                                      dataset = 3)
mu44, std44, cov44 = GPobj.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = means,
                                      jitters = jitters,
                                      time = np.linspace(time.min(), time.max(), 500),
                                      dataset = 4)

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.set_title('Fits')
ax1.fill_between(np.linspace(time.min(), time.max(), 500), 
                 mu11+std11, mu11-std11, color="grey", alpha=0.5)
ax1.plot(np.linspace(time.min(), time.max(), 500), mu11, "k--", alpha=1, lw=1.5)
ax1.errorbar(time,rv, rverr, fmt = "b.")
ax1.set_ylabel("RVs")

ax2.fill_between(np.linspace(time.min(), time.max(), 500), 
                 mu22+std22, mu22-std22, color="grey", alpha=0.5)
ax2.plot(np.linspace(time.min(), time.max(), 500), mu22, "k--", alpha=1, lw=1.5)
ax2.errorbar(time,fwhm, fwhmerr, fmt = "b.")
ax2.set_ylabel("FWHM")

ax3.fill_between(np.linspace(time.min(), time.max(), 500), 
                 mu33+std33, mu33-std33, color="grey", alpha=0.5)
ax3.plot(np.linspace(time.min(), time.max(), 500), mu33, "k--", alpha=1, lw=1.5)
ax3.errorbar(time,bis, biserr, fmt = "b.")
ax3.set_ylabel("BIS")

ax4.fill_between(np.linspace(time.min(), time.max(), 500), 
                 mu44+std44, mu44-std44, color="grey", alpha=0.5)
ax4.plot(np.linspace(time.min(), time.max(), 500), mu44, "k--", alpha=1, lw=1.5)
ax4.errorbar(time, rhk, rhkerr, fmt = "b.")
ax4.set_ylabel("R'hk")
plt.show()


#### corner plots
import corner
fig = corner.corner(samples[:,0:8], 
                    labels=["eta2", "eta3", "eta4", "s", 
                            "w1", "w2", "w3", "w4"],
                    show_titles=True)
fig = corner.corner(samples[:,8:13], 
                    labels=["kep1 P", "kep1 K", "kep1 e", "kep1 w", "kep1 phi"],
                    show_titles=True)
fig = corner.corner(samples[:,13:18], 
                    labels=["kep2 P", "kep2 K", "kep2 e", "kep2 w", "kep2 phi"],
                    show_titles=True)
fig = corner.corner(samples[:,18:22], 
                    labels=["offset1", "offset2", "offset3", "offset4"],
                    show_titles=True)
fig = corner.corner(samples[:,22:26], 
                    labels=["jitter 1", "jitter 2", "jitter 3", "jitter 4"],
                    show_titles=True)