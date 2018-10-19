#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(2102018)
import matplotlib.pyplot as plt
plt.close('all')

from gprn.complexGP import complexGP
from gprn.simpleGP import simpleGP
from gprn import weightFunction, nodeFunction, meanFunction

data_file = "corot7_harps.rdb"
results_file = "test_corot7_3.npy"


##### Data .rdb file #####
time, rv, rverr, fwhm, bis, rhk, rhkerr = np.loadtxt(data_file, 
                                                     skiprows=112, unpack=True, 
                            usecols=(0, 1, 2, 3, 4, 5, 6))

#removing NaNs
time = time[~np.isnan(rhk)]

rv = rv[~np.isnan(rhk)]
rv_mean = np.mean(rv)
rv = (rv-rv_mean)*1000 + rv_mean
rverr = rverr[~np.isnan(rhk)]*1000

fwhm = fwhm[~np.isnan(rhk)]
bis = bis[~np.isnan(rhk)]
rhkerr = rhkerr[~np.isnan(rhk)]
rhk = rhk[~np.isnan(rhk)]

#remaning errors
rms_fwhm = np.sqrt((1./fwhm.size*np.sum(fwhm**2)))
fwhmerr = 0.001*rms_fwhm * np.ones(fwhm.size)
rms_bis = np.sqrt((1./bis.size*np.sum(bis**2)))
biserr = 0.10*rms_bis * np.ones(bis.size)


##### MCMC data #####
samples = np.load(results_file)


import corner
fig = corner.corner(samples[:,:-1], 
                    labels=["$\eta_2$", "$\eta_3$", "$\eta_4$", "s", "$\eta_1$"],
                    show_titles=True)

#checking the likelihood that matters to us
values = np.where(samples[:,-1] > -2500)
#values = np.where(samples[:,-1] < -800)
likelihoods = samples[values,-1].T
plt.figure()
plt.hist(likelihoods)
plt.title("Likelihoood")
plt.xlabel("Value")
plt.ylabel("Samples")

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
weight = weightFunction.Constant(w1[0])
weight_values = [w1[0]]
means = [None]

GPobj = complexGP(nodes, weight, weight_values, means, time, rv, rverr)
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