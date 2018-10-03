#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(2102018)
import matplotlib.pyplot as plt
plt.close('all')

from gprn.complexGP import complexGP
from gprn import weightFunction, nodeFunction


soap_file = "data/1spot_50points.rdb"
samples_file = 'test_1spot50points_2nodes2datasets_RVsBIS.npy'


phase, rv, bis = np.loadtxt(soap_file, skiprows=2, unpack=True, usecols=(0, 2, 3))
t = 25.05 * phase 
rv = rv * 1000
rms_rv = np.sqrt(1./rv.size * np.sum(rv**2))
rverr = np.random.uniform(0.1, 0.2, t.size) * rms_rv * np.ones(rv.size)
bis = bis * 1000
rms_bis = np.sqrt(1./bis.size * np.sum(bis**2))
biserr = np.random.uniform(0.1, 0.2, t.size) * rms_bis * np.ones(bis.size)

data = np.load(samples_file)
nodes = [nodeFunction.QuasiPeriodic(0,0, 0, 0.0)]
weight = weightFunction.Constant(0)
weight_values = [0, 0]
means= [None, None]

GPobj = complexGP(nodes, weight, weight_values, means, t, 
                  rv, rverr, bis, biserr)

##### checking the likelihood that matters to us #####
samples = data
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
