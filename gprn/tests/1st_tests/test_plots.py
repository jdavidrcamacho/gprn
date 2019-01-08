#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

from gprn.simpleGP import simpleGP
from gprn import weightFunction, nodeFunction


soap_file = "data/SOAP_2spots.rdb"
samples_file = 'test_2spots_M32QP.npy'


##### Data #####
phase, rv = np.loadtxt(soap_file, skiprows=2, unpack=True, usecols=(0, 2))
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
node = nodeFunction.QuasiPeriodic(0,0,0,0)
weight = weightFunction.Matern32(0,0)

gpOBJ = simpleGP(node = node, weight = weight, mean = None, 
                 time = t, y = rv, yerr = rverr)



samples = np.load(samples_file)
##### checking the likelihood that matters to us #####
values = np.where(samples[:,-1] > -5000)
#values = np.where(samples[:,-1] < -2500)
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
final_weight = weightFunction.Matern32(w1[0],w2[0])
mu22, std22, cov22 = gpOBJ.predict_gp(node = final_node, weight= final_weight, 
                                      time = np.linspace(t.min(), t.max(), 500))
plt.figure()
plt.plot(np.linspace(t.min(), t.max(), 500), mu22, "k--", alpha=1, lw=1.5)
plt.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu22+std22, mu22-std22, color="grey", alpha=0.5)
plt.plot(t,rv,"b.")
plt.ylabel("RVs")