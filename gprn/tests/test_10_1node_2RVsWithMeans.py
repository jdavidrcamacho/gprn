#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(2102018)
import matplotlib.pyplot as plt
plt.close('all')

from gprn.complexGP import complexGP
from gprn import weightFunction, nodeFunction, meanFunction



##### Data #####
phase, rv = np.loadtxt("data/1spot_100points.rdb",
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
weight = weightFunction.Constant(0)
weight_values = [10, 5]
means1 = [meanFunction.Sine(10, np.pi, 0), 
        meanFunction.Sine(5, np.pi/2, 0)]
GPobj1 = complexGP(nodes, weight, weight_values, means1, t, 
                  rv1, rverr1, rv2, rverr2)
loglike = GPobj1.log_likelihood(nodes, weight, weight_values, means1)
print(loglike)

means2 = [meanFunction.Keplerian(10,10, 0.5, 0,0), 
        meanFunction.Keplerian(7, 14, 0.9, 0,0)]
GPobj2 = complexGP(nodes, weight, weight_values, means2, t, 
                  rv1, rverr1, rv2, rverr2)
loglike = GPobj2.log_likelihood(nodes, weight, weight_values, means2)
print(loglike)

means3 = [meanFunction.Keplerian(10,10, 0.5, 0,0) + meanFunction.Keplerian(7, 14, 0.9, 0,0),
        meanFunction.Sine(10, np.pi, 0)]
GPobj3 = complexGP(nodes, weight, weight_values, means3, t, 
                  rv1, rverr1, rv2, rverr2)
loglike = GPobj3.log_likelihood(nodes, weight, weight_values, means3)
print(loglike)


##### final plots #####
mu11, std11, cov11 = GPobj1.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = means1,
                                      time = np.linspace(t.min(), t.max(), 500),
                                      dataset = 1)
mu22, std22, cov22 = GPobj1.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = means1,
                                      time = np.linspace(t.min(), t.max(), 500),
                                      dataset = 2)

f, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.set_title('GP object 1')
ax1.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu11+std11, mu11-std11, color="grey", alpha=0.5)
ax1.plot(np.linspace(t.min(), t.max(), 500), mu11, "k--", alpha=1, lw=1.5)
ax1.errorbar(t,rv1, rverr1, fmt = "b.")
ax1.set_ylabel("RVs")
ax2.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu22+std22, mu22-std22, color="grey", alpha=0.5)
ax2.plot(np.linspace(t.min(), t.max(), 500), mu22, "k--", alpha=1, lw=1.5)
ax2.errorbar(t, rv2, rverr2, fmt = "b.")
ax2.set_ylabel("RVs")
plt.show()

#####

mu11, std11, cov11 = GPobj2.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = means2,
                                      time = np.linspace(t.min(), t.max(), 500),
                                      dataset = 1)
mu22, std22, cov22 = GPobj2.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = means2,
                                      time = np.linspace(t.min(), t.max(), 500),
                                      dataset = 2)

f, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.set_title('GP object 2')
ax1.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu11+std11, mu11-std11, color="grey", alpha=0.5)
ax1.plot(np.linspace(t.min(), t.max(), 500), mu11, "k--", alpha=1, lw=1.5)
ax1.errorbar(t,rv1, rverr1, fmt = "b.")
ax1.set_ylabel("RVs")
ax2.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu22+std22, mu22-std22, color="grey", alpha=0.5)
ax2.plot(np.linspace(t.min(), t.max(), 500), mu22, "k--", alpha=1, lw=1.5)
ax2.errorbar(t, rv2, rverr2, fmt = "b.")
ax2.set_ylabel("RVs")
plt.show()

#####

mu11, std11, cov11 = GPobj3.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = means3,
                                      time = np.linspace(t.min(), t.max(), 500),
                                      dataset = 1)
mu22, std22, cov22 = GPobj3.predict_gp(nodes = nodes, weight = weight, 
                                      weight_values = weight_values, means = means3,
                                      time = np.linspace(t.min(), t.max(), 500),
                                      dataset = 2)

f, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.set_title('GP object 3')
ax1.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu11+std11, mu11-std11, color="grey", alpha=0.5)
ax1.plot(np.linspace(t.min(), t.max(), 500), mu11, "k--", alpha=1, lw=1.5)
ax1.errorbar(t,rv1, rverr1, fmt = "b.")
ax1.set_ylabel("RVs")
ax2.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu22+std22, mu22-std22, color="grey", alpha=0.5)
ax2.plot(np.linspace(t.min(), t.max(), 500), mu22, "k--", alpha=1, lw=1.5)
ax2.errorbar(t, rv2, rverr2, fmt = "b.")
ax2.set_ylabel("RVs")
plt.show()

