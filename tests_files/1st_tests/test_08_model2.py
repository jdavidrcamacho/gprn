#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(8102018)
import matplotlib.pyplot as plt
plt.close('all')

from gprn.complexGP import complexGP
from gprn import weightFunction, nodeFunction

##### MODEL 2
#
#    P  ->  W11  ->  RVs
#    P  ->  W21  ->  BIS
#   SE  ->  W12  ->  RVs
#   SE  ->  W22  ->  BIS
#

##### Data #####
phase, rv, bis = np.loadtxt("data/1spot_25points.rdb", skiprows=2, unpack=True, 
                            usecols=(0, 2, 3))

t = 25.05 * phase 
rv = rv * 1000
rms_rv = np.sqrt(1./rv.size * np.sum(rv**2))
rverr = np.random.uniform(0.1, 0.2, t.size) * rms_rv * np.ones(rv.size)
bis = bis * 1000
rms_bis = np.sqrt(1./bis.size * np.sum(bis**2))
biserr = np.random.uniform(0.1, 0.2, t.size) * rms_bis * np.ones(bis.size)

#f, (ax1, ax2) = plt.subplots(2, sharex=True)
#ax1.set_title(' ')
#ax1.errorbar(t,rv, rverr, fmt = "b.")
#ax1.set_ylabel("RVs")
#ax2.errorbar(t, bis, biserr, fmt = "b.")
#ax2.set_ylabel("BIS")
#plt.show()

nodes = [nodeFunction.Periodic(1, 25, 0.1), 
         nodeFunction.SquaredExponential(100, 0.1)]
weight = weightFunction.SquaredExponential(0, 100)
weight_values = [np.sqrt(2)*rms_rv, 0, np.sqrt(2)*rms_bis, 0]
means = [None, None]

GPobj = complexGP(nodes, weight, weight_values, means, t, 
                  rv, rverr, bis, biserr)
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
ax1.set_title('Model 2')
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
