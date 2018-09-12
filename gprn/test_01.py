#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

from gprn.simpleGP import simpleGP
from gprn import weightFunction, nodeFunction

##### Data #####
phase, rv = np.loadtxt("data/SOAP_2spots.rdb",
                                  skiprows=2, unpack=True, 
                                  usecols=(0, 2))
t = 25.05 * phase
rv = 1000 * rv
rms_rv = np.sqrt((1./rv.size*np.sum(rv**2)))
rverr = 0.10*rms_rv * np.ones(rv.size)

plt.figure()
plt.errorbar(t, rv, rverr, fmt = '.')
plt.close()

##### Our GP #####
weight = weightFunction.SquaredExponential(1)
node = nodeFunction.QuasiPeriodic(0.1, 25, 0.5, 0.1)

gpOBJ = simpleGP(node = node, weight = weight, mean = None, 
                 time = t, y = rv, yerr = rverr)

##### Log marginal likelihood #####
gpOBJ.log_likelihood(node, weight, 10, mean = None)

#### Plots #####
mu11, std11, cov11 = gpOBJ.predict_gp(weight_value = 50, time = np.linspace(t.min(), t.max(), 500))

plt.figure()
plt.plot(np.linspace(t.min(), t.max(), 500), mu11, "k--", alpha=1, lw=1.5)
plt.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu11+std11, mu11-std11, color="grey", alpha=0.5)
plt.plot(t,rv,"b.")
plt.ylabel("RVs")