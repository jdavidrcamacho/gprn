#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

from gprn.complexGP import complexGP
from gprn import weightFunction, nodeFunction



##### Data #####
phase, rv = np.loadtxt("data/1spot_25points.rdb",
                                  skiprows=2, unpack=True, 
                                  usecols=(0, 2))


t = 25.05 * phase 
rv1 = 1000 * rv * np.linspace(1, 0.2, t.size)
rms_rv1 = np.sqrt(1./rv1.size * np.sum(rv1**2))
rverr1 = np.random.uniform(0.1, 0.2, t.size) * rms_rv1 * np.ones(rv1.size)

rv2 = 100 * rv * np.linspace(1, 0.2, t.size)
rms_rv2 = np.sqrt(1./rv1.size * np.sum(rv2**2))
rverr2 = np.random.uniform(0.1, 0.2, t.size) * rms_rv2 * np.ones(rv1.size)



plt.figure()
plt.errorbar(t, rv1, rverr1, fmt = '.')
plt.errorbar(t, rv2, rverr2, fmt = '.')
plt.close()

nodes = [nodeFunction.QuasiPeriodic(1, 10, 1, 0.1)] #,nodeFunction.Periodic(1,15, 0.5)]
weight = weightFunction.Constant(100)
weight_values = [10, 5] #,3,4]
means= [None, None]

GPobj = complexGP(nodes, weight, weight_values, means, t, rv1, rverr1, rv2, rverr2)

loglike = GPobj.log_likelihood(nodes, weight, weight_values, means)
print(loglike)