#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

from gprn.simpleGP import simpleGP
from gprn import weightFunction, nodeFunction

import scipy.optimize as op


##### Data #####
phase, rv = np.loadtxt("data/SOAP_2spots.rdb",
                                  skiprows=2, unpack=True, 
                                  usecols=(0, 2))
t = 25.05 * phase
rv = 1000 * rv
rms_rv = np.sqrt((1./rv.size * np.sum(rv**2)))
rverr = np.random.uniform(0.1, 0.5, 100) * rms_rv * np.ones(rv.size)

plt.figure()
plt.errorbar(t, rv, rverr, fmt = '.')
plt.close()

##### Our GP #####
node = nodeFunction.QuasiPeriodic(40.0, 10.0, 1.1, 0.1)
weight = weightFunction.Constant(1.1)

gpOBJ = simpleGP(node = node, weight = weight, mean = None, 
                 time = t, y = rv, yerr = rverr)

##### Log marginal likelihood #####
log_like = gpOBJ.log_likelihood(node, weight, mean = None)
print(log_like)

grad_log_like = gpOBJ.log_likelihood_gradient(node, weight, mean=None)
print(grad_log_like)


#####Log marginal likelihood to use in scipi.optimize
def likelihood(p):
    node_params = node.params_size
    new_node = nodeFunction.QuasiPeriodic(*p[:node_params])
    new_weight = weightFunction.Constant(*p[node_params:])
    ll = gpOBJ.log_likelihood(new_node, new_weight, mean = None)
    return -ll if np.isfinite(ll) else 1e25

def grad_likelihood(p):
    node_params = node.params_size
    new_node = nodeFunction.QuasiPeriodic(*p[:node_params])
    new_weight = weightFunction.Constant(*p[node_params:])
    grad = -gpOBJ.log_likelihood_gradient(new_node, new_weight, mean=None)
    return grad

##### Optimization routine
p0 = np.log(np.concatenate((node.pars, weight.pars)))
results = op.minimize(likelihood, p0, jac = grad_likelihood, 
                      options={'disp': True})
print('result = ', np.exp(results.x))