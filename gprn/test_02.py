#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(12345)
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
node = nodeFunction.QuasiPeriodic(1000, 25, 1.1, 0.1)
weight = weightFunction.SquaredExponential(1, 1.1)

gpOBJ = simpleGP(node = node, weight = weight, mean = None, 
                 time = t, y = rv, yerr = rverr)

##### Log marginal likelihood and fit #####
log_like = gpOBJ.log_likelihood(node, weight, mean = None)
print('initial log like =', log_like)
mu11, std11, cov11 = gpOBJ.predict_gp(node = node, weight= weight, 
                                      time = np.linspace(t.min(), t.max(), 500))
plt.figure()
plt.plot(np.linspace(t.min(), t.max(), 500), mu11, "k--", alpha=1, lw=1.5)
plt.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu11+std11, mu11-std11, color="grey", alpha=0.5)
plt.plot(t,rv,"b.")
plt.ylabel("RVs")


#####Log marginal likelihood to use in scipi.optimize #####
def likelihood(p):
    node_params = node.params_size
    new_node = nodeFunction.QuasiPeriodic(*p[:node_params])
    new_weight = weightFunction.SquaredExponential(*p[node_params:])
    ll = gpOBJ.log_likelihood(new_node, new_weight, mean = None)
    return -ll if np.isfinite(ll) else 1e25

def grad_likelihood(p):
    node_params = node.params_size
    new_node = nodeFunction.QuasiPeriodic(*p[:node_params])
    new_weight = weightFunction.SquaredExponential(*p[node_params:])
    grad = -gpOBJ.log_likelihood_gradient(new_node, new_weight, mean=None)
    return grad


##### Optimization routine #####
p0 = np.log(np.concatenate((node.pars, weight.pars)))
results = op.minimize(likelihood, p0, jac = grad_likelihood, method = 'BFGS',
                      options={'disp': True})
print('result = ', np.exp(results.x))


##### Final results and fit #####
node_params = node.params_size
final_node = nodeFunction.QuasiPeriodic(*results.x[:node_params])
final_weight = weightFunction.SquaredExponential(*results.x[node_params:])
log_like = gpOBJ.log_likelihood(final_node, final_weight, mean = None)
print('final log like', log_like)
mu22, std22, cov22 = gpOBJ.predict_gp(node = final_node, weight= final_weight, 
                                      time = np.linspace(t.min(), t.max(), 500))
plt.figure()
plt.plot(np.linspace(t.min(), t.max(), 500), mu22, "k--", alpha=1, lw=1.5)
plt.fill_between(np.linspace(t.min(), t.max(), 500), 
                 mu22+std22, mu22-std22, color="grey", alpha=0.5)
plt.plot(t,rv,"b.")
plt.ylabel("RVs")