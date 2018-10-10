#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#np.random.seed(12345)
import matplotlib.pyplot as plt
plt.close('all')

from gprn.simpleGP import simpleGP
from gprn import weightFunction, nodeFunction

import scipy.optimize as op

##### Data #####
phase, rv = np.loadtxt("data/SOAP_1spot.rdb",
                                  skiprows=2, unpack=True, 
                                  usecols=(0, 2))
phase = np.concatenate((phase,1+phase, 2+phase))
rv = np.concatenate((rv,rv, rv))

t = 25.05 * phase 
rv = 1000 * rv * np.linspace(1, 0.2, t.size)
rms_rv = np.sqrt((1./rv.size * np.sum(rv**2)))
rverr = np.random.uniform(0.1, 0.2, t.size) * rms_rv * np.ones(rv.size)

plt.figure()
plt.errorbar(t, rv, rverr, fmt = '.')
plt.close()


#FINAL RESULT
#Aperiodic length scale = 198.2127666306486 +192.90659222684334 -61.40148235349159
#Kernel period = 29.286125227817745 +0.5366007665145496 -8.420880548771816
#Periodic length scale = 1.240356424292018 +0.024811438104522132 -0.22814266790456128
#Kernel wn = 0.10295035900779088 +0.057776360558100495 -0.008023630892870517
#
#weight = 1.2272235085352101 +0.041498308571144005 -0.045090516652675605
#weight_l = 326.1221554562227 +303.61159764212084 -170.43182230623023
#
#likelihood = -1994.4596277438975 +249.23328579903205 -640.8433431740434
##### Our GP #####
node = nodeFunction.QuasiPeriodic(500, 30, 1.1, 0.1)
weight = weightFunction.SquaredExponential(10, 1.1)

node = nodeFunction.QuasiPeriodic(198.2127666306486, 29.286125227817745,
                                  1.240356424292018, 0.10295035900779088)

weight = weightFunction.SquaredExponential(1.2272235085352101, 326.1221554562227)



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
    grad = gpOBJ.log_likelihood_gradient(new_node, new_weight, mean=None)
    return -grad


##### Optimization routine #####
p0 = np.log(np.concatenate((node.pars, weight.pars)))
results = op.minimize(likelihood, p0, jac = grad_likelihood, method='BFGS',
                      options={'disp': True})
print('result = ', np.exp(results.x))


###### Final results and fit #####
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