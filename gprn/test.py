#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(230190)
import matplotlib.pyplot as plt
import matplotlib; matplotlib.style.use('default')
#%matplotlib notebook
plt.rcParams['figure.figsize'] = [15, 5]
plt.close('all')

#from gprn import weightFunction, nodeFunction 
from gprn.covFunction import SquaredExponential, Periodic, QuasiPeriodic, RQP, Exponential, Matern32, Matern52, RationalQuadratic
from gprn.meanFunction import Constant
from gprn.meanField import inference
from gprn.utils import run_mcmc
from tedi import process, kernels, means

from ipywidgets import interact, interactive
###### Data .rdb file #####
time,rv,rverr,bis,biserr,fwhm,fwhmerr = np.loadtxt("/home/camacho/Dropbox/Sun47points163span.txt", 
                                                   skiprows = 1, unpack = True, 
                                                   usecols = (0,1,2,3,4,5,6))

val1, val1err = rv, rverr
###### GP object #####
kernel = kernels.Periodic(1, 1, 35, 1)
mean = means.Constant(0)
tedibear = process.GP(kernel, mean, time, val1, val1err)
###### GPRN object #####
GPRN = inference(1, time, val1, val1err)

#plt.figure()
#plt.errorbar(time, val1, val1err, fmt='o')











neta1 = 1
neta3 = 27.08
neta4 = 1.23
ns = 1e-5
weta1 = 1
weta2 = 14.07
ws = 1e-5
meanval = 101.97
jitt = 1.0
iternum = 100

extention = 20

#GPRN
nodes = [Periodic(neta1, neta4, neta3, ns)]
weight1 = [SquaredExponential(weta1, weta2, ws)]
meangprn = [Constant(meanval)]
jitter = [jitt]
tstar = np.linspace(time.min()-extention, time.max()+extention, 1000)



elbo, mu, var = GPRN.optVarParams(nodes, weight1, meangprn, jitter, iterations = iternum)
fit = GPRN.Prediction(nodes, weight1, meangprn, tstar, mu)
#plt.plot(tstar, fit[0,:].T, '--', label='GPRN')
#plt.legend()
print(elbo)
