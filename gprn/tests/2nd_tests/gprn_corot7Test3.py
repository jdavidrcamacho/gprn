#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#np.random.seed(2102018)
import matplotlib.pyplot as plt
plt.close('all')

from gprn.complexGP import complexGP
from gprn import weightFunction, nodeFunction, meanFunction

data_plots = False
fit_plots = False
###### Data .rdb file #####
time,rv,rverr,fwhm,fwhmerr,bis,biserr,rhk,rhkerr = np.loadtxt("corot7_clean.rdb", 
                                                              skiprows=102, unpack=True, 
                                                              usecols=(0,1,2,3,4,5,6,7,8))


##### GP object #####
nodes = [nodeFunction.QuasiPeriodic(3.28, 22.21, 0.93, 0)]
weight = weightFunction.Constant(0)
weight_values = [9.31, 2, 1, 1]
means = [meanFunction.Constant(30),
         meanFunction.Constant(6.45), 
         meanFunction.Constant(0.025), 
         meanFunction.Constant(-4.75)]
jitters =[1, 1, 1, 1]
 
GPobj = complexGP(nodes, weight, weight_values, means, jitters, time, 
                  rv, rverr, fwhm, fwhmerr, bis, biserr, rhk, rhkerr)
loglike = GPobj.new_log_like(nodes, weight, weight_values, means, jitters)
#print(loglike)

##### fitplots #####
#mu11, std11, cov11 = GPobj.predict_gp(nodes = nodes, weight = weight, 
#                                      weight_values = weight_values, means = means,
#                                      jitters = jitters,
#                                      time = np.linspace(time.min(), time.max(), 500),
#                                      dataset = 1)
#mu22, std22, cov22 = GPobj.predict_gp(nodes = nodes, weight = weight, 
#                                      weight_values = weight_values, means = means,
#                                      jitters = jitters,
#                                      time = np.linspace(time.min(), time.max(), 500),
#                                      dataset = 2)
#mu33, std33, cov33 = GPobj.predict_gp(nodes = nodes, weight = weight, 
#                                      weight_values = weight_values, means = means,
#                                      jitters = jitters,
#                                      time = np.linspace(time.min(), time.max(), 500),
#                                      dataset = 3)
#mu44, std44, cov44 = GPobj.predict_gp(nodes = nodes, weight = weight, 
#                                      weight_values = weight_values, means = means,
#                                      jitters = jitters,
#                                      time = np.linspace(time.min(), time.max(), 500),
#                                      dataset = 4)

from scipy import stats
# log-likelihood
def loglike(theta):
    eta2,eta3,eta4,s, w1,w2,w3,w4, c1,c2,c3,c4, j1,j2,j3,j4 = theta
    new_node = [nodeFunction.QuasiPeriodic(eta2, eta3, eta4, s)]
    new_weight_values = [w1,w2,w3,w4]
    new_mean = [meanFunction.Constant(c1), 
                meanFunction.Constant(c2),
                meanFunction.Constant(c3),
                meanFunction.Constant(c4)]
    new_jitt = [j1,j2,j3,j4]
    return GPobj.new_log_like(new_node, weight, new_weight_values, 
                              new_mean, new_jitt)

def prior_transform(utheta):
    #quasi-periodic kernel
    eta2 = stats.uniform(0, 16)
    eta3 = stats.uniform(10, 40- 10) 
    eta4 = stats.uniform(0, 1 -np.exp(-10)) 
    s = stats.uniform(0, 2)
    #weight functions
    w1 = stats.uniform(0, 50)
    w2 = stats.uniform(0, 50)
    w3 = stats.uniform(0, 50)
    w4 = stats.uniform(0, 50)
    #mean functions
    c1 = stats.uniform(10, 50 -10)
    c2 = stats.uniform(1, 10 -1)
    c3 = stats.uniform(0, 10)
    c4 = stats.uniform(2, 8 -2)
    #jitters
    j1 = stats.uniform(0, 2)
    j2 = stats.uniform(0, 20)
    j3 = stats.uniform(0, 10)
    j4= stats.uniform(0, 1)
    return eta2.rvs(),eta3.rvs(),eta4.rvs(),s.rvs(), w1.rvs(),w2.rvs(),w3.rvs(), \
            w4.rvs(), c1.rvs(),c2.rvs(),c3.rvs(),c4.rvs(), j1.rvs(),j2.rvs(),j3.rvs(),j4.rvs()

##### dynesty #####
import dynesty
dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=1)
dsampler.run_nested(maxiter=5000)
dres = dsampler.results

from dynesty import plotting as dyplot
# trace plot
fig, axes = dyplot.traceplot(dsampler.results,
                             labels=["eta2", "eta3", "eta4", "s", "w1", "w2", 
                                      "w3", "w4", "c1", "c2", "c3", "c4", "j1",
                                      "j2", "j3", "j4"])
fig.tight_layout()
# summary plot
fig, axes = dyplot.runplot(dsampler.results)
# corner plot
fig, axes = dyplot.cornerplot(dsampler.results, 
                              labels=["eta2", "eta3", "eta4", "s", "w1", "w2", 
                                      "w3", "w4", "c1", "c2", "c3", "c4", "j1",
                                      "j2", "j3", "j4"],
                              show_titles=True)


##### Plots #####
if data_plots:
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    ax1.set_title('RVs, fwhm, BIS and Rhk')
    ax1.errorbar(time,rv, rverr, fmt = "b.")
    ax1.set_ylabel("RVs")
    ax2.errorbar(time,fwhm, fwhmerr, fmt = "r.")
    ax2.set_ylabel("fwhm")
    ax3.errorbar(time,bis, biserr, fmt = "g.")
    ax3.set_ylabel("BIS")
    ax4.errorbar(time,rhk, rhkerr, fmt = "y.")
    ax4.set_ylabel("Rhk")
    plt.show()

#if fit_plots:
#    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
#    ax1.set_title('Fits')
#    ax1.fill_between(np.linspace(time.min(), time.max(), 500), 
#                     mu11+std11, mu11-std11, color="grey", alpha=0.5)
#    ax1.plot(np.linspace(time.min(), time.max(), 500), mu11, "k--", alpha=1, lw=1.5)
#    ax1.errorbar(time,rv, rverr, fmt = "b.")
#    ax1.set_ylabel("RVs")
#    
#    ax2.fill_between(np.linspace(time.min(), time.max(), 500), 
#                     mu22+std22, mu22-std22, color="grey", alpha=0.5)
#    ax2.plot(np.linspace(time.min(), time.max(), 500), mu22, "k--", alpha=1, lw=1.5)
#    ax2.errorbar(time,fwhm, fwhmerr, fmt = "b.")
#    ax2.set_ylabel("FWHM")
#    
#    ax3.fill_between(np.linspace(time.min(), time.max(), 500), 
#                     mu33+std33, mu33-std33, color="grey", alpha=0.5)
#    ax3.plot(np.linspace(time.min(), time.max(), 500), mu33, "k--", alpha=1, lw=1.5)
#    ax3.errorbar(time,bis, biserr, fmt = "b.")
#    ax3.set_ylabel("BIS")
#    
#    ax4.fill_between(np.linspace(time.min(), time.max(), 500), 
#                     mu44+std44, mu44-std44, color="grey", alpha=0.5)
#    ax4.plot(np.linspace(time.min(), time.max(), 500), mu44, "k--", alpha=1, lw=1.5)
#    ax4.errorbar(time, rhk, rhkerr, fmt = "b.")
#    ax4.set_ylabel("R'hk")
#    plt.show()
