#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(2102018)
import matplotlib.pyplot as plt
plt.close('all')

##### Data .rdb file #####
time, rv, rverr, fwhm, bis, rhk, rhkerr = np.loadtxt("corot7_harps.rdb", 
                                                     skiprows=2, unpack=True, 
                            usecols=(0, 1, 2, 3, 4, 5, 6))

stuff = rhk
##removing NaNs
#time = time[~np.isnan(rhk)]
#rv = rv[~np.isnan(rhk)]
#rv_mean = np.mean(rv)
#rv = (rv-rv_mean)*1000 + rv_mean
#rverr = rverr[~np.isnan(rhk)]*1000
#fwhm = fwhm[~np.isnan(rhk)]
#bis = bis[~np.isnan(rhk)]
#rhkerr = rhkerr[~np.isnan(rhk)]
#rhk = rhk[~np.isnan(rhk)]
#
#remaning errors
#rms_fwhm = np.sqrt((1./fwhm.size*np.sum(fwhm**2)))
#fwhmerr = 0.001*rms_fwhm * np.ones(fwhm.size)
#rms_bis = np.sqrt((1./bis.size*np.sum(bis**2)))
#biserr = 0.10*rms_bis * np.ones(bis.size)

rhk1 = rhk[~np.isnan(rhk)]
rms_rhk = np.sqrt((1./rhk1.size*np.sum(rhk1**2)))


f, (ax1) = plt.subplots(1, sharex=True)
ax1.set_title('rhk')
ax1.errorbar(time, rhk, rhkerr, fmt = "b.")
ax1.set_ylabel("rhk")
plt.show()



import scipy.interpolate
a, b = rhk[64:69], rhk[78:83]
rhk_inter = np.concatenate((a, b), axis=None)
a, b = time[64:69], time[78:83]
time_inter = np.concatenate((a, b), axis=None)
y_interp = scipy.interpolate.BSpline(time_inter, rhk_inter, 3)
print(y_interp(time[64:83]))

rhk[69:78] = y_interp(time[69:78])
rhkerr[69:78] = 0.01*rms_rhk

f, (ax1) = plt.subplots(1, sharex=True)
ax1.set_title('rhk')
ax1.plot(time[64:83], y_interp(time[64:83]), 'o-')
ax1.set_ylabel("rhk")
plt.show()