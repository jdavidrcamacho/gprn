#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:01:24 2019

@author: joaocamacho
"""

import numpy as np
import matplotlib.pylab as plt

time1,rv,rverr,fwhm,fwhmerr,bis,biserr,rhk1,rhkerr1 = np.loadtxt("corot7_clean.rdb", 
                                                              skiprows=102, unpack=True, 
                                                              usecols=(0,1,2,3,4,5,6,7,8))


time,rv,rverr,fwhm,fwhmerr,bis,biserr,rhk,rhkerr = np.loadtxt("corot7_outlierclean.rdb", 
                                                              skiprows=102, unpack=True, 
                                                              usecols=(0,1,2,3,4,5,6,7,8))


plt.figure()
plt.plot(time, rhk, '*') #without outlier
plt.plot(time1, rhk1, '.') #with outlier
plt.savefig('outlier.png')
plt.figure()
plt.boxplot(rhk1)
plt.savefig('boxplot.png')
values = np.where(rhk > -4.82)
data = np.loadtxt("corot7_clean.rdb", skiprows=102, unpack=True, usecols=(0,1,2,3,4,5,6,7,8))
new_data = data[:,values].T
new_data = new_data.reshape(-1, 9)