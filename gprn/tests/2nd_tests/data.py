#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#np.random.seed(2102018)
import matplotlib.pyplot as plt
plt.close('all')

from gprn.complexGP import complexGP
from gprn import weightFunction, nodeFunction, meanFunction

##### Data files #####
time, rv, rverr, fwhm, bis, rhk, rhkerr = np.loadtxt("corot7_harps.rdb", 
                                                     skiprows=2, unpack=True, 
                                                     usecols=(0, 1, 2, 3, 4, 5, 6))
#remaning errors
fwhmerr = 2.35 * rverr
biserr = 2.0* rverr


newtime, newrv, newrverr = np.loadtxt("corot7.txt", skiprows=2, unpack=True, 
                                      usecols=(0, 1, 2))

time = np.concatenate((newtime[:69],newtime[78:]))
rv = np.concatenate((newrv[:69],newrv[78:]))
rverr = np.concatenate((newrverr[:69],newrverr[78:]))

data = np.stack((time, rv, rverr, fwhm, fwhmerr, bis, biserr, rhk, rhkerr))
data = data.T
np.savetxt('corot7_clean.rdb', data, delimiter='\t', 
           header ="time\trv\trverr\tfwhm\tfwhmerr\tbis\tbiserr\trhk\trhkerr", 
           comments='')