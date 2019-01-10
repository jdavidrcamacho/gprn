#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats


def loguniform(low=0, high=1, size=None):
    """
        Log-Uniform distribution
        
                logU(a, b) ~ exp(U(log(a), log(b))
        
        In this distribution, the log transformed random variable is assumed to 
    be uniformly distributed.
        Parameters:
            low = minimum value
            high = maximum value
        Returns:
            integer if size=None
            array if size=value
    """
    if size is None:
        return np.exp(stats.uniform(low, high -low).rvs())
    else:
        logU = []
        for i in range(size):
            logU.append(np.exp(stats.uniform(low, high -low).rvs()))
        return logU