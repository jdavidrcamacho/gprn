#for multiprocessing
import os
os.environ["OMP_NUM_THREADS"] = "1"
ncpu = 4
iterations = 10
max_n = iterations

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

from gprn.covFunction import SquaredExponential, Periodic, Sum
from gprn.meanFunction import Constant
from gprn.simpleMeanField import inference

import emcee
import corner

time2count = [10, 20, 50, 100, 200]#, 300, 400, 500, 600, 700, 800, 900, 1000]

finalTimes = []
for i, j in enumerate(time2count):
    #data
    np.random.seed(230190)
    t = np.linspace(10,100, j)
    y = np.random.randn() * 10*np.sin(2*np.pi*t/30)
    yerr = np.random.random(t.size)
    
    GPRN = inference(1, t, y, yerr)
    nodes = [Periodic(1, 0.5, 30, 1e-5)]
    weight = [SquaredExponential(10, 1000, 1e-5)]
    means = [Constant(0)]
    jitter = [0.1]
    
    from time import time
    
    times = []
    for i in range(0,10):
        start = time()
        MU, VAR = None, None
        elbo, MU, VAR = GPRN.optVarParams(nodes, weight, means, jitter, 
                                          iterations = 1000, mu=MU, var=VAR)
        end = time()
        times.append(end-start)
    av = np.mean(times)
    print('Average time = {0} sec'.format(av))
    finalTimes.append(av)
    
plt.figure()
plt.plot(time2count, finalTimes, '-*')
plt.xlabel('Number of points')
plt.ylabel('Average time (s)')