import numpy as np 
np.random.seed(42)

import matplotlib.pyplot as plt

import george
from george.utils import multivariate_gaussian_samples

from tedi import process, kernels
from tedi.kernels import QuasiPeriodic


from gprn.complexGP import complexGP
from gprn import weightFunction, nodeFunction, meanFunction

time, y, yerr = np.loadtxt('corot7.txt', skiprows=2)[106:,:].T
tt = np.linspace(time[0], time[-1], 1000)
# time = np.linspace(0, 10, y.size)


# one quasi-periodic node
QP_eta2 = 5 # eta2, evolutionary time scale
QP_P = 11 # periodicity
QP_eta4 = 0.5 # length scale of the periodic component
wn = 0

nodes = [nodeFunction.QuasiPeriodic(QP_eta2, QP_P, QP_eta4, wn)]

weight = weightFunction.Constant(0)
weight_values = [10, 40, 20, 1e-2]

# weight = weightFunction.SquaredExponential(0, 3)
# weight_values = [10, 40, 20, 1e-2]  

# means = [meanFunction.Constant(50),  # RVs
#          meanFunction.Constant(0),   # FWHM
#          meanFunction.Constant(0),   # BIS
#          meanFunction.Constant(-4.95),   # log R'hk
#         ]

means = [meanFunction.Constant(50),  # RVs
         meanFunction.Constant(0),   # FWHM
         meanFunction.Constant(0),   # BIS
         meanFunction.Constant(0)   # log R'hk
        ]

jitters = [1, 10, 5, 1e-6]
#jitters = 4*[0]

data = 4*[y,yerr]
GPobj = complexGP(nodes, weight, weight_values, means, jitters, time, *data)

samples = []

fig, axs = plt.subplots(len(means), 1, figsize=(6,8))
for i, ax in enumerate(axs):
    K = GPobj._covariance_matrix(nodes, weight, weight_values, time, i+1)
    K += np.diagflat(jitters[i]**2 * np.ones_like(time))

    mean = means[i](time)
    ys = multivariate_gaussian_samples(K, 1, mean=mean)

    ys_error = np.abs(np.random.randn(time.size) * jitters[i])

    samples.append(ys)
    samples.append(ys_error)

    ax.plot(time, ys, '-o')
    ax.errorbar(time, ys, ys_error, fmt='.')
    ax.set(xlabel='Time [days]')

axs[0].set(ylabel='RV [m/s]')
axs[1].set(ylabel='FWHM [m/s]')
axs[2].set(ylabel='BIS [m/s]')
axs[3].set(ylabel="log R'hk")


fig.tight_layout()
plt.show()

sampled_data = np.stack((time, samples[0], samples[1], samples[2], 
                         samples[4], samples[6], samples[7]))
sampled_data = sampled_data.T
np.savetxt('sampled_data.rdb', sampled_data, delimiter='\t', 
           header ="time\trv\trverr\tfwhm\tbis\trhk\trhkerr", 
           comments='')


#np.savetxt('simulated1.rdb', header='bjd\tvrad\tsvrad\n---\t----\t-----',
#           comments='', fmt='%-9.5f', # delimiter='\t',
#           X=np.c_[time, samples[0]])
