"""
Auto-correlation and cross-correlation functions
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


from scipy import fftpack
from astroML.time_series import periodogram

# from .periodogram import lomb_scargle


def ACF_scargle(t, y, dy, n_omega=2**10, omega_max=100):
    """Compute the Auto-correlation function via Scargle's method

    Parameters
    ----------
    t : array_like
        times of observation.  Assumed to be in increasing order.
    y : array_like
        values of each observation.  Should be same shape as t
    dy : float or array_like
        errors in each observation.
    n_omega : int (optional)
        number of angular frequencies at which to evaluate the periodogram
        default is 2^10
    omega_max : float (optional)
        maximum value of omega at which to evaluate the periodogram
        default is 100

    Returns
    -------
    ACF, t : ndarrays
        The auto-correlation function and associated times
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if y.shape != t.shape:
        raise ValueError("shapes of t and y must match")

    dy = np.asarray(dy) * np.ones(y.shape)

    d_omega = omega_max * 1. / (n_omega + 1)
    omega = d_omega * np.arange(1, n_omega + 1)

    # recall that P(omega = 0) = (chi^2(0) - chi^2(0)) / chi^2(0)
    #                          = 0

    # compute P and shifted full-frequency array
    # P = periodogram.lomb_scargle(t, y, dy, omega,
    #                              generalized=True)
    P = periodogram.LombScargle(t, y, dy).power(omega / (2 * np.pi))
    P = np.concatenate([[0], P, P[-2::-1]])

    # compute PW, the power of the window function
    # PW = lomb_scargle(t, np.ones(len(t)), dy, omega,
    #                   generalized=False, subtract_mean=False)
    PW = periodogram.LombScargle(t, np.ones(len(t)), dy, fit_mean=False,
                                 center_data=False).power(omega / (2 * np.pi))
    PW = np.concatenate([[0], PW, PW[-2::-1]])

    # compute the  inverse fourier transform of P and PW
    rho = fftpack.ifft(P).real
    rhoW = fftpack.ifft(PW).real

    ACF = fftpack.fftshift(rho / rhoW) / np.sqrt(2)
    N = len(ACF)
    dt = 2 * np.pi / N / (omega[1] - omega[0])
    t = dt * (np.arange(N) - N // 2)

    return ACF, t


def ACF_EK(t, y, dy, bins=20):
    """Auto-correlation function via the Edelson-Krolik method

    Parameters
    ----------
    t : array_like
        times of observation.  Assumed to be in increasing order.
    y : array_like
        values of each observation.  Should be same shape as t
    dy : float or array_like
        errors in each observation.
    bins : int or array_like (optional)
        if integer, the number of bins to use in the analysis.
        if array, the (nbins + 1) bin edges.
        Default is bins=20.

    Returns
    -------
    ACF : ndarray
        The auto-correlation function and associated times
    err : ndarray
        the error in the ACF
    bins : ndarray
        bin edges used in computation
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if y.shape != t.shape:
        raise ValueError("shapes of t and y must match")

    if t.ndim != 1:
        raise ValueError("t should be a 1-dimensional array")

    dy = np.asarray(dy) * np.ones(y.shape)

    # compute mean and standard deviation of y
    w = 1. / dy / dy
    w /= w.sum()
    mu = np.dot(w, y)
    sigma = np.std(y, ddof=1)

    dy2 = dy[:, None]

    dt = t - t[:, None]
    UDCF = ((y - mu) * (y - mu)[:, None] / np.sqrt(
        (sigma**2 - dy**2) * (sigma**2 - dy2**2)))

    # determine binning
    bins = np.asarray(bins)
    if bins.size == 1:
        dt_min = dt.min()
        dt_max = dt.max()
        bins = np.linspace(dt_min, dt_max + 1E-10, bins + 1)

    ACF = np.zeros(len(bins) - 1)
    M = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        flag = (dt >= bins[i]) & (dt < bins[i + 1])
        M[i] = flag.sum()
        ACF[i] = np.sum(UDCF[flag])

    ACF /= M

    return ACF, np.sqrt(2. / M), bins


def DCF_EK(t, y1, y2, dy1, dy2, bins=20):
    """Cross-correlation function via the Edelson-Krolik method

    Parameters
    ----------
    t : array_like
        times of observation.  Assumed to be in increasing order.
    y1, y2 : array_like
        values of each observation of each data train.  Should be same shape as t
    dy1, dy2 : float or array_like
        errors in each observation of each data train.
    bins : int or array_like (optional)
        if integer, the number of bins to use in the analysis.
        if array, the (nbins + 1) bin edges.
        Default is bins=20.

    Returns
    -------
    DCF : ndarray
        The discrete cross-correlation function and associated times
    err : ndarray
        the error in the ACF
    bins : ndarray
        bin edges used in computation
    """
    t = np.asarray(t)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    if y1.shape != t.shape or y2.shape != t.shape:
        raise ValueError("shapes of t and y1 and y2 must match")

    if t.ndim != 1:
        raise ValueError("t should be a 1-dimensional array")

    dy1 = np.asarray(dy1) * np.ones(y1.shape)
    dy2 = np.asarray(dy2) * np.ones(y2.shape)

    # compute mean and standard deviation of y1
    w1 = 1. / dy1 / dy1
    w1 /= w1.sum()
    mu1 = np.dot(w1, y1)
    sigma1 = np.std(y1, ddof=1)

    # compute mean and standard deviation of y2
    w2 = 1. / dy2 / dy2
    w2 /= w2.sum()
    mu2 = np.dot(w2, y2)
    sigma2 = np.std(y2, ddof=1)

    # dy12 = dy1[:, None]
    # dy22 = dy2[:, None]

    dt = t - t[:, None]
    UDCF = ((y1 - mu1) * (y2 - mu2)[:, None] / np.sqrt(
        (sigma1**2 - dy1**2) * (sigma2**2 - dy2**2)))

    # determine binning
    bins = np.asarray(bins)
    if bins.size == 1:
        dt_min = dt.min()
        dt_max = dt.max()
        bins = np.linspace(dt_min, dt_max + 1E-10, bins + 1)

    ACF = np.zeros(len(bins) - 1)
    M = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        flag = (dt >= bins[i]) & (dt < bins[i + 1])
        M[i] = flag.sum()
        ACF[i] = np.sum(UDCF[flag])

    ACF /= M

    return ACF, np.sqrt(2. / M), bins


from data import harps, espresso, full
from styler import params, figwidth, colors

figs = {
    1: True,
    2: False,
    3: False,
}

harpscolor = colors[0]
esprcolor = colors[1]

if figs[1]:
    with plt.rc_context(params):
        fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True,
                                       figsize=(figwidth, 0.9 * figwidth))

        kw = dict(lw=1)

        # HARPS RV ACF
        EKbins = np.linspace(0, 100, 50)
        std = harps.vrad.std()
        C_EK, C_EK_err, bins = ACF_EK(harps.time, harps.vrad / std, harps.error, bins=EKbins)
        t_EK = 0.5 * (bins[1:] + bins[:-1])
        m = ~np.isnan(C_EK)

        ax1.plot(t_EK[m], C_EK[m], color=harpscolor, **kw)


        # HARPS FWHM ACF
        EKbins = np.linspace(0, 100, 50)
        C_EK, C_EK_err, bins = ACF_EK(harps.time, harps.extras.fwhm, harps.extras.fwhm_err, bins=EKbins)
        t_EK = 0.5 * (bins[1:] + bins[:-1])
        m = ~np.isnan(C_EK)

        ax1.plot(t_EK[m], C_EK[m], color=harpscolor, ls='--', **kw)



        # HARPS R'hk ACF
        # EKbins = np.linspace(0, 100, 71)
        # C_EK, C_EK_err, bins = ACF_EK(harps.time, harps.extras.rhk, harps.extras.sig_rhk, bins=EKbins)
        # t_EK = 0.5 * (bins[1:] + bins[:-1])
        # m = ~np.isnan(C_EK)
        # ax1.plot(t_EK[m], C_EK[m], color=harpscolor, ls=':', **kw)

        # # HARPS BIS ACF
        # EKbins = np.linspace(0, 100, 20)
        # C_EK, C_EK_err, bins = ACF_EK(harps.time, harps.extras.bis_span, 2*harps.error, bins=EKbins)
        # t_EK = 0.5 * (bins[1:] + bins[:-1])
        # m = ~np.isnan(C_EK)
        # print(C_EK)

        # ax1.plot(t_EK[m], C_EK[m], color=harpscolor, ls=':', **kw)


        # ESPRESSO RV ACF
        EKbins = np.linspace(0, 100, 41)
        C_EK, C_EK_err, bins = ACF_EK(espresso.time, espresso.vrad,
                                      espresso.error, bins=EKbins)
        t_EK = 0.5 * (bins[1:] + bins[:-1])
        m = ~np.isnan(C_EK)

        # ax1.plot(t_EK[m], C_EK[m], color=esprcolor, **kw, ls='--')

        # HARPS RV-FWHM DCF
        # EKbins = np.linspace(-50, 50, 41)
        EKbins = np.linspace(-15, 15, 31)
        C_EK, C_EK_err, bins = DCF_EK(harps.time, harps.vrad,
                                      harps.extras.fwhm, harps.error,
                                      harps.error, bins=EKbins)
        t_EK = 0.5 * (bins[1:] + bins[:-1])
        m = ~np.isnan(C_EK)
        ax2.plot(t_EK[m], C_EK[m], harpscolor, **kw)


        # HARPS RV-BIS or RV-Rhk DCF
        # EKbins = np.linspace(-15, 15, 41)
        # C_EK, C_EK_err, bins = DCF_EK(harps.time, harps.vrad, harps.extras.rhk,
        #                             harps.error, harps.extras.sig_rhk, bins=EKbins)
        # # C_EK, C_EK_err, bins = DCF_EK(harps.time, harps.vrad, harps.extras.bis_span,
        # #                             harps.error, harps.error, bins=EKbins)
        # t_EK = 0.5 * (bins[1:] + bins[:-1])
        # m = ~np.isnan(C_EK)
        # ax2.plot(t_EK[m], C_EK[m], harpscolor, ls='--', **kw)

        # ESPRESSO RV-FWHM DCF
        EKbins = np.linspace(-15, 15, 21)

        coef = np.polyfit(espresso.time, espresso.extras.fwhm, deg=1,
                          w=1 / espresso.extras.fwhm_err)
        fit = np.polyval(coef, espresso.time)

        # C_EK, C_EK_err, bins = DCF_EK(espresso.time, espresso.vrad,
        #                               (espresso.extras.fwhm - fit), 
        #                               espresso.error, espresso.extras.fwhm_err, 
        #                               bins=EKbins)
        # t_EK = 0.5 * (bins[1:] + bins[:-1])
        # m = ~np.isnan(C_EK)
        # ax2.plot(t_EK[m], C_EK[m], color=esprcolor, **kw, ls='--')


        # # ESPRESSO RV-BIS DCF
        # EKbins = np.linspace(-15, 15, 21)

        # C_EK, C_EK_err, bins = DCF_EK(espresso.time, espresso.vrad,
        #                               espresso.extras.bis_span, 
        #                               espresso.error, 2*espresso.error,
        #                               bins=EKbins)
        # t_EK = 0.5 * (bins[1:] + bins[:-1])
        # m = ~np.isnan(C_EK)
        # ax2.plot(t_EK[m], C_EK[m], color=esprcolor, **kw, ls='--')


        ax1.axhline(y=0, color='k', alpha=0.2)
        ax2.axhline(y=0, color='k', alpha=0.2)
        ax2.axvline(x=0, color='k', alpha=0.2)

        # leg = ['HARPS', 'ESPRESSO']
        leg = ['RVs', 'FWHM']
        ax1.legend(leg, ncol=1, #bbox_to_anchor=(0.48, 0.95),
                   fontsize=8, frameon=False)
        ax1.set(xlabel='Time lag [days]', ylabel='Discrete ACF')
        ax1.set(xlim=(0,100))
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        # ax.tick_params(which='minor', length=4, color='r')
        # ax1.xaxis.grid(True, which='minor', alpha=0.1)
        # ax1.set_xticks(np.arange(0, 101, 10))

        ax2.set(xlabel='Time lag [days]', ylabel='Discrete CF \, RV - FWHM')
        # ax2.set(xlim=(-14, 14), ylim=(-0.5, 1))
        # ax2.minorticks_on()
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        # ax2.set_xticks(np.arange(-50, 51, 10))

        # ax1.set(ylim=(-1,1))

        fig.savefig('../img/correlation_functions.pdf')
    plt.close('all')

# sys.exit(0)

# EKbins = np.linspace(-15, 15, 31)
# C_EK, C_EK_err, bins = DCF_EK(harps.time, harps.vrad, harps.extras.fwhm,
#                               harps.error, harps.error, bins=EKbins)
# t_EK = 0.5 * (bins[1:] + bins[:-1])
# m = ~np.isnan(C_EK)

# plt.plot(t_EK[m], C_EK[m], '-')

# # EKbins = np.linspace(-15, 15, 50)
# C_EK, C_EK_err, bins = DCF_EK(espresso.time, espresso.vrad,
#                               espresso.extras.fwhm, espresso.error,
#                               espresso.error, bins=EKbins)
# t_EK = 0.5 * (bins[1:] + bins[:-1])
# m = ~np.isnan(C_EK)
# # plt.plot(t_EK[m], C_EK[m], '-')

# plt.axhline(y=0, color='k', alpha=0.2)
# plt.axvline(x=0, color='k', alpha=0.2)

# plt.show()
