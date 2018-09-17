#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#because it makes my life easier down the line
pi, exp, sine, cosine = np.pi, np.exp, np.sin, np.cos

class nodeFunction(object):
    """
        Definition the node functions (kernels) of our GPRN
    """
    def __init__(self, *args):
        """
            Puts all kernel arguments in an array pars
        """
        self.pars = np.array(args)

    def __call__(self, r):
        """
            r = t - t' 
            Not sure if this is a good approach since will make our life harder 
        when defining certain non-stationary kernels, e.g linear kernel.
        """
        raise NotImplementedError

    def __repr__(self):
        """
            Representation of each kernel instance
        """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))


##### Squared exponential ######################################################
class SquaredExponential(nodeFunction):
    """
        Squared Exponential kernel, also known as radial basis function or RBF 
    kernel in other works.
        Parameters:
            theta = amplitude
            ell = length-scale
            wn = white noise
    """
    def __init__(self, ell, wn):
        super(SquaredExponential, self).__init__(ell, wn)
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and anisotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        try:
            return exp(-0.5 * r**2 / self.ell**2)  \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return exp(-0.5 * r**2 / self.ell**2)

class dSquaredExponential_dell(SquaredExponential):
    """
        Log-derivative in order to ell
    """
    def __init__(self, ell, wn):
        super(dSquaredExponential_dell, self).__init__(ell, wn)
        self.ell = ell
        self.wn = wn
        
    def __call__(self, r):
        return (r**2 / self.ell**2) * exp(-0.5 * r**2 / self.ell**2)

class dSquaredExponential_dwn(SquaredExponential):
    """
        Log-derivative in order to the wn
    """
    def __init__(self, ell, wn):
        super(dSquaredExponential_dwn, self).__init__(ell, wn)
        self.ell = ell
        self.wn = wn
        
    def __call__(self, r):
        return
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


##### Periodic #################################################################
class Periodic(nodeFunction):
    """
        Definition of the periodic kernel.
        Parameters:
            theta = amplitude
            ell = lenght scale
            P = period
            wn = white noise
    """
    def __init__(self, ell, P, wn):
        super(Periodic, self).__init__(ell, P, wn)
        self.ell = ell
        self.P = P
        self.wn = wn
        self.type = 'non-stationary and isotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        try:
            return exp( -2 * sine(pi*np.abs(r)/self.P)**2 / self.ell**2) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return exp( -2 * sine(pi*np.abs(r)/self.P)**2 / self.ell**2)

class dPeriodic_dell(Periodic):
    """
        Log-derivative in order to ell
    """
    def __init__(self, ell, P, wn):
        super(dPeriodic_dell, self).__init__(ell, P, wn)
        self.ell = ell
        self.P = P
        self.wn = wn

    def __call__(self, r):
        return 4 * sine(pi*np.abs(r)/self.P)**2 / self.ell **2 \
                * exp( -2 * sine(pi*np.abs(r)/self.P)**2 / self.ell**2)

class dPeriodic_dP(Periodic):
    """
        Log-derivative in order to P
    """
    def __init__(self, ell, P, wn):
        super(dPeriodic_dP, self).__init__(ell, P, wn)
        self.ell = ell
        self.P = P
        self.wn = wn

    def __call__(self, r):
        return 4*pi*r * cosine(pi*np.abs(r)/self.P) * sine(pi*np.abs(r)/self.P) \
                * exp(-2 * sine(pi * np.abs(r) /self.P)**2 / self.ell**2) \
                / (self.ell**2 * self.P)

class dPeriodic_dwn(Periodic):
    """
        Log-derivative in order to wn
    """
    def __init__(self, ell, P, wn):
        super(dPeriodic_dwn, self).__init__(ell, P, wn)
        self.ell = ell
        self.P = P
        self.wn = wn

    def __call__(self, r):
        return
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


##### Quasi Periodic ###########################################################
class QuasiPeriodic(nodeFunction):
    """
        This kernel is the product between the exponential sine squared kernel 
    and the squared exponential kernel, commonly known as the quasi-periodic 
    kernel.
        Parameters:
            ell_e = evolutionary time scale
            P = kernel periodicity
            ell_p = length scale of the periodic component
            wn = white noise
    """
    def __init__(self, ell_e, P, ell_p, wn):
        super(QuasiPeriodic, self).__init__(ell_e, P, ell_p, wn)
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 4    #number of derivatives in this kernel
        self.params_size = 4    #number of hyperparameters

    def __call__(self, r):
        try:
            return exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                       /self.ell_p**2 - r**2/(2*self.ell_e**2)) \
                       + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                       /self.ell_p**2 - r**2/(2*self.ell_e**2))

class dQuasiPeriodic_delle(QuasiPeriodic):
    """
        Log-derivative in order to ell_e
    """
    def __init__(self, ell_e, P, ell_p, wn):
        super(dQuasiPeriodic_delle, self).__init__(ell_e, P, ell_p, wn)
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
        
    def __call__(self, r):
        return (r**2 * exp(-0.5 * r**2 / self.ell_e**2 \
                           -2 * sine(pi * np.abs(r) / self.P)**2 \
                           / self.ell_p**2)) / self.ell_e**2

class dQuasiPeriodic_dP(QuasiPeriodic):
    """
        Log-derivative in order to P
    """
    def __init__(self, ell_e, P, ell_p, wn):
        super(dQuasiPeriodic_dP, self).__init__(ell_e, P, ell_p, wn)
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
        
    def __call__(self, r):
        return (4 * pi * r * cosine(pi * np.abs(r) / self.P) \
                * sine(pi * np.abs(r) / self.P) \
                * exp(-0.5 * r**2 / self.ell_e**2 \
                     - 2 * sine(pi * np.abs(r) /self.P)**2 / self.ell_p**2)) \
                / (self.ell_p**2 * self.P)

class dQuasiPeriodic_dellp(QuasiPeriodic):
    """
        Log-derivative in order to ell_p
    """
    def __init__(self, ell_e, P, ell_p, wn):
        super(dQuasiPeriodic_dellp, self).__init__(ell_e, P, ell_p, wn)
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
        
    def __call__(self, r):
        return (4 * sine(pi * np.abs(r) /self.P)**2 * exp(-0.5 * r**2 \
                / self.ell_e**2 -2 * sine(pi*np.abs(r) / self.P)**2 \
                      / self. ell_p**2)) / self.ell_p**2

class dQuasiPeriodic_dwn(QuasiPeriodic):
    """
        Log-derivative in order to wn
    """
    def __init__(self, ell_e, P, ell_p, wn):
        super(dQuasiPeriodic_dwn, self).__init__(ell_e, P, ell_p, wn)
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


### END