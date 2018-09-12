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
#    def __init__(self, theta, ell, wn):
#        super(SquaredExponential, self).__init__(theta, ell, wn)
#        self.theta = theta
#        self.ell = ell
#        self.wn = wn
#        self.type = 'stationary and anisotropic'
#
#    def __call__(self, r):
#        try:
#            return self.theta**2 * exp(-0.5 * r**2 / self.ell**2)  \
#                    +self.wn**2 * np.diag(np.diag(np.ones_like(r)))
#        except ValueError:
#            return self.theta**2 * exp(-0.5 * r**2 / self.ell**2)
    def __init__(self, ell, wn):
        # AMPLITUDE = 1
        super(SquaredExponential, self).__init__(ell, wn)
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and anisotropic'

    def __call__(self, r):
        try:
            return exp(-0.5 * r**2 / self.ell**2)  \
                    +self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return exp(-0.5 * r**2 / self.ell**2)
        

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
#    def __init__(self, theta, ell, P, wn):
#        super(Periodic, self).__init__(theta, ell, P, wn)
#        self.theta = theta
#        self.ell = ell
#        self.P = P
#        self.wn = wn
#        self.type = 'non-stationary and isotropic'
#
#    def __call__(self, r):
#        try:
#            return self.theta**2 * exp( -2 * sine(pi*np.abs(r)/self.P)**2 \
#                                       /self.ell**2) \
#                    +self.wn**2 * np.diag(np.diag(np.ones_like(r)))
#        except ValueError:
#            return self.theta**2 * exp( -2 * sine(pi*np.abs(r)/self.P)**2 \
#                                       /self.ell**2)
    def __init__(self, ell, P, wn):
        # AMPLITUDE = 1
        super(Periodic, self).__init__(ell, P, wn)
        self.ell = ell
        self.P = P
        self.wn = wn
        self.type = 'non-stationary and isotropic'

    def __call__(self, r):
        try:
            return exp( -2 * sine(pi*np.abs(r)/self.P)**2 / self.ell**2) \
                    +self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return exp( -2 * sine(pi*np.abs(r)/self.P)**2 / self.ell**2)

##### Quasi Periodic ###########################################################
class QuasiPeriodic(nodeFunction):
    """
        This kernel is the product between the exponential sine squared kernel 
    and the squared exponential kernel, commonly known as the quasi-periodic 
    kernel.
        Parameters:
            theta = kernel amplitude
            ell_e = evolutionary time scale
            ell_p = length scale of the Periodic component
            P = kernel Periodicity
            wn = white noise
    """
#    def __init__(self, theta, ell_e, P, ell_p, wn):
#        super(QuasiPeriodic, self).__init__(theta, ell_e, P, ell_p, wn)
#        self.theta = theta
#        self.ell_e = ell_e
#        self.P = P
#        self.ell_p = ell_p
#        self.wn = wn
#        self.type = 'non-stationary and anisotropic'
#
#    def __call__(self, r):
#        try:
#            return self.theta**2 * exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
#                                       /self.ell_p**2 - r**2/(2*self.ell_e**2)) \
#                    +self.wn**2 * np.diag(np.diag(np.ones_like(r)))
#        except ValueError:
#            return self.theta**2 * exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
#                                       /self.ell_p**2 - r**2/(2*self.ell_e**2))
    def __init__(self, ell_e, P, ell_p, wn):
        # AMPLITUDE = 1
        super(QuasiPeriodic, self).__init__(ell_e, P, ell_p, wn)
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
        self.type = 'non-stationary and anisotropic'

    def __call__(self, r):
        try:
            return exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                                       /self.ell_p**2 - r**2/(2*self.ell_e**2)) \
                    +self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                                       /self.ell_p**2 - r**2/(2*self.ell_e**2))