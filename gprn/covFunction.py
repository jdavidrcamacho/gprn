#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#because it makes my life easier down the line
pi, exp, sine, cosine, sqrt = np.pi, np.exp, np.sin, np.cos, np.sqrt

class covFunction(object):
    """
        Definition the covariance functions (kernels) of our GPRN, by default 
    and because it simplifies my life, all kernels include a white noise term
    """
    def __init__(self, *args):
        """
            Puts all kernel arguments in an array pars
        """
        self.pars = np.array(args, dtype=float)

    def __call__(self, r, t1 = None, t2=None):
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

    def __add__(self, b):
        return Sum(self, b)
    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        return Multiplication(self, b)
    def __rmul__(self, b):
        return self.__mul__(b)


class _operator(covFunction):
    """ 
        To allow operations between two kernels 
    """
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2
        self.kerneltype = 'complex'

    @property
    def pars(self):
        return np.append(self.k1.pars, self.k2.pars)


class Sum(_operator):
    """ 
        To allow the sum of kernels
    """
    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)

    def __call__(self, r):
        return self.k1(r) + self.k2(r)


class Multiplication(_operator):
    """ 
        To allow the multiplication of kernels 
    """
    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)

    def __call__(self, r):
        return self.k1(r) * self.k2(r)


##### Constant #################################################################
class Constant(covFunction):
    """
        This kernel returns its constant argument c with white noise
        Parameters:
            c = constant
            wn = white noise amplitude
    """
    def __init__(self, c, wn):
        super(Constant, self).__init__(c, wn)
        self.c = c
        self.wn = wn
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.c**2 * np.ones_like(r) \
                        + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.c**2 * np.ones_like(r)

class dConstant_dc(Constant):
    """
        Log-derivative in order to c
    """
    def __init__(self, c, wn):
        super(dConstant_dc, self).__init__(c, wn)
        self.c = c
        self.wn = wn

    def __call__(self, r):
        return 2 * self.c**2 * np.ones_like(r)

class dConstant_dwn(Constant):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, c, wn):
        super(dConstant_dwn, self).__init__(c, wn)
        self.c = c
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)
        
        
##### White Noise ##############################################################
class WhiteNoise(covFunction):
    """
        Definition of the white noise kernel.
        Parameters
            wn = white noise amplitude
    """
    def __init__(self, wn):
        super(WhiteNoise, self).__init__(wn)
        self.wn = wn
        self.type = 'stationary'
        self.derivatives = 1    #number of derivatives in this kernel
        self.params_size = 1    #number of hyperparameters

    def __call__(self, r):
        return self.wn**2 * np.diag(np.diag(np.ones_like(r)))

class dWhiteNoise_dwn(WhiteNoise):
    """
        Log-derivative in order to the amplitude
    """
    def __init__(self, wn):
        super(dWhiteNoise_dwn, self).__init__(wn)
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)        
        
##### Squared exponential ######################################################
class SquaredExponential(covFunction):
    """
        Squared Exponential kernel, also known as radial basis function or RBF 
    kernel in other works.
        Parameters:
            theta = amplitude
            ell = length-scale
            wn = white noise
    """
    def __init__(self, theta, ell, wn):
        super(SquaredExponential, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and anisotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * exp(-0.5 * r**2 / self.ell**2) \
                    + self.wn**2 *np.diag(np.diag(np.ones_like(r))) 
        except ValueError:
            return self.theta**2 * exp(-0.5 * r**2 / self.ell**2)

class dSquaredExponential_dtheta(SquaredExponential):
    """
        Log-derivative in order to theta
    """
    def __init__(self, theta, ell, wn):
        super(dSquaredExponential_dtheta, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn
        
    def __call__(self, r):
        return 2 * self.theta**2 * exp(-0.5 * r**2 / self.ell**2) 

class dSquaredExponential_dell(SquaredExponential):
    """
        Log-derivative in order to ell
    """
    def __init__(self, theta, ell, wn):
        super(dSquaredExponential_dell, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn
        
    def __call__(self, r):
        return self.theta**2 * (r**2 / self.ell**2) * exp(-0.5 * r**2 / self.ell**2)

class dSquaredExponential_dwn(SquaredExponential):
    """
        Log-derivative in order to the wn
    """
    def __init__(self, theta, ell, wn):
        super(dSquaredExponential_dwn, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn
        
    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)
        
        
##### Periodic #################################################################
class Periodic(covFunction):
    """
        Definition of the periodic kernel.
        Parameters:
            theta = amplitude
            ell = lenght scale
            P = period
            wn = white noise
    """
    def __init__(self, theta, ell, P, wn):
        super(Periodic, self).__init__(theta, ell, P, wn)
        self.theta = theta
        self.ell = ell
        self.P = P
        self.wn = wn
        self.type = 'non-stationary and isotropic'
        self.derivatives = 4    #number of derivatives in this kernel
        self.params_size = 4    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell**2) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell**2)

class dPeriodic_dtheta(Periodic):
    """
        Log-derivative in order to theta
    """
    def __init__(self, theta, ell, P, wn):
        super(dPeriodic_dtheta, self).__init__(theta, ell, P, wn)
        self.theta = theta
        self.ell = ell
        self.P = P
        self.wn = wn

    def __call__(self, r):
        return 2*self.theta**2 * exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell**2)

class dPeriodic_dell(Periodic):
    """
        Log-derivative in order to ell
    """
    def __init__(self, theta, ell, P, wn):
        super(dPeriodic_dell, self).__init__(theta, ell, P, wn)
        self.theta = theta
        self.ell = ell
        self.P = P
        self.wn = wn

    def __call__(self, r):
        return self.theta**2 * 4*sine(pi*np.abs(r)/self.P)**2/self.ell**2 \
                * exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell**2)

class dPeriodic_dP(Periodic):
    """
        Log-derivative in order to P
    """
    def __init__(self, theta, ell, P, wn):
        super(dPeriodic_dP, self).__init__(theta, ell, P, wn)
        self.theta = theta
        self.ell = ell
        self.P = P
        self.wn = wn

    def __call__(self, r):
        return self.theta**2 * 4*pi*r*cosine(pi*np.abs(r)/self.P) \
                * sine(pi*np.abs(r)/self.P) \
                * exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell**2) \
                / (self.ell**2*self.P)

class dPeriodic_dwn(Periodic):
    """
        Log-derivative in order to wn
    """
    def __init__(self, theta, ell, P, wn):
        super(dPeriodic_dwn, self).__init__(theta, ell, P, wn)
        self.theta = theta
        self.ell = ell
        self.P = P
        self.wn = wn

    def __call__(self, r):
        try:
            return 2*self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)
        
        
##### Quasi Periodic ###########################################################
class QuasiPeriodic(covFunction):
    """
        This kernel is the product between the exponential sine squared kernel 
    and the squared exponential kernel, commonly known as the quasi-periodic 
    kernel.
        Parameters:
            theta = amplitude
            ell_e = evolutionary time scale
            P = kernel periodicity
            ell_p = length scale of the periodic component
            wn = white noise
    """
    def __init__(self, theta, ell_e, P, ell_p, wn):
        super(QuasiPeriodic, self).__init__(theta, ell_e, P, ell_p, wn)
        self.theta = theta
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 5    #number of derivatives in this kernel
        self.params_size = 5    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * exp(-2*sine(pi*r/self.P)**2 \
                       /self.ell_p**2 - r**2/(2*self.ell_e**2)) \
                       + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * exp(-2*sine(pi*np.abs(r)/self.P)**2 \
                       /self.ell_p**2 - r**2/(2*self.ell_e**2))

class dQuasiPeriodic_dtheta(QuasiPeriodic):
    """
        Log-derivative in order to theta
    """
    def __init__(self, theta, ell_e, P, ell_p, wn):
        super(dQuasiPeriodic_dtheta, self).__init__(theta, ell_e, P, ell_p, wn)
        self.theta = theta
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        return 2*self.theta**2 \
                * exp(-2*sine(pi*r/self.P)**2/self.ell_p**2-r**2/(2*self.ell_e**2))

class dQuasiPeriodic_delle(QuasiPeriodic):
    """
        Log-derivative in order to ell_e
    """
    def __init__(self, theta, ell_e, P, ell_p, wn):
        super(dQuasiPeriodic_delle, self).__init__(theta, ell_e, P, ell_p, wn)
        self.theta = theta
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        return self.theta**2 * (r**2*exp(-0.5*r**2/self.ell_e**2 \
                           -2*sine(pi*np.abs(r)/self.P)**2 \
                           / self.ell_p**2)) / self.ell_e**2

class dQuasiPeriodic_dP(QuasiPeriodic):
    """
        Log-derivative in order to P
    """
    def __init__(self, theta, ell_e, P, ell_p, wn):
        super(dQuasiPeriodic_dP, self).__init__(theta, ell_e, P, ell_p, wn)
        self.theta = theta
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        return self.theta**2 * (4*pi*r*cosine(pi*np.abs(r)/self.P) \
                * sine(pi*np.abs(r)/self.P) * exp(-0.5*r**2/self.ell_e**2 \
                     - 2*sine(pi*np.abs(r) /self.P)**2/self.ell_p**2)) \
                / (self.ell_p**2*self.P)

class dQuasiPeriodic_dellp(QuasiPeriodic):
    """
        Log-derivative in order to ell_p
    """
    def __init__(self, theta, ell_e, P, ell_p, wn):
        super(dQuasiPeriodic_dellp, self).__init__(theta, ell_e, P, ell_p, wn)
        self.theta = theta
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        return self.theta**2 * (4*sine(pi*np.abs(r)/self.P)**2 *exp(-0.5*r**2 \
                / self.ell_e**2 -2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2)) \
                / self.ell_p**2

class dQuasiPeriodic_dwn(QuasiPeriodic):
    """
        Log-derivative in order to wn
    """
    def __init__(self, theta, ell_e, P, ell_p, wn):
        super(dQuasiPeriodic_dwn, self).__init__(theta, ell_e, P, ell_p, wn)
        self.theta = theta
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)
        
        
##### Rational Quadratic #######################################################
class RationalQuadratic(covFunction):
    """
        Definition of the rational quadratic kernel.
        Parameters:
            theta = amplitude
            alpha = weight of large and small scale variations
            ell = characteristic lenght scale to define the kernel "smoothness"
            wn = white noise amplitude
    """
    def __init__(self, theta, alpha, ell, wn):
        super(RationalQuadratic, self).__init__(theta, alpha, ell, wn)
        self.theta = theta
        self.alpha = alpha
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and anisotropic'
        self.derivatives = 4    #number of derivatives in this kernel
        self.params_size = 4    #number of hyperparameters

    def __call__(self, r):
        try: 
            return self.theta**2 /(1+r**2/(2*self.alpha*self.ell**2))**(-self.alpha) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 /(1+r**2/(2*self.alpha*self.ell**2))**(-self.alpha)

class dRationalQuadratic_dtheta(RationalQuadratic):
    """
        Log-derivative in order to theta
    """
    def __init__(self, theta, alpha, ell, wn):
        super(dRationalQuadratic_dtheta, self).__init__(theta, alpha, ell, wn)
        self.theta = theta
        self.alpha = alpha
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        return 2*self.theta**2 /(1+r**2/(2*self.alpha*self.ell**2))**(-self.alpha)

class dRationalQuadratic_dalpha(RationalQuadratic):
    """
        Log-derivative in order to alpha
    """
    def __init__(self, theta, alpha, ell, wn):
        super(dRationalQuadratic_dalpha, self).__init__(theta, alpha, ell, wn)
        self.theta = theta
        self.alpha = alpha
        self.ell = ell
        self.wn = wn

    def __call(self, r):
        return self.theta**2 * self.alpha*(r**2 / (2*self.alpha*self.ell**2 \
                    *(r**2/(2*self.alpha*self.ell**2)+1)) \
                    -np.log(r**2/(2*self.alpha*self.ell**2)+1)) \
                    /(1+r**2/(2*self.alpha*self.ell**2))**self.alpha

class dRationalQuadratic_dell(RationalQuadratic):
    """
        Log-derivative in order to ell
    """
    def __init__(self, theta, alpha, ell, wn):
        super(dRationalQuadratic_dell, self).__init__(theta, alpha, ell, wn)
        self.theta = theta
        self.alpha = alpha
        self.ell = ell
        self.wn = wn

    def __call(self, r):
        return self.theta**2 * r**2*(1+r**2/(2*self.alpha*self.ell**2))**(-1-self.alpha) \
                /self.ell**2

class dRationalQuadratic_dwn(RationalQuadratic):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, theta, alpha, ell, wn):
        super(dRationalQuadratic_dwn, self).__init__(theta, alpha, ell, wn)
        self.theta = theta 
        self.alpha = alpha
        self.ell = ell
        self.wn = wn
        
    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)
        
        
##### RQP kernel ###############################################################
class RQP(covFunction):
    """
        Definition of the product between the exponential sine squared kernel 
    and the rational quadratic kernel that we called RQP kernel.
        If I am thinking this correctly then this kernel should tend to the
    QuasiPeriodic kernel as alpha increases, although I am not sure if we can
    say that it tends to the QuasiPeriodic kernel as alpha tends to infinity.
        Parameters:
            theta = amplitude
            alpha = alpha of the rational quadratic kernel
            ell_e and ell_p = aperiodic and periodic lenght scales
            P = periodic repetitions of the kernel
            wn = white noise amplitude
    """
    def __init__(self, theta, alpha, ell_e, P, ell_p, wn):
        super(RQP, self).__init__(theta, alpha, ell_e, P, ell_p, wn)
        self.theta = theta
        self.alpha = alpha
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 6    #number of derivatives in this kernel
        self.params_size = 6    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2) \
                        *(1+r**2/(2*self.alpha*self.ell_e**2))**(-self.alpha) \
                        + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 *exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2) \
                        *(1+r**2/(2*self.alpha*self.ell_e**2))**(-self.alpha)

class dRQP_dtheta(RQP):
    """
        Log-derivative in order to theta
    """
    def __init__(self, theta, alpha, ell_e, P, ell_p, wn):
        super(dRQP_dtheta, self).__init__(theta, alpha, ell_e, P, ell_p, wn)
        self.theta = theta
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        return 2*self.theta**2 * exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2) \
                        *(1+r**2/(2*self.alpha*self.ell_e**2))**(-self.alpha)

class dRQP_dalpha(RQP):
    """
        Log-derivative in order to alpha
    """
    def __init__(self, theta, alpha, ell_e, P, ell_p, wn):
        super(dRQP_dalpha, self).__init__(theta, alpha, ell_e, P, ell_p, wn)
        self.theta = theta
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        return self.theta**2 * self.alpha*(r**2/(2*self.alpha*self.ell_e**2 \
                                   * (r**2/(2*self.alpha*self.ell_e**2)+1)) \
                 - np.log(r**2/(2*self.alpha*self.ell_e**2) +1) \
                 * exp(-(2*sine(pi*r/self.P)**2)/self.ell_p**2)) \
                 /(1 + r**2/(2*self.alpha*self.ell_e**2))**self.alpha

class dRQP_delle(RQP):
    """
        Log-derivative in order to ell_e
    """
    def __init__(self, theta, alpha, ell_e, P, ell_p, wn):
        super(dRQP_delle, self).__init__(theta, alpha, ell_e, P, ell_p, wn)
        self.theta = theta
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        return self.theta**2*r**2*(1+r**2/(2*self.alpha*self.ell_e**2))**(-1-self.alpha) \
                * exp(-(2*sine(pi*r/self.P)**2)/self.ell_p**2)/self.ell_e**2

class dRQP_dP(RQP):
    """
        Log-derivative in order to P
    """
    def __init__(self, theta, alpha, ell_e, P, ell_p, wn):
        super(dRQP_dP, self).__init__(theta, alpha, ell_e, P, ell_p, wn)
        self.theta = theta
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        return self.theta**2 * (4*pi*r*cosine(pi*r/self.P)*sine(pi*r/self.P) \
                * exp(-(2*sine(pi*r/self.P)**2)/self.ell_p**2)) \
                / (self.ell_p**2 \
                   * (1+r**2/(2*self.alpha*self.ell_e**2))**self.alpha * self.P)

class dRQP_dellp(RQP):
    """
        Log-derivative in order to ell_p
    """
    def __init__(self, theta, alpha, ell_e, P, ell_p, wn):
        super(dRQP_dellp, self).__init__(theta, alpha, ell_e, P, ell_p, wn)
        self.theta = theta
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call(self, r):
        return self.theta**2 * (4*sine(pi*r/self.P)**2 \
                * exp(-(2*sine(pi*r/self.P)**2) / self.ell_p**2)) \
                / (self.ell_p**2 * (1+r**2/(2*self.alpha*self.ell_e**2))**self.alpha)

class dRQP_dwn(RQP):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, theta, alpha, ell_e, P, ell_p, wn):
        super(dRQP_dwn, self).__init__(theta, alpha, ell_e, P, ell_p, wn)
        self.theta = theta
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


##### Cosine ###################################################################
class Cosine(covFunction):
    """
        Definition of the cosine kernel.
        Parameters:
            theta  =amplitude
            P = period
            wn = white noise amplitude
    """
    def __init__(self, theta, P, wn):
        super(Cosine, self).__init__(theta, P, wn)
        self.theta = theta
        self.P = P
        self.wn = wn
        self.type = 'non-stationary and isotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * cosine(2*pi*np.abs(r) / self.P) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 *cosine(2*pi*np.abs(r) / self.P)

class dCosine_dtheta(Cosine):
    """
        Log-derivative in order to theta
    """
    def __init__(self, theta, P, wn):
        super(dCosine_dtheta, self).__init__(theta, P, wn)
        self.theta = theta
        self.P = P
        self.wn = wn

    def __call__(self, r):
        return self.theta**2 * cosine(2*pi*np.abs(r) / self.P)

class dCosine_dP(Cosine):
    """
        Log-derivative in order to P
    """
    def __init__(self, theta, P, wn):
        super(dCosine_dP, self).__init__(theta, P, wn)
        self.theta = theta
        self.P = P
        self.wn = wn

    def __call__(self, r):
        return self.theta**2 * r*pi*sine(2*pi*np.abs(r)/self.P)/self.P

class dCosine_dwn(Cosine):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, theta, P, wn):
        super(dCosine_dwn, self).__init__(theta, P, wn)
        self.theta = theta
        self.P = P
        self.wn = wn

    def __call__(self, r):
        try:
            return 2*self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)
        
        
##### Laplacian ##############################################################
class Laplacian(covFunction):
    """
        Definition of the Laplacian kernel.
        Parameters:
            theta = amplitude
            ell = characteristic lenght scale
            wn = white noise amplitude
    """
    def __init__(self, theta, ell, wn):
        super(Laplacian, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and isotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r): 
        try:
            return self.theta**2 * exp(-np.abs(r)/self.ell) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * exp(-np.abs(r)/self.ell) 

class dLaplacian_dtheta(Laplacian):
    """
        Log-derivative in order to theta
    """
    def __init__(self, theta, ell, wn):
        super(dLaplacian_dtheta, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        return 2*self.theta**2 * exp(-np.abs(r)/self.ell) 

class dLaplacian_dell(Laplacian):
    """
        Log-derivative in order to ell
    """
    def __init__(self, theta, ell, wn):
        super(dLaplacian_dell, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        return -0.5*self.theta**2 * r * exp(- np.abs(r)/self.ell) / self.ell

class dLaplacian_dwn(Laplacian):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, theta, ell, wn):
        super(dLaplacian_dwn, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)
        
        
##### Exponential ##############################################################
class Exponential(covFunction):
    """
        Definition of the exponential kernel.
        Parameters:
            theta = amplitude
            ell = characteristic lenght scale
            wn = white noise amplitude
    """
    def __init__(self, theta, ell, wn):
        super(Exponential, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and isotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r): 
        try:
            return self.theta**2 * exp(-np.abs(r)/(2*self.ell**2)) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * exp(-np.abs(r)/self.ell) 

class dExpoential_dtheta(Exponential):
    """
        Log-derivative in order to theta
    """
    def __init__(self, theta, ell, wn):
        super(dExpoential_dtheta, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        raise 2*self.theta**2 * exp(-np.abs(r)/(2*self.ell**2))

class dExpoential_dell(Exponential):
    """
        Log-derivative in order to ell
    """
    def __init__(self, theta, ell, wn):
        super(dExpoential_dell, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        raise -0.5 * self.theta**2 * r * exp(- np.abs(r)/self.ell) / self.ell

class dExpoential_dwn(Exponential):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, theta, ell, wn):
        super(dExpoential_dwn, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)
        
        
##### Matern 3/2 ###############################################################
class Matern32(covFunction):
    """
        Definition of the Matern 3/2 kernel. This kernel arise when setting 
    v=3/2 in the matern family of kernels
        Parameters:
            theta = amplitude
            ell = characteristic lenght scale
            wn = white noise amplitude
    """
    def __init__(self, theta, ell, wn):
        super(Matern32, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and isotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * (1.0+np.sqrt(3.0)*np.abs(r)/self.ell) \
                        *np.exp(-np.sqrt(3.0)*np.abs(r) / self.ell) \
                        + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * (1.0+np.sqrt(3.0)*np.abs(r)/self.ell) \
                        *np.exp(-np.sqrt(3.0)*np.abs(r) / self.ell)

class dMatern32_dtheta(Matern32):
    """
        Log-derivative in order to theta
    """
    def __init__(self, theta, ell, wn):
        super(dMatern32_dtheta, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        return 2*self.theta**2 * (1.0+np.sqrt(3.0)*np.abs(r)/self.ell) \
                        *np.exp(-np.sqrt(3.0)*np.abs(r) / self.ell)

class dMatern32_dell(Matern32):
    """
        Log-derivative in order to ell
    """
    def __init__(self, theta, ell, wn):
        super(dMatern32_dell, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        return self.theta**2 * (sqrt(3)*r*(1+(sqrt(3)*r)/self.ell) \
                *exp(-(sqrt(3)*r) / self.ell)) / self.ell \
                -(sqrt(3) * r * exp(-(sqrt(3)*r) / self.ell))/self.ell

class dMatern32_dwn(Matern32):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, theta, ell, wn):
        super(dMatern32_dwn, self).__init__(theta, ell, wn)
        self.theta = theta 
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)
        
        
#### Matern 5/2 ################################################################
class Matern52(covFunction):
    """
        Definition of the Matern 5/2 kernel. This kernel arise when setting 
    v=5/2 in the matern family of kernels
        Parameters:
            theta = amplitude
            ell = characteristic lenght scale  
            wn = white noise amplitude
    """
    def __init__(self, theta, ell, wn):
        super(Matern52, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and isotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * (1.0+( 3*np.sqrt(5)*self.ell*np.abs(r) \
                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                           *exp(-np.sqrt(5.0)*np.abs(r)/self.ell) \
                           + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * (1.0+( 3*np.sqrt(5)*self.ell*np.abs(r) \
                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                           *exp(-np.sqrt(5.0)*np.abs(r)/self.ell)

class dMatern52_dtheta(Matern52):
    """
        Log-derivative in order to theta
    """
    def __init__(self, theta, ell, wn):
        super(dMatern52_dtheta, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        return 2*self.theta**2 * (1.0+(3*np.sqrt(5)*self.ell*np.abs(r) \
                           +5*np.abs(r)**2)/(3*self.ell**2)) \
                           *exp(-np.sqrt(5.0)*np.abs(r)/self.ell)

class dMatern52_dell(Matern52):
    """
        Log-derivative in order to ell
    """
    def __init__(self, theta, ell, wn):
        super(dMatern52_dell, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        return self.theta**2 * self.ell * ((sqrt(5)*r*(1+(sqrt(5)*r) \
                                 /self.ell+(5*r**2)/(3*self.ell**2)) \
                             *exp(-(sqrt(5)*r)/self.ell)) \
            /self.ell**2 +(-(sqrt(5)*r)/self.ell**2-(10*r**2) \
                           /(3*self.ell**3)) \
                           *exp(-(sqrt(5)*r)/self.ell))

class dMatern52_dwn(Matern52):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, theta, ell, wn):
        super(dMatern52_dwn, self).__init__(theta, ell, wn)
        self.theta = theta
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)
        
        
#### Linear ####################################################################
class Linear(covFunction):
    """
        Definition of the Linear kernel.
            theta = amplitude (should we even have an amplitude???)
            c = constant
            wn = white noise amplitude
    """
    def __init__(self, theta, c, wn):
        super(Linear, self).__init__(theta, c, wn)
        self.theta = theta
        self.c = c
        self.wn = wn
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r, t1, t2):
        try:
            return  self.theta**2 * (t1 - self.c) * (t2 - self.c) \
                + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return  (t1 - self.c) * (t2 - self.c)

class dLinear_dtheta(Linear):
    """
        Log-derivative in order to theta???
    """
    def __init__(self, theta, c, wn):
        super(dLinear_dtheta, self).__init__(theta, c, wn)
        self.theta = theta
        self.c = c
        self.wn = wn

    def __call__(self, r, t1, t2):
        return 2*self.theta**2 * (t1 - self.c) * (t2 - self.c) 

class dLinear_dc(Linear):
    """
        Log-derivative in order to c
    """
    def __init__(self, theta, c, wn):
        super(dLinear_dc, self).__init__(theta, c, wn)
        self.thta = theta
        self.c = c
        self.wn = wn

    def __call__(self, r, t1, t2):
        return self.theta**2 * self.c * (-t1 - t2 + 2*self.c)

class dLinear_dwn(Linear):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, theta, c, wn):
        super(dLinear_dwn, self).__init__(theta, c, wn)
        self.theta = theta
        self.c = c
        self.wn = wn

    def __call__(self, r, t1, t2):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)
        
        
##### Gamma-exponential ########################################################
class GammaExp(covFunction):
    """
        Definition of the gamma-exponential kernel
            theta = amplitude
            gamma = shape parameter ( 0 < gamma <= 2)
            ell = lenght scale
            wn = white noise amplitude
    """
    def __init__(self, theta, gamma, ell, wn):
        super(GammaExp, self).__init__(theta, gamma, ell, wn)
        self.theta = theta
        self.gamma = gamma
        self.ell = ell
        self.wn = wn
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 4    #number of derivatives in this kernel
        self.params_size = 4    #number of hyperparameters

    def __call__(self, r):
        try: 
            return self.theta**2 * exp(-(np.abs(r)/self.ell)**self.gamma) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * exp(-(np.abs(r)/self.ell) ** self.gamma) 

class dGammaExp_dtheta(GammaExp):
    """
        Log-derivative in order to theta
    """
    def __init__(self, theta, gamma, ell, wn):
        super(dGammaExp_dtheta, self).__init__(theta, gamma, ell, wn)
        self.theta = theta
        self.gamma = gamma
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        return 2*self.theta**2 * exp(-(np.abs(r)/self.ell)**self.gamma)

class dGammaExp_dgamma(GammaExp):
    """
        Log-derivative in order to ell
    """
    def __init__(self, theta, gamma, ell, wn):
        super(dGammaExp_dgamma, self).__init__(theta, gamma, ell, wn)
        self.theta = theta
        self.gamma = gamma
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        return - self.theta**2 * self.gamma * (np.abs(r)/self.ell)**self.gamma \
                *np.log(np.abs(r)/self.ell) * exp(-(np.abs(r)/self.ell)**self.gamma)

class dGammaExp_dell(GammaExp):
    """
        Log-derivative in order to gamma
    """
    def __init__(self, theta, gamma, ell, wn):
        super(dGammaExp_dell, self).__init__(theta, gamma, ell, wn)
        self.theta = theta 
        self.gamma = gamma
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        return self.theta**2 * (np.abs(r)/self.ell)**self.gamma \
                * self.gamma * exp(-(np.abs(r)/self.ell)**self.gamma)

class dGammaExp_dwn(GammaExp):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, theta, gamma, ell, wn):
        super(dGammaExp_dwn, self).__init__(theta, gamma, ell, wn)
        self.theta = theta
        self.gamma = gamma
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)
        
        
##### Polinomial ###############################################################
class Polynomial(covFunction):
    """
        Definition of the polinomial kernel
            theta = amplitude ???
            a = real value > 0
            b = real value >= 0
            c = integer value
            wn = white noise amplitude
    """
    def __init__(self, theta, a, b, c, wn):
        super(Polynomial, self).__init__(theta, a, b, c, wn)
        self.theta = theta
        self.a = a
        self.b = b
        self.c = c
        self.wn = wn
        self.type = 'unknown'
        self.derivatives = 5    #number of derivatives in this kernel
        self.params_size = 5    #number of hyperparameters

    def __call__(self, r, t1, t2):
        try: 
            return self.theta**2 * (self.a * t1 * t2 + self.b)**self.c \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return (self.a * t1 * t2 + self.b)**self.c 

class dPolynomial_dtheta(Polynomial):
    """
        Log-derivative in order to theta
    """
    def __init__(self, theta, a, b, c, wn):
        super(dPolynomial_dtheta, self).__init__(theta, a, b, c, wn)
        self.theta = theta
        self.a = a
        self.b = b
        self.c = c
        self.wn = wn

    def __call__(self, r, t1, t2):
        return 2*self.theta**2 * (self.a * t1 * t2 + self.b)**self.c 

class dPolynomial_da(Polynomial):
    """
        Log-derivative in order to a
    """
    def __init__(self, theta, a, b, c, wn):
        super(dPolynomial_da, self).__init__(theta, a, b, c, wn)
        self.theta = theta
        self.a = a
        self.b = b
        self.c = c
        self.wn = wn

    def __call__(self, r, t1, t2):
        return self.theta**2 * self.c*t1*t2*(self.b+self.a*t1*t2)**(self.c-1)*self.a

class dPolynomial_db(Polynomial):
    """
        Log-derivative in order to b
    """
    def __init__(self, theta, a, b, c, wn):
        super(dPolynomial_db, self).__init__(theta, a, b, c, wn)
        self.theta = theta
        self.a = a
        self.b = b
        self.c = c
        self.wn = wn

    def __call__(self, r, t1, t2):
        return self.theta**2 * self.c*(self.b+self.a*t1*t2)**(self.c-1)*self.b

class dPolynomial_dc(Polynomial):
    """
        Log-derivative in order to c
    """
    def __init__(self, theta, a, b, c, wn):
        super(dPolynomial_dc, self).__init__(theta, a, b, c, wn)
        self.theta = theta
        self.a = a
        self.b = b
        self.c = c
        self.wn = wn

    def __call__(self, r, t1, t2):
        return self.theta**2 * self.c*(self.b+self.a*t1*t2)**self.c \
                *np.log(self.a*t1*t2+self.b)

class dPolynomial_dwn(Polynomial):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, theta, a, b, c, wn):
        super(dPolynomial_dwn, self).__init__(theta, a, b, c, wn)
        self.theta = theta
        self.a = a
        self.b = b
        self.c = c
        self.wn = wn

    def __call__(self, r):
        try:
            return 2*self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


### END

