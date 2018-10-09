#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#because it makes my life easier down the line
pi, exp, sine, cosine, sqrt = np.pi, np.exp, np.sin, np.cos, np.sqrt

class nodeFunction(object):
    """
        Definition the node functions (kernels) of our GPRN, by default and 
    because it simplifies my life, all kernels include a white noise term
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


##### Constant #################################################################
class Constant(nodeFunction):
    """
        This kernel returns its constant argument c with white noise
        Parameters:
            c = constant
            wn = white noise amplitude
    """
    def __init__(self, c, wn):
        super(Constant, self).__init__(c, wn)
        self.c = c
        self.c = wn
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.c * np.ones_like(r) \
                        + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.c * np.ones_like(r)

class dConstant_dc(Constant):
    """
        Log-derivative in order to c
    """
    def __init__(self, c, wn):
        super(dConstant_dc, self).__init__(c, wn)
        self.c = c
        self.wn = wn

    def __call__(self, r):
        return self.c * np.ones_like(r)

class dConstant_dwn(Constant):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, c, wn):
        super(dConstant_dc, self).__init__(c, wn)
        self.c = c
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


##### White Noise ##############################################################
class WhiteNoise(nodeFunction):
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
        return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))


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
            return exp(-0.5 * r**2 / self.ell**2) \
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


##### Rational Quadratic #######################################################
class RationalQuadratic(nodeFunction):
    """
        Definition of the rational quadratic kernel.
        Parameters:
            alpha = weight of large and small scale variations
            ell = characteristic lenght scale to define the kernel "smoothness"
            wn = white noise amplitude
    """
    def __init__(self, alpha, ell, wn):
        super(RationalQuadratic, self).__init__(alpha, ell, wn)
        self.alpha = alpha
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and anisotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        try: 
            return 1 / (1+ r**2/ (2*self.alpha*self.ell**2))**self.alpha \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return 1 / (1+ r**2/ (2*self.alpha*self.ell**2))**self.alpha

class dRationalQuadratic_dalpha(RationalQuadratic):
    """
        Log-derivative in order to alpha
    """
    def __init__(self, alpha, ell, wn):
        super(dRationalQuadratic_dalpha, self).__init__(alpha, ell, wn)
        self.alpha = alpha
        self.ell = ell
        self.wn = wn

    def __call(self, r):
        return ((r**2/(2*self.alpha*self.ell**2*(r**2/(2*self.alpha*self.ell**2)+1))\
                 - np.log(r**2/(2*self.alpha*self.ell**2)+1)) * self.alpha) \
                    / (1+r**2/(2*self.alpha*self.ell**2))**self.alpha

class dRationalQuadratic_dell(RationalQuadratic):
    """
        Log-derivative in order to ell
    """
    def __init__(self, alpha, ell, wn):
        super(dRationalQuadratic_dell, self).__init__(alpha, ell, wn)
        self.alpha = alpha
        self.ell = ell
        self.wn = wn

    def __call(self, r):
        return r**2 * (1+r**2/(2*self.alpha*self.ell**2))**(-1-self.alpha) \
                * 1 / self.ell**2

class dRationalQuadratic_dwn(RationalQuadratic):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, alpha, ell, wn):
        super(dRationalQuadratic_dwn, self).__init__(alpha, ell, wn)
        self.alpha = alpha
        self.ell = ell
        self.wn = wn
        
    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


##### RQP kernel ###############################################################
class RQP(nodeFunction):
    """
        Definition of the product between the exponential sine squared kernel 
    and the rational quadratic kernel that we called RQP kernel.
        If I am thinking this correctly then this kernel should tend to the
    QuasiPeriodic kernel as alpha increases, although I am not sure if we can
    say that it tends to the QuasiPeriodic kernel as alpha tends to infinity.
        Parameters:
            ell_e and ell_p = aperiodic and periodic lenght scales
            alpha = alpha of the rational quadratic kernel
            P = periodic repetitions of the kernel
            wn = white noise amplitude
    """
    def __init__(self, alpha, ell_e, P, ell_p, wn):
        super(RQP, self).__init__(alpha, ell_e, P, ell_p, wn)
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 5    #number of derivatives in this kernel
        self.params_size = 5    #number of hyperparameters

    def __call__(self, r):
        try:
            return exp(- 2*sine(pi*np.abs(r)/self.P)**2 / self.ell_p**2) \
                        /(1+ r**2/ (2*self.alpha*self.ell_e**2))**self.alpha \
                        + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return exp(- 2*sine(pi*np.abs(r)/self.P)**2 / self.ell_p**2) \
                        /(1+ r**2/ (2*self.alpha*self.ell_e**2))**self.alpha

class dRQP_dalpha(RQP):
    """
        Log-derivative in order to alpha
    """
    def __init__(self, alpha, ell_e, P, ell_p, wn):
        super(dRQP_dalpha, self).__init__(alpha, ell_e, P, ell_p, wn)
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        return self.alpha * ((r**2 / (2*self.alpha \
                         *self.ell_e**2*(r**2/(2*self.alpha*self.ell_e**2)+1)) \
            -np.log(r**2/(2*self.alpha*self.ell_e**2)+1)) \
            *exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2)) \
            /(1+r**2/(2*self.alpha*self.ell_e**2))**self.alpha

class dRQP_delle(RQP):
    """
        Log-derivative in order to ell_e
    """
    def __init__(self, alpha, ell_e, P, ell_p, wn):
        super(dRQP_delle, self).__init__(alpha, ell_e, P, ell_p, wn)
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        return (r**2*(1+r**2/(2*self.alpha*self.ell_e**2))**(-1-self.alpha) \
                *exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2))/self.ell_e**2

class dRQP_dP(RQP):
    """
        Log-derivative in order to P
    """
    def __init__(self, alpha, ell_e, P, ell_p, wn):
        super(dRQP_dP, self).__init__(alpha, ell_e, P, ell_p, wn)
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call__(self, r):
        return (4*pi*r*cosine(pi*np.abs(r)/self.P)*sine(pi*np.abs(r)/self.P) \
                *exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2)) \
            /(self.ell_p**2*(1+r**2/(2*self.alpha*self.ell_e**2))^self.alpha*self.P)

class dRQP_dellp(RQP):
    """
        Log-derivative in order to ell_p
    """
    def __init__(self, alpha, ell_e, P, ell_p, wn):
        super(dRQP_dellp, self).__init__(alpha, ell_e, P, ell_p, wn)
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn

    def __call(self, r):
        return (4*sine(pi*np.abs(r)/self.P)**2 \
                *exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2)) \
                /(self.ell_p**2*(1+r**2/(2*self.alpha*self.ell_e**2))**self.alpha)

class dRQP_dwn(RQP):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, alpha, ell_e, P, ell_p, wn):
        super(dRQP_dwn, self).__init__(alpha, ell_e, P, ell_p, wn)
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
class Cosine(nodeFunction):
    """
        Definition of the cosine kernel.
        Parameters:
            P = period
            wn = white noise amplitude
    """
    def __init__(self, P, wn):
        super(Cosine, self).__init__(P, wn)
        self.P = P
        self.wn = wn
        self.type = 'non-stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        try:
            return cosine(2*pi*np.abs(r) / self.P) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return cosine(2*pi*np.abs(r) / self.P)

class dCosine_dP(Cosine):
    """
        Log-derivative in order to P
    """
    def __init__(self, P, wn):
        super(dCosine_dP, self).__init__(P, wn)
        self.P = P
        self.wn = wn

    def __call__(self, r):
        return r * pi * sine(2*pi*np.abs(r) / self.P) / self.P

class dCosine_dwn(Cosine):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, P, wn):
        super(dCosine_dwn, self).__init__(P, wn)
        self.P = P
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


##### Exponential ##############################################################
class Exponential(nodeFunction):
    """
        Definition of the exponential kernel. This kernel arises when 
    setting v=1/2 in the matern family of kernels
        Parameters:
            ell = characteristic lenght scale
            wn = white noise amplitude
    """
    def __init__(self, ell, wn):
        super(Exponential, self).__init__(ell, wn)
        self.ell = ell
        self.wn
        self.type = 'stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r): 
        try:
            return exp(- np.abs(r)/self.ell) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return exp(- np.abs(r)/self.ell) 

class dExpoential_dell(Exponential):
    """
        Log-derivative in order to ell
    """
    def __init__(self, ell, wn):
        super(dExpoential_dell, self).__init__(ell, wn)
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        return -0.5 * r * exp(- np.abs(r)/self.ell) / self.ell

class dExpoential_dwn(Exponential):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, ell, wn):
        super(dExpoential_dell, self).__init__(ell, wn)
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)

##### Matern 3/2 ###############################################################
class Matern32(nodeFunction):
    """
        Definition of the Matern 3/2 kernel. This kernel arise when setting 
    v=3/2 in the matern family of kernels
        Parameters:
            ell = characteristic lenght scale
            wn = white noise amplitude
    """
    def __init__(self, ell, wn):
        super(Matern32, self).__init__(ell, wn)
        self.ell = ell
        self.type = 'stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        try:
            return (1.0 + np.sqrt(3.0)*np.abs(r)/self.ell) \
                        *np.exp(-np.sqrt(3.0)*np.abs(r) / self.ell) \
                        + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return (1.0 + np.sqrt(3.0)*np.abs(r)/self.ell) \
                        *np.exp(-np.sqrt(3.0)*np.abs(r) / self.ell)

class dMatern32_dell(Matern32):
    """
        Log-derivative in order to ell
    """
    def __init__(self, ell, wn):
        super(dMatern32_dell, self).__init__(ell, wn)
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        return (sqrt(3) * r * (1+ (sqrt(3) * r) / self.ell) \
                *exp(-(sqrt(3)*r) / self.ell)) / self.ell \
                -(sqrt(3) * r * exp(-(sqrt(3)*r) / self.ell))/self.ell

class dMatern32_dwn(Matern32):
    """
        Log-derivative in order to the white noise
    """
    def __init__(self, ell, wn):
        super(dMatern32_dwn, self).__init__(ell, wn)
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


#### Matern 5/2 ################################################################
class Matern52(nodeFunction):
    """
        Definition of the Matern 5/2 kernel. This kernel arise when setting 
    v=5/2 in the matern family of kernels
        Parameters:
            ell = characteristic lenght scale  
            wn = white noise amplitude
    """
    def __init__(self, ell, wn):
        super(Matern52, self).__init__(ell, wn)
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        try:
            return (1.0 + ( 3*np.sqrt(5)*self.ell*np.abs(r) \
                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                           *exp(-np.sqrt(5.0)*np.abs(r)/self.ell) \
                           + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return (1.0 + ( 3*np.sqrt(5)*self.ell*np.abs(r) \
                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                           *exp(-np.sqrt(5.0)*np.abs(r)/self.ell)

class dMatern52_dell(Matern52):
    """
        Log-derivative in order to ell
    """
    def __init__(self, ell, wn):
        super(dMatern52_dell, self).__init__(ell, wn)
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        return self.ell * ((sqrt(5)*r*(1+(sqrt(5)*r) \
                                 /self.ell+(5*r**2)/(3*self.ell**2)) \
                             *exp(-(sqrt(5)*r)/self.ell)) \
            /self.ell**2 +(-(sqrt(5)*r)/self.ell**2-(10*r**2) \
                           /(3*self.ell**3)) \
                           *exp(-(sqrt(5)*r)/self.ell))

class dMatern52_dwn(Matern52):
    """
        Log-derivative in order to the weight
    """
    def __init__(self, ell, wn):
        super(dMatern52_dwn, self).__init__(ell, wn)
        self.ell = ell
        self.wn = wn

    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)

#### Linear ####################################################################
class Linear(nodeFunction):
    """
        Definition of the Linear kernel.
            c = constant
            wn = white noise amplitude
    """
    def __init__(self, c, wn):
        super(Linear, self).__init__(c, wn)
        self.c = c
        self.wn = wn
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r, t1, t2):
        try:
            return  (t1 - self.c) * (t2 - self.c) \
                + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return  (t1 - self.c) * (t2 - self.c)


### END
