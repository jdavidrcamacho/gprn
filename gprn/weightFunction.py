#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#because it makes my life easier down the line
pi, exp, sine, cosine, sqrt = np.pi, np.exp, np.sin, np.cos, np.sqrt

class weightFunction(object):
    """
        Definition the weight functions (kernels) of our GPRN.
        Kernels not fully implemented yet:
            Matern32, and Matern52
    """
    def __init__(self, *args):
        """
            Puts all kernel arguments in an array pars
        """
        self.pars = np.array(args)

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

#Not working yet!
#    def __minus__(self, b):
#        return Minus(self, b)
#    def __rminus__(self, b):
#        return self.__minus__(b)
#
#
#class _operator(weightFunction):
#    """ 
#        To allow operations between two kernels 
#    """
#    def __init__(self, k1):
#        self.k1 = k1
#
#    @property
#    def pars(self):
#        return np.append(self.k1.pars)
#
#
#class Minus(_operator):
#    """ 
#        To allow a "minus" linear kernel
#    """
#    def __repr__(self):
#        return "-{0}".format(self.k1)
#
#    def __call__(self, r):
#        return -self.k1(r)


##### Constant #################################################################
class Constant(weightFunction):
    """
        This kernel returns its constant argument c 
        Parameters:
            c = constant
    """
    def __init__(self, c):
        super(Constant, self).__init__(c)
        self.c = c
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 1    #number of derivatives in this kernel
        self.params_size = 1    #number of hyperparameters

    def __call__(self, r):
        return self.c * np.ones_like(r)

class dConstant_dc(Constant):
    """
        Log-derivative in order to c
    """
    def __init__(self, c):
        super(dConstant_dc, self).__init__(c)
        self.c = c

    def __call__(self, r):
        return self.c * np.ones_like(r)

##### White Noise ##############################################################
class WhiteNoise(weightFunction):
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
class SquaredExponential(weightFunction):
    """
        Squared Exponential kernel, also known as radial basis function or RBF 
    kernel in other works.
        Parameters:
            weight = weight/amplitude of the kernel
            ell = length-scale
    """
    def __init__(self, weight, ell):
        super(SquaredExponential, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell
        self.type = 'stationary and anisotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        return self.weight**2 * exp(-0.5 * r**2 / self.ell**2)

class dSquaredExponential_dweight(SquaredExponential):
    """
        Log-derivative in order to the weight
    """
    def __init__(self, weight, ell):
        super(dSquaredExponential_dweight, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell

    def __call__(self, r):
        return 2 * self.weight**2 * exp(-0.5 * r**2 / self.ell**2)

class dSquaredExponential_dell(SquaredExponential):
    """
        Log-derivative in order to the ell
    """
    def __init__(self, weight, ell):
        super(dSquaredExponential_dell, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell

    def __call__(self, r):
        return (r**2 * self.weight**2 / self.ell**2) \
                * exp(-0.5 * r**2 / self.ell**2)


##### Periodic #################################################################
class Periodic(weightFunction):
    """
        Definition of the periodic kernel.
        Parameters:
            weight = weight/amplitude of the kernel
            ell = lenght scale
            P = period
    """
    def __init__(self, weight, ell, P):
        super(Periodic, self).__init__(weight, ell, P)
        self.weight = weight
        self.ell = ell
        self.P = P
        self.type = 'non-stationary and isotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        return self.weight**2 * exp( -2 * sine(pi*np.abs(r)/self.P)**2 /self.ell**2)

class dPeriodic_dweight(Periodic):
    """
        Log-derivative in order to the weight
    """
    def __init__(self, weight, ell, P):
        super(dPeriodic_dweight, self).__init__(weight, ell, P)
        self.weight = weight
        self.ell = ell
        self.P = P

    def __call__(self, r):
        return 2 * self.weight**2 * exp(-2 * sine(pi * np.abs(r) / self.P)**2 \
                                        / self.ell**2)

class dPeriodic_dell(Periodic):
    """
        Log-derivative in order to ell
    """
    def __init__(self, weight, ell, P):
        super(dPeriodic_dell, self).__init__(weight, ell, P)
        self.weight = weight
        self.ell = ell
        self.P = P

    def __call__(self, r):
        return (4* self.weight**2 * sine(pi * np.abs(r) / self.P)**2 \
                *exp(-2 * sine(pi * np.abs(r) / self.P)**2 \
                     / self.ell**2)) / self.ell**2

class dPeriodic_dP(Periodic):
    """
        Log-derivative in order to P
    """
    def __init__(self, weight, ell, P):
        super(dPeriodic_dP, self).__init__(weight, ell, P)
        self.weight = weight
        self.ell = ell
        self.P = P

    def __call__(self, r):
        return (4 * pi * r * self.weight**2 \
                * cosine(pi*np.abs(r) / self.P) *sine(pi*np.abs(r) / self.P) \
                * exp(-2 * sine(pi*np.abs(r) / self.P)**2 / self.ell**2)) \
                / (self.ell**2 * self.P)


##### Quasi Periodic ###########################################################
class QuasiPeriodic(weightFunction):
    """
        This kernel is the product between the exponential sine squared kernel 
    and the squared exponential kernel, commonly known as the quasi-periodic 
    kernel.
        Parameters:
            weight = weight/amplitude of the kernel
            ell_e = evolutionary time scale
            ell_p = length scale of the Periodic component
            P = kernel Periodicity
    """
    def __init__(self, weight, ell_e, P, ell_p):
        super(QuasiPeriodic, self).__init__(weight, ell_e, P, ell_p)
        self.weight = weight
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 4    #number of derivatives in this kernel
        self.params_size = 4    #number of hyperparameters

    def __call__(self, r):
        return self.weight**2 *exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                                   /self.ell_p**2 - r**2/(2*self.ell_e**2))

class dQuasiPeriodic_dweight(Periodic):
    """
            Log-derivative in order to the weight
    """
    def __init__(self, weight, ell_e, P, ell_p):
        super(dQuasiPeriodic_dweight, self).__init__(weight, ell_e, P, ell_p)
        self.weight = weight
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call(self, r):
        return 2 * self.weight**2 *exp(-2 * sine(pi*np.abs(r)/self.P)**2 \
                                   /self.ell_p**2 - r**2/(2*self.ell_e**2))

class dQuasiPeriodic_delle(QuasiPeriodic):
    """
        Log-derivative in order to ell_e
    """
    def __init__(self, weight, ell_e, P, ell_p):
        super(dQuasiPeriodic_delle, self).__init__(weight, ell_e, P, ell_p)
        self.weight = weight
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call(self, r):
        return (r**2 * self.weight**2 / self.ell_e**2) \
                *exp(-2 * sine(pi*np.abs(r)/self.P)**2 \
                     /self.ell_p**2 - r**2/(2*self.ell_e**2))

class dQuasiPeriodic_dP(QuasiPeriodic):
    """
        Log-derivative in order to P
    """
    def __init__(self, weight, ell_e, P, ell_p):
        super(dQuasiPeriodic_dP, self).__init__(weight, ell_e, P, ell_p)
        self.weight = weight
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call(self, r):
        return 4 * pi * r * self.w**2 \
                * cosine(pi*np.abs(r)/self.P) * sine(pi*np.abs(r)/self.P) \
                * exp(-2 * sine(pi * np.abs(r)/self.P)**2 \
                      /self.ell_p**2 - r**2/(2*self.ell_e**2)) \
                      / (self.ell_p**2 * self.P)

class dQuasiPeriodic_dellp(QuasiPeriodic):
    """
        Log-derivative in order to ell_p
    """
    def __init__(self, weight, ell_e, P, ell_p):
        super(dQuasiPeriodic_dellp, self).__init__(weight, ell_e, P, ell_p)
        self.weight = weight
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call(self, r):
        return  4 * self.w**2 * sine(pi*r/self.P)**2 \
                * exp(-2 * sine(pi*np.abs(r)/self.P)**2 \
                      /self.ell_p**2 - r**2/(2*self.ell_e**2)) / self.ell_p**2


##### Rational Quadratic #######################################################
class RationalQuadratic(weightFunction):
    """
        Definition of the rational quadratic kernel.
        Parameters:
            weight = weight/amplitude of the kernel
            alpha = weight of large and small scale variations
            ell = characteristic lenght scale to define the kernel "smoothness"
    """
    def __init__(self, weight, alpha, ell):
        super(RationalQuadratic, self).__init__(weight, alpha, ell)
        self.weight = weight
        self.alpha = alpha
        self.ell = ell
        self.type = 'stationary and anisotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        return self.weight**2 / (1+ r**2/ (2*self.alpha*self.ell**2))**self.alpha

class dRationalQuadratic_dweight(RationalQuadratic):
    """
        Log-derivative in order to the weight
    """
    def __init__(self, weight, alpha, ell):
        super(dRationalQuadratic_dweight, self).__init__(weight, alpha, ell)
        self.weight = weight
        self.alpha = alpha
        self.ell = ell

    def __call__(self, r):
        return 2 * self.weight**2 \
                / (1+ r**2/ (2*self.alpha*self.ell**2))**self.alpha

class dRationalQuadratic_dalpha(RationalQuadratic):
    """
        Log-derivative in order to alpha
    """
    def __init__(self, weight, alpha, ell):
        super(dRationalQuadratic_dalpha, self).__init__(weight, alpha, ell)
        self.weight = weight
        self.alpha = alpha
        self.ell = ell

    def __call(self, r):
        return ((r**2/(2*self.alpha*self.ell**2*(r**2/(2*self.alpha*self.ell**2)+1))\
                 - np.log(r**2/(2*self.alpha*self.ell**2)+1)) \
                    * self.weight**2 * self.alpha) \
                    / (1+r**2/(2*self.alpha*self.ell**2))**self.alpha

class dRationalQuadratic_dell(RationalQuadratic):
    """
        Log-derivative in order to ell
    """
    def __init__(self, weight, alpha, ell):
        super(dRationalQuadratic_dell, self).__init__(weight, alpha, ell)
        self.weight = weight
        self.alpha = alpha
        self.ell = ell

    def __call(self, r):
        return r**2 * (1+r**2/(2*self.alpha*self.ell**2))**(-1-self.alpha) \
                * self.w**2 / self.ell**2


##### RQP kernel ###############################################################
class RQP(weightFunction):
    """
        Definition of the product between the exponential sine squared kernel 
    and the rational quadratic kernel that we called RQP kernel.
        If I am thinking this correctly then this kernel should tend to the
    QuasiPeriodic kernel as alpha increases, although I am not sure if we can
    say that it tends to the QuasiPeriodic kernel as alpha tends to infinity.
        Parameters:
            weight = weight/amplitude of the kernel
            ell_e and ell_p = aperiodic and periodic lenght scales
            alpha = alpha of the rational quadratic kernel
            P = periodic repetitions of the kernel
    """
    def __init__(self, weight, alpha, ell_e, P, ell_p):
        super(RQP, self).__init__(weight, alpha, ell_e, P, ell_p)
        self.weight = weight
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 5    #number of derivatives in this kernel
        self.params_size = 5    #number of hyperparameters

    def __call__(self, r):
        return self.weight**2 * exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                                    / self.ell_p**2) \
                    /(1+ r**2/ (2*self.alpha*self.ell_e**2))**self.alpha

class dRQP_dweight(RQP):
    """
        Log-derivative in order to the weight
    """
    def __init__(self, weight, alpha, ell_e, P, ell_p):
        super(dRQP_dweight, self).__init__(weight, alpha, ell_e, P, ell_p)
        self.weight = weight
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call(self, r):
        return 2 * self.weight**2 * exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                                    / self.ell_p**2) \
                    /(1+ r**2/ (2*self.alpha*self.ell_e**2))**self.alpha

class dRQP_dalpha(RQP):
    """
        Log-derivative in order to alpha
    """
    def __init__(self, weight, alpha, ell_e, P, ell_p):
        super(dRQP_dweight, self).__init__(weight, alpha, ell_e, P, ell_p)
        self.weight = weight
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call__(self, r):
        return self.alpha * ((r**2 / (2*self.alpha \
                         *self.ell_e**2*(r**2/(2*self.alpha*self.ell_e**2)+1)) \
            -np.log(r**2/(2*self.alpha*self.ell_e**2)+1)) \
            *self.w**2*exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2)) \
            /(1+r**2/(2*self.alpha*self.ell_e**2))**self.alpha

class dRQP_delle(RQP):
    """
        Log-derivative in order to ell_e
    """
    def __init__(self, weight, alpha, ell_e, P, ell_p):
        super(dRQP_dweight, self).__init__(weight, alpha, ell_e, P, ell_p)
        self.weight = weight
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call__(self, r):
        return (r**2*(1+r**2/(2*self.alpha*self.ell_e**2))**(-1-self.alpha) \
                *self.w**2 \
                *exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2))/self.ell_e**2

class dRQP_dP(RQP):
    """
        Log-derivative in order to P
    """
    def __init__(self, weight, alpha, ell_e, P, ell_p):
        super(dRQP_dweight, self).__init__(weight, alpha, ell_e, P, ell_p)
        self.weight = weight
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call__(self, r):
        return (4*pi*r*self.w**2*cosine(pi*np.abs(r)/self.P)*sine(pi*np.abs(r)/self.P) \
                *exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2)) \
            /(self.ell_p**2*(1+r**2/(2*self.alpha*self.ell_e**2))^self.alpha*self.P)

class dRQP_dellp(RQP):
    """
        Log-derivative in order to ell_p
    """
    def __init__(self, weight, alpha, ell_e, P, ell_p):
        super(dRQP_dweight, self).__init__(weight, alpha, ell_e, P, ell_p)
        self.weight = weight
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call(self, r):
        return (4*self.w**2*sine(pi*np.abs(r)/self.P)**2 \
                *exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2)) \
                /(self.ell_p**2*(1+r**2/(2*self.alpha*self.ell_e**2))**self.alpha)


##### Cosine ###################################################################
class Cosine(weightFunction):
    """
        Definition of the cosine kernel.
        Parameters:
            weight = weight/amplitude of the kernel
            P = period
    """
    def __init__(self, weight, P):
        super(Cosine, self).__init__(weight, P)
        self.weight = weight
        self.P = P
        self.type = 'non-stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        return self.weight**2 * cosine(2*pi*np.abs(r) / self.P)

class dCosine_dweight(Cosine):
    """
        Log-derivative in order to the weight
    """
    def __init__(self, weight, P):
        super(dCosine_dweight, self).__init__(weight, P)
        self.weight = weight
        self.P = P

    def __call__(self, r):
        return 2 * self.weight**2 * cosine(2*pi*np.abs(r) / self.P)

class dCosine_dP(Cosine):
    """
        Log-derivative in order to P
    """
    def __init__(self, weight, P):
        super(dCosine_dP, self).__init__(weight, P)
        self.weight = weight
        self.P = P

    def __call__(self, r):
        return self.weight**2 * r * pi * sine(2*pi*np.abs(r) / self.P) / self.P


##### Exponential ##############################################################
class Exponential(weightFunction):
    """
        Definition of the exponential kernel. This kernel arises when 
    setting v=1/2 in the matern family of kernels
        Parameters:
            weight = weight/amplitude of the kernel
            ell = characteristic lenght scale
    """
    def __init__(self, weight, ell):
        super(Exponential, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell
        self.type = 'stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r): 
        return self.weight**2 * exp(- np.abs(r)/self.ell)

class dExponential_dweight(Exponential):
    """
        Log-derivative in order to the weight
    """
    def __init__(self, weight, ell):
        super(dExponential_dweight, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell

    def __call__(self, r):
        return 2 * self.weight**2 * exp(- np.abs(r)/self.ell)

class dExpoential_dell(Exponential):
    """
        Log-derivative in order to ell
    """
    def __init__(self, weight, ell):
        super(dExpoential_dell, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell

    def __call__(self, r):
        return -0.5 * self.weight**2 * r * exp(- np.abs(r)/self.ell) / self.ell


##### Matern 3/2 ###############################################################
class Matern32(weightFunction):
    """
        Definition of the Matern 3/2 kernel. This kernel arise when setting 
    v=3/2 in the matern family of kernels
        Parameters:
            weight = weight/amplitude of the kernel
            theta = amplitude of the kernel
            ell = characteristic lenght scale
    """
    def __init__(self, weight, ell):
        super(Matern32, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell
        self.type = 'stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        return self.weight**2 *(1.0 + np.sqrt(3.0)*np.abs(r)/self.ell) \
                    *np.exp(-np.sqrt(3.0)*np.abs(r) / self.ell)

class dMatern32_dweight(Matern32):
    """
        Log-derivative in order to the weight
    """
    def __init__(self, weight, ell):
        super(dMatern32_dweight, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell

    def __call__(self, r):
        return 2 * self.weight**2 *(1.0 + np.sqrt(3.0)*np.abs(r)/self.ell) \
                    *np.exp(-np.sqrt(3.0)*np.abs(r) / self.ell)

class dMatern32_dell(Matern32):
    """
        Log-derivative in order to ell
    """
    def __init__(self, weight, ell):
        super(dMatern32_dell, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell

    def __call__(self, r):
        return (sqrt(3) * r * (1+ (sqrt(3) * r) / self.ell) \
                *exp(-(sqrt(3)*r) / self.ell) * self.weight**2) / self.ell \
                -(sqrt(3) * r * exp(-(sqrt(3)*r) / self.ell)*self.weight**2)/self.ell


#### Matern 5/2 ################################################################
class Matern52(weightFunction):
    """
        Definition of the Matern 5/2 kernel. This kernel arise when setting 
    v=5/2 in the matern family of kernels
        Parameters:
            weight = weight/amplitude of the kernel
            theta = amplitude of the kernel
            ell = characteristic lenght scale  
    """
    def __init__(self, weight, ell):
        super(Matern52, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell
        self.type = 'stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r):
        return self.weight**2 * (1.0 + ( 3*np.sqrt(5)*self.ell*np.abs(r) \
                                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                                          *exp(-np.sqrt(5.0)*np.abs(r)/self.ell)

class dMatern52_dweight(Matern52):
    """
        Log-derivative in order to the weight
    """
    def __init__(self, weight, ell):
        super(dMatern52_dweight, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell

    def __call__(self, r):
        return 2 * self.weight**2 * (1.0 + ( 3*np.sqrt(5)*self.ell*np.abs(r) \
                                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                                          *exp(-np.sqrt(5.0)*np.abs(r)/self.ell)

class dMatern52_dell(Matern52):
    """
        Log-derivative in order to ell
    """
    def __init__(self, weight, ell):
        super(dMatern52_dell, self).__init__(weight, ell)
        self.weight = weight
        self.ell = ell

    def __call__(self, r):
        return self.ell * ((sqrt(5)*r*(1+(sqrt(5)*r) \
                                 /self.ell+(5*r**2)/(3*self.ell**2)) \
                             *exp(-(sqrt(5)*r)/self.ell)*self.weight**2) \
            /self.ell**2 +(-(sqrt(5)*r)/self.ell**2-(10*r**2) \
                           /(3*self.ell**3)) \
                           *exp(-(sqrt(5)*r)/self.ell)*self.weight**2)


#### Linear ####################################################################
class Linear(weightFunction):
    """
        Definition of the Linear kernel.
            weight = weight/amplitude of the kernel
            c = constant
    """
    def __init__(self, weight, c):
        super(Linear, self).__init__(weight, c)
        self.weight = weight
        self.c = c
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_size = 2    #number of hyperparameters

    def __call__(self, r, t1, t2):
        return self.weight**2 * (t1 - self.c) * (t2 - self.c)

### END