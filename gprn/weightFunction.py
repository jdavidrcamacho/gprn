#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#because it makes my life easier down the line
pi, exp, sine, cosine = np.pi, np.exp, np.sin, np.cos

class weightFunction(object):
    """
        Definition the weight functions (kernels) of our GPRN
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

    def __call__(self, r):
        return self.c * np.ones_like(r)

    def dConstant_dc(self, r):
        """
            Log-derivative in order to c
        """
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

    def __call__(self, r):
        return self.wn**2 * np.diag(np.diag(np.ones_like(r)))

    def dWhiteNoise_dwn(self, r):
        """
            Log-derivative in order to the amplitude
        """
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

    def __call__(self, r):
        return self.weight**2 * exp(-0.5 * r**2 / self.ell**2)

    def dSquaredExponential_dweight(self, r):
        """
            Log-derivative in order to the weight
        """
        return 2 * self.weight**2 * exp(-0.5 * r**2 / self.ell**2)
    
    def dSquaredExponential_dell(self, r):
        """
            Log-derivative in order to the ell
        """
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

    def __call__(self, r):
        return self.weight**2 * exp( -2 * sine(pi*np.abs(r)/self.P)**2 /self.ell**2)

    def dPeriodic_dweight(self, r):
        """
            Log-derivative in order to the weight
        """
        return 2 * self.weight**2 * exp(-2 * sine(pi * np.abs(r) / self.P)**2 \
                                        / self.ell**2)

    def dPeriodic_dell(self, r):
        """
            Log-derivative in order to ell
        """
        return (4* self.weight**2 * sine(pi * np.abs(r) / self.P)**2 \
                *exp(-2 * sine(pi * np.abs(r) / self.P)**2 \
                     / self.ell**2)) / self.ell**2

    def dPeriodic_dP(self, r):
        """
            Log-derivative in order to P
        """
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

    def __call__(self, r):
        return self.weight**2 *exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                                   /self.ell_p**2 - r**2/(2*self.ell_e**2))

    def dQuasiPeriodic_dweight(self, r):
        """
            Log-derivative in order to the weight
        """
        raise Exception("Not implemented yet")

    def dQuasiPeriodic_delle(self, r):
        """
            Log-derivative in order to ell_e
        """
        raise Exception("Not implemented yet")

    def dQuasiPeriodic_dP(self, r):
        """
            Log-derivative in order to P
        """
        raise Exception("Not implemented yet")

    def dQuasiPeriodic_dellp(self, r):
        """
            Log-derivative in order to ell_p
        """
        raise Exception("Not implemented yet")


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

    def __call__(self, r):
        return self.weight**2 / (1+ r**2/ (2*self.alpha*self.ell**2))**self.alpha

    def dRationalQuadratic_dweight(self, r):
        """
            Log-derivative in order to the weight
        """
        raise Exception("Not implemented yet")

    def dRationalQuadratic_dalpha(self, r):
        """
            Log-derivative in order to alpha
        """
        raise Exception("Not implemented yet")

    def dRationalQuadratic_dell(self, r):
        """
            Log-derivative in order to ell
        """
        raise Exception("Not implemented yet")
        

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

    def __call__(self, r):
        return self.weight**2 * exp(- (2*sine(pi*np.abs(r)/self.P)**2) \
                                    / self.ell_p**2) \
                    /(1+ r**2/ (2*self.alpha*self.ell_e**2))**self.alpha

    def dRQP_dweight(self, r):
        """
            Log-derivative in order to the weight
        """
        raise Exception("Not implemented yet")

    def dRQP_dalpha(self, r):
        """
            Log-derivative in order to alpha
        """
        raise Exception("Not implemented yet")

    def dRQP_delle(self, r):
        """
            Log-derivative in order to ell_e
        """
        raise Exception("Not implemented yet")

    def dRQP_dP(self, r):
        """
            Log-derivative in order to P
        """
        raise Exception("Not implemented yet")

    def dRQP_dellp(self, r):
        """
            Log-derivative in order to ell_p
        """
        raise Exception("Not implemented yet")


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

    def __call__(self, r):
        return self.weight**2 * cosine(2*pi*np.abs(r) / self.P)

    def dCosine_dweight(self, r):
        """
            Log-derivative in order to the weight
        """
        raise Exception("Not implemented yet")

    def dCosine_dP(self, r):
        """
            Log-derivative in order to P
        """
        raise Exception("Not implemented yet")
        

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

    def __call__(self, r): 
        return self.weight**2 * exp(- np.abs(r)/self.ell)

    def dExponential_dweight(self, r):
        """
            Log-derivative in order to the weight
        """
        raise Exception("Not implemented yet)

    def dExpoential_dell(self, r):
        """
            Log-derivative in order to ell
        """
        raise Exception("Not implemented yet")


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

    def __call__(self, r):
        return self.weight**2 *(1.0 + np.sqrt(3.0)*np.abs(r)/self.ell) \
                    *np.exp(-np.sqrt(3.0)*np.abs(r) / self.ell)

    def dMatern32_dweight(self, r):
        """
            Log-derivative in order to the weight
        """
        raise Exception("Not implemented yet")


    def dMatern32_dell(self, r):
        """
            Log-derivative in order to ell
        """
        raise Exception("Not implemented yet")


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

    def __call__(self, r):
        return self.weight**2 * (1.0 + ( 3*np.sqrt(5)*self.ell*np.abs(r) \
                                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                                          *exp(-np.sqrt(5.0)*np.abs(r)/self.ell)

    def dMatern52_dweight(self, r):
        """
            Log-derivative in order to the weight
        """
        raise Exception("Not implemented yet")

    def dMatern52_dell(self, r):
        """
            Log-derivative in order to ell
        """
        raise Exception("Not implemented yet")


### END