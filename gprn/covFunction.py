"""
Covariance functions to use on the GPRN
"""
import numpy as np
#because it makes my life easier down the line
PI, EXP, SINE, COSINE, SQRT = np.pi, np.exp, np.sin, np.cos, np.sqrt

class covFunction():
    """
    Definition the covariance functions (kernels) of our GPRN, by default and
    because it simplifies my life, all kernels include a white noise term
    """
    def __init__(self, *args):
        """ Puts all kernel arguments in an array pars """
        self.pars = np.array(args, dtype=float)
        self.pars[self.pars > 1e50] = 1e50
        #self.pars[self.pars < 1e-50] = 1e-50
    def __call__(self, r, t1=None, t2=None):
        """
        r = t - t'
        Not sure if this is a good approach since will make our life harder
        when defining certain non-stationary kernels, e.g linear kernel.
        """
        raise NotImplementedError

    def __repr__(self):
        """ Representation of each kernel instance """
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
    """ To allow operations between two kernels """
    def __init__(self, k1, k2):
        super(_operator, self).__init__(k1, k2)
        self.k1 = k1
        self.k2 = k2
        self.kerneltype = 'complex'

    @property
    def pars(self):
        """ Parameters og the two kernels """
        return np.append(self.k1.pars, self.k2.pars)


class Sum(_operator):
    """ To allow the sum of kernels """
    def __call__(self, r):
        return self.k1(r) + self.k2(r)

    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)



class Multiplication(_operator):
    """ To allow the multiplication of kernels """
    def __call__(self, r):
        return self.k1(r) * self.k2(r)

    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)



##### Constant #################################################################
class Constant(covFunction):
    """
    This kernel returns its constant argument c with white noise

    Parameters
    ----------
    c: float
        Constant
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, c, wn):
        super(Constant, self).__init__(c, wn)
        self.tag = 'C'
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


##### White Noise ##############################################################
class WhiteNoise(covFunction):
    """
    Definition of the white noise kernel.

    Parameters
    ----------
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, wn):
        super(WhiteNoise, self).__init__(wn)
        self.tag = 'WN'
        self.wn = wn
        self.type = 'stationary'
        self.derivatives = 1    #number of derivatives in this kernel
        self.params_size = 1    #number of hyperparameters

    def __call__(self, r):
        if r[0, :].shape == r[:, 0].shape:
            return self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        return self.wn**2 * np.ones_like(r)


##### Squared exponential ######################################################
class SquaredExponential(covFunction):
    """
    Squared Exponential kernel, also known as radial basis function or RBF
    kernel in other works.

    Parameterstoo-many-arguments
    ----------
    theta: float
        Amplitude
    ell: float
        Length-scale
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, theta, ell, wn):
        super(SquaredExponential, self).__init__(theta, ell, wn)
        self.tag = 'SE'
        self.theta = theta
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and anisotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * EXP(-0.5 * r**2 / self.ell**2) \
                    + self.wn**2 *np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * EXP(-0.5 * r**2 / self.ell**2)


##### Periodic #################################################################
class Periodic(covFunction):
    """
    Definition of the periodic kernel.

    Parameters
    ----------
    theta: float
        Amplitude
    ell: float
        Lenght scale
    P: float
        Periodtoo-many-arguments
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, theta, ell, P, wn):
        super(Periodic, self).__init__(theta, ell, P, wn)
        self.tag = 'P'
        self.theta = theta
        self.ell = ell
        self.P = P
        self.wn = wn
        self.type = 'non-stationary and isotropic'
        self.derivatives = 4    #number of derivatives in this kernel
        self.params_size = 4    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * EXP(-2*SINE(PI*np.abs(r)/self.P)**2/self.ell**2) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * EXP(-2*SINE(PI*np.abs(r)/self.P)**2/self.ell**2)


##### Quasi Periodic ###########################################################
class QuasiPeriodic(covFunction):
    """
    This kernel is the product between the exponential sine squared kernel
    and the squared exponential kernel, commonly known as the quasi-periodic
    kernel

    Parameters
    ----------
    theta: float
        Amplitude
    ell_e: float
        Evolutionary time scale
    P: float
        Kernel periodicity
    ell_p: float
        Length scale of the periodic component
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, theta, ell_e, P, ell_p, wn):
        super(QuasiPeriodic, self).__init__(theta, ell_e, P, ell_p, wn)
        self.tag = 'QP'
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
            return self.theta**2 * EXP(-2*SINE(PI*r/self.P)**2 \
                       /self.ell_p**2 - r**2/(2*self.ell_e**2)) \
                       + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * EXP(-2*SINE(PI*np.abs(r)/self.P)**2 \
                       /self.ell_p**2 - r**2/(2*self.ell_e**2))


##### Rational Quadratic #######################################################
class RationalQuadratic(covFunction):
    """
    Definition of the rational quadratic kernel

    Parameters
    ----------

    theta: float
        Amplitude
    alpha: float
        Weight of large and small scale variations
    ell: float
        Characteristic lenght scale to define the kernel "smoothness"
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, theta, alpha, ell, wn):
        super(RationalQuadratic, self).__init__(theta, alpha, ell, wn)
        self.tag = 'RQ'
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


##### RQP kernel ###############################################################
class RQP(covFunction):
    """
    Definition of the product between the exponential sine squared kernel
    and the rational quadratic kernel that we called RQP kernel.
    If I am thinking this correctly then this kernel should tend to the
    QuasiPeriodic kernel as alpha increases, although I am not sure if we can
    say that it tends to the QuasiPeriodic kernel as alpha tends to infinity.

    Parameters
    ----------
    theta: float
        Amplitude
    alpha: float
        Alpha of the rational quadratic kernel
    ell_e, ell_p: float
        Aperiodic and periodic lenght scales
    P: float
        Periodic repetitions of the kernel
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, theta, alpha, ell_e, P, ell_p, wn):
        super(RQP, self).__init__(theta, alpha, ell_e, P, ell_p, wn)
        self.tag = 'RQP'
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
            return self.theta**2 * EXP(-2*SINE(PI*np.abs(r)/self.P)**2/self.ell_p**2) \
                        *(1+r**2/(2*self.alpha*self.ell_e**2))**(-self.alpha) \
                        + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 *EXP(-2*SINE(PI*np.abs(r)/self.P)**2/self.ell_p**2) \
                        *(1+r**2/(2*self.alpha*self.ell_e**2))**(-self.alpha)


##### CoSINE ###################################################################
class CoSINE(covFunction):
    """
    Definition of the cosine kernel

    Parameters
    ----------
    theta: float
        Amplitude
    P: float
        Period
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, theta, P, wn):
        super(CoSINE, self).__init__(theta, P, wn)
        self.tag = 'COS'
        self.theta = theta
        self.P = P
        self.wn = wn
        self.type = 'non-stationary and isotroPIc'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * COSINE(2*PI*np.abs(r) / self.P) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 *COSINE(2*PI*np.abs(r) / self.P)


##### Laplacian ##############################################################
class Laplacian(covFunction):
    """
    Definition of the Laplacian kernel

    Parameters
    ----------
    theta: float
        Amplitude
    ell: float
        Characteristic lenght scale
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, theta, ell, wn):
        super(Laplacian, self).__init__(theta, ell, wn)
        self.tag = 'LAP'
        self.theta = theta
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and isotroPIc'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * EXP(-np.abs(r)/self.ell) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * EXP(-np.abs(r)/self.ell)


##### Exponential ##############################################################
class Exponential(covFunction):
    """
    Definition of the exponential kernel

    Parameters
    ----------
    theta: float
        Amplitude
    ell: float
        Characteristic lenght scale
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, theta, ell, wn):
        super(Exponential, self).__init__(theta, ell, wn)
        self.tag = 'EXP'
        self.theta = theta
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and isotroPIc'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * EXP(-np.abs(r)/(2*self.ell**2)) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * EXP(-np.abs(r)/self.ell)


##### Matern 3/2 ###############################################################
class Matern32(covFunction):
    """
    Definition of the Matern 3/2 kernel. This kernel arise when setting
    v=3/2 in the matern family of kernels

    Parameters
    ----------
    theta: float
        Amplitude
    ell: float
        Characteristic lenght scale
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, theta, ell, wn):
        super(Matern32, self).__init__(theta, ell, wn)
        self.tag = 'M32'
        self.theta = theta
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and isotroPIc'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * (1.0+SQRT(3.0)*np.abs(r)/self.ell) \
                        *np.exp(-SQRT(3.0)*np.abs(r) / self.ell) \
                        + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * (1.0+SQRT(3.0)*np.abs(r)/self.ell) \
                        *np.exp(-SQRT(3.0)*np.abs(r) / self.ell)


#### Matern 5/2 ################################################################
class Matern52(covFunction):
    """
    Definition of the Matern 5/2 kernel. This kernel arise when setting v=5/2
    in the matern family of kernels

    Parameters
    ----------
    theta: float
        Amplitude
    ell: float
        Characteristic lenght scale
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, theta, ell, wn):
        super(Matern52, self).__init__(theta, ell, wn)
        self.tag = 'M52'
        self.theta = theta
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and isotroPIc'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_size = 3    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2*(1.0+(3*SQRT(5)*self.ell*np.abs(r) \
                           + 5*np.abs(r)**2)/(3*self.ell**2)) \
                           * EXP(-SQRT(5.0)*np.abs(r)/self.ell) \
                           + self.wn**2* np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2*(1.0+(3*SQRT(5)*self.ell*np.abs(r) \
                           +5*np.abs(r)**2)/(3* self.ell**2)) \
                           *EXP(-SQRT(5.0)*np.abs(r)/self.ell)


#### Linear ####################################################################
class Linear(covFunction):
    """
    Definition of the Linear kernel

    Parameters
    ----------
    theta: float
        Amplitude (should we even have an amplitude???)
    c: float
        Constant
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, theta, c, wn):
        super(Linear, self).__init__(theta, c, wn)
        self.tag = 'LIN'
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


##### Gamma-exponential ########################################################
class GammaEXP(covFunction):
    """
    Definition of the gamma-exponential kernel

    Parameters
    ----------
    theta: float
        Amplitude
    gamma: float
        Shape parameter ( 0 < gamma <= 2)
    ell: float
        Lenght scale
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, theta, gamma, ell, wn):
        super(GammaEXP, self).__init__(theta, gamma, ell, wn)
        self.tag = 'GEXP'
        self.theta = theta
        self.gamma = gamma
        self.ell = ell
        self.wn = wn
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 4    #number of derivatives in this kernel
        self.params_size = 4    #number of hyperparameters

    def __call__(self, r):
        try:
            return self.theta**2 * EXP(-(np.abs(r)/self.ell)**self.gamma) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.theta**2 * EXP(-(np.abs(r)/self.ell) ** self.gamma)


##### Polinomial ###############################################################
class Polynomial(covFunction):
    """
    Definition of the polinomial kernel

    Parameters
    ----------
    theta: float
        Amplitude ???
    a: float
        Real value > 0
    b: foat
        Real value >= 0
    c: int
        Integer value
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, theta, a, b, c, wn):
        super(Polynomial, self).__init__(theta, a, b, c, wn)
        self.tag = 'POLY'
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


##### Piecewise ################################################################
class Piecewise(covFunction):
    """
    WARNING: EXPERIMENTAL KERNEL
    
    Parameters
    ----------
    """
    def __init__(self, eta3):
        super(Piecewise, self).__init__(eta3)
        self.eta3 = eta3
        self.type = 'unknown'
        self.derivatives = 0    #number of derivatives in this kernel
        self.params_number = 0    #number of hyperparameters
    def __call__(self, r):
        r = r/(0.5*self.eta3)
        piecewise = (3*np.abs(r) +1) * (1 - np.abs(r))**3
        piecewise = np.where(np.abs(r)>1, 0, piecewise)
        return piecewise


### END
