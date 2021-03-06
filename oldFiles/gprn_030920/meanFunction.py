#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from functools import wraps

__all__ = ['Constant', 'Linear', 'Parabola', 'Cubic', 'Keplerian']

def array_input(f):
    """ Decorator to provide the __call__ methods with an array """
    @wraps(f)
    def wrapped(self, t):
        t = np.atleast_1d(t)
        r = f(self, t)
        return r
    return wrapped


class MeanModel(object):
    _parsize = 0
    def __init__(self, *pars):
        self.pars = list(pars)
        #np.array(pars, dtype=float)
        
    def __repr__(self):
        """ Representation of each instance """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))
        
    @classmethod
    def initialize(cls):
        """ Initialize instance, setting all parameters to 0. """
        return cls( *([0.]*cls._parsize) )
    
    def __add__(self, b):
        return Sum(self, b)
    def __radd__(self, b):
        return self.__add__(b)
    
    
class Sum(MeanModel):
    """ Sum of two mean functions """
    def __init__(self, m1, m2):
        self.m1, self.m2 = m1, m2
        
    @property
    def _parsize(self):
        return self.m1._parsize + self.m2._parsize
    
    @property
    def pars(self):
        return self.m1.pars + self.m2.pars
    
    def initialize(self):
        return
    
    def __repr__(self):
        return "{0} + {1}".format(self.m1, self.m2)
    
    @array_input
    def __call__(self, t):
        return self.m1(t) + self.m2(t)
    
    
class Constant(MeanModel):
    """  A constant offset mean function """
    _parsize = 1
    def __init__(self, c):
        super(Constant, self).__init__(c)
        
    @array_input
    def __call__(self, t):
        return np.full(t.shape, self.pars[0])
    
    
class Linear(MeanModel):
    """ 
    A linear mean function
    m(t) = slope * t + intercept 
    """
    _parsize = 2
    def __init__(self, slope, intercept):
        super(Linear, self).__init__(slope, intercept)
        
    @array_input
    def __call__(self, t):
        tmean = t.mean()
        return self.pars[0] * (t-tmean) + self.pars[1]
    
    
class Parabola(MeanModel):
    """ 
    A 2nd degree polynomial mean function
    m(t) = quad * t**2 + slope * t + intercept 
    """
    _parsize = 3
    def __init__(self, quad, slope, intercept):
        super(Parabola, self).__init__(quad, slope, intercept)
        
    @array_input
    def __call__(self, t):
        return np.polyval(self.pars, t)
    
    
class Cubic(MeanModel):
    """ 
    A 3rd degree polynomial mean function
    m(t) = cub * t**3 + quad * t**2 + slope * t + intercept 
    """
    _parsize = 4
    def __init__(self, cub, quad, slope, intercept):
        super(Cubic, self).__init__(cub, quad, slope, intercept)
        
    @array_input
    def __call__(self, t):
        return np.polyval(self.pars, t)
    
    
class Sine(MeanModel):
    """ 
        A sinusoidal mean function
        m(t) = amplitude**2 * sine( (2*pi*t/P) + phase) + displacement
    """
    _parsize = 3
    def __init__(self, amp, P, phi, D):
        super(Sine, self).__init__(amp, P, phi, D)
        
    @array_input
    def __call__(self, t):
        return self.pars[0] * np.sin((2*np.pi*t/self.pars[1]) + self.pars[2]) \
                + self.pars[3]
                
                
class oldKeplerian(MeanModel):
    """
    Keplerian function with T0
    tan[phi(t) / 2 ] = sqrt(1+e / 1-e) * tan[E(t) / 2] = true anomaly
    E(t) - e*sin[E(t)] = M(t) = eccentric anomaly
    M(t) = (2*pi*t/tau) + M0 = Mean anomaly
    P  = period in days
    e = eccentricity
    K = RV amplitude in m/s 
    w = longitude of the periastron
    T0 = time of periastron passage
    
    Then we have RV = K[cos(w+v) + e*cos(w)] + sis_vel
    """
    _parsize = 5
    def __init__(self, P, K, e, w, T0):
        super(oldKeplerian, self).__init__(P, K, e, w, T0)
        
    @array_input
    def __call__(self, t):
        P, K, e, w, T0 = self.pars
        #mean anomaly
        Mean_anom = 2*np.pi*(t-T0)/P
        #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
        E0 = Mean_anom + e*np.sin(Mean_anom) + 0.5*(e**2)*np.sin(2*Mean_anom)
        #mean anomaly -> M0=E0 - e*sin(E0)
        M0 = E0 - e*np.sin(E0)
        niter=0
        while niter < 100:
            aux = Mean_anom - M0
            E1 = E0 + aux/(1 - e*np.cos(E0))
            M1 = E0 - e*np.sin(E0)
            niter += 1
            E0 = E1
            M0 = M1
        nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E0/2))
        RV = K*(e*np.cos(w)+np.cos(w+nu))
        return RV


class Keplerian(MeanModel):
    """
    Keplerian function with phi
    tan[phi(t) / 2 ] = sqrt(1+e / 1-e) * tan[E(t) / 2] = true anomaly
    E(t) - e*sin[E(t)] = M(t) = eccentric anomaly
    M(t) = (2*pi*t/tau) + M0 = Mean anomaly
    P  = period in days
    e = eccentricity
    K = RV amplitude in m/s 
    w = longitude of the periastron
    phi = orbital phase
    sys_vel = offset
    
    Then we have RV = K[cos(w+v) + e*cos(w)] + sys_vel
    """
    _parsize = 6
    def __init__(self, P, K, e, w, phi, sys_vel):
        super(Keplerian, self).__init__(P, K, e, w, phi, sys_vel)
        
    @array_input
    def __call__(self, t):
        P, K, e, w, phi = self.pars
        #mean anomaly
        T0 = t[0] - (P*phi)/(2.*np.pi)
        Mean_anom = 2*np.pi*(t-T0)/P
        #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
        E0 = Mean_anom + e*np.sin(Mean_anom) + 0.5*(e**2)*np.sin(2*Mean_anom)
        #mean anomaly -> M0=E0 - e*sin(E0)
        M0 = E0 - e*np.sin(E0)
        niter=0
        while niter < 100:
            aux = Mean_anom - M0
            E1 = E0 + aux/(1 - e*np.cos(E0))
            M1 = E0 - e*np.sin(E0)
            niter += 1
            E0 = E1
            M0 = M1
        nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E0/2))
        RV = K*(e*np.cos(w)+np.cos(w+nu)) + sys_vel
        return RV
    
    
class corot7Keplerian(MeanModel):
    """
    Keplerian function with phi for two planets (for the CoRoT-7 case)
    tan[phi(t) / 2 ] = sqrt(1+e / 1-e) * tan[E(t) / 2] = true anomaly
    E(t) - e*sin[E(t)] = M(t) = eccentric anomaly
    M(t) = (2*pi*t/tau) + M0 = Mean anomaly
    P  = period in days
    e = eccentricity
    K = RV amplitude in m/s 
    w = longitude of the periastron
    phi = orbital phase
    sis_vel = offset
    
    RV = K[cos(w+v) + e*cos(w)] + K[cos(w+v) + e*cos(w)] + sys_vel
    """
    _parsize = 11
    def __init__(self, P1, K1, e1, w1, phi1, P2, K2, e2, w2, phi2, sys_vel):
        super(corot7Keplerian, self).__init__(P1, K1, e1, w1, phi1,
                                         P2, K2, e2, w2, phi2, sys_vel)
        
    @array_input
    def __call__(self, t):
        P1, K1, e1, w1, phi1, P2, K2, e2, w2, phi2, sis_vel = self.pars
        #mean anomaly for the 1st planet
        T0 = t[0] - (P1*phi1)/(2.*np.pi)
        Mean_anom = 2*np.pi*(t-T0)/P1
        #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
        E0 = Mean_anom + e1*np.sin(Mean_anom) + 0.5*(e1**2)*np.sin(2*Mean_anom)
        #mean anomaly -> M0=E0 - e*sin(E0)
        M0 = E0 - e1*np.sin(E0)
        niter=0
        while niter < 500:
            aux = Mean_anom - M0
            E1 = E0 + aux/(1 - e1*np.cos(E0))
            M1 = E0 - e1*np.sin(E0)
            niter += 1
            E0 = E1
            M0 = M1
        nu1 = 2*np.arctan(np.sqrt((1+e1)/(1-e1))*np.tan(E0/2))
        #mean anomaly for the 2nd planet
        T0 = t[0] - (P2*phi2)/(2.*np.pi)
        Mean_anom = 2*np.pi*(t-T0)/P2
        #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
        E0 = Mean_anom + e2*np.sin(Mean_anom) + 0.5*(e2**2)*np.sin(2*Mean_anom)
        #mean anomaly -> M0=E0 - e*sin(E0)
        M0 = E0 - e2*np.sin(E0)
        niter=0
        while niter < 500:
            aux = Mean_anom - M0
            E1 = E0 + aux/(1 - e2*np.cos(E0))
            M1 = E0 - e2*np.sin(E0)
            niter += 1
            E0 = E1
            M0 = M1
        nu2 = 2*np.arctan(np.sqrt((1+e2)/(1-e2))*np.tan(E0/2))
        RV = K1*(e1*np.cos(w1)+np.cos(w1+nu1)) \
                + K2*(e2*np.cos(w2)+np.cos(w2+nu2)) + sys_vel
        return RV