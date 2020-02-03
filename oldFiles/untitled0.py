#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:38:01 2019

@author: camacho
"""

    def _jittELBO(self, params, node, weight, mean, mu, sigF, sigW):
        """
        Function used to optimeze the jitter terms
        
        Parameters
        ----------
        params : array
            Jitter and means parameters
        node : array
            Node functions of the GPRN
        weight: array
            Weights of the GPRN
        mean: array
            Mean fucntions of the GPRN
        mu: array
            Means of the variational parameters
        sigF: array
            Covariance of the variational parameters (nodes)
        sigW: array
            Covariance of the variational parameters (weights)
        Returns
        -------
        ExpLogLike: float
            Minus expected log-likelihood
        """
        #to separate the means between the nodes and weights
        muF, muW = self._u_to_fhatW(mu.flatten())
        jitter = np.array([])
        for p in range(self.p):
            jitter = np.append(jitter, params[p])
        
        parsUsed = self.p
        for p in range(self.p):
            if mean[p] == None:
                pass
            else:
                howBig = len(mean[p].pars) 
                parsToUse = params[parsUsed:parsUsed+howBig]
                mean[p].pars = np.exp(np.array(parsToUse))
                parsUsed += howBig

        #Expected log-likelihood
        ExpLogLike = self._expectedLogLike(node, weight, mean, jitter,
                                           sigF, muF, sigW, muW)
        return -ExpLogLike