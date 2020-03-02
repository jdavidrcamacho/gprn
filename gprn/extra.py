#log-likelihood
def elboCalc(theta, MU, VAR):
    amp11,ell11,p11, amp22,ell22, m11,j11= theta
    logprior = +amp1.logpdf(amp11) + ell1.logpdf(ell11) + p1.logpdf(p11) \
                +amp2.logpdf(amp22) + ell2.logpdf(ell22) \
                +m1.logpdf(m11)  +j1.logpdf(j11) 
    if np.isinf(logprior):
        return logprior
    
    nodes = [Periodic(amp11, ell11, p11, 1e-5)]
    weight = [SquaredExponential(amp22, ell22, 1e-5)]
    means = [Constant(m11)]
    jitter = [j11]
    
    #elbo = GPRN.EvidenceLowerBound(nodes, weight, means, jitter)
    elbo, MU, VAR = GPRN.optVarParams(nodes, weight, means, jitter, 
                                      iterations = 10, 
                                      updateVarParams = True, 
                                      updateMeanParams = False, 
                                      updateJittParams = False, 
                                      updateHyperParams = False, 
                                      mu=MU, var=VAR)

    return logprior + elbo