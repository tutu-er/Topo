"""Proper predictive EIG for a declared 1D finite Gaussian mixture toy.

Topology priors are an explicit toy assumption. No AC scores, fitted model
probabilities or historical data enter. Gauss-Hermite order comparisons
are convergence diagnostics, not rigorous integration-error certificates.
"""
from functools import lru_cache
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp


@lru_cache(maxsize=12)
def _normal_quadrature(degree):
    nodes,weights=hermgauss(degree)
    nodes=np.sqrt(2.)*nodes;weights=weights/np.sqrt(np.pi)
    nodes.setflags(write=False);weights.setflags(write=False)
    return nodes,weights


def gaussian_mixture_eig_1d(means,prior,sigma,*,degree=80):
    """I(S;Y), Y|S=a ~ N(means[a],sigma²), via standard-normal quadrature."""
    means,prior=np.asarray(means,float),np.asarray(prior,float)
    if (means.ndim!=1 or not len(means) or prior.shape!=means.shape
            or not np.isfinite(means).all() or not np.isfinite(prior).all()
            or np.any(prior<=0) or abs(float(prior.sum())-1.)>1e-12
            or not np.isfinite(sigma) or sigma<=0):
        raise ValueError('finite means, positive declared probability vector and positive sigma required')
    if isinstance(degree,bool) or int(degree)!=degree or degree<2 or degree>256:
        raise ValueError('quadrature degree must be an integer in [2,256]')
    z,weights=_normal_quadrature(int(degree));logprior=np.log(prior)
    conditional=[]
    for a in range(len(means)):
        shift=(means[a]-means)/sigma
        mixture=logsumexp(logprior[None,:]-.5*(z[:,None]+shift[None,:])**2,axis=1)
        conditional.append(float(weights@(-.5*z*z-mixture)))
    value=float(prior@np.asarray(conditional))
    entropy=float(-prior@logprior)
    return dict(value_nats=value,prior_entropy_nats=entropy,conditional_KL_nats=conditional,
        means=means.tolist(),prior=prior.tolist(),sigma=float(sigma),degree=int(degree),
        scope='proper mutual information under declared 1D Gaussian toy; quadrature estimate',
        integration_error_certified=False,AC_fit_calls=0)
