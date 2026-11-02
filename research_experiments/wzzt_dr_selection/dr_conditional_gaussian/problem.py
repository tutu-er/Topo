"""Continuous signed Gaussian DR and frozen-candidate conditional probability.

All residuals use micro p.u.^2.  This is a probability evaluator, not a new
topology identifier.  Its analytic derivative is of the finite conditional
integration approximation at regular points; global interchange of derivative
and expectation is NOT asserted by numerical derivative checks.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import null_space
from conditional_integral import conditional_probability


class ConditionalProblem:
    def __init__(self, packet, samples, seed):
        self.p = packet
        self.shape = tuple(packet['shape'])
        if len(self.shape)!=2 or any(int(v)!=v or v<=0 for v in self.shape):
            raise ValueError('positive resource-by-slot shape required')
        cv=np.broadcast_to(np.asarray(packet['cv'],float),self.shape)
        elasticity=np.broadcast_to(np.asarray(packet['elasticities'],float),self.shape)
        if not np.isfinite(cv).all() or np.any(cv<0):raise ValueError('finite nonnegative CV required')
        if not np.isfinite(elasticity).all() or np.any(elasticity<=0):raise ValueError('positive finite elasticity required')
        if not np.isfinite(packet['dt']) or packet['dt']<=0:raise ValueError('positive finite slot duration required')
        self.m = int(packet['mu_C'].size)
        self.samples = int(samples)
        if self.samples!=samples or self.samples<=0:raise ValueError('positive integer integration count required')
        self.weights = np.r_[np.ones(self.m), -packet['kappa']*np.ones(self.m)]/self.m
        self.b = np.r_[packet['L_C']@packet['direction'], packet['L_F']@packet['direction']]
        if not np.any(self.b!=0):
            raise ValueError('ineffective conditioning direction: no nonzero residual factor')
        # A structurally fixed zero term contributes exactly zero; it has no
        # moving coefficient in this restricted control problem. Nonzero weak
        # terms are never dropped or thresholded.
        for offset,key in [(0,'C'),(self.m,'F')]:
            fixed=(packet['mu_'+key]==0)&np.all(packet['L_'+key]==0,axis=1)
            fixed&=np.tile(np.all(packet['effect_'+key]==0,axis=1),self.shape[1])
            self.weights[offset:offset+self.m][fixed]=0.
        rng = np.random.default_rng(seed)
        eta = rng.normal(size=(samples, packet['complement'].shape[1]))
        zbase = eta@packet['complement'].T
        self.a0 = np.concatenate([
            packet['mu_C'][None]+zbase@packet['L_C'].T,
            packet['mu_F'][None]+zbase@packet['L_F'].T], axis=1)
        zresponse = rng.normal(size=(samples, int(np.prod(self.shape))))
        # Optional fixed factor preserves correlated execution multipliers.
        if packet.get('response_factor') is not None:
            factor=np.asarray(packet['response_factor'],float);d=int(np.prod(self.shape))
            if factor.shape!=(d,d) or not np.isfinite(factor).all():raise ValueError('finite square response factor required')
            if not np.allclose(np.sum(factor*factor,axis=1),1.,atol=1e-12,rtol=1e-12):raise ValueError('response factor rows must have unit variance')
            zresponse = zresponse@factor.T
        self.multiplier = 1.+np.asarray(packet['cv'])*zresponse.reshape((samples,)+self.shape)
        self.last_x = self.last_result = None
        self.evaluations = 0

    def evaluate(self, flat_u):
        u = np.asarray(flat_u, float).reshape(self.shape)
        if self.last_x is not None and np.array_equal(u, self.last_x):
            return self.last_result
        delivered = self.multiplier*u[None]
        ec = np.einsum('nr,srt->stn', self.p['effect_C'], delivered).reshape(self.samples,-1)
        ef = np.einsum('nr,srt->stn', self.p['effect_F'], delivered).reshape(self.samples,-1)
        aa = self.a0+np.concatenate([ec,ef],axis=1)
        values, gradients, invalid, diagnostics = [], [], [], []
        for i in range(self.samples):
            r = conditional_probability(aa[i], self.b, self.weights, self.p['threshold'])
            values.append(r['probability'])
            diagnostics.append(r.get('diagnostics', {}))
            if not r['gradient_valid']:
                invalid.append(i)
                gradients.append(np.zeros(self.shape))
                continue
            ga = r['coeff_grad_a']
            gc = ga[:self.m].reshape(self.shape[1],-1)@self.p['effect_C']
            gf = ga[self.m:].reshape(self.shape[1],-1)@self.p['effect_F']
            gradients.append((gc+gf).T*self.multiplier[i])
        result = dict(probability=float(np.mean(values)),
            gradient=np.mean(gradients,axis=0).ravel() if not invalid else None,
            invalid_indices=invalid, conditional_values=np.asarray(values),
            conditional_mc_se=float(np.std(values,ddof=1)/np.sqrt(self.samples)) if self.samples>1 else 0.,
            gradient_scope='finite fixed integration approximation only; no global smoothness theorem',
            diagnostics=diagnostics)
        self.last_x = u.copy();self.last_result = result;self.evaluations += 1
        return result

    def cost(self, flat_u):
        u=np.asarray(flat_u).reshape(self.shape)
        return float(self.p['dt']*np.sum(u*u/self.p['elasticities']))

    def cost_gradient(self, flat_u):
        u=np.asarray(flat_u).reshape(self.shape)
        return (2*self.p['dt']*u/self.p['elasticities']).ravel()


def integration_direction(mu_c,mu_f,lc,lf,kappa):
    """Fix a history-only nonzero base factor; no future or true topology."""
    g=(np.sign(mu_c)@lc-kappa*np.sign(mu_f)@lf)/mu_c.size
    if np.linalg.norm(g)==0:
        _,singular,vh=np.linalg.svd(np.vstack([lc,lf]),full_matrices=False)
        if singular.size==0 or singular[0]==0:
            raise ValueError('no nonzero historical factor for the conditioning direction')
        g=vh[0]
    if np.linalg.norm(g)==0:
        raise ValueError('no nonzero historical factor for the proposed smoothing direction')
    direction=g/np.linalg.norm(g)
    return direction,null_space(direction[None,:])
