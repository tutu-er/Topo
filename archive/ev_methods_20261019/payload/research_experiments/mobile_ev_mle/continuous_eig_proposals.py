"""Exact sequential conditional-TN box proposal, not a box-truncated MVN.

Math and pre-execution protocol are in continuous_eig_Honly_v3_plan.md.
This module changes only q. It contains no observations, truth or likelihood.
"""
from __future__ import annotations

from dataclasses import dataclass,field
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import erf, log_ndtr, logsumexp, ndtri_exp

from .continuous_eig_sampling import _frozen, _integer, LOG2PI


def clip_last_bit_roundoff(value,lower,upper,*,name,diagnostics=None):
    """Clip only proven floating roundoff; a wider excursion is a failure."""
    value,lower,upper = np.broadcast_arrays(np.asarray(value,float),np.asarray(lower,float),np.asarray(upper,float))
    clipped = np.minimum(np.maximum(value,lower),upper)
    correction = np.abs(clipped-value)
    scale = np.maximum(1.,np.maximum(np.abs(value),np.maximum(np.abs(lower),np.abs(upper))))
    allowed = 128*np.finfo(float).eps*scale
    if not np.isfinite(value).all() or np.any(correction > allowed):
        raise FloatingPointError(name+" leaves its box by more than 128 eps relative roundoff tolerance")
    if diagnostics is not None:
        row = diagnostics.setdefault(name,dict(maximum_absolute_correction=0.,clipped_value_count=0,
                                               tolerance="128*machine_epsilon*max(1,abs(value),abs(bounds))"))
        row["maximum_absolute_correction"] = max(row["maximum_absolute_correction"],float(correction.max(initial=0.)))
        row["clipped_value_count"] += int((correction > 0).sum())
    return clipped


def log_standard_normal_interval(a,b):
    """Stable log[Phi(b)-Phi(a)], including probabilities under exp underflow."""
    a,b = np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a >= b):
        raise ValueError("finite nonempty conditional normal interval required")
    answer = np.empty(a.shape)
    left,right = b <= 0,a >= 0
    middle = ~(left | right)
    if np.any(left):
        la,lb = log_ndtr(a[left]),log_ndtr(b[left])
        answer[left] = lb+np.log(-np.expm1(la-lb))
    if np.any(right):
        lsa,lsb = log_ndtr(-a[right]),log_ndtr(-b[right])
        answer[right] = lsa+np.log(-np.expm1(lsb-lsa))
    if np.any(middle):
        answer[middle] = np.log(.5*(erf(b[middle]/np.sqrt(2))-erf(a[middle]/np.sqrt(2))))
    if not np.isfinite(answer).all():
        raise FloatingPointError("conditional normalizer is nonfinite; no sample can be silently excluded")
    return answer


def sample_standard_normal_interval(a,b,rng,*,diagnostics=None):
    """Inverse-CDF draw with log-CDF or log-survival deep-tail evaluation."""
    a,b = np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    logz = log_standard_normal_interval(a,b)
    # A PRNG float equal to zero has no mathematical relevance but makes log
    # inversion infinite. Use the nearest representable interior uniform.
    u = np.maximum(rng.random(a.shape),np.nextafter(0.,1.))
    logu = np.log(u)
    answer = np.empty(a.shape)
    right = a >= 0
    if np.any(right):
        logs = np.logaddexp(log_ndtr(-b[right]),logu[right]+logz[right])
        answer[right] = -ndtri_exp(logs)
    other = ~right
    if np.any(other):
        logp = np.logaddexp(log_ndtr(a[other]),logu[other]+logz[other])
        # A crossing-zero interval can make logp round to zero at the last
        # PRNG float below one. Use its corresponding log-survival instead.
        upper_tail = logp > -np.log(2.)
        values = np.empty_like(logp)
        values[~upper_tail] = ndtri_exp(logp[~upper_tail])
        if np.any(upper_tail):
            logs = np.logaddexp(log_ndtr(-b[other][upper_tail]),
                               np.log1p(-u[other][upper_tail])+logz[other][upper_tail])
            values[upper_tail] = -ndtri_exp(logs)
        answer[other] = values
    if not np.isfinite(answer).all():
        raise FloatingPointError("conditional tail inverse failed; no redraw or hidden exclusion allowed")
    # Correct last-bit inversion roundoff to a closed interval, never reject.
    return clip_last_bit_roundoff(answer,a,b,name="eta_interval_roundoff",diagnostics=diagnostics)


@dataclass(frozen=True)
class SequentialGaussianBoxProposal:
    component: object
    means_phi: np.ndarray
    covariances_phi: np.ndarray
    uniform_mass: float = .2
    last_sample_diagnostics: dict = field(default_factory=dict,init=False,repr=False,compare=False)

    def __post_init__(self):
        d = len(self.component.lower)
        means,cov = np.asarray(self.means_phi,float),np.asarray(self.covariances_phi,float)
        if means.ndim != 2 or means.shape[1] != d or not len(means):
            raise ValueError("scaled means must have shape (component,beta_dimension)")
        if not np.isfinite(means).all():
            raise ValueError("scaled means must be finite")
        if cov.shape != (len(means),d,d) or not np.isfinite(cov).all():
            raise ValueError("scaled covariance matrices must match all means")
        if not 0 < self.uniform_mass <= 1:
            raise ValueError("uniform coverage probability must be in (0,1]")
        for matrix in cov:
            if not np.allclose(matrix,matrix.T,atol=1e-12,rtol=1e-10):
                raise ValueError("covariance must be symmetric")
            np.linalg.cholesky(matrix)
        object.__setattr__(self,"means_phi",_frozen(means))
        object.__setattr__(self,"covariances_phi",_frozen(cov))

    @classmethod
    def from_gaussian(cls,proposal):
        return cls(proposal.component,proposal.means_phi,proposal.covariances_phi,proposal.uniform_mass)

    def _layout(self,index):
        matrix = self.covariances_phi[index]
        order = np.argsort(-np.diag(matrix),kind="stable")
        permuted = matrix[np.ix_(order,order)]
        L = np.linalg.cholesky(permuted)
        mean = self.means_phi[index,order]
        lower = (self.component.lower/self.component.nominal_beta)[order]
        upper = (self.component.upper/self.component.nominal_beta)[order]
        return order,L,mean,lower,upper

    def sequential_logpdf(self,beta,index):
        """One component's exact beta-coordinate density with all conditional Z."""
        beta = np.asarray(beta,float)
        if beta.shape[-1] != len(self.component.lower):
            raise ValueError("wrong beta dimension")
        original_shape = beta.shape[:-1]
        flat = beta.reshape(-1,beta.shape[-1])
        supported = np.isfinite(self.component.prior_logpdf(flat))
        answer = np.full(len(flat),-np.inf)
        if not np.any(supported):
            return answer.reshape(original_shape)
        order,L,mean,lower,upper = self._layout(index)
        phi = (flat[supported]/self.component.nominal_beta)[:,order]
        eta = solve_triangular(L,(phi-mean).T,lower=True).T
        logdensity = np.zeros(len(phi))
        for j in range(len(mean)):
            past = eta[:,:j]@L[j,:j]
            m = mean[j]+past
            a,b = (lower[j]-m)/L[j,j],(upper[j]-m)/L[j,j]
            logz = log_standard_normal_interval(a,b)
            logdensity += -.5*eta[:,j]**2-.5*LOG2PI-np.log(L[j,j])-logz
        logdensity -= np.log(self.component.nominal_beta).sum()
        if not np.isfinite(logdensity).all():
            raise FloatingPointError("sequential joint density nonfinite inside target box")
        answer[supported] = logdensity
        return answer.reshape(original_shape)

    def logpdf(self,beta):
        uniform = self.component.prior_logpdf(beta)
        if self.uniform_mass == 1:
            return uniform
        seq = logsumexp(np.stack([self.sequential_logpdf(beta,k) for k in range(len(self.means_phi))]),axis=0)-np.log(len(self.means_phi))
        return np.logaddexp(np.log(self.uniform_mass)+uniform,np.log1p(-self.uniform_mass)+seq)

    def _sample_one(self,count,index,rng):
        order,L,mean,lower,upper = self._layout(index)
        eta = np.zeros((count,len(mean)))
        phi = np.zeros_like(eta)
        for j in range(len(mean)):
            m = mean[j]+eta[:,:j]@L[j,:j]
            a,b = (lower[j]-m)/L[j,j],(upper[j]-m)/L[j,j]
            eta[:,j] = sample_standard_normal_interval(a,b,rng,diagnostics=self.last_sample_diagnostics)
            phi[:,j] = clip_last_bit_roundoff(m+L[j,j]*eta[:,j],lower[j],upper[j],
                          name="phi_conditional_roundoff",diagnostics=self.last_sample_diagnostics)
        unpermuted = np.empty_like(phi)
        unpermuted[:,order] = phi
        # Last-bit multiplication may produce a one-ulp box excursion.
        raw_beta = unpermuted*self.component.nominal_beta
        # Express any final multiplication correction in phi units so a tiny
        # weak-edge canonical parameter cannot hide a relatively large clip.
        clipped_phi = clip_last_bit_roundoff(raw_beta/self.component.nominal_beta,
            self.component.lower/self.component.nominal_beta,self.component.upper/self.component.nominal_beta,
            name="beta_multiplication_roundoff_in_phi_units",diagnostics=self.last_sample_diagnostics)
        final_beta = clipped_phi*self.component.nominal_beta
        beta = np.minimum(np.maximum(final_beta,self.component.lower),self.component.upper)
        correction_phi = np.abs(beta-final_beta)/self.component.nominal_beta
        allowed_phi = 128*np.finfo(float).eps*np.maximum(1.,np.abs(clipped_phi))
        if np.any(correction_phi > allowed_phi):
            raise FloatingPointError("final beta multiplication leaves box beyond scaled 128 eps roundoff tolerance")
        record = self.last_sample_diagnostics.setdefault("beta_final_roundoff_in_phi_units",dict(
            maximum_absolute_correction=0.,clipped_value_count=0,tolerance="128 eps * max(1,abs(phi))"))
        record["maximum_absolute_correction"] = max(record["maximum_absolute_correction"],float(correction_phi.max(initial=0.)))
        record["clipped_value_count"] += int((correction_phi > 0).sum())
        return beta

    def sample(self,count,rng):
        count = _integer(count,"theta count")
        c = self.component
        self.last_sample_diagnostics.clear()
        beta = c.lower+(c.upper-c.lower)*rng.random((count,len(c.lower)))
        if self.uniform_mass < 1:
            use_seq = rng.random(count) >= self.uniform_mass
            choice = rng.integers(len(self.means_phi),size=count)
            for k in range(len(self.means_phi)):
                rows = use_seq & (choice == k)
                if np.any(rows):
                    beta[rows] = self._sample_one(int(rows.sum()),k,rng)
        if not np.isfinite(c.prior_logpdf(beta)).all():
            raise RuntimeError("sequential proposal left target box despite exact conditional intervals")
        return beta,self.logpdf(beta)

    def payload(self):
        return dict(density="uniform_plus_sequential_conditional_TN_Gaussian",standard_box_truncated_MVN=False,
            uniform_mass=self.uniform_mass,means_phi=self.means_phi.tolist(),
            covariances_phi=self.covariances_phi.tolist(),phi_scale=self.component.nominal_beta.tolist(),
            coordinate_orders=[self._layout(k)[0].tolist() for k in range(len(self.means_phi))],
            ordering="scaled covariance diagonal descending, stable index ties",
            conditional_normalizers="past-dependent Zj included exactly",
            normalizer_tails="log_ndtr plus logdiff using expm1; crossing zero uses erf",
            inverse_tails="ndtri_exp of logCDF/logSurvival",
            beta_Jacobian_included=True,permutation_abs_Jacobian=1.,
            original_engineering_covariance_equals_actual_truncated_covariance=False,
            all_draws_inside_target_prior_box=True,outside_points_redrawn=False,
            sampling_roundoff_diagnostics=dict(self.last_sample_diagnostics),
            target_prior=self.component.name,engineering_proposal_is_prior_or_CI=False)
