"""Normalized white Gaussian guidance for the exact nonlinear latent target.

Finite differences define q only. Meter density p0/q is restored; no linear
voltage model enters the target. Implementation/toy authorization only until
parent freezes an actual-H experiment; this module contains no MC driver.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
from scipy.special import logsumexp

from .continuous_eig_sampling import LatentEstimate,LOG2PI,log_ess


@dataclass(frozen=True)
class WhiteGaussianGuidance:
    mean: np.ndarray
    precision_cholesky: np.ndarray
    logdet_precision: np.ndarray

    @classmethod
    def build(cls,H,residual):
        H,residual = np.asarray(H,float),np.asarray(residual,float)
        if H.ndim < 2 or residual.shape != H.shape[:-1] or H.shape[-1] < 1:
            raise ValueError("H[...,voltage,white] and residual[...,voltage] must match")
        if not np.isfinite(H).all() or not np.isfinite(residual).all():
            raise ValueError("guidance H/residual must be finite")
        d = H.shape[-1]
        with np.errstate(over="raise",invalid="raise"):
            K = np.eye(d)+np.einsum("...ki,...kj->...ij",H,H)
            L = np.linalg.cholesky(K)
            rhs = np.einsum("...ki,...k->...i",H,residual)
            # NumPy 2 treats batched rhs[...,d] as matrices unless a final
            # singleton right-hand-side axis is explicit.
            mean = np.linalg.solve(K,rhs[...,None])[...,0]
            logdet = 2*np.log(np.diagonal(L,axis1=-2,axis2=-1)).sum(axis=-1)
        if not np.isfinite(mean).all() or not np.isfinite(logdet).all():
            raise FloatingPointError("unrepresentable Gaussian guidance; no covariance clipping or model removal")
        return cls(mean,L,logdet)

    @property
    def dimension(self):
        return self.mean.shape[-1]

    def sample_guided(self,z):
        z = np.asarray(z,float)
        z = np.broadcast_to(z,self.mean.shape[:-1]+z.shape[-2:])
        if z.shape[-1] != self.dimension or not np.isfinite(z).all():
            raise ValueError("white normals must be finite and match dimension")
        displacement = np.linalg.solve(np.swapaxes(self.precision_cholesky,-1,-2),np.swapaxes(z,-1,-2))
        answer = self.mean[...,None,:]+np.swapaxes(displacement,-1,-2)
        if not np.isfinite(answer).all():
            raise FloatingPointError("guided sample became nonfinite")
        return answer

    def logpdf_guided(self,w):
        delta = np.asarray(w,float)-self.mean[...,None,:]
        white = np.einsum("...li,...ij->...lj",delta,self.precision_cholesky)
        with np.errstate(over="raise",invalid="raise"):
            value = .5*self.logdet_precision[...,None]-.5*self.dimension*LOG2PI-.5*np.sum(white**2,axis=-1)
        if not np.isfinite(value).all():
            raise FloatingPointError("guided density cannot be represented")
        return value

    def logpdf_baseline(self,w):
        with np.errstate(over="raise",invalid="raise"):
            value = -.5*self.dimension*LOG2PI-.5*np.sum(np.asarray(w,float)**2,axis=-1)
        if not np.isfinite(value).all():
            raise FloatingPointError("baseline density cannot be represented")
        return value

    def logpdf_mixture(self,w,baseline_mass=.2):
        if not 0 < baseline_mass <= 1:
            raise ValueError("baseline coverage mass must be in (0,1]")
        p0 = self.logpdf_baseline(w)
        if baseline_mass == 1:
            return p0
        return np.logaddexp(np.log(baseline_mass)+p0,np.log1p(-baseline_mass)+self.logpdf_guided(w))

    def sample_mixture(self,z,uniforms,baseline_mass=.2):
        if not 0 < baseline_mass <= 1:
            raise ValueError("baseline coverage mass must be in (0,1]")
        guided = self.sample_guided(z)
        baseline = np.broadcast_to(z,guided.shape)
        uniforms = np.broadcast_to(np.asarray(uniforms,float),guided.shape[:-1])
        if not np.isfinite(uniforms).all() or np.any(uniforms < 0) or np.any(uniforms >= 1):
            raise ValueError("mixture uniforms must be in [0,1)")
        w = np.where((uniforms < baseline_mass)[...,None],baseline,guided)
        log_ratio = self.logpdf_baseline(w)-self.logpdf_mixture(w,baseline_mass)
        if np.any(log_ratio > -np.log(baseline_mass)+1e-10):
            raise FloatingPointError("meter/proposal ratio exceeds the full mixture coverage bound")
        return w,log_ratio


@dataclass(frozen=True)
class GuidedLatentEstimate(LatentEstimate):
    guidance_diagnostics: dict = field(default_factory=dict)
    prefix_estimates: dict = field(default_factory=dict,repr=False)

    def diagnostics(self):
        value = super().diagnostics()
        value["guided_latent_proposal"] = self.guidance_diagnostics
        return value


def _response(component,beta,observation,meter,x,engine,action):
    m = component.candidate.m
    inv = engine.inverse_ac(component.candidate,beta[:,None,None,:m],beta[:,None,None,m:],
        x[...,:3],x[...,3:6],x[...,6],x[...,7],root_v=observation.root_v[None,:,None])
    pbase,qbase = x[...,:3],x[...,3:6]
    if action is not None:
        pbase=pbase-action.p[None,:,None,:];qbase=qbase-action.q[None,:,None,:]
    logbg = engine.background_logpdf(pbase,qbase)
    logsrc = engine.source_logpdf(inv["source_p"],inv["source_q"])
    rectangle = (x[...,6] >= 0)&(x[...,6] <= 1)&(x[...,7] >= 0)&(x[...,7] <= .5)
    numerical = np.asarray(inv["valid"],bool)&np.asarray(inv.get("high_branch",inv["valid"]),bool)
    positive = (np.isfinite(inv["v"]).all(axis=-1)&(np.asarray(inv["v"]) > 0).all(axis=-1)&
                np.isfinite(inv["source_p"])&np.isfinite(inv["source_q"])&
                np.isfinite(inv["jacobian_abs"])&(np.asarray(inv["jacobian_abs"]) > 0))
    raw_usable = numerical&positive
    certified = np.asarray(inv.get("certified_domain",inv["valid"]),bool)
    valid = raw_usable&certified&rectangle
    eligible = np.isfinite(logbg)&rectangle
    if np.any(eligible&~valid):
        raise RuntimeError("inverse failed inside proven background/master/parameter domain; cannot assign zero density or guidance fallback")
    return inv,logbg,logsrc,rectangle,eligible,valid,raw_usable


def guided_latent_likelihood(component,beta,observation,meter,latent_normals,mixture_uniforms,
                             engine,*,action=None,baseline_mass=.2,fd_white_step=.02,prefix_samples=()):
    """Exact nonlinear likelihood expectation using a beta/time-specific q.

    normals (T,L,8) and uniforms (T,L) must be independent across times; they
    may be paired across theta/actions. No actual future or truth is accepted.
    """
    beta = np.asarray(beta,float)
    z,u = np.asarray(latent_normals,float),np.asarray(mixture_uniforms,float)
    if beta.ndim != 2 or beta.shape[1] != 2*component.candidate.m:
        raise ValueError("beta shape must be (theta,2*edge_count)")
    if z.ndim != 3 or z.shape[0] != observation.t or z.shape[2] != 8 or z.shape[1] < 1 or u.shape != z.shape[:2]:
        raise ValueError("independent-time normals (T,L,8) and uniforms (T,L) required")
    if not np.isfinite(z).all() or not np.isfinite(u).all() or np.any(u < 0) or np.any(u >= 1):
        raise ValueError("finite proposal random variables required")
    if np.any(observation.root_v != 1.):
        raise ValueError("nominal proof requires known root=1")
    if action is not None:
        if action.p.shape != observation.p.shape:
            raise ValueError("action must match observation horizon")
        action.validate_nominal_domain()
    if not np.isfinite(fd_white_step) or fd_white_step <= 0:
        raise ValueError("positive finite FD white step required")
    prefixes=tuple(sorted(set(int(n) for n in prefix_samples)))
    if any(n<1 or n>=z.shape[1] for n in prefixes):
        raise ValueError("requested integration prefixes must be positive and smaller than full L")
    supported = np.isfinite(component.prior_logpdf(beta))
    if not np.all(supported):
        n,t = len(beta),observation.t
        times=np.full((n,t),-np.inf);ess=np.zeros((n,t));count=np.zeros((n,t),int)
        sub=None
        if np.any(supported):
            sub=guided_latent_likelihood(component,beta[supported],observation,meter,z,u,engine,
                action=action,baseline_mass=baseline_mass,fd_white_step=fd_white_step,prefix_samples=prefixes)
            times[supported]=sub.log_by_time;ess[supported]=sub.time_ess;count[supported]=sub.time_positive_count
        prefix_results={}
        for n_prefix in prefixes:
            pt=np.full((n,t),-np.inf);pe=np.zeros((n,t));pc=np.zeros((n,t),int)
            ps=None if sub is None else sub.prefix_estimates[n_prefix]
            if ps is not None:
                pt[supported]=ps.log_by_time;pe[supported]=ps.time_ess;pc[supported]=ps.time_positive_count
            prefix_results[n_prefix]=GuidedLatentEstimate(pt.sum(axis=1),pt,pe,pc,
                0 if ps is None else ps.invalid_inverse_count,n_prefix,int((~supported).sum()),
                0 if ps is None else ps.background_outside_point_count,
                0 if ps is None else ps.master_rectangle_outside_point_count,
                0 if ps is None else ps.source_prior_outside_supported_point_count,
                {} if ps is None else ps.guidance_diagnostics)
        return GuidedLatentEstimate(times.sum(axis=1),times,ess,count,
            0 if sub is None else sub.invalid_inverse_count,z.shape[1],int((~supported).sum()),
            0 if sub is None else sub.background_outside_point_count,
            0 if sub is None else sub.master_rectangle_outside_point_count,
            0 if sub is None else sub.source_prior_outside_supported_point_count,
            {} if sub is None else sub.guidance_diagnostics,prefix_results)
    B,T,L = len(beta),observation.t,z.shape[1]
    m0 = np.column_stack((observation.p-meter.bias_p,observation.q-meter.bias_q,
                          observation.master_p,observation.master_q))
    scale = np.r_[meter.p_sigma,meter.q_sigma,meter.master_p_sigma,meter.master_q_sigma]
    anchor = np.broadcast_to(m0[None,:,None,:],(B,T,1,8))
    response = _response(component,beta,observation,meter,anchor,engine,action)
    anchor_v=response[0]["v"][...,0,:]
    anchor_ok=response[-1][...,0]
    perturb=np.concatenate((np.eye(8),-np.eye(8)),axis=0)
    xfd=np.broadcast_to(m0[None,:,None,:]+fd_white_step*scale*perturb[None,None,:,:],(B,T,16,8))
    fdresponse = _response(component,beta,observation,meter,xfd,engine,action)
    fd_ok=fdresponse[-1].all(axis=-1)
    vfd=fdresponse[0]["v"]
    A=np.swapaxes((vfd[...,:8,:]-vfd[...,8:,:])/(2*fd_white_step),-1,-2)
    guidance_ok=anchor_ok&fd_ok&np.isfinite(A).all(axis=(-1,-2))
    H=np.where(guidance_ok[...,None,None],A/meter.v_sigma[None,None,:,None],0.)
    residual=(observation.v-meter.bias_v)[None]-anchor_v
    r=np.where(guidance_ok[...,None],residual/meter.v_sigma,0.)
    guidance=WhiteGaussianGuidance.build(H,r)
    w,logratio=guidance.sample_mixture(z,u,baseline_mass)
    x=m0[None,:,None,:]+scale*w
    inv,logbg,logsrc,rectangle,eligible,valid,_ = _response(component,beta,observation,meter,x,engine,action)
    supported_target=valid&np.isfinite(logbg)&np.isfinite(logsrc)
    kernel=np.full(valid.shape,-np.inf)
    if np.any(supported_target):
        measured_v=np.broadcast_to((observation.v-meter.bias_v)[None,:,None,:],inv["v"].shape)
        with np.errstate(over="raise",invalid="raise",divide="raise"):
            residual_v=(measured_v[supported_target]-inv["v"][supported_target])/meter.v_sigma
            logv=np.sum(-.5*residual_v**2-np.log(meter.v_sigma)-.5*LOG2PI,axis=-1)
            logjac=np.log(np.asarray(inv["jacobian_abs"])[supported_target])
            target=(logratio[supported_target]+logv+logbg[supported_target]+
                    logsrc[supported_target]+logjac)
        if not (np.isfinite(logratio[supported_target]).all() and np.isfinite(logv).all() and
                np.isfinite(logjac).all() and np.isfinite(target).all()):
            raise FloatingPointError("nonfinite supported Gaussian target; numerical_failure, not probability zero")
        kernel[supported_target]=target
    diagnostics=dict(name="white_meter_plus_voltage_guided_Gaussian_mixture",baseline_mass=baseline_mass,
        meter_ratio_restored=True,full_mixture_density_used=True,linear_V_used_in_target=False,
        finite_difference_only_shapes_q=True,FD_white_step=fd_white_step,
        guidance_available_beta_time_count=int(guidance_ok.sum()),
        guidance_fallback_outside_target_support_count=int((~guidance_ok).sum()),
        fallback_is_normalized_baseline_not_model_deletion=True,
        minimum_precision_cholesky_diagonal=float(np.diagonal(guidance.precision_cholesky,axis1=-2,axis2=-1).min()),
        maximum_absolute_white_mean=float(np.abs(guidance.mean).max()),
        maximum_log_meter_ratio=float(logratio.max()),mixture_coverage_log_bound=float(-np.log(baseline_mass)),
        time_clouds_must_be_independent=True,proposal_beta_dependent=True,
        guidance_clipping_or_jitter_applied=False)
    def make_estimate(n):
        k=kernel[...,:n];v=valid[...,:n];bg=logbg[...,:n];src=logsrc[...,:n]
        rec=rectangle[...,:n];el=eligible[...,:n]
        times=logsumexp(k,axis=-1)-np.log(n)
        diag=dict(diagnostics);diag["maximum_log_meter_ratio"]=float(logratio[...,:n].max())
        diag["integration_prefix_of_same_per_time_cloud"]=n<L
        return GuidedLatentEstimate(times.sum(axis=1),times,log_ess(k),np.isfinite(k).sum(axis=-1),
            int((~v).sum()),n,0,int((~np.isfinite(bg)).sum()),int((~rec).sum()),
            int((el&v&~np.isfinite(src)).sum()),diag)
    full=make_estimate(L)
    full.prefix_estimates.update({n:make_estimate(n) for n in prefixes})
    return full
