"""One final affine Bayesian update after a complete-AC MAP candidate.

The original proper prior and the historical likelihood are each used once.
This is a local linear approximation to a nonlinear posterior, not an exact
nonlinear posterior or a Hessian-based Laplace approximation.
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from .. import ev_bayes_response_ac as old
from .. import ev_bayes_response_truncation as trunc
from ..symmetric_ac_local import CHANNELS, whiten_observations


DEFAULT_TRUNCATION=dict(seed=0,power=10,max_power=18,replicates=8,min_ess_fraction=0.)


def affine_local_posterior(prior_mean,prior_covariance,observed,expansion_point,
                           mean,jacobian,line_prior,*,truncation_options=None,
                           log_whitening_jacobian=0.):
    """Use only the original prior and one final affine likelihood.

    All coordinates are w=(eta,z), not optimizer-whitened state coordinates.
    The evidence includes the Gaussian log determinant and Zpost/Zprior.
    ``log_whitening_jacobian`` converts a whitened-meter density to raw units.
    """
    began=perf_counter();m0=np.asarray(prior_mean,float);C0=np.asarray(prior_covariance,float)
    y=np.asarray(observed,float);point=np.asarray(expansion_point,float)
    f=np.asarray(mean,float);J=np.asarray(jacobian,float);d=len(line_prior['mean'])
    if m0.ndim!=1 or C0.shape!=(len(m0),len(m0)) or point.shape!=m0.shape:
        raise ValueError('matching original joint prior/expansion dimensions required')
    if y.ndim!=1 or f.shape!=y.shape or J.shape!=(len(y),len(m0)) or not 0<d<=len(m0):
        raise ValueError('matching mean/Jacobian and positive line dimension required')
    if not all(np.isfinite(a).all() for a in (m0,C0,y,point,f,J)):
        raise ValueError('finite prior, data, expansion and derivatives required')
    if not np.isfinite(log_whitening_jacobian):raise ValueError('finite whitening Jacobian required')
    if (not np.allclose(m0[:d],line_prior['mean'],rtol=1e-11,atol=1e-13)
            or not np.allclose(C0[:d,:d],line_prior['cov'],rtol=1e-11,atol=1e-13)
            or np.max(abs(C0[:d,d:]),initial=0.)>1e-13):
        raise ValueError('original line-prior mean/covariance and independent state prior required')
    C0=.5*(C0+C0.T);L0=np.linalg.cholesky(C0)
    offset=f-J@point;residual=y-offset-J@m0
    # Work in original-prior standard coordinates. This avoids subtracting
    # two large covariance matrices and uses det(I+AA')=det(I+A'A).
    A=J@L0;G=np.eye(len(m0))+A.T@A;factor=cho_factor(G,lower=True)
    rhs=A.T@residual;step=cho_solve(factor,rhs)
    untr_mean=m0+L0@step
    untr_cov=L0@cho_solve(factor,L0.T);untr_cov=.5*(untr_cov+untr_cov.T)
    logdet=float(2*np.log(np.diag(factor[0])).sum())
    # At the affine posterior mean, likelihood+prior gives the same quadratic
    # form as residual'(I+A A')^-1 residual, without catastrophic subtraction.
    affine_residual=residual-A@step
    quadratic=float(affine_residual@affine_residual+step@step)
    gaussian_log_density=float(-.5*(quadratic+logdet+len(y)*np.log(2*np.pi)))
    settings={**DEFAULT_TRUNCATION,**(truncation_options or {})};settings.pop('return_samples',None)
    line=trunc.bounded_line_moments(untr_mean[:d],untr_cov[:d,:d],line_prior['bounds'],**settings)
    result=dict(status=line['status'],success=False,line_dimension=d,
        untruncated=dict(mean=untr_mean,cov=untr_cov),bounded=None,
        line_prior=deepcopy(line_prior),integration_options=settings,
        line_integration={k:v for k,v in line.items() if k not in ('samples','weights','log_weights')},
        gaussian_log_density=gaussian_log_density,gaussian_logdet=logdet,
        gaussian_quadratic=quadratic,affine_offset=offset,expansion_point=point.copy(),
        expansion_mean=f.copy(),affine_jacobian=J.copy(),prior_mean=m0.copy(),
        original_prior_used_once=True,historical_likelihood_used_once=True,
        old_topology_weights_used=False,local_nonlinear_posterior_approximation=True,
        center_scope='accepted first-order MAP candidate; local minimum unproven',
        nonlinear_local_minimum_certified=False,global_map_certified=False,true_Hessian_used=False,
        data_dependent_affine_evidence_is_nonlinear_approximation=True,
        exact_nonlinear_evidence=False,log_whitening_jacobian=float(log_whitening_jacobian),
        log_evidence_white=None,log_evidence_raw=None,seconds=perf_counter()-began)
    if line['status']!='ok':return result
    logZ0=float(line_prior['log_mass'])
    if not np.isfinite(logZ0):raise ValueError('proper original line-prior mass required')
    bounded=old.restore_joint_moments(untr_mean,untr_cov,d,line['mean'],line['cov'])
    evidence=float(gaussian_log_density+line['log_mass']-logZ0)
    result.update(status='ok',success=True,bounded=bounded,
        logZprior=logZ0,logZpost=float(line['log_mass']),
        log_evidence_white=evidence,log_evidence_raw=evidence+float(log_whitening_jacobian),
        seconds=perf_counter()-began)
    return result


def whitening_log_jacobian(history,meter):
    total=0.
    for name in CHANNELS:
        shape=np.asarray(getattr(history,name)).shape
        sigma=np.broadcast_to(np.asarray(getattr(meter,name+'_sigma'),float),shape)
        if not np.isfinite(sigma).all() or np.any(sigma<=0):raise ValueError('positive meter sigmas required')
        total-=float(np.log(sigma).sum())
    return total


def final_local_posterior(problem,map_result,*,truncation_options=None):
    """Adapter for an accepted first-order MAP candidate, not a minimum proof."""
    if not (map_result.get('success',False) and map_result.get('converged',True)):
        return dict(status='map_failure',success=False,map_status=map_result.get('status'),
            local_nonlinear_posterior_approximation=True,old_topology_weights_used=False)
    w=np.asarray(map_result['map_w'],float)
    try:evaluation=problem.evaluate(w)
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
        return dict(status='map_AC_failure',success=False,error=type(exc).__name__+': '+str(exc),
            old_topology_weights_used=False)
    if not evaluation.get('physical_ok',True):
        return dict(status='map_physical_failure',success=False,
            physical_failures=evaluation.get('physical_failures'),old_topology_weights_used=False)
    result=affine_local_posterior(problem.prior_mean,problem.prior_covariance,
        whiten_observations(problem.history,problem.meter),w,evaluation['mean'],
        evaluation['Jacobian'],problem.line_prior,truncation_options=truncation_options,
        log_whitening_jacobian=whitening_log_jacobian(problem.history,problem.meter))
    result['candidate']=problem.candidate.name
    if result['success']:
        center=result['bounded']['mean']
        try:check=problem.evaluate(center)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            result.update(status='bounded_center_AC_failure',success=False,
                error=type(exc).__name__+': '+str(exc))
            return result
        affine=result['affine_offset']+result['affine_jacobian']@center
        difference=np.asarray(check['mean'])-affine
        result['bounded_center_AC_check']=dict(physical_ok=check.get('physical_ok',True),
            physical_failures=check.get('physical_failures'),
            affine_error_whitened_RMS=float(np.sqrt(np.mean(difference**2))),
            affine_error_whitened_max=float(np.max(abs(difference))),
            MAP_to_bounded_center_norm=float(np.linalg.norm(center-w)))
        if not check.get('physical_ok',True):result.update(status='bounded_center_physical_failure',success=False)
    return result


def future_marginal(local,K_D,K_FD,K_FF):
    """Continue the original GP without conditioning on history a second time."""
    if not local.get('success'):
        return dict(status='historical_posterior_failure',success=False,
            historical_status=local.get('status'))
    KD,KFD,KFF=map(lambda x:np.asarray(x,float),(K_D,K_FD,K_FF));d=local['line_dimension']
    h=len(local['untruncated']['mean'])-d;future=len(KFF)
    if KD.shape!=(h,h) or KFD.shape!=(future,h) or KFF.shape!=(future,future):
        raise ValueError('process history/cross/future dimensions disagree')
    factor=cho_factor(.5*(KD+KD.T),lower=True)
    A=cho_solve(factor,KFD.T).T
    innovation=KFF-A@KFD.T
    innovation,roundoff=old.symmetric_psd(innovation,name='GP conditional future covariance')
    def extend(record):
        m=np.asarray(record['mean']);C=np.asarray(record['cov'])
        mean=np.r_[m[:d],A@m[d:]]
        cross=C[:d,d:]@A.T
        covariance=np.block([[C[:d,:d],cross],[cross.T,A@C[d:,d:]@A.T+innovation]])
        covariance,correction=old.symmetric_psd(covariance,name='future joint covariance')
        return dict(mean=mean,cov=covariance,covariance_roundoff=correction)
    return dict(status='ok',success=True,line_dimension=d,
        untruncated=extend(local['untruncated']),bounded=extend(local['bounded']),
        line_prior=deepcopy(local['line_prior']),integration_options=deepcopy(local['integration_options']),
        line_integration=local['line_integration'],
        log_evidence_white=local['log_evidence_white'],log_evidence_raw=local['log_evidence_raw'],
        future_conditional_matrix=A,future_innovation_covariance=innovation,
        future_innovation_roundoff=roundoff,history_used_again=False,
        line_state_cross_retained=True,local_nonlinear_posterior_approximation=True)


def build_forecast_context(candidates,future_results,future_meter,*,horizon,
                           base_kw=20.,root_v=None,options=None):
    """Package a complete candidate bank for the unchanged AC propagation API.

    A fresh uniform topology prior is used with this update's affine evidence.
    No old posterior weights are accepted as inputs.
    """
    candidates=tuple(candidates);names=[c.name for c in candidates]
    if int(horizon)!=horizon or horizon<1 or not np.isfinite(base_kw) or base_kw<=0:
        raise ValueError('positive integer horizon and positive finite base_kw required')
    if not candidates or len(set(names))!=len(names) or set(future_results)!=set(names):
        raise ValueError('exact complete candidate bank required')
    failures={name:r.get('status') for name,r in future_results.items() if not r.get('success')}
    if failures:raise ValueError('required candidate posterior failure: '+str(failures))
    n=candidates[0].n;width=2*n+2;rows=[];updates={};evidence=[]
    for candidate in candidates:
        future=future_results[candidate.name];d=future['line_dimension'];bounded=future['bounded'];untr=future['untruncated']
        if candidate.n!=n or d!=2*candidate.m or len(bounded['mean'])!=d+horizon*width:
            raise ValueError('candidate/future dimension mismatch')
        mean=np.asarray(bounded['mean']);C=np.asarray(bounded['cov']);r,x=trunc.eta_to_rx(mean[:d]);latent=mean[d:].reshape(horizon,width)
        rows.append(dict(candidate=candidate,r=r,x=x,line_sha256=old._line_id(r,x),
            line_dimension=d,joint_mean=mean,joint_covariance=C,latent_mean=latent))
        evidence.append(float(future['log_evidence_white']))
        updates[candidate.name]=dict(line_prior=future['line_prior'],
            TV_untruncated_future=dict(mean=untr['mean'],cov=untr['cov']),
            TV_posterior_future=dict(mean=mean,cov=C),future_latent_mean=latent,
            future_latent_covariance=C[d:,d:],TV_integration_options=future['integration_options'],
            TV_line_integration=future['line_integration'],
            log_evidence_white=future['log_evidence_white'],log_evidence_raw=future['log_evidence_raw'],
            historical_likelihood_used_once=True,original_prior_used_once=True,
            local_nonlinear_posterior_approximation=True)
    evidence=np.asarray(evidence)
    if not np.isfinite(evidence).all():raise ValueError('finite complete affine evidence bank required')
    log_weights=evidence-logsumexp(evidence)
    root=np.ones(horizon) if root_v is None else np.asarray(root_v,float)
    if root.shape!=(horizon,) or not np.isfinite(root).all() or np.any(root<=0):raise ValueError('invalid root voltage')
    opts=old._options(options)
    context=dict(rows=rows,prior=np.exp(log_weights),log_prior=log_weights,
        log_prior_all_history=log_weights.copy(),log_prior_evidence_only=log_weights.copy(),
        future_meter=future_meter,horizon=horizon,n=n,base_kw=float(base_kw),root_v=root,
        source_interface=1,options=opts,
        model_scope='complete-AC historical first-order MAP candidate; final affine bounded posterior')
    record=dict(candidate_updates=updates,log_prior=log_weights,prior=np.exp(log_weights),
        log_prior_evidence_only=log_weights.copy(),options=opts,n=n,horizon=horizon,base_kw=float(base_kw),
        topology_prior='uniform fresh original prior',old_topology_weights_used=False,
        historical_likelihood_used_once=True,original_prior_used_once=True,
        evidence_scope='proper final affine approximation with logdet and Zpost/Zprior',
        source_interface=1,line_state_cross_retained=True,root_v=root,
        nonlinear_local_minimum_certified=False,global_map_certified=False,true_Hessian_used=False,
        local_nonlinear_posterior_approximation=True)
    return context,record
