"""Local AC/affine checks without probability-calibration or acceptance claims.

The meter output f is already whitened by its declared standard deviations.
The original Gaussian prior is used only to measure displacements and the
objective. These diagnostics neither refit data nor change a posterior.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor,cho_solve,null_space,solve_triangular


def _inputs(problem,map_w):
    point=np.asarray(map_w,float);m0=np.asarray(problem.prior_mean,float)
    C0=np.asarray(problem.prior_covariance,float)
    if point.ndim!=1 or m0.shape!=point.shape or C0.shape!=(len(point),len(point)):
        raise ValueError('matching historical point and original prior required')
    if not all(np.isfinite(a).all() for a in (point,m0,C0)):
        raise ValueError('finite historical point and original prior required')
    L0=np.linalg.cholesky(.5*(C0+C0.T))
    evaluated=problem.evaluate(point,with_jacobian=True)
    f=np.asarray(evaluated['mean'],float);J=np.asarray(evaluated['Jacobian'],float)
    y=np.asarray(problem.observed,float)
    if f.shape!=y.shape or J.shape!=(len(y),len(point)):
        raise ValueError('historical response and Jacobian dimensions disagree')
    return point,m0,L0,evaluated,f,J,y


def _channels(problem,total,channel_sizes=None):
    if channel_sizes is None and hasattr(problem,'history'):
        T,n=problem.history.t,problem.history.n
        channel_sizes=dict(p=T*n,q=T*n,v=T*n,master_p=T,master_q=T)
    if channel_sizes is None:channel_sizes={'all':total}
    if any(isinstance(v,bool) or int(v)!=v or v<1 for v in channel_sizes.values()):
        raise ValueError('positive integer channel sizes required')
    if sum(channel_sizes.values())!=total:raise ValueError('channel sizes must cover every whitened output once')
    result={};start=0
    for name,size in channel_sizes.items():result[name]=slice(start,start+int(size));start+=int(size)
    return result


def _summary(value):
    value=np.asarray(value,float)
    return dict(count=int(value.size),RMS=float(np.sqrt(np.mean(value**2))),
        maximum_absolute=float(np.max(abs(value))),signed_mean=float(value.mean()),
        squared_norm=float(value@value))


def center_closure(problem,map_w,bounded_mean,*,channel_sizes=None):
    """Check f_AC(mean) against the final affine model and its likelihood.

    e=f_AC(mean)-[f_AC(MAP)+J_MAP(mean-MAP)]. The exact likelihood difference
    is r_affine'e+||e||²/2. A small RMS alone is insufficient in many channels;
    the likelihood difference records the accumulated effect explicitly.
    """
    began=perf_counter();point,m0,L0,evaluated,f,J,y=_inputs(problem,map_w)
    center=np.asarray(bounded_mean,float)
    if center.shape!=point.shape or not np.isfinite(center).all():raise ValueError('finite matching bounded mean required')
    checked=problem.evaluate(center,with_jacobian=False)
    f_ac=np.asarray(checked['mean'],float);affine=f+J@(center-point)
    if f_ac.shape!=y.shape or not np.isfinite(f_ac).all():raise ValueError('finite bounded-center AC response required')
    error=f_ac-affine;ra=affine-y;rc=f_ac-y
    direct=float(.5*(rc@rc-ra@ra));identity=float(ra@error+.5*(error@error))
    delta=center-point;prior_step=solve_triangular(L0,delta,lower=True)
    center_prior=solve_triangular(L0,center-m0,lower=True)
    prior_penalty=.5*float(center_prior@center_prior)
    channel={name:dict(closure=_summary(error[block]),AC_residual=_summary(rc[block]),
        affine_residual=_summary(ra[block])) for name,block in _channels(problem,len(y),channel_sizes).items()}
    d=problem.line_dimension
    return dict(status='ok',channels=channel,overall_closure=_summary(error),
        likelihood_difference=dict(AC_minus_affine_half_SSE=direct,
            residual_cross_term=float(ra@error),remainder_half_squared_norm=float(.5*(error@error)),
            exact_identity_value=identity,identity_roundoff=direct-identity,
            AC_objective=float(.5*(rc@rc)+prior_penalty),
            affine_objective=float(.5*(ra@ra)+prior_penalty),prior_terms_identical=True),
        displacement=dict(prior_standardized_squared_norm=float(prior_step@prior_step),
            GN_squared_norm=float(prior_step@prior_step+(J@delta)@(J@delta)),
            line_coordinate_norm=float(np.linalg.norm(delta[:d])),
            state_coordinate_norm=float(np.linalg.norm(delta[d:])),
            mixed_raw_coordinate_norm_not_used=True),
        physical_ok=bool(checked.get('physical_ok',True)),physical_failures=checked.get('physical_failures',[]),
        MAP_physical_ok=bool(evaluated.get('physical_ok',True)),
        scope='one-point local approximation check, not posterior or predictive calibration',
        changes_posterior=False,inference_acceptance_gate=False,seconds=perf_counter()-began)


def weak_profile_curvature(problem,map_w,untruncated_covariance,*,directions=2,
                           standard_deviation_steps=(.5,.25),max_backtracks=12,
                           shrink=.5,activity_tolerance=1e-8,feasibility_tolerance=1e-10):
    """Compare GN and complete-objective curvature in compensated line directions.

    In original-prior standardized coordinates, eliminate the state block
    from I+J'J. The weakest line eigenvectors are lifted back with their best
    local state compensation. Each requested step is a fraction of that
    direction's untruncated line-marginal SD. Physical feasibility may shorten
    both sides equally; all attempted endpoints remain in the report.
    """
    began=perf_counter();point,m0,L0,evaluated,f,J,y=_inputs(problem,map_w)
    C=np.asarray(untruncated_covariance,float);d=problem.line_dimension;s=len(point)-d
    if C.shape!=(len(point),len(point)) or not np.isfinite(C).all():
        raise ValueError('finite matching untruncated covariance required')
    if isinstance(directions,bool) or int(directions)!=directions or directions<1:
        raise ValueError('positive integer direction count required')
    if isinstance(max_backtracks,bool) or int(max_backtracks)!=max_backtracks or max_backtracks<0:
        raise ValueError('nonnegative integer backtrack count required')
    if not 0<shrink<1 or activity_tolerance<0 or feasibility_tolerance<0:
        raise ValueError('valid shrink and nonnegative numerical tolerances required')
    steps=tuple(float(h) for h in standard_deviation_steps)
    if not steps or not all(np.isfinite(h) and h>0 for h in steps):raise ValueError('positive finite SD fractions required')
    A=J@L0;Ae,Av=A[:,:d],A[:,d:]
    Hvv=np.eye(s)+Av.T@Av;Hve=Av.T@Ae
    state_gain=-cho_solve(cho_factor(Hvv,lower=True),Hve) if s else np.empty((0,d))
    Schur=np.eye(d)+Ae.T@Ae+Hve.T@state_gain;Schur=.5*(Schur+Schur.T)
    line_chol=L0[:d,:d]
    white_C=solve_triangular(line_chol,C[:d,:d],lower=True)
    white_C=solve_triangular(line_chol,white_C.T,lower=True).T
    white_C=.5*(white_C+white_C.T)
    constraints=np.asarray(problem.A,float);bounds=np.asarray(problem.h,float)
    slack=bounds-constraints@point[:d]
    active=np.flatnonzero(slack<=activity_tolerance*(1+abs(bounds)))
    tangent=null_space(constraints[active]@line_chol) if len(active) else np.eye(d)
    reduced=tangent.T@Schur@tangent
    eigen,vectors=np.linalg.eigh(.5*(reduced+reduced.T)) if tangent.shape[1] else (np.empty(0),np.empty((0,0)))
    initial_prior=solve_triangular(L0,point-m0,lower=True);r=f-y
    objective0=.5*float(r@r+initial_prior@initial_prior)
    def trial_at(w):
        line_violation=float(np.max(constraints@w[:d]-bounds,initial=0.))
        result=dict(line_domain_violation=line_violation,physical_ok=False)
        if line_violation>feasibility_tolerance:
            result['physical_failures']=['line_domain'];return result
        try:
            response=problem.evaluate(w,with_jacobian=False)
            result.update(physical_ok=bool(response.get('physical_ok',True)),
                          physical_failures=response.get('physical_failures',[]))
            if result['physical_ok']:
                residual=np.asarray(response['mean'],float)-y
                prior=solve_triangular(L0,w-m0,lower=True)
                objective=.5*float(residual@residual+prior@prior)
                if not np.isfinite(objective):raise ValueError('nonfinite complete objective')
                result['objective']=objective
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            result.update(physical_ok=False,error=type(exc).__name__+': '+str(exc))
        return result
    rows=[]
    for index in range(min(int(directions),len(eigen))):
        v=tangent@vectors[:,index];variance=float(v@white_C@v)
        if variance<=0:raise ValueError('positive line-marginal directional variance required')
        sd=np.sqrt(variance);white_direction=np.r_[v*sd,state_gain@(v*sd)]
        direction=L0@white_direction
        GN=float(white_direction@white_direction+(J@direction)@(J@direction))
        slope=float((J@direction)@r+white_direction@initial_prior)
        row=dict(index=index,line_profile_eigenvalue=float(eigen[index]),
            line_marginal_SD_prior_units=float(sd),line_direction_prior_units=v,
            state_compensation_prior_units=state_gain@v,direction_original_units=direction,
            GN_curvature=GN,first_derivative=slope,steps=[])
        for requested in steps:
            h=requested;attempts=[];accepted=None
            for backtrack in range(int(max_backtracks)+1):
                plus,minus=trial_at(point+h*direction),trial_at(point-h*direction)
                current=dict(h_standard_deviations=h,backtrack=backtrack,plus=plus,minus=minus)
                attempts.append(current)
                if plus['physical_ok'] and minus['physical_ok']:
                    curvature=float((plus['objective']+minus['objective']-2*objective0)/h**2)
                    scale=max(abs(plus['objective']),abs(minus['objective']),abs(objective0),1.)
                    accepted=dict(status='ok',h_standard_deviations=h,complete_objective_curvature=curvature,
                        GN_curvature=GN,residual_curvature_difference=curvature-GN,
                        curvature_ratio=curvature/GN if GN>0 else None,
                        floating_cancellation_scale=float(16*np.finfo(float).eps*scale/h**2),
                        central_first_derivative=float((plus['objective']-minus['objective'])/(2*h)))
                    break
                h*=shrink
            row['steps'].append(dict(requested_h_standard_deviations=requested,
                result=accepted or dict(status='no_symmetric_physical_step'),attempts=attempts))
        rows.append(row)
    return dict(status='ok' if rows else 'no_symmetric_line_tangent',directions=rows,
        line_dimension=d,state_dimension=s,active_line_indices=active,active_line_slacks=slack[active],
        line_tangent_dimension=int(tangent.shape[1]),activity_tolerance=float(activity_tolerance),
        objective_at_MAP_candidate=objective0,MAP_physical_ok=bool(evaluated.get('physical_ok',True)),
        Schur_line_marginal_covariance_identity_relative_error=float(
            np.linalg.norm(Schur@white_C-np.eye(d))/max(np.sqrt(d),1.)),
        scope='selected feasible profile directions only; positive curvature is not a minimum certificate',
        full_true_Hessian_computed=False,posterior_calibration_claim=False,
        changes_posterior=False,inference_acceptance_gate=False,seconds=perf_counter()-began)


def diagnose_posterior(problem,map_w,bounded_mean,untruncated_covariance,*,
                       channel_sizes=None,**curvature_options):
    return dict(center_closure=center_closure(problem,map_w,bounded_mean,channel_sizes=channel_sizes),
        profile_curvature=weak_profile_curvature(problem,map_w,untruncated_covariance,**curvature_options),
        scope='local numerical diagnostics; held-out calibration and evaluation remain separate')
