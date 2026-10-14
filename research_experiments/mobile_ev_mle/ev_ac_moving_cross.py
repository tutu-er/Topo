"""AC mixed-derivative proxy in moving total-P/Q coordinates.

The proxy is bilinear in action and competitor increment. Its box may be an
explicit outer relaxation of the original physical box; the two are exposed
separately. No truth, future noise, or fitted bias enters construction.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.optimize import lsq_linear

from .symmetric_ac_local import mean_and_jacobian


STATE = ('p', 'q', 'source_p', 'source_q')


def state_vector(values):
    return np.column_stack([np.asarray(values[k], float) for k in STATE]).ravel()


def vector_state(vector, n):
    value = np.asarray(vector, float).reshape(-1, 2*n+2)
    return dict(p=value[:, :n], q=value[:, n:2*n],
                source_p=value[:, 2*n], source_q=value[:, 2*n+1])


def channel_slices(channels, times, n):
    answer = {}; start = 0
    for channel in channels:
        width = n if channel in ('p', 'q', 'v') else 1
        answer[channel] = slice(start, start+times*width)
        start += times*width
    return answer


def channel_norms(vector, channels, times, n):
    value = np.asarray(vector, float)
    blocks = channel_slices(channels, times, n)
    if len(value) != max((s.stop for s in blocks.values()), default=0):
        raise ValueError('channel rows do not match declared times and terminals')
    return {key: dict(norm=float(np.linalg.norm(value[sl])),
        max_abs=float(np.max(abs(value[sl]), initial=0.)), rows=sl.stop-sl.start)
        for key, sl in blocks.items()}


def control_map(n, horizon, site, base_kw):
    if n < 1 or horizon < 1 or not 0 <= site < n or base_kw <= 0:
        raise ValueError('valid terminal count, horizon, site and power base required')
    U = np.zeros((horizon*(2*n+2), horizon))
    U[np.arange(horizon)*(2*n+2)+site, np.arange(horizon)] = 1./base_kw
    return U


def affine_box_range(matrix, lower, upper):
    matrix, lower, upper = [np.asarray(v, float) for v in (matrix, lower, upper)]
    if matrix.ndim != 2 or lower.shape != (matrix.shape[1],) or upper.shape != lower.shape:
        raise ValueError('box range dimensions differ')
    if np.any(lower > upper) or not all(np.isfinite(v).all() for v in (matrix,lower,upper)):
        raise ValueError('finite ordered box required')
    return (np.maximum(matrix,0)@lower+np.minimum(matrix,0)@upper,
            np.maximum(matrix,0)@upper+np.minimum(matrix,0)@lower)


def transform_domain(old_center, old_lower, old_upper, shift, reference_kw,
                     action_lower, action_upper):
    """Exact action box plus explicit common outer and safe-intersection boxes.

    Physical parameter equals base_center + shift @ u + z. The action-specific
    z box preserves the original absolute parameter box exactly. The outer
    box includes every action-specific box. The safe box instead restricts
    the model and must not substitute for the original physical domain.
    """
    center, lo, hi, shift, u0 = [np.asarray(v,float) for v in
        (old_center,old_lower,old_upper,shift,reference_kw)]
    if shift.shape != (len(center),len(u0)) or lo.shape != center.shape or hi.shape != center.shape:
        raise ValueError('parameter domain/control dimensions differ')
    if np.any(lo >= hi):
        raise ValueError('strictly ordered original increment box required')
    physical_lo, physical_hi = center+lo, center+hi
    base = center-shift@u0
    shift_lo, shift_hi = affine_box_range(shift,action_lower,action_upper)
    return dict(base_center=base, physical_lower=physical_lo, physical_upper=physical_hi,
        action_lower=np.asarray(action_lower,float), action_upper=np.asarray(action_upper,float),
        shift=shift, outer_lower=physical_lo-base-shift_hi,
        outer_upper=physical_hi-base-shift_lo,
        safe_lower=physical_lo-base-shift_lo, safe_upper=physical_hi-base-shift_hi,
        outer_scope='common outer relaxation of original action-specific physical parameter domain',
        safe_scope='common intersection; additional restriction, not original physical domain')


def original_bounds(model, u):
    domain=model['domain']; u=np.asarray(u,float)
    shift=np.asarray(domain['shift'])@u
    base=np.asarray(domain['base_center'])
    return (np.asarray(domain['physical_lower'])-base-shift,
            np.asarray(domain['physical_upper'])-base-shift)


def physical_parameter(model, u, z):
    domain=model['domain']
    return (np.asarray(domain['base_center'])+np.asarray(domain['shift'])@np.asarray(u,float)
            +np.asarray(z,float))


def parameter_increment(model, u, physical):
    return (np.asarray(physical,float)-np.asarray(model['domain']['base_center'])
            -np.asarray(model['domain']['shift'])@np.asarray(u,float))


def matrix_at(model, u):
    return np.asarray(model['A0'])+np.einsum('t,tij->ij',np.asarray(u,float),np.asarray(model['At']))


def residual(model, u, z):
    u,z=np.asarray(u,float),np.asarray(z,float)
    return base_residual(model,u)+matrix_at(model,u)@z


def base_residual(model, u):
    u=np.asarray(u,float)
    r=np.asarray(model['c'])+np.asarray(model['B'])@u
    if 'control_curvature_difference' in model:
        du=u-np.asarray(model['reference_kw'])
        r=r+.5*np.asarray(model['control_curvature_difference'])@(du*du)
    return r


def fixed_action_qp(model, u, *, domain='original', tol=1e-10, max_iter=2000):
    """Convex inner QP with feasible primal and numeric tangent lower bounds."""
    began=perf_counter(); u=np.asarray(u,float)
    if u.shape != (np.asarray(model['B']).shape[1],) or not np.isfinite(u).all():
        raise ValueError('finite action with declared horizon required')
    if domain == 'original': lo,hi=original_bounds(model,u)
    elif domain == 'outer':
        lo,hi=np.asarray(model['domain']['outer_lower']),np.asarray(model['domain']['outer_upper'])
    else: raise ValueError('domain must be original or outer')
    A=matrix_at(model,u); r0=base_residual(model,u)
    norm=np.linalg.norm(A,axis=0); scale=np.ones(len(norm)); scale[norm>0]=1./norm[norm>0]
    solved=lsq_linear(A*scale,-r0,bounds=(lo/scale,hi/scale),method='bvls',tol=tol,max_iter=max_iter)
    z=solved.x*scale; r=r0+A@z; primal=float(r@r)
    gradient=2*A.T@r; corner=np.where(gradient>=0,lo,hi)
    raw=primal+float(gradient@(corner-z))
    guard=64*np.finfo(float).eps*max(1.,primal,float(abs(gradient)@(abs(corner)+abs(z))))
    lower=max(0.,min(primal,raw-guard))
    feasible=bool(np.all(z>=lo-1e-9) and np.all(z<=hi+1e-9) and np.isfinite(r).all())
    physical=physical_parameter(model,u,z); d=model['domain']
    violation=float(max(0.,np.max(np.asarray(d['physical_lower'])-physical),
        np.max(physical-np.asarray(d['physical_upper']))))
    return dict(success=bool(solved.success and feasible),status=int(solved.status),
        message=str(solved.message),z=z,physical_parameter=physical,residual=r,
        primal_upper=primal,tangent_lower_numeric=lower,numerical_gap=primal-lower,
        physical_bound_violation=violation,domain=domain,seconds=perf_counter()-began,
        scope='declared moving-center bilinear proxy at fixed action; not complete nonlinear AC')


def relative_score_bounds(competitor, reference):
    """Valid cost-difference interval; cross lower minus self upper, and converse."""
    return dict(lower=.5*(competitor['tangent_lower_numeric']-reference['primal_upper']),
        upper=.5*(competitor['primal_upper']-reference['tangent_lower_numeric']),
        scope='numeric interval for half of competitor minus self proxy cost')


def future_geometry(candidate, theta, state, root_v, meter, channels):
    return mean_and_jacobian(candidate,theta[:candidate.m],theta[candidate.m:],
        state['p'],state['q'],state['source_p'],state['source_q'],root_v,meter,channels=channels)


def mixed_control_derivatives(candidate, theta, future, meter, channels, *,
                              site, base_kw, step_kw=1e-3):
    """Differentiate analytic AC Jacobians along EV at each independent time.

    The two batched derivative calls perturb every time. Independent AC solves
    imply each row belongs to exactly one time; splitting those rows recovers
    each individual action partial. Tests compare individual-time differences.
    """
    if not np.isfinite(step_kw) or step_kw<=0: raise ValueError('positive finite mixed difference step required')
    horizon=len(future['p']); width=2*candidate.n+2
    plus={key:np.asarray(future[key],float).copy() for key in STATE}
    minus={key:np.asarray(future[key],float).copy() for key in STATE}
    plus['p'][:,site]+=step_kw/base_kw; minus['p'][:,site]-=step_kw/base_kw
    gp=future_geometry(candidate,np.asarray(theta),plus,future['root_v'],meter,channels)
    gm=future_geometry(candidate,np.asarray(theta),minus,future['root_v'],meter,channels)
    dl=(gp['line_jacobian']-gm['line_jacobian'])/(2*step_kw)
    dw=(gp['state_jacobian']-gm['state_jacobian'])/(2*step_kw)
    rows=len(gp['mean']); line=np.zeros((horizon,rows,2*candidate.m)); state=np.zeros((horizon,rows,horizon*width))
    slices=channel_slices(channels,horizon,candidate.n)
    for channel,sl in slices.items():
        channel_width=candidate.n if channel in ('p','q','v') else 1
        for time in range(horizon):
            row=slice(sl.start+time*channel_width,sl.start+(time+1)*channel_width)
            line[time,row]=dl[row]; state[time,row]=dw[row]
    return dict(line=line,state=state,step_kw=float(step_kw),ac_derivative_calls=2,
        scheme='central difference of complete AC analytic Jacobian, batched independent times')


def build_from_saved(model, competitor, future_meter, *, action_lower=None,
                     action_upper=None, step_kw=1e-3, include_mixed=True):
    """Convert an old absolute-total-P/Q model without changing its fixed-u domain."""
    full=model['full']; C=np.asarray(full['C'],float); old_B=np.asarray(full['B'],float)
    u0=np.asarray(model['reference_kw'],float); horizon=len(u0); n=competitor.n
    if action_lower is None: action_lower=np.zeros(horizon)
    if action_upper is None: action_upper=np.full(horizon,6.)
    hrows=len(model['geometry_b']['history']['mean']); lines=int(full['line_columns'])
    hstates=int(full['history_state_columns']); nstates=int(full['future_state_columns'])
    if lines!=2*competitor.m or nstates!=horizon*(2*n+2):
        raise ValueError('saved model column dimensions disagree with candidate')
    shift=np.zeros((C.shape[1],horizon))
    shift[lines+hstates:]=control_map(n,horizon,int(model['site']),float(model['base_kw']))
    B=old_B+C@shift
    ref=np.asarray(full['reference_residual'],float)
    theta=np.asarray(full['parameter_center'],float)[:lines]
    future={key:np.asarray(model['forecast_b'][key],float) for key in STATE}
    future['root_v']=np.asarray(model['forecast_b']['root_v'],float)
    mixed=mixed_control_derivatives(competitor,theta,future,future_meter,tuple(model['channels']),
        site=int(model['site']),base_kw=float(model['base_kw']),step_kw=step_kw)
    At=np.zeros((horizon,C.shape[0],C.shape[1]))
    if include_mixed:
        At[:,hrows:,:lines]=-mixed['line']
        At[:,hrows:,lines+hstates:]=-mixed['state']
    A0=C-np.einsum('t,tij->ij',u0,At)
    domain=transform_domain(full['parameter_center'],full['lower'],full['upper'],shift,u0,
        action_lower,action_upper)
    return dict(name=model['name']+'__moving_cross',a=model['a'],b=model['b'],
        c=ref-B@u0,B=B,A0=A0,At=At,reference_kw=u0,reference_residual=ref,
        C_at_reference=C,old_B=old_B,domain=domain,channels=list(model['channels']),
        history_rows=hrows,history_state_columns=hstates,line_columns=lines,
        future_state_columns=nstates,site=int(model['site']),base_kw=float(model['base_kw']),
        future_background=vector_state(state_vector(future)-shift[lines+hstates:]@u0,n),
        future_root_v=future['root_v'],mixed_derivatives=mixed,
        coordinate='z=(delta_line,delta_history_state,future_state-background-Uu)',
        default_exchange_domain='outer',original_domain_preserved_at_fixed_u=True,
        residual_scope='first-order source response; competitor u-times-parameter mixed AC terms only',
        omissions=['line-times-background_state_increment','parameter-only and state-only curvature',
                   'control-squared source/competitor curvature','higher orders'],
        original_AC_global_certificate=False)


def self_saved_model(pair):
    """Explicit self model for action-dependent reference-cost evaluation."""
    answer=dict(pair); answer.update(name=pair['a']+'__self',b=pair['a'],full=pair['self_model']['full'],
        forecast_b=pair['forecast_a'],geometry_b=pair['geometry_a'])
    return answer


def add_control_curvature(model, saved_pair, reference_candidate, future_meter,
                          *, step_kw=1e-3):
    """Add source minus competitor control² term; fixed-action inner stays QP.

    This residual is quadratic in control, bilinear in control/parameters.
    Fixed-explanation cost is quartic in control, so a bilinear exchange MILP
    cannot use it unchanged. Return a separate copy for diagnostic use.
    """
    forecast=saved_pair['forecast_a']
    theta=np.asarray(saved_pair['self_model']['full']['parameter_center'])[:2*reference_candidate.m]
    a_mixed=mixed_control_derivatives(reference_candidate,theta,forecast,future_meter,
        tuple(saved_pair['channels']),site=int(saved_pair['site']),
        base_kw=float(saved_pair['base_kw']),step_kw=step_kw)
    U=control_map(reference_candidate.n,len(saved_pair['reference_kw']),
        int(saved_pair['site']),float(saved_pair['base_kw']))
    ha=np.column_stack([a_mixed['state'][t]@U[:,t] for t in range(U.shape[1])])
    hb=np.column_stack([np.asarray(model['mixed_derivatives']['state'])[t]@U[:,t]
        for t in range(U.shape[1])])
    hrows=model['history_rows']; ha_full=np.vstack((np.zeros((hrows,U.shape[1])),ha))
    hb_full=np.vstack((np.zeros((hrows,U.shape[1])),hb))
    answer=dict(model); answer.update(control_curvature_reference=ha_full,
        control_curvature_competitor=hb_full,control_curvature_difference=ha_full-hb_full,
        control_curvature_step_kw=step_kw,
        residual_scope='control² source-minus-competitor plus control-times-parameter AC expansion',
        exchange_MILP_compatible_without_modification=False,
        omissions=['line-times-background_state_increment','parameter-only and state-only curvature','higher orders'])
    return answer
