"""Fixed output contrasts and exact interval-gap MILPs for a declared model pair.

The two training line vectors never change.  An SVD chooses one whitened output
direction per time; the source-box support term is retained even after a
numerical projection.  No validation, truth or future-noise argument exists.
The MILP certificate concerns its frozen affine contrast, not nonlinear AC.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import csr_matrix, hstack

from .ev_fixed_response import FixedResponse
from . import ev_fixed_train_design as fixed
from .ev_original_domain_milp import _solve_mip


DEFAULT_OPTIONS = dict(solver_seconds=2., total_seconds=35., backend='gurobi',
    rcond=1e-10, acceptance_atol=1e-8, acceptance_rtol=1e-10)


def interval_gap(d, low, high):
    """Distance of the scalar interval [d+low,d+high] from zero."""
    d, low, high = np.broadcast_arrays(*map(lambda v: np.asarray(v, float), (d, low, high)))
    if not np.isfinite([d, low, high]).all() or np.any(low > high):
        raise ValueError('finite ordered scalar intervals required')
    return np.maximum.reduce((np.zeros_like(d), d+low, -d-high))


def box_support_interval(coefficients, lower, upper):
    c, lo, hi = np.broadcast_arrays(*map(lambda v: np.asarray(v, float),
                                        (coefficients, lower, upper)))
    if not np.isfinite([c, lo, hi]).all() or np.any(lo > hi):
        raise ValueError('finite ordered support box required')
    return (np.sum(c*np.where(c >= 0, lo, hi), axis=-1),
            np.sum(c*np.where(c >= 0, hi, lo), axis=-1))


def contrast_direction(delta, control_jacobian, Ga, Gb, *, projection='joint', rcond=1e-10):
    """Choose a deterministic unit direction; report discarded source columns.

    Average projection leaves more signal and pays for its source leakage by
    the original physical source box.  Joint projection is more conservative.
    """
    delta, B, Ga, Gb = (np.asarray(v, float) for v in (delta, control_jacobian, Ga, Gb))
    m = len(delta)
    if delta.shape != (m,) or B.ndim != 2 or B.shape[0] != m or Ga.shape != (m, 2) or Gb.shape != Ga.shape:
        raise ValueError('matching output and two-component source dimensions required')
    if not all(np.isfinite(v).all() for v in (delta, B, Ga, Gb)) or not 0 < rcond < 1:
        raise ValueError('finite directions and 0<rcond<1 required')
    if projection not in ('joint', 'bounded_average'):
        raise ValueError('unknown source projection')
    G = np.column_stack((Ga, Gb)) if projection == 'joint' else (Ga+Gb)/2
    U, singular, _ = np.linalg.svd(G, full_matrices=False)
    cutoff = rcond*(singular[0] if len(singular) else 0.)
    rank = int(np.sum(singular > cutoff))
    P = np.eye(m)-U[:, :rank]@U[:, :rank].T
    pd, pB = P@delta, P@B
    norm = float(np.linalg.norm(pd)); origin = 'projected_nominal_mean_difference'
    if norm > 1e-12:
        w = pd/norm
    elif np.linalg.norm(pB) > 1e-12:
        w = np.linalg.svd(pB, full_matrices=False)[0][:, 0]
        origin = 'projected_control_sensitivity'
    else:
        w = np.zeros(m); origin = 'no_resolved_contrast_signal'
    if np.any(w):
        pivot = int(np.argmax(abs(w)))
        if w[pivot] < 0: w = -w
    return w, dict(projection=projection, source_singular_values=singular,
        source_rank=rank, rank_cutoff=float(cutoff), direction_origin=origin,
        direction_norm=float(np.linalg.norm(w)), nominal_signal_norm=float(np.linalg.norm(delta)),
        projected_signal_norm=norm, projected_control_norm=float(np.linalg.norm(pB)),
        source_leakage_a=w@Ga, source_leakage_b=w@Gb,
        original_source_columns_retained_for_support=True)


def _sigma(context):
    T, n = context['control_shape']; meter = context['future_meter']
    return np.column_stack((fixed._broadcast(meter.v_sigma, (T,n), 'V sigma'),
        fixed._broadcast(meter.master_p_sigma, (T,), 'root P sigma'),
        fixed._broadcast(meter.master_q_sigma, (T,), 'root Q sigma')))


def build_contrast(context, anchor_kw, *, projection='joint', rcond=1e-10, fixed_weights=None):
    began = perf_counter()
    if not context.get('success') or len(context.get('bank', ())) != 2:
        raise ValueError('a successful, predeclared two-candidate context required')
    u = np.asarray(anchor_kw, float); T, n = context['control_shape']
    if u.shape != (T,n) or not np.isfinite(u).all():
        raise ValueError('finite matching anchor required')
    sigma = _sigma(context); responses = {}; outputs = {}
    supplied_weights=None if fixed_weights is None else np.asarray(fixed_weights,float)
    if supplied_weights is not None and (supplied_weights.shape!=(T,n+2)
            or not np.isfinite(supplied_weights).all()
            or np.any(np.linalg.norm(supplied_weights,axis=1)>1+1e-9)):
        raise ValueError('finite fixed contrast weights with norm <= 1 required')
    for c in context['bank']:
        line = context['lines'][c.name]
        if fixed._line_id(line['r'], line['x']) != line['sha256']:
            raise ValueError('training line snapshot changed')
        r = FixedResponse(c, line['r'], line['x'], method='ac')
        s = context['forecasts'][c.name]
        out = r.evaluate(context['background']['p']+u/context['base_kw'],
            context['background']['q'], s['source_p'], s['source_q'], context['root_v'])
        responses[c.name], outputs[c.name] = r, out
    names = [c.name for c in context['bank']]; a, b = names
    delta = (outputs[a]['values']-outputs[b]['values'])/sigma
    jac = {k:v['jac']/sigma[:, :, None] for k,v in outputs.items()}
    weights, records, beta = [], [], np.zeros((T,T*n))
    alpha = np.zeros(T); low = np.zeros(T); high = np.zeros(T)
    bounds_lo = np.asarray(context['source_lower']).reshape(T,2)
    bounds_hi = np.asarray(context['source_upper']).reshape(T,2)
    sa = np.column_stack([context['forecasts'][a][key] for key in ('source_p','source_q')])
    sb = np.column_stack([context['forecasts'][b][key] for key in ('source_p','source_q')])
    for t in range(T):
        B = (jac[a][t, :, :n]-jac[b][t, :, :n])/context['base_kw']
        Ga, Gb = jac[a][t, :, -2:], jac[b][t, :, -2:]
        w, row = contrast_direction(delta[t], B, Ga, Gb, projection=projection, rcond=rcond)
        if supplied_weights is not None:
            w=supplied_weights[t].copy()
            row.update(direction_origin='fixed_nominal_prediction_direction',
                direction_norm=float(np.linalg.norm(w)),source_leakage_a=w@Ga,source_leakage_b=w@Gb)
        weights.append(w); beta[t,t*n:(t+1)*n] = w@B
        alpha[t] = w@delta[t]-beta[t]@u.ravel()
        coefficients = np.r_[w@Ga,-w@Gb]
        l, h = box_support_interval(coefficients, np.r_[bounds_lo[t]-sa[t],bounds_lo[t]-sb[t]],
            np.r_[bounds_hi[t]-sa[t],bounds_hi[t]-sb[t]])
        low[t],high[t] = l,h
        row.update(source_support_low=float(l),source_support_high=float(h))
        records.append(row)
    return dict(success=True,names=names,anchor_kw=u.copy(),weights=np.array(weights),
        alpha=alpha,beta=beta,source_low=low,source_high=high,sigma=sigma,
        direction_records=records,responses=responses,context=context,
        line_sha256={name:context['lines'][name]['sha256'] for name in names},
        construction_seconds=perf_counter()-began,
        source_bound_scope='exact original-box support for frozen affine source derivatives',
        global_AC_source_elimination_certified=False)


def evaluate_contrast(model, power_kw):
    began = perf_counter(); context = model['context']; u = np.asarray(power_kw,float)
    if u.shape != tuple(context['control_shape']) or not np.isfinite(u).all():
        raise ValueError('finite matching power required')
    outputs = {}
    for name in model['names']:
        line=context['lines'][name]
        if fixed._line_id(line['r'],line['x']) != model['line_sha256'][name]:
            raise ValueError('fixed training lines changed')
        s=context['forecasts'][name]
        outputs[name]=model['responses'][name].evaluate(context['background']['p']+u/context['base_kw'],
            context['background']['q'],s['source_p'],s['source_q'],context['root_v'])
    a,b=model['names']; w=model['weights']; sig=model['sigma']
    raw=(outputs[a]['values']-outputs[b]['values'])/sig
    d=np.einsum('to,to->t',w,raw)
    affine=model['alpha']+model['beta']@u.ravel()
    gap=interval_gap(d,model['source_low'],model['source_high'])
    affine_gap=interval_gap(affine,model['source_low'],model['source_high'])
    nominal_signal_squared=float(d@d)
    voltage_margin=float(min(min(v['diagnostics']['min_voltage']-.85,
        1.15-v['diagnostics']['max_voltage']) for v in outputs.values()))
    leak={name:np.einsum('to,tos->ts',w,outputs[name]['jac'][:,:,-2:]/sig[:,:,None]) for name in (a,b)}
    return dict(success=bool(voltage_margin>=0),score=float(np.sum(gap)),scalar_AC_contrast=d,
        scalar_affine_contrast=affine,scalar_affine_error=d-affine,source_adjusted_gaps=gap,
        affine_source_adjusted_gaps=affine_gap,
        nominal_projected_half_SSE=nominal_signal_squared/2,
        affine_source_distance_lower_bound_from_L1=float(np.sum(affine_gap)**2/(2*len(gap))),
        source_adjusted_AC_contrast_proxy_squared=float(np.sum(gap)**2/(2*len(gap))),
        lower_bound_scope='frozen affine model only; AC values use frozen source support',
        terminal_and_internal_voltage_margin=voltage_margin,source_derivative_at_action=leak,
        seconds=perf_counter()-began,full_AC_source_distance_certificate=False)


def solve_interval_gap_milp(domain, alpha, beta, source_low, source_high, *, solver_seconds=2.,backend='gurobi',start=None,scenario_groups=None):
    """Globally optimize sum max(0,d+low,-d-high) in the frozen operation MILP."""
    began=perf_counter(); op=domain.as_dict(); U=np.asarray(op['u_map'],float)
    alpha=np.asarray(alpha,float); beta=np.asarray(beta,float)
    low,high=np.asarray(source_low,float),np.asarray(source_high,float)
    R=len(alpha); N=len(op['lower'])
    if alpha.shape!=(R,) or beta.shape!=(R,U.shape[0]) or low.shape!=(R,) or high.shape!=(R,):
        raise ValueError('matching scalar contrast arrays required')
    if not all(np.isfinite(v).all() for v in (alpha,beta,low,high)) or np.any(low>high):
        raise ValueError('finite ordered scalar source intervals required')
    dl,dh=box_support_interval(beta,op['control_lower'],op['control_upper']); dl+=alpha;dh+=alpha
    vu=np.maximum.reduce((np.zeros(R),dh+low,-dl-high))
    L=np.column_stack((np.zeros(R),dl+low,-dh-high))
    M=vu[:,None]-L
    groups=None
    if scenario_groups is not None:
        groups=[np.asarray(g,int) for g in scenario_groups]
        if not groups or any(g.ndim!=1 or not len(g) or np.any(g<0) or np.any(g>=R) for g in groups):
            raise ValueError('nonempty valid scenario index groups required')
        if sorted(np.concatenate(groups).tolist())!=list(range(R)):
            raise ValueError('scenario groups must partition all contrast rows')
    width=R+3*R+int(groups is not None)
    constraints=[LinearConstraint(hstack((csr_matrix(c.A),csr_matrix((c.A.shape[0],width))),format='csr'),c.lb,c.ub) for c in op['constraints']]
    for t in range(R):
        affine_coeff=beta[t]@U
        for k,(coeff,const) in enumerate(((np.zeros(N),0.),(affine_coeff,alpha[t]+low[t]),(-affine_coeff,-alpha[t]-high[t]))):
            row=np.zeros(N+width);row[:N]=-coeff;row[N+t]=1.
            constraints.append(LinearConstraint(row,const,np.inf))
            row=row.copy();row[N+R+3*t+k]=M[t,k]
            constraints.append(LinearConstraint(row,-np.inf,const+M[t,k]))
        row=np.zeros(N+width);row[N+R+3*t:N+R+3*t+3]=1.
        constraints.append(LinearConstraint(row,1.,1.))
    objective=np.zeros(N+width)
    if groups is None:
        objective[N:N+R]=-1.
    else:
        objective[-1]=-1.
        for g in groups:
            row=np.zeros(N+width);row[-1]=1.;row[N+g]=-1.
            constraints.append(LinearConstraint(row,-np.inf,0.))
    initial=None
    if start is not None:
        sx=np.asarray(start,float);d=alpha+beta@(U@sx)
        pieces=np.column_stack((np.zeros(R),d+low,-d-high))
        bits=np.zeros((R,3));bits[np.arange(R),np.argmax(pieces,axis=1)]=1
        initial=np.r_[sx,pieces.max(axis=1),bits.ravel()]
        if groups is not None:initial=np.r_[initial,min(float(np.sum(pieces.max(axis=1)[g])) for g in groups)]
    lower=np.r_[op['lower'],np.zeros(R),np.zeros(3*R)]
    upper=np.r_[op['upper'],vu,np.ones(3*R)]
    integrality=np.r_[op['integrality'],np.zeros(R,int),np.ones(3*R,int)]
    if groups is not None:
        lower=np.r_[lower,0.];upper=np.r_[upper,min(float(vu[g].sum()) for g in groups)];integrality=np.r_[integrality,0]
    result=_solve_mip(objective,integrality,Bounds(lower,upper),
        constraints,time_limit=solver_seconds,backend=backend,start=initial)
    x=None if result.x is None else np.asarray(result.x[:N]);audit=domain.check(x) if x is not None else dict(feasible=False)
    power=None if x is None else (U@x).reshape(domain.indices.u.shape)
    gaps=None if power is None else interval_gap(alpha+beta@power.ravel(),low,high)
    actual=None if gaps is None else (float(np.sum(gaps)) if groups is None else min(float(gaps[g].sum()) for g in groups))
    bound=getattr(result,'mip_dual_bound',None);gap=getattr(result,'mip_gap',None)
    return dict(success=bool(x is not None and audit['feasible']),x=x,power_kw=power,
        proxy_value=actual,solver_value=None if result.fun is None else -float(result.fun),
        proxy_best_bound=None if bound is None or not np.isfinite(bound) else -float(bound),
        proxy_mip_gap=None if gap is None or not np.isfinite(gap) else float(gap),
        status=int(result.status),message=str(result.message),operation=audit,
        contrast_lower=dl,contrast_upper=dh,gap_upper=vu,big_M=M,
        added_binary_count=3*R,added_continuous_count=R+int(groups is not None),seconds=perf_counter()-began,
        scenario_groups=groups,scenario_count=1 if groups is None else len(groups),
        bound_scope='fixed affine interval-gap MILP only',global_AC_gap=None)


def select_contrast_controls(context,domain,initial_powers,*,projection='joint',options=None,seed=0,prediction_scenarios=None):
    began=perf_counter();config=dict(DEFAULT_OPTIONS);config.update(options or {})
    if set(config)-set(DEFAULT_OPTIONS):raise ValueError('unsupported contrast options')
    for k in ('solver_seconds','total_seconds'):
        if not np.isfinite(config[k]) or config[k]<=0:raise ValueError('positive finite budget required')
    deadline=began+config['total_seconds']
    if not initial_powers:raise ValueError('public initial plans required')
    anchor=initial_powers.get('square',next(iter(initial_powers.values())))
    model=build_contrast(context,anchor,projection=projection,rcond=config['rcond'])
    models=[model];scenario_names=['nominal']
    for scenario in prediction_scenarios or []:
        if perf_counter()>=deadline:
            raise RuntimeError('planning budget exhausted before all declared scenarios were built')
        meta=scenario.get('metadata',{})
        if (meta.get('training_only') is not True or any(meta.get(k) is not False for k in
                ('uses_validation','uses_truth','uses_future_noise'))):
            raise ValueError('prediction scenarios must declare training-only provenance')
        name=scenario['name']
        if name in scenario_names:raise ValueError('distinct prediction scenario names required')
        p,q=(np.asarray(scenario[k],float) for k in ('p','q'))
        if p.shape!=context['control_shape'] or q.shape!=p.shape or not np.isfinite([p,q]).all():
            raise ValueError('finite matching background scenarios required')
        new_context=dict(context);new_context['background']=dict(p=p.copy(),q=q.copy())
        models.append(build_contrast(new_context,anchor,projection=projection,rcond=config['rcond'],fixed_weights=model['weights']))
        scenario_names.append(name)
    def evaluate_all(u):
        values=[]
        for m in models:
            if perf_counter()>=deadline:
                return dict(success=False,score=None,scenario_evaluations=values,
                    failure='planning budget exhausted; incomplete scenario scores are not used')
            values.append(evaluate_contrast(m,u))
        if len(values)==1:return values[0]
        return dict(success=all(v['success'] for v in values),score=min(v['score'] for v in values),
            scenario_evaluations=values,scenario_names=scenario_names,
            criterion='minimum fixed-contrast AC score over declared training scenarios',
            global_AC_source_certificate=False)
    serial_model={k:v for k,v in model.items() if k not in ('responses','context')}
    record=dict(method='fixed_source_bounded_contrast_L1',projection=projection,configuration=config,
        model=serial_model,initial_candidates=[],uses_truth=False,uses_validation=False,
        uses_future_noise=False,random_action_proposals=False,source_fits_in_selector=0,
        one_trained_parameter_vector=True,line_parameters_fixed=True,
        acceptance_scope='complete AC nominal contrast with frozen affine source-support subtraction',
        full_AC_profile_score_is_separate_diagnostic=True,global_AC_gap=None,seed=int(seed))
    record['scenario_names']=scenario_names
    record['scenario_models']=[{k:v for k,v in m.items() if k not in ('responses','context')} for m in models]
    record['prediction_scenario_coverage_certified']=False
    best=None;seen=set()
    ordered_initials=([('square',initial_powers['square'])] if 'square' in initial_powers else [])
    ordered_initials.extend((k,v) for k,v in initial_powers.items() if k!='square')
    record['score_tie_policy']='retain public squared-power baseline before other initial plans'
    for origin,power in ordered_initials:
        if perf_counter()>=deadline:
            record['initialization_budget_exhausted']=True
            break
        if power is None:continue
        u=np.asarray(power,float);key=u.tobytes()
        if key in seen:continue
        seen.add(key);realized=fixed._realize(domain,u)
        row=dict(origin=origin,operation=realized['audit'])
        if realized['success']:
            ev=evaluate_all(u);row.update(evaluation=ev)
            if ev['success'] and (best is None or ev['score']>best['score']+config['acceptance_atol']+config['acceptance_rtol']*abs(best['score'])):
                best=dict(origin=origin,power=u.copy(),x=realized['x'],score=ev['score'],evaluation=ev)
        record['initial_candidates'].append(row)
    if best is None:
        record.update(success=False,failure='no physically valid initial plan',seconds=perf_counter()-began)
        return None,record
    record['initial_best_score']=best['score'];remaining=config['total_seconds']-(perf_counter()-began)
    if remaining>0:
        R=len(model['alpha']);groups=[np.arange(i*R,(i+1)*R) for i in range(len(models))] if len(models)>1 else None
        step=solve_interval_gap_milp(domain,np.concatenate([m['alpha'] for m in models]),
            np.vstack([m['beta'] for m in models]),np.concatenate([m['source_low'] for m in models]),
            np.concatenate([m['source_high'] for m in models]),scenario_groups=groups,
            solver_seconds=min(config['solver_seconds'],remaining),backend=config['backend'],start=best['x'])
        record['MILP']=step
        if step['success'] and perf_counter()<deadline:
            ev=evaluate_all(step['power_kw']);record['candidate_evaluation']=ev
            tolerance=config['acceptance_atol']+config['acceptance_rtol']*abs(best['score'])
            record['candidate_accepted']=bool(ev['success'] and ev['score']>best['score']+tolerance)
            if record['candidate_accepted']:
                best=dict(origin='contrast_MILP',power=step['power_kw'],x=step['x'],score=ev['score'],evaluation=ev)
        elif step['success']:
            record['candidate_acceptance_not_attempted']='planning budget exhausted; feasible initial plan retained'
    counters={scenario_names[i]+'/'+name:r.counters for i,m in enumerate(models) for name,r in m['responses'].items()}
    record.update(success=True,selected_origin=best['origin'],selected_score=best['score'],
        selected_evaluation=best['evaluation'],selected_operation=domain.check(best['x']),selected_operation_x=best['x'],
        informative_proxy=bool(np.any(model['weights']) and best['score']>config['acceptance_atol']),
        response_counters=counters,seconds=perf_counter()-began)
    record['time_budget_exceeded']=record['seconds']>config['total_seconds']
    return best['power'].copy(),record
