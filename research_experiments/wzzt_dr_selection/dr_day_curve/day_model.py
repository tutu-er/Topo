"""Daily curve model; all powers are mean modulation in kW.

The two fitted historical predictors and their output maps stay frozen.
The reference predictor is a declared forecasting hypothesis, not truth.
Whole-day random multipliers preserve each device's daily energy equality.
"""
from __future__ import annotations

import math
import numpy as np

DT = 0.25
MICRO = 1.e6
NUMERICAL_MARGIN_MICRO = 1.e-4


def canonical_prices(mean_u, elasticities):
    u = np.asarray(mean_u, float)
    lam = np.asarray(elasticities, float)
    if u.ndim != 2 or lam.shape != (u.shape[0],) or np.any(lam <= 0):
        raise ValueError('resource-by-time mean curve and positive elasticities required')
    return np.max(np.abs(u), axis=1) / lam


def payment(mean_u, prices):
    u = np.asarray(mean_u, float); c = np.asarray(prices, float)
    if c.shape != (u.shape[0],) or np.any(c < 0):
        raise ValueError('nonnegative daily prices required')
    return float(c @ (DT * np.maximum(-u, 0.).sum(axis=1)))


def resource_audit(problem, mean_u, prices=None, tolerance=1.e-8):
    u = np.asarray(mean_u, float)
    j, t = len(problem['elasticities']), problem['slots']
    if u.shape != (j, t) or not np.isfinite(u).all():
        raise ValueError('finite resource-by-time mean modulation required')
    lam = np.asarray(problem['elasticities']); d = np.asarray(problem['factors'])
    c = canonical_prices(u, lam) if prices is None else np.asarray(prices, float)
    maximum = float(np.max(np.abs(DT*u.sum(axis=1)), initial=0.))
    maximum = max(maximum, float(np.max(-c, initial=0.)),
                  float(np.max(c-np.asarray(problem['price_caps']), initial=0.)),
                  float(np.max(np.abs(u)-lam[:,None]*c[:,None], initial=0.)))
    corners=[]
    for z in [(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)]:
        eta=1+d@np.array(z); actual=eta[:,None]*u
        prefix=DT*np.cumsum(actual,axis=1)
        v=max(float(np.max(np.asarray(problem['power_lower'])-actual,initial=0.)),
              float(np.max(actual-np.asarray(problem['power_upper']),initial=0.)),
              float(np.max(np.asarray(problem['energy_lower'])-prefix,initial=0.)),
              float(np.max(prefix-np.asarray(problem['energy_upper']),initial=0.)))
        # Individual availability is checked without conflating resources.
        v=max(v,float(np.max(np.abs(actual)[~np.asarray(problem['available'],bool)],initial=0.)))
        fee=payment(actual,c);maximum=max(maximum,v)
        if problem['budget'] is not None:maximum=max(maximum,fee-problem['budget'])
        corners.append(dict(factors=list(z),payment=fee,violation=max(v,0.)))
    expected=payment(u,c)
    hard=float(((1+np.abs(d).sum(axis=1))*c)@(DT*np.maximum(-u,0.).sum(axis=1)))
    if problem['budget'] is not None:maximum=max(maximum,hard-problem['budget'])
    return dict(feasible=maximum<=tolerance,maximum_violation=max(maximum,0.),
                prices=c.tolist(),expected_payment=expected,hard_payment_bound=hard,corners=corners,
                payment_cap=problem['budget'],payment_cap_enforced=problem['budget'] is not None)


def margin_values(problem, scenarios, mean_u):
    """Exact paired validation margins, in micro p.u.^2; no fitted weights move."""
    u=np.asarray(mean_u,float)
    result=[]
    for s in scenarios:
        effect=np.einsum('ij,jt->ti',np.asarray(problem['contrast']),
                         (1+np.asarray(problem['factors'])@np.asarray(s['z']))[:,None]*u)
        bad=np.asarray(s['base_bad_micro'])+effect
        m=float(problem['history_component_micro']+problem['day_weight']*(np.abs(bad).mean()
            -(1+problem['new_tolerance'])*s['reference_mae_micro'])-problem['epsilon_micro'])
        m0=float(s['margin_zero_micro'])
        target=float(s.get('target_micro',.2*max(-m0,0.)+NUMERICAL_MARGIN_MICRO))
        result.append(dict(margin_micro=m,baseline_margin_micro=m0,increment_micro=m-m0,
                           target_micro=target,surplus_micro=m-target,passed=m>=target-1.e-7))
    return result


def scenario_summary(problem, scenarios, mean_u):
    a=margin_values(problem,scenarios,mean_u);S=len(a);required=math.ceil(.9*S)
    surplus=sorted((x['surplus_micro'] for x in a),reverse=True)
    return dict(denominator=S,required=required,passed=sum(x['passed'] for x in a),
                feasibility_slack_micro=max(0.,-surplus[required-1]),
                lower90_surplus_micro=surplus[required-1],
                mean_margin_micro=float(np.mean([x['margin_micro'] for x in a])),
                mean_increment_micro=float(np.mean([x['increment_micro'] for x in a])),
                rows=a)


def build_problem(pool, history, resources, *, reference_id='refined', budget=None, template_role='training'):
    from research_experiments.wzzt_dr_selection.dr_frozen_distance.frozen_pool import _power,_target
    models={x['candidate_id']:x for x in pool['models']}
    if reference_id not in models or len(models)!=2:raise ValueError('two frozen historical candidates required')
    other=next(x for x in models if x!=reference_id);f=models[reference_id];b=models[other]
    labels=pool['labels'];n=len(labels);j=len(resources)
    training=[];historical={}
    for key,model in models.items():
        losses=[]
        for scene in history['validation']:
            p,q=_power(scene,labels);y=_target(scene,p.shape)
            losses.extend(np.abs(y-p@np.asarray(model['R']).T-q@np.asarray(model['X']).T).ravel())
        historical[key]=float(np.mean(losses))
    H=historical[other]-1.03*historical[reference_id]
    if template_role not in ('training','validation'):raise ValueError('historical observed day role required')
    for scene in history[template_role]:
        p,q=_power(scene,labels);y=_target(scene,p.shape)
        if p.shape!=(96,n):raise ValueError('three complete observed training days required')
        reference_residual=(y-p@np.asarray(f['R']).T-q@np.asarray(f['X']).T)*MICRO
        training.append(dict(base_bad_micro=(y-p@np.asarray(b['R']).T-q@np.asarray(b['X']).T)*MICRO,
            base_reference_micro=reference_residual,
            reference_mae_micro=float(np.abs(reference_residual).mean())))
    if len(training)!=3 or len(history['validation'])!=3:raise ValueError('all288/288 historical rows retained')
    dr=np.asarray(f['R'])-np.asarray(b['R']);dx=np.asarray(f['X'])-np.asarray(b['X'])
    powerlo=[];powerhi=[];elo=[];ehi=[];available=[];contrast=[]
    for r in resources:
        node=labels.index(r['node_label']);base=np.broadcast_to(np.asarray(r['baseline_kw'],float),(96,))
        lower=np.broadcast_to(np.asarray(r.get('power_lower_kw',0.),float),(96,))-base
        upper=np.broadcast_to(np.asarray(r['power_upper_kw'],float),(96,))-base
        powerlo.append(np.maximum(lower,-1.));powerhi.append(np.minimum(upper,1.))
        elo.append(np.broadcast_to(np.asarray(r['prefix_delta_lower_kwh'],float),(96,)))
        ehi.append(np.broadcast_to(np.asarray(r['prefix_delta_upper_kwh'],float),(96,)))
        available.append(np.broadcast_to(np.asarray(r.get('available',True),bool),(96,)))
        ratio=np.broadcast_to(np.asarray(r.get('reactive_ratio',.35),float),(96,))
        if not np.all(ratio==ratio[0]):raise ValueError('this pilot retains original constant Q/P per resource')
        contrast.append((dr[:,node]+ratio[0]*dx[:,node])*MICRO/pool['base_kw'])
    problem=dict(slots=96,n=n,labels=labels,reference_id=reference_id,competitor_id=other,
        historical_losses=historical,history_component_micro=.6*H*MICRO,
        day_weight=.4,new_tolerance=.02,epsilon_micro=1.e-6,
        elasticities=np.full(j,.5),price_caps=np.full(j,1./(.5*1.2)),
        factors=np.array([[.15,.05],[.15,-.05]]),contrast=np.asarray(contrast).T,
        power_lower=np.asarray(powerlo),power_upper=np.asarray(powerhi),
        energy_lower=np.asarray(elo),energy_upper=np.asarray(ehi),available=np.asarray(available),
        budget=None if budget is None else float(budget),training=training,
        forecast_scope='Yfuture=Yhistorical+mu_reference(DR); paired training-day P/Q/Y residual block, reference-conditional synthetic working law')
    return problem


def draw_scenarios(problem, samples, seed, balanced=False):
    rng=np.random.default_rng(seed);result=[]
    for k in range(samples):
        day=k%3 if balanced else int(rng.integers(3));z=rng.uniform(-1,1,2)
        template=problem['training'][day]
        m0=float(problem['history_component_micro']+problem['day_weight']*(np.abs(template['base_bad_micro']).mean()
            -(1+problem['new_tolerance'])*template['reference_mae_micro'])-problem['epsilon_micro'])
        result.append(dict(day=int(day),z=z.tolist(),**template,margin_zero_micro=m0,
                           target_micro=.2*max(-m0,0.)+NUMERICAL_MARGIN_MICRO))
    return result


def alternative_forecast_summary(problem,scenarios,mean_u):
    """Same coarse-exit target, but competitor instead generates DR voltages.

This changes the conditional forecast hypothesis, not the fitted matrices or
observed historical residuals. It is a sensitivity check, not a truth oracle.
"""
    u=np.asarray(mean_u,float);result=[]
    for s in scenarios:
        effect=np.einsum('ij,jt->ti',np.asarray(problem['contrast']),
            (1+np.asarray(problem['factors'])@np.asarray(s['z']))[:,None]*u)
        reference=np.asarray(s['base_reference_micro'])-effect
        m=float(problem['history_component_micro']+problem['day_weight']*(np.abs(s['base_bad_micro']).mean()
            -(1+problem['new_tolerance'])*np.abs(reference).mean())-problem['epsilon_micro'])
        target=s['target_micro']
        result.append(dict(margin_micro=m,increment_micro=m-s['margin_zero_micro'],
                           target_micro=target,passed=m>=target-1.e-7))
    return dict(denominator=len(result),passed=sum(x['passed'] for x in result),
        mean_increment_micro=float(np.mean([x['increment_micro'] for x in result])),
        forecast_scope='competing coarse predictor generates voltage response; no test truth used',rows=result)
