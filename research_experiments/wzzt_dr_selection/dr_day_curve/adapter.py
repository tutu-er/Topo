"""Translate two frozen 24h mean curves to the existing resource-day DTOs.

This module does not optimize, fit parameters, run AC or read case data.
One random multiplier is held over each resource's whole curve; it is not a
per-slot noise law. The caller must freeze curves/prices before actual data.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Mapping
import numpy as np

from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import (
    Battery,DayModule,ResourceDay,evaluate_schedule,
)
from research_experiments.wzzt_dr_selection.dr_conic_pricing.resources import resource_contract_data

DT=.25
SLOTS=96


def _plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Mapping):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    if isinstance(value,Battery):return _plain(asdict(value))
    return deepcopy(value)


def _resource(value):
    if isinstance(value,ResourceDay):
        return _plain(asdict(value)),value
    if not isinstance(value,Mapping):raise TypeError('resources must be existing ResourceDay objects or DTO mappings')
    original=_plain(dict(value));decoded=deepcopy(original)
    if isinstance(decoded.get('battery'),Mapping):decoded['battery']=Battery(**decoded['battery'])
    return original,ResourceDay(**decoded)


def _priced(dto,price):
    result=deepcopy(dto);result['incentive_per_kwh']=float(price)
    if isinstance(result.get('battery'),Mapping):result['battery']=Battery(**result['battery'])
    return ResourceDay(**result)


def _finite(value,shape,name):
    a=np.asarray(value,float)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'{name} must be finite with shape{shape}')
    return a.copy()


def _dummy_profile(resource):
    # A zero-price resource still needs a legal nonzero DayModule because the
    # existing paid-shift DTO rejects empty reduction profiles. Amplitude0
    # makes this dummy physically and financially inactive.
    available=np.flatnonzero(resource.available)
    if resource.service_deadline_slot is not None:
        available=available[available<resource.service_deadline_slot]
    if len(available)<2:raise ValueError('inactive resource needs two available slots for a legal zero-amplitude dummy')
    h=np.zeros(SLOTS);h[available[0]]=-1.;h[available[1]]=1.
    return h


def build_plan(resources,mean_u_kw,daily_prices,*,node_labels,
               elasticities=(.5,.5),response_factors=((.15,.05),(.15,-.05)),
               module_upper_kw=(1.,1.),budget=1.6,policy='day_curve',metadata=None):
    """Return plan/audit/contract_data using the original physical interfaces.

    ``mean_u_kw`` is[2,96] expected active-power increments. Each row is
    energy-neutral. ``prices`` are constant whole-day prices, a=lambda*c,
    h=u/a and actual u=(1+D*z)*u with one shared two-factor draw per day.

    |u_it|<=lambda_i*c_i is required, otherwise free h scaling would let the
    same curve be bought at an arbitrarily small price. The minimum price
    representing a fixed nonzero u is max|u|/lambda; a larger c remains a
    legal redundant representation but is flagged and never silently changed.
    The caller may canonicalize that representation before freezing it.

    ``budget=None`` leaves payment uncapped. All original power, energy,
    availability and whole-day support-corner checks still apply, and both
    expected and worst-case payment are reported. The old actual worker is
    not compatible with this uncapped DTO; this adapter performs no execution.
    """
    original=list(resources)
    if len(original)!=2:raise ValueError('this bounded day-curve adapter requires two declared resources')
    decoded=[_resource(r) for r in original];dtos=[r[0] for r in decoded];objects=[r[1] for r in decoded]
    if len({r.resource_id for r in objects})!=2:raise ValueError('resource IDs must be distinct')
    if len({r.node_label for r in objects})!=2:raise ValueError('two distinct resource nodes required')
    labels=list(node_labels)
    if not labels or len(labels)!=len(set(labels)):raise ValueError('explicit unique terminal order required')
    u=_finite(mean_u_kw,(2,SLOTS),'mean_u_kw');c=_finite(daily_prices,(2,),'daily_prices')
    lam=_finite(elasticities,(2,),'elasticities');d=_finite(response_factors,(2,2),'response_factors')
    caps=_finite(module_upper_kw,(2,),'module_upper_kw')
    if np.any(c<0) or np.any(c>2) or np.any(lam<=0) or np.any(caps<0):raise ValueError('retain price[0,2], positive elasticity and nonnegative original module caps')
    if np.any(np.abs(d).sum(axis=1)>1+1e-12):raise ValueError('whole-day response factor L1 norm must be<=1')
    cap_enforced=budget is not None
    if cap_enforced and (not np.isfinite(budget) or float(budget)!=1.6):
        raise ValueError('existing capped one-day worker contract keeps budget1.6; use None for uncapped model audit')
    if not isinstance(policy,str) or not policy:raise ValueError('policy must be a nonempty name')
    a=lam*c;peaks=np.max(np.abs(u),axis=1);canonical=peaks/lam
    if np.any(peaks>a+1e-10):raise ValueError('|mean_u| must be<=lambda*price; free h cannot create unpriced power')
    profiles=[];inactive=[]
    for i,resource in enumerate(objects):
        if c[i]==0:
            if np.any(u[i]!=0):raise ValueError('zero price requires exactly zero mean curve; no clipping is applied')
            h=_dummy_profile(resource);inactive.append(resource.resource_id)
        else:
            if a[i]==0:raise ValueError('positive-price response underflows numerically')
            if not np.any(u[i]!=0):raise ValueError('positive price with zero curve is redundant; canonicalize price to0 before freezing')
            h=u[i]/a[i]
        module=DayModule('day_curve_'+resource.resource_id,{resource.resource_id:h},float(caps[i]),
            'custom96-slot frozen energy-neutral curve; one whole-day random response multiplier')
        profiles.append(module)
    data=resource_contract_data(objects,profiles,node_labels=labels,elasticities=lam,
        price_upper=[2.,2.],response_factors=d)
    # Original resource DTOs remain unmodified in the plan; pricing is applied
    # to copies for the same net-resource fee audit used by the actual worker.
    priced=[_priced(dto,price) for dto,price in zip(dtos,c)]
    mean=evaluate_schedule(priced,profiles,a.tolist(),rho=None)
    if not mean['resource_feasible']:raise ValueError('mean curve violates original resource/module constraints')
    reconstructed=np.stack([a[i]*np.asarray(next(iter(m.profiles.values()))) for i,m in enumerate(profiles)])
    error=float(np.max(np.abs(reconstructed-u),initial=0.))
    if error>1e-10:raise AssertionError('custom curve normalization changed expected increments')
    hours=np.asarray(data['reduction_hours']);expected=float((hours*c)@a)
    hard=float(np.sum(hours*lam*(1+np.abs(d).sum(axis=1))*c*c))
    direct_expected=float(c@(DT*np.maximum(-u,0).sum(axis=1)))
    if abs(expected-direct_expected)>1e-10 or abs(expected-mean['total_incentive_payment'])>1e-9:
        raise AssertionError('per-contract and net-resource expected fee disagree')
    if cap_enforced and hard>float(budget)+1e-9:raise ValueError('curve/daily prices exceed the original hard budget')
    corners=[]
    for z in [(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)]:
        factor=1+d@np.asarray(z);actual=a*factor
        check=evaluate_schedule(priced,profiles,actual.tolist(),rho=None)
        if not check['resource_feasible']:raise ValueError('whole-day support corner violates original resource/module constraints')
        actual_u=reconstructed*factor[:,None]
        fee=float(c@(DT*np.maximum(-actual_u,0).sum(axis=1)))
        if abs(fee-check['total_incentive_payment'])>1e-9 or fee>hard+1e-9:
            raise AssertionError('corner fee violates independent settlement identity/cap')
        corners.append(dict(factors=list(z),maximum_violation=float(check['maximum_violation']),
            payment=fee,daily_increment_kwh=(DT*actual_u.sum(axis=1)).tolist()))
    plan=dict(status='frozen',policy=policy,resources=dtos,
        modules=[dict(module_id=m.module_id,profiles={k:np.asarray(v).tolist() for k,v in m.profiles.items()},
            amplitude_upper_kw=m.amplitude_upper_kw,description=m.description) for m in profiles],
        amplitudes=a.tolist(),prices=c.tolist(),elasticities=lam.tolist(),response_factors=d.tolist(),
        budget=float(budget) if cap_enforced else None,cap_enforced=cap_enforced,
        expected_payment=expected,hard_payment_bound=hard,
        baseline_semantics='already-contained original component; inject increments only once',
        metadata=_plain(metadata if metadata is not None else {}))
    audit=dict(status='verified',numerical_feasible=True,slots=SLOTS,interval_hours=DT,
        mean_curve_reconstruction_max_error_kw=error,mean_daily_increment_kwh=(DT*u.sum(axis=1)).tolist(),
        expected_payment=expected,direct_expected_payment=direct_expected,hard_payment_bound=hard,
        cap_enforced=cap_enforced,payment_cap=float(budget) if cap_enforced else None,
        existing_actual_worker_compatible=cap_enforced,
        exact_box_worst_payment=max(r['payment'] for r in corners),support_corners=corners,
        canonical_daily_prices=canonical.tolist(),price_scale_slack=(c-canonical).tolist(),
        price_shape_scale_redundancy='daily c and free h are redundant; canonical c=max|u|/lambda preserves u and lowers fee; not behavior learning',
        input_prices_changed=False,no_clipping=True,inactive_dummy_resources=inactive,
        zero_increment_plan=bool(not np.any(u!=0)),
        downstream_component_check_needed=bool(np.any(u!=0)),
        physical_execution='not performed; original AC/measurement bridge still required',
        time_law='one whole-day multiplier per resource; independent per-slot multipliers would not preserve daily energy')
    return dict(plan=plan,audit=audit,contract_data=data)
