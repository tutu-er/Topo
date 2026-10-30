"""Coded response primitives; same fixed power/energy contract for all policies."""
from __future__ import annotations
import numpy as np
from research_experiments.wzzt_dr_selection.dr_competition_bridge.coded_shapes import pair_template,contract_quantities

TEMPLATES=('same','opposed','alternating_opposed')
POLICIES=('zero','single','rotation','same','opposed','alternating','input_best','wzzt_best')


def contract_spec(kind):
    profile=pair_template(kind)
    quantity=contract_quantities(profile)
    if not (np.allclose(quantity['max_power_kw'],.2) and np.allclose(quantity['reduced_energy_kWh'],.2)
            and np.allclose(quantity['added_energy_kWh'],.2) and np.allclose(quantity['net_energy_kWh'],0)):
        raise ValueError('template violates frozen power/energy quantities')
    return dict(template=kind,interval_hours=.25,window_slots=list(range(76,84)),
                pair_power_kw=profile.tolist(),max_power_kw=.2,reduced_energy_kWh=.2,
                added_energy_kWh=.2,reactive_ratio=.35,capacity_kw=45.,four_event_worst_budget=1.6,
                probability_by_price=[0.,.5,.75],probability_curve_invariant='synthetic preapproved symmetric device band, not empirically calibrated',
                alternating_means='within-event quarter-hour alternation; no cross-event sign flip')


def directions(kind,positions,n):
    if kind not in TEMPLATES or len(positions)!=2 or len(set(positions))!=2:
        raise ValueError('one declared two-site template required')
    out=np.ones((2,n),dtype=int)
    if kind!='same':out[:,positions[1]]=-1
    return out


def expand_prices(menu,positions,n):
    small=np.asarray(menu)
    if small.shape!=(2,2) or not np.isin(small,[0,1,2]).all() or small.sum()>4:
        raise ValueError('two-site prices outside the shared budget')
    out=np.zeros((2,n),dtype=int);out[:,positions]=small
    return out


def apply_response(p_pu,q_pu,price_row,participants,*,base_kw=200.,template_kind='same',active_positions=None):
    """Give every participant its whole fixed coded curve; never clip inputs."""
    p,q=np.asarray(p_pu,float).copy(),np.asarray(q_pu,float).copy()
    prices,z=np.asarray(price_row),np.asarray(participants,bool)
    if p.ndim!=2 or p.shape[0]!=96 or q.shape!=p.shape or prices.shape!=(p.shape[1],) or z.shape!=prices.shape:
        raise ValueError('96 aligned samples and terminal controls required')
    if not np.isin(prices,[0,1,2]).all() or np.any(z&(prices==0)):
        raise ValueError('invalid price or zero-price participation')
    if active_positions is None or len(active_positions)!=2:
        raise ValueError('active pair must be explicit')
    if any(z[j] for j in range(p.shape[1]) if j not in active_positions):
        raise ValueError('this pilot only offers contracts to the active pair')
    if not np.isfinite(base_kw) or base_kw<=0:raise ValueError('invalid power base')
    profile=pair_template(template_kind)
    delta=np.zeros_like(p)
    delta[np.ix_(range(76,84),active_positions)]=profile*z[active_positions]
    p+=delta/base_kw;q+=.35*delta/base_kw
    energy=delta.sum(axis=0)*.25
    return p,q,dict(actual_payment=.2*float(prices@z),participants=int(z.sum()),
        net_energy_change_kWh=energy.tolist(),energy_return_pass=bool(np.max(abs(energy),initial=0.)<=1e-12),
        max_absolute_net_power_kw=float(np.max(abs(p))*base_kw),
        capacity_pass=bool((abs(p)*base_kw<=45.+1e-12).all()),
        nonfinite_values=bool(not np.isfinite(p).all() or not np.isfinite(q).all()),
        curve=template_kind,device_reserve_requires_separate_provider_check=True,
        physical_load_not_clipped=True,flexible_component_empirically_calibrated=False)
