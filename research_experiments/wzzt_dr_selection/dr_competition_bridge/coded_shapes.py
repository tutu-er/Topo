"""Same-power, same-energy signed load-shift templates for a next experiment.

Algebra/contract primitives only. Not connected to an actual experiment here.
Device baseline, power margins and cumulative-energy envelope must be supplied
by the flexibility provider; aggregate-feeder feasibility is insufficient.
"""
from __future__ import annotations
import numpy as np


def pair_template(kind='opposed'):
    forward=np.r_[np.full(4,-.2),np.full(4,.2)]
    if kind=='same':return np.column_stack((forward,forward))
    if kind=='opposed':return np.column_stack((forward,-forward))
    if kind=='alternating_opposed':
        wave=np.array([-.2,.2]*4)
        return np.column_stack((wave,-wave))
    raise ValueError('unknown signed-shift template')


def contract_quantities(profile,dt=.25):
    a=np.asarray(profile,dtype=float)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('finite time-by-device profile required')
    return dict(max_power_kw=np.max(abs(a),axis=0),
        reduced_energy_kWh=np.maximum(-a,0).sum(axis=0)*dt,
        added_energy_kWh=np.maximum(a,0).sum(axis=0)*dt,
        net_energy_kWh=a.sum(axis=0)*dt,
        maximum_prefix_energy_kWh=np.max(abs(np.cumsum(a,axis=0)*dt),axis=0))


def feasible_on_declared_device_band(profile,baseline_kw,power_upper_kw,prefix_energy_lower,prefix_energy_upper,dt=.25):
    """Check supplied device reserve bounds without inferring them from topology.

    Prefix bounds apply to the incremental charging-energy trajectory, already
    adjusted by the provider for efficiency, SOC, arrival/departure and deadline.
    This helper does not estimate user availability or participation probability.
    """
    a=np.asarray(profile,float);b=np.asarray(baseline_kw,float)
    upper=np.broadcast_to(np.asarray(power_upper_kw,float),a.shape)
    lo=np.broadcast_to(np.asarray(prefix_energy_lower,float),a.shape)
    hi=np.broadcast_to(np.asarray(prefix_energy_upper,float),a.shape)
    if a.shape!=b.shape or not all(np.isfinite(x).all() for x in (a,b,upper,lo,hi)):
        raise ValueError('aligned finite reserve inputs required')
    cumulative=np.cumsum(a,axis=0)*dt
    return bool((b+a>=-1e-12).all() and (b+a<=upper+1e-12).all()
        and (cumulative>=lo-1e-12).all() and (cumulative<=hi+1e-12).all()
        and np.max(abs(cumulative[-1]),initial=0)<=1e-12)


def atom_response(support,delta):
    delta=np.asarray(delta,float);z=np.zeros(delta.shape[-1])
    z[list(support)]=1.
    return np.sum(delta*z,axis=-1)[...,None]*z


def complementary_inputs(c,d,n):
    """Two candidate-based balanced directions; no true graph or line weights."""
    c,d=set(c),set(d)
    if not(c-d and c&d and d-c):raise ValueError('a genuine crossing required')
    i,j,k=min(c-d),min(c&d),min(d-c)
    first=np.zeros(n);first[j]=1;first[i]=-1
    second=np.zeros(n);second[j]=1;second[k]=-1
    return np.stack((first,second)),dict(exclusive_C=i,intersection=j,exclusive_D=k)
