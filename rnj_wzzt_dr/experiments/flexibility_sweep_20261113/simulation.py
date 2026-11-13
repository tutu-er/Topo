"""Existing original AC generator plus declared customer-device baselines."""
from __future__ import annotations

import numpy as np

from ...observations import Observation
from ..unified_posterior_dr_20261112.physical import generate_background, measured_channels, open_runtime
from ..customer_flexibility_20261113.model import CustomerCase
from .config import STUDY_SEED


def runtime(run):
    return open_runtime(run, 'main')[0]


def slice_observation(o, indices):
    return Observation(o.p[indices], o.q[indices], o.y[indices], o.labels,
                       None if o.voltage is None else o.voltage[indices],
                       None if o.root is None else o.root[indices])


def join_observations(days):
    return Observation(np.concatenate([o.p for o in days]), np.concatenate([o.q for o in days]),
        np.concatenate([o.y for o in days]), days[0].labels,
        np.concatenate([o.voltage for o in days]), np.concatenate([o.root for o in days]))


def sample_history(days, count):
    """48: uniform 30-minute full day; 96: 15-minute day; 192: two full days."""
    if count == 48:
        observation = slice_observation(days[0], np.arange(0, 96, 2))
    elif count == 96:
        observation = days[0]
    elif count == 192:
        if len(days) != 2:
            raise ValueError('Two historical days are required')
        observation = join_observations(days)
    else:
        raise ValueError('The locked grid uses 48, 96 or 192 observations')
    split = count*2//3
    return observation, slice_observation(observation, slice(0, split)), slice_observation(observation, slice(split, None))


def background(module, resource_case, spec, replicate):
    profile = 'independent_scale' if spec['profile'] == 'synchronous' else 'baseline'
    level = .25 if profile == 'independent_scale' else 1.
    b = generate_background(module, dict(case='paper15', replicate=int(replicate),
        voltage_sigma=spec['voltage_sigma'], pq_sigma=.005, root_sigma=.003,
        study_seed=STUDY_SEED, profile=profile, level=level))
    # Reorder a complete generated 24-hour path to the existing 08:00-08:00
    # device contract. No observations are selected using their outcomes.
    for key in ('p', 'q', 'root', 'voltage'):
        b[key] = np.roll(b[key], -32, axis=0)
    b['noise'] = [np.roll(z, -32, axis=0) for z in b['noise']]
    if spec['profile'] == 'evening_peak':
        clock = (8+np.arange(96)/4) % 24
        distance = (clock-19+12) % 24-12
        factor = 1+.35*np.exp(-.5*(distance/2)**2)
        b['p'] = b['p']*factor[:, None]
        b['q'] = b['q']*factor[:, None]
    if tuple(b['labels']) != resource_case.labels:
        raise ValueError('Original physical terminal order changed')
    dp, dq = resource_case.device_baseline_pq_kw()
    b['p'] = b['p']+dp/b['network']['base_kw']
    b['q'] = b['q']+dq/b['network']['base_kw']
    b['resource_case_id'] = resource_case.config['id']
    # The old pre-overlay voltage is not an execution result. Full device-day
    # voltages are calculated separately, after policy sealing for test days.
    b.pop('voltage')
    return b


def observe(module, b, u=None, resource_case=None, response=None):
    p, q = b['p'].copy(), b['q'].copy()
    audit = None
    if u is not None:
        actual = u if response is None else u*response[None]
        audit = resource_case.audit(actual)
        if not audit['passed']:
            raise ArithmeticError('Delivered contract violates the original linear resource domain')
        dp, dq = resource_case.aggregate(resource_case.delivery(actual))
        p += dp/b['network']['base_kw']
        q += dq/b['network']['base_kw']
    v, ac = module._ac_voltages(b['network'], b['labels'], p, q, b['root'])
    return measured_channels(b, p, q, v), dict(ac=ac, device=audit,
        voltage_cells_outside_screen=int(((v < .95) | (v > 1.05)).sum()),
        minimum_voltage_pu=float(v.min()), maximum_voltage_pu=float(v.max())), (p, q, v)


def forecast(b, replicate):
    rng = np.random.default_rng(np.random.SeedSequence([STUDY_SEED, replicate, 991]))
    power = np.column_stack((b['p'], b['q']))
    return power*(1+.05*rng.normal(size=power.shape))


def response(resource_count, replicate):
    # A whole-day resource multiplier preserves linearity and per-resource
    # energy equalities. Values <=1 preserve the convex domain containing zero.
    rng = np.random.default_rng(np.random.SeedSequence([STUDY_SEED, replicate, 773]))
    return rng.uniform(.9, 1., resource_count)
