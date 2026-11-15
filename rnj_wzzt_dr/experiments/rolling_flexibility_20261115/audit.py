"""Independent saved-array audit for quarter-hour rolling flexibility.

This module never calls the production scorer, device executor, optimization
model or AC solver. Runtime versions are checked against their own archived
sources. Saved actions, scores, device states and payments are reconstructed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from scipy.stats import norm

from ..distribution_gate_20261114.probability import PairIntegral
from ..flexibility_resource_validation_20261114 import operational_audit as electrical


DT = .25
STEPS = 96
ROOT = Path(__file__).resolve().parents[3]


def finite(value, *, ndim=None):
    result = np.asarray(value, float)
    if (ndim is not None and result.ndim != ndim) or not np.isfinite(result).all():
        raise ValueError('Expected finite numeric data of the declared dimension')
    return result


def strict_winner(sse, internal, *, cells, reference_mse, tie):
    """The original physical-MSE tie rule, independently expressed."""
    sse, internal = finite(sse, ndim=1), finite(internal, ndim=1)
    if (sse.shape != internal.shape or not sse.size or np.any(sse < 0)
            or np.any(internal < 0) or np.any(internal != internal.astype(int))
            or cells <= 0 or reference_mse <= 0 or tie < 0):
        raise ValueError('Invalid original selection inputs')
    scaled = sse / (cells * reference_mse)
    eligible = np.flatnonzero(scaled <= scaled.min() + tie)
    return int(min(eligible, key=lambda j: (internal[j], scaled[j], j)))


def profile_scores(sse, families, sigma2):
    """Minimize cumulative SSE within a family, then normalize over families."""
    sse = finite(sse, ndim=1)
    if len(families) != len(sse) or not len(sse) or sigma2 <= 0 or np.any(sse < 0):
        raise ValueError('Invalid family score inputs')
    groups = list(dict.fromkeys(map(str, families)))
    family_ids = list(map(str, families))
    points = [min((i for i, g in enumerate(family_ids) if g == family),
                  key=lambda i: (sse[i], i)) for family in groups]
    values = sse[points]
    exponent = -(values - values.min()) / (2 * sigma2)
    weights = np.exp(exponent)
    weights /= weights.sum()
    return dict(families=groups, points=points, sse=values, weights=weights,
                interpretation='Family-minimum cumulative frozen SSE; profile evidence')


def cumulative_scores(H, history_sse, P, Q, Y, *, internal, families,
                      initial_cells, sigma2, reference_mse, tie):
    """Return initial state plus each prefix, consuming every row exactly once."""
    H, history = finite(H, ndim=3), finite(history_sse, ndim=1)
    P, Q, Y = [finite(a, ndim=2) for a in (P, Q, Y)]
    if (P.shape != Q.shape or P.shape != Y.shape or H.shape != (len(history), P.shape[1], 2*P.shape[1])
            or len(P) > STEPS or initial_cells <= 0 or len(families) != len(history)):
        raise ValueError('Inconsistent frozen-bank or quarter-hour observations')
    errors = Y[None] - np.einsum('tc,knc->ktn', np.c_[P, Q], H)
    increments = np.square(errors).sum(axis=2).T
    scores = np.vstack((history, history[None] + np.cumsum(increments, axis=0)))
    result = []
    for k, sse in enumerate(scores):
        cells = initial_cells + k*P.shape[1]
        selected = strict_winner(sse, internal, cells=cells, reference_mse=reference_mse, tie=tie)
        profile = profile_scores(sse, families, sigma2)
        foreign = [j for j, g in enumerate(families) if g != families[selected]]
        competitor = min(foreign, key=lambda j: (sse[j], internal[j], j)) if foreign else None
        result.append(dict(step=k, cells=cells, sse=sse, native_target=selected,
                           competitor_native=competitor, profile=profile))
    return dict(increments=increments, snapshots=result)


def frozen_means(H, validation_power, validation_y, sigma2):
    """Independent original row-scale calibration, once per native point."""
    H = finite(H, ndim=3)
    X, Y = finite(validation_power, ndim=2), finite(validation_y, ndim=2)
    if X.shape != (len(Y), H.shape[2]) or Y.shape[1] != H.shape[1] or sigma2 <= 0:
        raise ValueError('Invalid historical mean calibration')
    mu = np.einsum('tc,knc->ktn', X, H)
    residual = Y[None] - mu
    factor = 1 + (mu*residual).sum(axis=1)/(np.square(mu).sum(axis=1) + len(X)*sigma2)
    return H*factor[:, :, None]


def power_map(resources, labels, base_kw):
    if base_kw <= 0 or len(set(labels)) != len(labels):
        raise ValueError('Invalid node identities or power base')
    mapping = np.zeros((2*len(labels), len(resources)))
    for j, row in enumerate(resources):
        unit = row['action_unit']
        if unit not in ('kW', 'kvar'):
            raise ValueError('Unknown resource action unit')
        node = list(labels).index(row['node_label'])
        mapping[node + (len(labels) if unit == 'kvar' else 0), j] = 1/base_kw
    return mapping


def reconstruct_delivery(service, extra_request, true_eta, resources):
    """Firm local service tracks directly; one daily device gain scales extra."""
    service, extra = finite(service, ndim=2), finite(extra_request, ndim=2)
    if service.shape != extra.shape or service.shape[1] != len(resources):
        raise ValueError('Service and additional-request dimensions differ')
    if isinstance(true_eta, dict):
        gains = np.asarray([true_eta[r['physical_device_id']] for r in resources], float)
    else:
        gains = finite(true_eta, ndim=1)
    if gains.shape != (len(resources),) or np.any(gains < .9) or np.any(gains > 1):
        raise ValueError('Daily device response is outside [0.9,1]')
    for device in {r['physical_device_id'] for r in resources}:
        values = gains[[r['physical_device_id'] == device for r in resources]]
        if np.any(values != values[0]):
            raise ValueError('One physical device must share its P/Q response gain')
    extra_delivered = extra*gains[None]
    return dict(extra_delivered=extra_delivered, actual_total=service+extra_delivered,
                requested_total=service+extra, daily_column_gains=gains)


def time_window(clock, limits):
    a, b = limits
    return ((clock >= a) & (clock < b)) if a < b else ((clock >= a) | (clock < b))


def activation_prices(spec, resources):
    """Quarter-hour instantaneous prices; duration is applied during payment."""
    clock = (8 + (np.arange(STEPS)+.5)*DT) % 24
    pricing = spec['pricing']
    multiplier = np.ones(STEPS)
    for hours in pricing['peak_clock_windows']:
        multiplier[time_window(clock, hours)] = pricing['clock_hour_multiplier']['morning_evening_peak']
    for hours in pricing['night_clock_windows']:
        multiplier[time_window(clock, hours)] = pricing['clock_hour_multiplier']['night']
    devices = {d['id']: d for d in spec['devices']}
    plus = np.empty((STEPS, len(resources)))
    minus = np.empty_like(plus)
    for j, row in enumerate(resources):
        if row['action_unit'] == 'kW':
            d = devices[row['physical_device_id']]
            plus[:, j] = multiplier*d['fee_positive_yuan_per_kwh']
            minus[:, j] = multiplier*d['fee_negative_yuan_per_kwh']
        elif row['action_unit'] == 'kvar':
            plus[:, j] = minus[:, j] = spec['reactive_fee_yuan_per_kvarh']
        else:
            raise ValueError('Unknown resource price unit')
    return plus, minus


def cumulative_fees(requested_total, actual_total, plus, minus, capacity_prices, *, reserved_tail_peaks=None):
    """Charge total changes from the original baseline; daily peak once."""
    req, actual, positive, negative = [finite(a, ndim=2) for a in
                                      (requested_total, actual_total, plus, minus)]
    cap = finite(capacity_prices, ndim=1)
    if (not req.shape == actual.shape == positive.shape == negative.shape
            or cap.shape != (req.shape[1],) or (cap < 0).any()
            or (positive < 0).any() or (negative < 0).any()):
        raise ValueError('Payment dimensions or nonnegative prices differ')
    activation = DT*(positive*np.maximum(actual, 0) + negative*np.maximum(-actual, 0)).sum(axis=1)
    committed = abs(req) if reserved_tail_peaks is None else finite(reserved_tail_peaks, ndim=2)
    if committed.shape != req.shape or (committed < 0).any():
        raise ValueError('Invalid non-cancellable remaining-plan capacity reservations')
    if np.any(committed + 1e-12 < abs(req)):
        raise ValueError('Reserved tail capacity omits its executed request')
    peaks = np.maximum.accumulate(committed, axis=0)
    increments = np.diff(np.vstack((np.zeros((1, req.shape[1])), peaks)), axis=0)
    availability = increments@cap
    return dict(activation_by_step=activation, availability_by_step=availability,
                cumulative_activation=np.cumsum(activation), cumulative_availability=np.cumsum(availability),
                cumulative_total=np.cumsum(activation+availability), purchased_peak=peaks,
                requested_active_or_reactive_absolute_hours=DT*abs(req).sum(axis=0),
                delivered_active_or_reactive_absolute_hours=DT*abs(actual).sum(axis=0))


def thermal_states(device, power):
    """Exact substep interpolation of the previously declared hourly model."""
    power = finite(power, ndim=1)
    rho_hour = float(device['hourly_temperature_retention'])
    C = float(device['heat_capacity_kwh_per_C'])
    if not 0 < rho_hour <= 1 or C <= 0:
        raise ValueError('Invalid declared thermal dynamics')
    rho = rho_hour**DT
    gain = DT/C if rho_hour == 1 else (1-rho)/((1-rho_hour)*C)
    demand = finite(device['heat_demand_kw'], ndim=1)[:len(power)]
    ambient = device['ambient_temperature_C']
    values = [float(device['initial_temperature_C'])]
    for p, d in zip(power, demand, strict=True):
        values.append(ambient+rho*(values[-1]-ambient)+gain*(device['cop']*p-d))
    return np.asarray(values)


def battery_states(device, power):
    power = finite(power, ndim=1)
    energy = [float(device['initial_soc']*device['energy_kwh'])]
    for p in power:
        rate = device['charging_efficiency']*max(p, 0) + min(p, 0)/device['discharging_efficiency']
        energy.append(energy[-1]+DT*rate)
    return np.asarray(energy)/device['energy_kwh']


def device_prefix(spec, resources, actual_delta):
    """Rebuild physical states and backward service reachability from catalog."""
    delta = finite(actual_delta, ndim=2)
    k, r = delta.shape
    if k > STEPS or r != len(resources):
        raise ValueError('Invalid executed prefix length/resource identity')
    clock = (8 + (np.arange(STEPS)+.5)*DT) % 24
    lower, upper = np.zeros((STEPS, r)), np.zeros((STEPS, r))
    baseline = np.zeros((STEPS, r))
    states, devices = {}, {}
    max_box = max_circle = max_state = max_terminal = max_reach = 0.
    for d in spec['devices']:
        did, kind, rating = d['id'], d['kind'], float(d['power_kw'])
        cols = {x['action_unit']: j for j, x in enumerate(resources) if x['physical_device_id'] == did}
        pcol, qcol = cols['kW'], cols.get('kvar')
        p0, q0 = finite(d['baseline_p_kw'], ndim=1), finite(d['baseline_q_kvar'], ndim=1)
        if p0.shape != (STEPS,) or q0.shape != p0.shape:
            raise ValueError('Device baseline must contain all 96 fixed points')
        baseline[:, pcol] = p0
        if qcol is not None:
            baseline[:, qcol] = q0
        P = p0[:k]+delta[:, pcol]
        Q = q0[:k]+(0 if qcol is None else delta[:, qcol])
        if kind == 'pv':
            available = finite(d['available_generation_kw'], ndim=1)
            lower[:, pcol] = 0.
            upper[:, pcol] = available
            online = available > 0
        elif kind == 'ev':
            online = time_window(clock, d['connected_clock_hours'])
            lower[:, pcol], upper[:, pcol] = -p0, rating*online-p0
            a, g = np.ones(STEPS), np.zeros(STEPS)
            c = np.full(STEPS, DT*d['charging_efficiency'])
            initial, target = 0., float(d['required_battery_energy_kwh'])
            lo_state, hi_state = 0., target
            state_units = 'kWh'
        elif kind == 'battery':
            charge, discharge = time_window(clock, d['charging_clock_hours']), time_window(clock, d['discharging_clock_hours'])
            lower[:, pcol] = -rating*discharge
            upper[:, pcol] = rating*charge
            a, g = np.ones(STEPS), np.zeros(STEPS)
            c = DT*np.where(charge, d['charging_efficiency'], np.where(discharge, 1/d['discharging_efficiency'], 0.))
            initial = d['initial_soc']*d['energy_kwh']
            target = d['terminal_soc']*d['energy_kwh']
            lo_state, hi_state = d['minimum_soc']*d['energy_kwh'], d['maximum_soc']*d['energy_kwh']
            state_units = 'kWh'
            online = np.zeros(STEPS, bool)
            for hours in d['reactive_clock_windows']:
                online |= time_window(clock, hours)
        elif kind == 'thermal':
            lower[:, pcol], upper[:, pcol] = -p0, rating-p0
            rho_hour, C = d['hourly_temperature_retention'], d['heat_capacity_kwh_per_C']
            rho = rho_hour**DT
            gain = DT/C if rho_hour == 1 else (1-rho)/((1-rho_hour)*C)
            a = np.full(STEPS, rho)
            g = (1-rho)*d['ambient_temperature_C']-gain*np.asarray(d['heat_demand_kw'])
            c = np.full(STEPS, gain*d['cop'])
            initial, target = d['initial_temperature_C'], d['terminal_temperature_C']
            lo_state, hi_state = d['minimum_temperature_C'], d['maximum_temperature_C']
            state_units = 'degC'
        else:
            raise ValueError('Unknown physical device kind')
        if qcol is not None:
            lower[online, qcol] = -d['reactive_cap_kvar']
            upper[online, qcol] = d['reactive_cap_kvar']
            max_circle = max(max_circle, float(np.maximum(0., (P*P+Q*Q)/d['inverter_kva']**2-1).max(initial=0.)))
        elif np.any(Q != 0):
            raise ValueError('Uncontrolled device Q changed from zero')
        devices[did] = dict(kind=kind, node_label=d['node_label'], p_kw=P, q_kvar=Q,
                            baseline_p_kw=p0[:k], baseline_q_kvar=q0[:k])
        if kind != 'pv':
            values = [float(initial)]
            for t in range(k):
                values.append(a[t]*values[-1]+g[t]+c[t]*P[t])
            values = np.asarray(values)
            reach_lo, reach_hi = np.empty(STEPS+1), np.empty(STEPS+1)
            reach_lo[-1] = reach_hi[-1] = target
            for t in range(STEPS-1, -1, -1):
                reach_lo[t] = max(lo_state, (reach_lo[t+1]-g[t]-c[t]*(p0[t]+upper[t, pcol]))/a[t])
                reach_hi[t] = min(hi_state, (reach_hi[t+1]-g[t]-c[t]*(p0[t]+lower[t, pcol]))/a[t])
            violation = max(0., lo_state-float(values.min()), float(values.max())-hi_state)
            reach = max(0., float(reach_lo[k]-values[-1]), float(values[-1]-reach_hi[k]))
            terminal = abs(float(values[-1]-target)) if k == STEPS else None
            max_state, max_reach = max(max_state, violation), max(max_reach, reach)
            max_terminal = max(max_terminal, terminal or 0.)
            row = dict(kind=kind, units=state_units, trajectory=values, current=float(values[-1]), target=target,
                       terminal_error=terminal, reachable_lower=float(reach_lo[k]), reachable_upper=float(reach_hi[k]),
                       remaining_reachable=reach <= 1e-7)
            if kind == 'battery':
                row['soc'] = values/d['energy_kwh']
            if kind == 'ev':
                row['remaining_energy_kwh'] = float(target-values[-1])
            states[did] = row
    max_box = max(0., float((lower[:k]-delta).max(initial=0.)), float((delta-upper[:k]).max(initial=0.)))
    return dict(passed=max(max_box, max_circle, max_state, max_terminal, max_reach) <= 1e-7,
                completed_steps=k, terminal_checked=k == STEPS, states=states, devices=devices,
                lower=lower, upper=upper, baseline=baseline, max_box_violation=max_box,
                max_circle_relative_violation=max_circle, max_state_violation=max_state,
                max_terminal_error=max_terminal, max_reachability_violation=max_reach)


def monitor_values(H, means, sigma2, covariance, pair, x, y, deviations=2.):
    """Same pre-observation pair under every frozen mean; a true interval union."""
    H, means = finite(H, ndim=3), finite(means, ndim=3)
    x, y = finite(x, ndim=1), finite(y, ndim=1)
    t, j = map(int, pair)
    difference = (H[t]-H[j])@x
    actual = float((np.square(y-H[j]@x).sum()-np.square(y-H[t]@x).sum())/(2*sigma2))
    expectation = (.5*(difference@difference)+(means@x-H[t]@x)@difference)/sigma2
    variance = max(0., float(difference@covariance@difference/sigma2**2))
    half = deviations*np.sqrt(variance)
    tolerance = 64*np.finfo(float).eps*max(1., abs(actual), float(np.max(abs(expectation), initial=0.)))
    fits = abs(actual-expectation) <= half+tolerance
    return dict(actual_new_logodds=actual, mean_under_each_native=expectation, variance=variance,
                interval_lower=expectation-half, interval_upper=expectation+half,
                explained_by_native=np.flatnonzero(fits).tolist(), outside_union=not bool(fits.any()),
                target_mean_explains=bool(fits[t]), competitor_mean_explains=bool(fits[j]),
                actual_minus_target_mean=float(actual-expectation[t]))


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def file_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(reference):
    path = Path(reference['path'])
    if file_ref(path)['sha256'] != reference['sha256']:
        raise ValueError('Source hash differs: '+str(path))
    return path


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def write(path, value):
    def convert(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=convert)+'\n', encoding='utf-8')


def catalog_resources(spec):
    rows = []
    for unit, devices in [('kW', spec['devices']), ('kvar', [d for d in spec['devices'] if d['kind'] in ('pv', 'battery')])]:
        for d in devices:
            rows.append(dict(physical_device_id=d['id'], node_label=d['node_label'], action_unit=unit,
                             contract_kw=d['power_kw'] if unit == 'kW' else None,
                             capacity_kvar=d.get('reactive_cap_kvar') if unit == 'kvar' else None))
    return rows


def quote_stages(quote):
    """Keep a preliminary solve when first-plan progress triggers another solve."""
    rows = quote_stages(quote['first_quote']) if isinstance(quote.get('first_quote'), dict) else []
    rows.append({key: quote.get(key) for key in ('status', 'feasible', 'optimizer_invoked',
        'source', 'cost_yuan', 'native_status', 'native_runtime', 'raw_native_bound_yuan',
        'lower_bound_yuan', 'relative_gap', 'seconds', 'seconds_budget', 'error')})
    return rows


def epoch_transition_allowed(previous_epoch, current_epoch, reset_steps, step):
    return (current_epoch == previous_epoch or (previous_epoch is None and current_epoch == 0)
            or (current_epoch in reset_steps and reset_steps[current_epoch] <= step))


def independent_quadratic_box_upper(q, a, c, lower, upper):
    """Center/radius outer bound, with all ignored couplings relaxed."""
    Q, A, lo, hi = [np.asarray(x, np.longdouble) for x in (q, a, lower, upper)]
    center, radius = (lo+hi)/2, (hi-lo)/2
    origin = np.longdouble(c)+np.sum(center*np.einsum('tij,tj->ti', Q, center))+2*np.sum(A*center)
    slope = np.einsum('tij,tj->ti', Q, center)+A
    spread = 2*np.sum(abs(slope)*radius)+np.sum(radius*np.einsum('tij,tj->ti', abs(Q), radius))
    return float(origin+spread)+1e-10*max(1., float(abs(origin)+abs(spread)))


def feedback_transition(prior, observation, *, changed, confirmation=8, recovery=8, caution=.25):
    state, outside, recovered = prior['state'], prior['outside_streak'], prior['recovery_streak']
    available = observation is not None
    if not available:
        outside = recovered = 0
        if state != 'pause' and changed: state = 'caution'
    elif observation['outside_union']:
        outside += 1; recovered = 0
        state = 'pause' if state == 'pause' or outside >= confirmation else 'caution'
    elif state == 'pause':
        outside = 0; recovered += 1
        if recovered >= recovery: state, recovered = 'caution', 0
    else:
        outside = recovered = 0; state = 'normal'
    if changed or (available and not observation['target_mean_explains']):
        if state != 'pause': state = 'caution'
    return dict(state=state, outside_streak=outside, recovery_streak=recovered,
                extra_action_fraction=0. if state == 'pause' else caution if state == 'caution' else 1.)


class GateAudit:
    """Full joint-Gaussian integrals, cached before taking remaining-time slices."""
    def __init__(self, bank, covariance, forecast, config):
        self.bank, self.covariance, self.forecast, self.config = bank, covariance, forecast, config
        self.cache = {}

    def integral(self, t, j, mode):
        law = 'forecast' if mode == 'mgf_forecast' else 'fixed'
        key = (t, j, law)
        if key not in self.cache:
            b = self.bank
            sd = .05*abs(self.forecast) if law == 'forecast' else np.zeros_like(self.forecast)
            self.cache[key] = PairIntegral(b['H'][t], b['H'][j], b['means'][t], b['sigma2'], self.covariance, sd, 0.)
        return self.cache[key]

    def evaluate(self, snapshot, shifted, power, mode):
        b, cfg = self.bank, self.config
        step, t = snapshot['step'], snapshot['native_target']
        competitors = [j for j, g in enumerate(b['topology_ids']) if g != b['topology_ids'][t]]
        count = len(set(b['topology_ids']))
        threshold = max(1e-7, (b['initial_cells']+96*len(b['labels']))*b['reference_mse']*b['tie']/(2*b['sigma2']),
                        np.log((count-1)*cfg['beta']/(1-cfg['beta'])) if count > 1 else 0.)
        rows = []
        for j in competitors:
            integral = self.integral(t, j, mode)
            h = (np.asarray(snapshot['sse'])[j]-np.asarray(snapshot['sse'])[t])/(2*b['sigma2'])
            alpha = (1-cfg['eta'])/len(competitors)
            mean_steps = np.einsum('ti,ij,tj->t', power, integral.Q, power)
            base_steps = np.einsum('ti,ij,tj->t', shifted, integral.Q, shifted)
            variance_steps = np.einsum('ti,ij,tj->t', power, integral.V, power)
            mean = float(h+mean_steps.sum()); variance = max(0., float(variance_steps.sum()))
            if mode == 'normal_fixed':
                margin = mean-threshold-norm.isf(alpha)*np.sqrt(variance)
                failure = float(norm.cdf((threshold-mean)/np.sqrt(variance))) if variance > 0 else float(mean <= threshold)
                contribution, base_contribution = mean_steps, base_steps
                selected_lambda = None
            else:
                eigenvalues = integral.eigenvalues[step:]
                projected = integral.projected_G[step:]
                selected_lambda = float(cfg['lambda_start'])
                while np.min(1+2*selected_lambda*eigenvalues) <= 0: selected_lambda *= .5
                eigen = 1+2*selected_lambda*eigenvalues
                def contributions(x):
                    square = np.einsum('ti,ij,tj->t', x, integral.Q, x)
                    linear = np.einsum('tij,tj->ti', projected, x)
                    return selected_lambda*square+.5*np.log(eigen).sum(1)-2*selected_lambda**2*np.sum(linear*linear/eigen, axis=1)
                contribution, base_contribution = contributions(power), contributions(shifted)
                logbound = selected_lambda*(threshold-h)-float(contribution.sum())
                margin = np.log(alpha)-logbound
                failure = float(np.exp(min(0., logbound)))
            rows.append(dict(target_native=t, competitor_native=j, h=float(h), threshold=float(threshold),
                alpha=float(alpha), margin=float(margin), pair_failure=failure, lambda_selected=selected_lambda,
                forward_mean_gain=float(mean_steps.sum()-base_steps.sum()),
                mean=mean, variance=variance, mean_steps=mean_steps, variance_steps=variance_steps,
                contribution=contribution, base_contribution=base_contribution))
        return dict(rows=rows, feasible=all(r['margin'] >= -1e-7 and r['forward_mean_gain'] >= -1e-8 for r in rows),
                    nominal_union_lower=max(0., 1-sum(r['pair_failure'] for r in rows)), threshold=float(threshold))

    def box_bounds(self, snapshot, shifted, mode, lower, upper, control_map, direct):
        if mode == 'normal_fixed': return []
        output = []; step = snapshot['step']
        for row in direct['rows']:
            integral = self.integral(row['target_native'], row['competitor_native'], mode)
            lam = row['lambda_selected']; eigen = 1+2*lam*integral.eigenvalues[step:]
            G = integral.projected_G[step:]
            F = lam*integral.Q[None]-2*lam**2*np.einsum('tki,tk,tkj->tij', G, 1/eigen, G, optimize=True)
            q = np.einsum('ir,tij,js->trs', control_map, F, control_map, optimize=True)
            a = np.einsum('ir,tij,tj->tr', control_map, F, shifted, optimize=True)
            c = (float(np.einsum('ti,tij,tj->', shifted, F, shifted))+.5*np.log(eigen).sum()
                 +lam*(row['h']-row['threshold'])+np.log(row['alpha']))
            bound = independent_quadratic_box_upper(q, a, c, lower, upper)
            output.append(dict(target_native=row['target_native'], competitor_native=row['competitor_native'],
                upper=bound, fixed_lambda=row['lambda_selected'], impossible=bound < -1e-7,
                scope='Outer resource box for fixed-lambda allocated-risk sufficient QCQP; not original probability or actual AC infeasibility'))
        return output


class Audit:
    def __init__(self):
        self.sources, self.maxima, self.rows, self.trajectories, self.errors = {}, {}, [], [], []

    def require(self, passed, message):
        if not passed: raise AssertionError(message)

    def same(self, name, saved, expected, tolerance=1e-8, *, rtol=1e-10):
        a, b = np.asarray(saved), np.asarray(expected)
        self.require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), name+': shape/finite mismatch')
        error = float(np.max(abs(a-b), initial=0.))
        self.maxima[name] = max(self.maxima.get(name, 0.), error)
        self.require(np.allclose(a, b, atol=tolerance, rtol=rtol), name+': values differ, max '+str(error))

    def file(self, path):
        reference = file_ref(path); self.sources[reference['path']] = reference
        return Path(path)

    def checked(self, ref):
        path = checked(ref); self.file(path); return path

    def refs(self, value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str): self.checked(value)
            else:
                for item in value.values(): self.refs(item)
        elif isinstance(value, list):
            for item in value: self.refs(item)

    def snapshot(self, saved, expected, bank):
        for key in ('step', 'cells', 'native_target', 'competitor_native'):
            self.require(saved[key] == expected[key], 'Cumulative snapshot differs: '+key)
        self.same('cumulative_SSE', saved['sse'], expected['sse'], 1e-12)
        profile = expected['profile']
        for g, point, sse, weight in zip(profile['families'], profile['points'], profile['sse'], profile['weights'], strict=True):
            self.require(saved['profile_family_points'][g] == point, 'Family profile point differs')
            self.same('family_profile_SSE', saved['profile_family_scores'][g], sse, 1e-12)
            self.same('family_profile_weight', saved['weights'][g], weight, 1e-12)
        self.require(saved['target_family'] == bank['topology_ids'][expected['native_target']], 'Target family differs')

    def device_check(self, saved, calculated):
        self.require(saved['passed'] == calculated['passed'], 'Physical acceptance flag differs')
        for key in ('completed_steps', 'terminal_checked'):
            self.require(saved[key] == calculated[key], 'Physical prefix length differs')
        for key in ('max_box_violation', 'max_circle_relative_violation', 'max_state_violation',
                    'max_terminal_error', 'max_reachability_violation'):
            self.same('physical_'+key, saved[key], calculated[key], 1e-8)
        self.require(set(saved['states']) == set(calculated['states']), 'Physical device state identities differ')
        for did, row in calculated['states'].items():
            for key in ('trajectory', 'current', 'target', 'reachable_lower', 'reachable_upper'):
                self.same('device_'+key, saved['states'][did][key], row[key], 1e-9)
            if row['terminal_error'] is not None:
                self.same('terminal_error', saved['states'][did]['terminal_error'], row['terminal_error'], 1e-9)

    def context(self, entry, source):
        self.refs(entry)
        score = read_json(self.checked(entry['score']))
        val = arrays(self.checked(entry['files']['validation.npz']))
        native = read_json(self.checked(entry['native_copy']))
        H = np.asarray([r['H'] for r in score['native_bank']])
        n = len(score['labels']); reconstructed = []
        active = []
        for row in native['frozen_candidates']:
            point = row['point']; matrix = np.zeros((n, 2*n)); groups = set()
            for support, r, x in zip(point['supports'], point['r'], point['x'], strict=True):
                block = np.ix_(support, support); matrix[:, :n][block] += r; matrix[:, n:][block] += x
                if max(r, x) > native['config']['zero_tol']: groups.add(tuple(sorted(support)))
            reconstructed.append(matrix); active.append(groups)
        self.same('frozen_native_support_matrices', H, reconstructed, 1e-14)
        X = np.c_[val['P'], val['Q']]
        sse = np.square(val['Y'][None]-np.einsum('tc,knc->ktn', X, H)).sum((1, 2))
        self.same('initial_validation_SSE', sse, [r['history_sse'] for r in score['native_bank']], 1e-12)
        representatives = [int(r['index']) for r in score['models']]
        self.same('frozen_common_validation_scale', score['sigma2'], min(sse[representatives])/val['Y'].size, 1e-15)
        means = frozen_means(H, X, val['Y'], score['sigma2'])
        bank = dict(H=H, means=means, initial_sse=sse, topology_ids=[r['topology_id'] for r in score['native_bank']],
                    internal=[r['internal'] for r in score['native_bank']], sigma2=score['sigma2'], labels=score['labels'],
                    initial_cells=score['cells'], reference_mse=score['reference_mse'], tie=score['validation_tie'])
        root_sd = 2*.0005*float(np.median(val['root']))**2
        individual_sd = 2*.0005*float(np.median(val['V']))**2
        covariance = root_sd**2*np.ones((n, n))+individual_sd**2*np.eye(n)
        spec = read_json(self.checked(entry['equipment_spec'])); resources = catalog_resources(spec)
        forecast = arrays(self.checked(entry['forecast_ref']))[entry.get('forecast_key', 'forecast00')]
        physical = arrays(self.checked(entry['physical_cache']['original']))
        bands = read_json(self.checked(entry['node_time_bandwidth'])); bands = arrays(self.checked(bands['arrays']))
        meta = source/'source_snapshot/topology_identification_standalone/cases'/entry['case']
        if not meta.exists(): meta = Path(read_json(source/'protocol.json')['source_run'])/'source_snapshot/topology_identification_standalone/cases'/entry['case']
        metadata = read_json(self.file(meta/'metadata.json'))
        def csv_rows(name):
            with self.file(meta/name).open(encoding='utf-8-sig', newline='') as stream: return list(csv.DictReader(stream))
        branches, buses = csv_rows('branches.csv'), csv_rows('buses.csv')
        children = {}
        for row in branches:
            if str(row.get('status', 'true')).lower() == 'true': children.setdefault(int(row['from_bus']), []).append(int(row['to_bus']))
        labels = list(bank['labels'])
        def descendants(node):
            values = {labels.index(node)} if node in labels else set()
            for child in children.get(node, []): values |= descendants(child)
            return values
        truth = {tuple(sorted(descendants(int(r['to_bus'])))) for r in branches if str(r.get('status', 'true')).lower() == 'true'}
        truth.discard(())
        proper = lambda groups: {g for g in groups if 1 < len(g) < n}
        correct = [proper(g) == proper(truth) for g in active]
        exact = [g == truth for g in active]
        return dict(bank=bank, spec=spec, resources=resources, forecast=forecast, physical=physical,
                    covariance=covariance, metadata=metadata, branches=branches, buses=buses, bands=bands,
                    internal_correct=correct, full_support_exact=exact,
                    true_family_ids=sorted({bank['topology_ids'][i] for i,v in enumerate(correct) if v}))

    def band(self, delta, bands, mapping, base, start):
        n = len(bands['kappa']); nodal = delta@mapping.T*base
        deviation = nodal[:, n:]-nodal[:, :n]*bands['kappa']
        width = np.repeat(bands['band_kvar'], 4, axis=0)[start:start+len(delta)]
        tolerance = 64*np.finfo(float).eps*(abs(nodal[:, n:])+abs(nodal[:, :n]*bands['kappa'])+width)
        violation = abs(deviation)-width
        return dict(passed=bool(np.all(violation <= tolerance)),
                    encoded_passed=bool(np.all(abs(deviation) <= width*(1-1e-7))),
                    maximum_violation_kvar=float(np.maximum(violation, 0.).max(initial=0.)))

    def progress(self, ledger, direct, mode, step, alpha_z):
        if ledger is None: return []
        offset = step-ledger['start_step']
        weights = np.asarray(ledger['weights'])[:, offset:]
        reference_variance = np.asarray(ledger['reference_variance'])[:, offset:]
        rows = []
        for i, pair in enumerate(direct['rows']):
            if mode == 'normal_fixed':
                need = pair['threshold']-pair['h']+alpha_z(pair['alpha'])*np.sqrt(reference_variance[i].sum())
            else:
                need = pair['lambda_selected']*(pair['threshold']-pair['h'])+np.log(1/pair['alpha'])
            deficit = max(0., float(need-pair['base_contribution'].sum()))
            remaining = float(weights[i].sum())
            self.require(not (deficit > 1e-10 and remaining == 0), 'Exhausted positive progress weights require explicit epoch reset')
            gamma = deficit*weights[i, 0]/remaining if remaining > 0 else 0.
            margin = float(pair['contribution'][0]-pair['base_contribution'][0]-gamma) if deficit > 0 else 0.
            rows.append(dict(gamma=float(gamma), margin=margin, deficit=deficit))
        return rows

    def trajectory(self, directory, entry, policy, repeat, protocol, ctx):
        complete = read_json(self.file(directory/'complete.json')); self.refs(complete)
        bank, spec, resources, forecast = [ctx[k] for k in ('bank', 'spec', 'resources', 'forecast')]
        saved_bank = read_json(self.file(directory/'bank.json'))
        for key, value in bank.items():
            if key in ('topology_ids', 'labels'): self.require(saved_bank[key] == value, 'Frozen bank identity changed')
            else: self.same('frozen_bank_'+key, saved_bank[key], value, 1e-12)
        if complete['completed_steps'] == 0 and complete['status'] != 'complete':
            self.require(complete.get('failure') is not None, 'Zero-length failed trajectory lacks reason')
            self.trajectories.append(dict(id=entry['id'], policy=policy, repeat=repeat,
                source=file_ref(directory/'complete.json'), status=complete['status'], audited_steps=0,
                expected_steps=96, missing_steps=96, final=complete,
                true_family_ids=ctx['true_family_ids'],
                internal_truth_covered=any(ctx['internal_correct']), full_support_truth_covered=any(ctx['full_support_exact'])))
            return
        config = protocol['config']; n = len(bank['labels']); mode = None if policy == 'no_dfs' else policy.removeprefix('rolling_')
        mapping = power_map(resources, bank['labels'], entry['base_kw'])
        plus, minus = activation_prices(spec, resources)
        cap_prices = np.asarray([config['kappa_p'] if r['action_unit'] == 'kW' else config['kappa_q'] for r in resources])
        random = arrays(self.file(directory/'randomness.npz')); noise = arrays(self.checked(entry['evaluation_noise']))
        self.refs(read_json(self.file(directory/'randomness.json')))
        for z, name in enumerate(('P', 'Q', 'V', 'root')):
            self.same('sealed_evaluation_noise_'+name, random['noise'+str(z)], noise[name][repeat], 0., rtol=0.)
        etas = {}
        for device in spec['devices']:
            key = ['equipment_flex_whole_device_v1', int(entry['noise_seed']), int(entry.get('original_input_index', entry['index'])), int(entry['history_repeat']), int(repeat), device['id']]
            digest = hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).digest()
            etas[device['id']] = np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest, dtype='<u4').tolist())).uniform(.9, 1.)
        eta = np.asarray([etas[r['physical_device_id']] for r in resources]); self.same('fixed_daily_eta', random['eta'], eta, 0., rtol=0.)
        gains = np.full(len(resources), config['initial_delivery_gain']); numer, denom = {}, {}
        prefix = np.empty((0, len(resources))); P, Q, Y, request_rows, tail_peaks, service_rows, extra_rows, extra_delivery = [], [], [], [], [], [], [], []
        paid = np.zeros(len(resources)); root_bill = 0.; previous_seal = previous_receipt = None
        feedback = dict(state='normal', outside_streak=0, recovery_streak=0)
        gate = GateAudit(bank, ctx['covariance'], forecast, config)
        records = [json.loads(line) for line in self.file(directory/'records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()] if complete['completed_steps'] else []
        self.require(len(records) == complete['completed_steps'], 'Completed-step ledger count differs')
        full_scores = lambda: cumulative_scores(bank['H'], bank['initial_sse'], np.asarray(P).reshape(-1, n), np.asarray(Q).reshape(-1, n), np.asarray(Y).reshape(-1, n),
                internal=bank['internal'], families=bank['topology_ids'], initial_cells=bank['initial_cells'], sigma2=bank['sigma2'], reference_mse=bank['reference_mse'], tie=bank['tie'])
        trajectory_rows = []; epoch_ledgers = {}; previous_ledger = None; epoch_reset_steps = {}
        for k, ledger_row in enumerate(records):
            self.require(ledger_row['step'] == k, 'Missing/duplicated/out-of-order quarter hour')
            folder = directory/f'step{k:03d}'; row = read_json(self.checked(ledger_row['receipt']))
            self.require(Path(ledger_row['receipt']['path']).resolve() == (folder/'receipt.json').resolve(), 'Receipt identity differs')
            seal = read_json(self.checked(row['seal'])); self.refs(seal)
            plan = arrays(self.checked(seal['plan'])); obs = arrays(self.checked(row['observation']))
            self.require(seal['step'] == k and seal['completed_steps'] == k and seal['information_cutoff_step'] == k-1, 'Information cutoff differs')
            self.require(seal['previous_seal'] == previous_seal and seal['previous_receipt'] == previous_receipt, 'Sealed causal chain broken')
            self.require(datetime.fromisoformat(seal['created_utc']) <= datetime.fromisoformat(row['receipt_utc']), 'Observation receipt predates request seal')
            self.snapshot(seal['snapshot'], full_scores()['snapshots'][-1], bank)
            self.require(seal['bank'] == file_ref(directory/'bank.json'), 'Frozen bank reference changed')
            self.same('estimated_gain_uses_past_only', plan['estimated_gains'], gains, 1e-12)
            self.same('original_forecast_tail', plan['forecast_remaining'], forecast[k:], 0., rtol=0.)
            service, extra = plan['service_tail'], plan['extra_tail']
            self.require(service.shape == extra.shape == (96-k, len(resources)), 'Remaining horizon/action dimensions differ')
            self.same('request_tail', plan['request_tail'], service+extra, 0., rtol=0.)
            shifted = forecast[k:]+service@mapping.T
            self.same('service_shifted_forecast', plan['shifted_forecast'], shifted, 1e-14)
            srv = device_prefix(spec, resources, np.vstack((prefix, service)))
            self.require(srv['passed'], 'Firm service tail cannot fulfill physical obligations')
            self.device_check(seal['service_receipt']['physical_audit'], srv)
            nominal = service+extra*gains
            planned = device_prefix(spec, resources, np.vstack((prefix, nominal)))
            endpoints = [device_prefix(spec, resources, np.vstack((prefix, service[0]+v*extra[0]))) for v in (.9, 1.)]
            quote = seal['quote']; claimed = quote.get('feasible') is True
            expected_fraction = 0. if feedback['state'] == 'pause' else config['caution_fraction'] if feedback['state'] == 'caution' else 1.
            self.same('feedback_dispatch_fraction', seal['extra_action_fraction'], expected_fraction, 0., rtol=0.)
            extra_lo = (planned['lower'][k:]-service)/gains
            extra_hi = (planned['upper'][k:]-service)/gains
            for endpoint in (.9, 1.):
                extra_lo[0] = np.maximum(extra_lo[0], (planned['lower'][k]-service[0])/endpoint)
                extra_hi[0] = np.minimum(extra_hi[0], (planned['upper'][k]-service[0])/endpoint)
            extra_lo[0] *= expected_fraction; extra_hi[0] *= expected_fraction
            self.require(np.all(extra >= extra_lo-1e-7) and np.all(extra <= extra_hi+1e-7), 'Extra request violates device box or caution first-step limit')
            if policy == 'no_dfs' or expected_fraction == 0: self.same('no_extra_in_pause_or_baseline', extra, np.zeros_like(extra), 0., rtol=0.)
            direct = None
            progress = []; certificate = None; box_checks = []; band_check = self.band(nominal, ctx['bands'], mapping, entry['base_kw'], k)
            if mode is not None:
                direct = gate.evaluate(seal['snapshot'], shifted, shifted+(extra*gains)@mapping.T, mode)
                if seal['progress_event'] is not None:
                    event = seal['progress_event']
                    self.require(event['event'] == 'explicit_epoch_reset', 'Unknown progress reset event')
                    self.require(event['epoch_id'] == seal['snapshot']['physical_epoch'], 'Reset event/snapshot epoch differs')
                    self.require(event['epoch_id'] not in epoch_reset_steps, 'Same epoch reset was rewritten')
                    epoch_reset_steps[event['epoch_id']] = k
                ledger = seal['progress_ledger']
                if ledger is not None:
                    epoch_id = ledger['epoch_id']
                    if epoch_id in epoch_ledgers:
                        self.require(ledger == epoch_ledgers[epoch_id], 'Progress weights/first plan changed inside an epoch')
                    else:
                        self.require(ledger['start_step'] == k and ledger['end_step'] == 96, 'First progress ledger starts at another observation time')
                        first = np.asarray(ledger['first_verified_plan'])
                        first_direct = gate.evaluate(seal['snapshot'], shifted, shifted+(first*gains)@mapping.T, mode)
                        self.require(first_direct['feasible'], 'First progress plan was not full-native gate feasible')
                        first_physical = device_prefix(spec, resources, np.vstack((prefix, service+first*gains)))
                        self.require(first_physical['passed'], 'First progress plan fails physical terminal tasks')
                        positive = np.asarray([np.maximum(r['contribution']-r['base_contribution'], 0.) for r in first_direct['rows']])
                        sums = positive.sum(axis=1, keepdims=True)
                        weights = np.divide(positive, sums, out=np.zeros_like(positive), where=sums > 0)
                        self.same('frozen_progress_positive', ledger['positive_contributions'], positive, 1e-7)
                        self.same('frozen_progress_weights', ledger['weights'], weights, 1e-8)
                        self.same('frozen_progress_variance', ledger['reference_variance'], [r['variance_steps'] for r in first_direct['rows']], 1e-7)
                        epoch_ledgers[epoch_id] = ledger
                    self.require(epoch_transition_allowed(None if previous_ledger is None else previous_ledger['epoch_id'], epoch_id, epoch_reset_steps, k),
                                 'Progress epoch changed without an earlier explicit receipt')
                    signature = ledger['signature']
                    self.require(signature['target'] == seal['snapshot']['native_target'] and signature['mode'] == mode, 'Progress epoch target/mode mismatch')
                    self.require(signature['competitors'] == [r['competitor_native'] for r in direct['rows']], 'Progress excludes a foreign native point')
                    self.require(signature['lambdas'] == [r['lambda_selected'] for r in direct['rows']], 'Progress silently changes its exponential parameter')
                    self.same('progress_epoch_response_map', signature['control_map'], mapping*gains[None], 1e-13)
                    progress = self.progress(ledger, direct, mode, k, norm.isf)
                    previous_ledger = ledger
                if claimed:
                    self.require(planned['passed'] and all(x['passed'] for x in endpoints), 'Claimed quote violates nominal/first-step physical domain')
                    self.require(direct['feasible'], 'Claimed quote fails independent full-native probability/forward guards')
                    self.require(band_check['passed'], 'Claimed quote violates original historical P/Q band')
                    self.require(all(r['margin'] >= -1e-7 for r in progress), 'Claimed quote postpones required first-step progress')
                elif np.any(extra):
                    raise AssertionError('Failed/unresolved quote executed nonzero extra')
                if 'gate_outer_box_upper' in quote:
                    box_checks = gate.box_bounds(seal['snapshot'], shifted, mode, extra_lo, extra_hi, mapping*gains[None], direct)
                    self.same('independent_probability_outer_box', quote['gate_outer_box_upper'], [r['upper'] for r in box_checks], 1e-7)
                    if quote['status'] == 'infeasible_by_probability_box_bound':
                        self.require(any(r['impossible'] for r in box_checks), 'No independent negative outer-box certificate')
                cost_request = DT*np.sum(plus[k:]*np.maximum(service+extra, 0.)+minus[k:]*np.maximum(-service-extra, 0.))
                reservation = np.maximum(np.max(abs(service), 0), np.max(abs(service+extra), 0))
                objective = float(cost_request+cap_prices@(np.maximum(paid, reservation)-paid))
                encoded = bool(planned['passed'] and band_check['encoded_passed'] and
                    all(r['margin'] >= 1e-8 and r['forward_mean_gain'] >= 0. for r in direct['rows']) and
                    all(r['margin'] >= -1e-9 for r in progress))
                if claimed and 'cost_yuan' in quote: self.same('remaining_quote_objective', quote['cost_yuan'], objective, 1e-8)
                raw = quote.get('raw_native_bound_yuan'); lower = quote.get('lower_bound_yuan')
                conflict = bool(raw is not None and claimed and raw > objective+1e-7*max(1., objective))
                if conflict: self.require(lower is None or lower <= 0, 'Native lower bound contradicts returned feasible cost but was retained')
                certificate = dict(objective_yuan=objective if claimed else None, objective_kind='Requested activation plus nonrefundable additional daily capacity',
                    encoded_domain_feasible=encoded, raw_native_bound_yuan=raw, reported_lower_bound_yuan=lower,
                    bound_conflict=conflict, native_status=quote.get('native_status'), source=quote.get('source'),
                    native_runtime=quote.get('native_runtime'), initial=quote.get('initial', []),
                    trajectory_bound_history_available=False,
                    global_optimality_scope='No whole-day feedback optimum; raw QCQP numerical bounds retained separately')
            actual = service[0]+eta*extra[0]
            for key, value in [('service_first', service[0]), ('extra_request_first', extra[0]), ('extra_delivered_first', eta*extra[0]), ('request_first', service[0]+extra[0]), ('total_actual_first', actual)]:
                self.same('action_'+key, row[key], value, 1e-12)
            self.same('observation_action', obs['total_actual_delta'], actual, 1e-12)
            self.same('observation_extra_delivery', obs['extra_delivered'], eta*extra[0], 1e-12)
            prefix = np.vstack((prefix, actual)); physical = device_prefix(spec, resources, prefix)
            self.require(physical['passed'], 'Actual equipment prefix/remaining firm service violates limits')
            self.device_check(row['physical'], physical)
            baseline = ctx['physical']; start = 288+k
            self.same('baseline_P_current_only', obs['base_P'], baseline['P'][start:start+1], 0., rtol=0.)
            self.same('baseline_Q_current_only', obs['base_Q'], baseline['Q'][start:start+1], 0., rtol=0.)
            delta = actual@mapping.T
            self.same('clean_net_P', obs['clean_P'], obs['base_P']+delta[:n], 1e-13)
            self.same('clean_net_Q', obs['clean_Q'], obs['base_Q']+delta[n:], 1e-13)
            self.same('physical_root', obs['clean_root'], baseline['root'][start:start+1], 0., rtol=0.)
            for j, key in enumerate(('P', 'Q', 'V', 'root')):
                sd = entry['pq_relative_sd'] if j < 2 else .0005
                self.same('meter_'+key, obs[key], obs['clean_'+key]*(1+sd*random['noise'+str(j)][k:k+1]), 1e-13)
            self.same('Y_squared_voltage', obs['Y'], obs['root'][:, None]**2-obs['V']**2, 1e-13)
            P.append(obs['P'][0]); Q.append(obs['Q'][0]); Y.append(obs['Y'][0])
            after = full_scores()['snapshots'][-1]; self.snapshot(row['snapshot_after'], after, bank)
            target = seal['snapshot']['native_target']; opponent = seal['snapshot']['competitor_native']
            self.require(seal['monitor']['pair'] == (None if opponent is None else [target, opponent]), 'Monitoring pair was not fixed before observation')
            monitored = None if opponent is None else monitor_values(bank['H'], bank['means'], bank['sigma2'], ctx['covariance'], (target, opponent), np.r_[P[-1], Q[-1]], Y[-1])
            if monitored is not None:
                for key in ('actual_new_logodds', 'mean_under_each_native', 'variance', 'interval_lower', 'interval_upper', 'actual_minus_target_mean'):
                    self.same('monitor_'+key, row['feedback_report'][key], monitored[key], 1e-8)
                for key in ('outside_union', 'explained_by_native', 'target_mean_explains', 'competitor_mean_explains'):
                    self.require(row['feedback_report'][key] == monitored[key], 'Monitor union/identity differs: '+key)
            next_feedback = feedback_transition(feedback, monitored, changed=target != after['native_target'], confirmation=round(config['confirmation_hours']/DT), recovery=round(config['recovery_hours']/DT), caution=config['caution_fraction'])
            for key in ('state', 'outside_streak', 'recovery_streak'):
                self.require(seal['feedback'][key] == feedback[key] and row['feedback_after'][key] == next_feedback[key], 'Feedback order/window differs')
            feedback = next_feedback
            for device in spec['devices']:
                did = device['id']; cols = [j for j, r in enumerate(resources) if r['physical_device_id'] == did]
                scales = np.asarray([resources[j]['contract_kw'] or resources[j]['capacity_kvar'] or 1. for j in cols])
                x, y = extra[0, cols]/scales, (eta*extra[0])[cols]/scales
                numer[did] = numer.get(did, 0.)+float(x@y); denom[did] = denom.get(did, 0.)+float(x@x)
                if denom[did] > 1e-16: gains[cols] = np.clip(numer[did]/denom[did], config['minimum_delivery_gain'], 1.)
            self.same('gain_after_observable_telemetry', row['gain_after'], gains, 1e-12)
            self.same('paid_capacity_before', seal['paid_capacity_before'], paid, 1e-12)
            reserve = np.maximum(abs(service).max(0), abs(plan['request_tail']).max(0))
            request_rows.append(plan['request_tail'][0]); tail_peaks.append(reserve)
            costs = cumulative_fees(request_rows, prefix, plus[:k+1], minus[:k+1], cap_prices, reserved_tail_peaks=tail_peaks)
            paid = costs['purchased_peak'][-1]
            for key, value in [('activation_yuan', costs['activation_by_step'][-1]), ('availability_increment_yuan', costs['availability_by_step'][-1]), ('cumulative_activation_yuan', costs['cumulative_activation'][-1]), ('cumulative_availability_yuan', costs['cumulative_availability'][-1]), ('cumulative_service_yuan', costs['cumulative_total'][-1]), ('paid_capacity', paid)]:
                self.same('cost_'+key, row['cost'][key], value, 1e-9)
            ac = {key[3:]: value for key, value in obs.items() if key.startswith('ac_')}
            observed = dict(labels=bank['labels'], true_P=obs['clean_P'], true_Q=obs['clean_Q'], true_V=obs['clean_V'], true_root=obs['clean_root'], V=obs['V'], root=obs['root'])
            operational, _, _ = electrical.electrical(self, ac, observed, ctx['metadata'], ctx['branches'], ctx['buses'])
            clock = (8+(k+.5)*DT) % 24; tariff = spec['pricing']['grid_buy_yuan_per_kwh']
            rate = tariff['night' if clock < 7 else 'evening' if 17 <= clock < 22 else 'day']
            root_P = float(ac['root_supply_kw'][0]); bill = DT*rate*(max(root_P, 0.)-.9*max(-root_P, 0.)); root_bill += bill
            self.same('root_energy_step', row['cost']['root_energy_yuan'], bill, 1e-9)
            self.same('root_energy_cumulative', row['cost']['cumulative_root_energy_yuan'], root_bill, 1e-8)
            service_rows.append(service[0]); extra_rows.append(extra[0]); extra_delivery.append(eta*extra[0])
            units = np.asarray([r['action_unit'] for r in resources])
            def amounts(values):
                values = np.asarray(values)
                return dict(P_kwh=float(DT*abs(values[units == 'kW']).sum()),
                            Q_kvarh=float(DT*abs(values[units == 'kvar']).sum()))
            calculated = dict(id=entry['id'], policy=policy, repeat=repeat, step=k, audit_status='passed',
                before_target=target, after_target=after['native_target'], internal_correct=bool(ctx['internal_correct'][after['native_target']]),
                full_support_exact=bool(ctx['full_support_exact'][after['native_target']]), native_sse=after['sse'],
                profile_weights={g:float(w) for g,w in zip(after['profile']['families'], after['profile']['weights'])},
                monitor=monitored, feedback=feedback, fees=row['cost'], operational=operational,
                quote_status=quote['status'], new_gate_feasible=quote.get('feasible'),
                true_family_ids=ctx['true_family_ids'], target_switch=target != after['native_target'],
                quantities={name:amounts(value) for name,value in [('service',service[0]),
                    ('extra_requested',extra[0]),('extra_delivered',eta*extra[0]),
                    ('total_requested',service[0]+extra[0]),('total_actual',actual)]},
                quote_seconds=quote.get('seconds'), round_budget_seconds=seal['round_budget_seconds'],
                round_planning_seconds=row.get('round_planning_seconds'),
                round_wall_seconds=row['round_wall_seconds'], cumulative_compute_seconds=row['cumulative_compute_seconds'],
                compute_budget_overrun=row['compute_budget_overrun'], quote_exception=row.get('quote_exception'),
                optimizer_invoked=quote.get('optimizer_invoked', False), gate=direct, progress=progress,
                quote_stages=quote_stages(quote),
                planned_band=band_check, delivered_band=self.band(actual[None], ctx['bands'], mapping, entry['base_kw'], k),
                certificate=certificate,
                probability_box_checks=box_checks,
                source_receipt=ledger_row['receipt'], physical_service_assumption=True)
            trajectory_rows.append(calculated); self.rows.append(calculated)
            previous_seal, previous_receipt = row['seal'], ledger_row['receipt']
        self.require(complete['expected_steps'] == 96 and complete['completed_steps'] == len(trajectory_rows), 'Trajectory denominator differs')
        if complete['status'] == 'complete':
            self.require(len(trajectory_rows) == 96, 'Complete trajectory must retain all 96 quarters')
            final = arrays(self.checked(complete['arrays']))
            for key, value in [('P', P), ('Q', Q), ('Y', Y), ('actual_delta', prefix), ('service', service_rows), ('extra_request', extra_rows), ('extra_delivered', extra_delivery)]:
                self.same('final_archive_'+key, final[key], value, 0., rtol=0.)
            self.snapshot(complete['final_snapshot'], full_scores()['snapshots'][-1], bank)
            for key, value in [('activation_yuan', costs['cumulative_activation'][-1]), ('availability_yuan', costs['cumulative_availability'][-1]), ('service_yuan', costs['cumulative_total'][-1]), ('root_energy_yuan', root_bill)]:
                self.same('final_'+key, complete[key], value, 1e-8)
            self.require(complete['evaluation']['native_map_correct'] == ctx['internal_correct'][trajectory_rows[-1]['after_target']], 'Final native correctness differs')
            self.require(complete['evaluation']['native_full_support_exact'] == ctx['full_support_exact'][trajectory_rows[-1]['after_target']], 'Final full-support correctness differs')
            if 'profile_true_weight' in complete['evaluation']:
                profile = full_scores()['snapshots'][-1]['profile']; true_families = {bank['topology_ids'][i] for i,v in enumerate(ctx['internal_correct']) if v}
                true_weight = sum(w for g,w in zip(profile['families'],profile['weights']) if g in true_families)
                self.same('final_profile_true_weight', complete['evaluation']['profile_true_weight'], true_weight, 1e-12)
        else:
            self.require(complete.get('failure') is not None, 'Failed trajectory lacks failure reason')
        record = dict(id=entry['id'], policy=policy, repeat=repeat, source=file_ref(directory/'complete.json'),
            status=complete['status'], audited_steps=len(trajectory_rows), expected_steps=96,
            missing_steps=96-len(trajectory_rows), final=complete,
            true_family_ids=ctx['true_family_ids'],
            internal_truth_covered=any(ctx['internal_correct']), full_support_truth_covered=any(ctx['full_support_exact']))
        self.trajectories.append(record)


def information_boundary(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_trajectory')
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
    seals = [n.lineno for n in calls if ast.unparse(n.func) == 'io.write' and n.args and "'seal.json'" in ast.unparse(n.args[0])]
    evaluator = [n.lineno for n in calls if ast.unparse(n.func) == 'Evaluator']
    steps = [n.lineno for n in calls if ast.unparse(n.func) == 'evaluator.step']
    if not len(seals) == len(evaluator) == len(steps) == 1 or not seals[0] < evaluator[0] < steps[0]:
        raise AssertionError('No verified plan-seal-before-evaluator source boundary')
    for node in ast.walk(function):
        if isinstance(node, ast.Attribute) and node.attr in ('_eta', '_noise', '_ctx'):
            raise AssertionError('Runner accesses private future evaluator data')
    return dict(seal_line=seals[0], evaluator_creation_line=evaluator[0], evaluator_step_line=steps[0],
                scope='Static source ordering plus saved hash chains; not an external trusted timestamp service')


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve completed and failed audit outputs')
    output.mkdir(parents=True); shutil.copyfile(__file__, output/'source_snapshot.py')
    audit = Audit(); protocol = read_json(audit.file(source/'protocol.json'))
    complete = read_json(audit.file(source/'complete.json'))
    entries = read_json(audit.checked(protocol['inputs']))
    if 'source_completion' in protocol: audit.checked(protocol['source_completion'])
    for ref in protocol['sources']:
        snapshot = source/'source_snapshot'/Path(ref['path']).resolve().relative_to(ROOT)
        audit.require(file_ref(audit.file(snapshot))['sha256'] == ref['sha256'], 'Runtime snapshot changed')
    boundary = information_boundary(source/'source_snapshot'/Path(__file__).relative_to(ROOT).with_name('run.py'))
    write(output/'protocol.json', dict(source=file_ref(__file__), source_protocol=file_ref(source/'protocol.json'),
        planning_boundary=boundary, source_helpers=[file_ref(sys.modules[PairIntegral.__module__].__file__), file_ref(electrical.__file__)],
        scope='Independent saved-array audit only; no fitting, optimization or AC execution'))
    for entry in entries:
        try: ctx = audit.context(entry, source)
        except Exception as exc:
            audit.errors.append(dict(id=entry['id'], stage='input', error=repr(exc), traceback=traceback.format_exc())); continue
        for repeat in range(protocol['repeats']):
            for policy in protocol['policies']:
                directory = source/'trajectories'/entry['id']/f'path{repeat:02d}'/policy
                try: audit.trajectory(directory, entry, policy, repeat, protocol, ctx)
                except Exception as exc:
                    audit.errors.append(dict(id=entry['id'], repeat=repeat, policy=policy, error=repr(exc), traceback=traceback.format_exc()))
                write(output/'progress.json', dict(audited_trajectories=len(audit.trajectories), audited_rounds=len(audit.rows), errors=len(audit.errors)))
    for path, ref in audit.sources.items():
        if file_ref(path) != ref: audit.errors.append(dict(source=path, error='Input changed during audit'))
    audited = sum(t['audited_steps'] for t in audit.trajectories)
    missing = sum(t['missing_steps'] for t in audit.trajectories)
    passed = (not audit.errors and len(audit.trajectories) == protocol['expected_trajectories']
              and audited == complete['completed_rounds'] and audited+missing == protocol['expected_rounds'])
    write(output/'records.json', audit.rows)
    result = dict(status='passed' if passed else 'failed', source_protocol=file_ref(source/'protocol.json'),
        expected_trajectories=protocol['expected_trajectories'], audited_trajectories=len(audit.trajectories),
        expected_rounds=protocol['expected_rounds'], audited_rounds=audited, missing_rounds=missing,
        original_failures=complete.get('failures', []), errors=audit.errors, maxima=audit.maxima,
        quote_status_counts=dict(Counter(row['quote_status'] for row in audit.rows)),
        claimed_gate_feasible_steps=sum(row['new_gate_feasible'] is True for row in audit.rows),
        native_optimizer_invocations=sum(bool(stage['optimizer_invoked']) for row in audit.rows for stage in row['quote_stages']),
        quote_exceptions=sum(t['final'].get('quote_exceptions', 0) for t in audit.trajectories),
        trajectories=audit.trajectories, records=file_ref(output/'records.json'),
        source_inputs=list(audit.sources.values()),
        scopes=dict(profile_weights='Family minimum after native cumulative SSE; no integrated Bayesian claim',
                    probability='Conditional remaining-plan declared model only', firm_service='New explicit exact local service tracking assumption',
                    device_telemetry='Exact delivered extra permits identifying a device gain after nonzero dispatch',
                    thermal_ratings='Unavailable; current, root apparent power and voltage are measured diagnostics'))
    write(output/'report.json', result); write(output/'complete.json', dict(report=file_ref(output/'report.json')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run, args.out)
    print({k: result[k] for k in ('status', 'audited_trajectories', 'audited_rounds', 'missing_rounds')})


if __name__ == '__main__':
    main()
