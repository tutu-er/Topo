"""Observed-only WZZT/AC/G-SQ planning on public reference-network units.

Each candidate keeps one historical R/X vector throughout planning. Future
terminal P/Q and source P/Q are signed forecasts from arrived observations,
with separately development-frozen P and Q forecasting configurations. MILPs
optimize continuous powers and a time-expanded route; whole routes/power
combinations are not enumerated. The complete AC forecast score is recorded
beside the MILP proxy, without using realized future loads or noise.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from time import perf_counter
from collections.abc import Mapping

import numpy as np

from .. import likelihood as lk
from .. import ev_mobile_v4_domain as nested
from ..ev_mobile_service_domain import MobileServiceConfig
from ..ev_bayes_small_study import complete_curve
from ..g_precision.context import FixedResponses, sigmas
from ..g_precision.coefficients import build_tasks_from_responses
from ..g_precision.study import precision_task
from ..block_joint_margin.study import _plan_record
from ..full_pipeline import passive, forecast, solver
from ..full_pipeline.study import _family, _random_plan
from . import inference


_OBSERVED_KEYS = frozenset(('history', 'history_meter', 'future_meter', 'timestamps',
                          'horizon', 'base_kw', 'public_parameters', 'network'))
_PUBLIC_KEYS = frozenset(('base_kw', 'base_kv', 'base_mva', 'observation_bus_ids',
                         'site_bus_ids', 'source_interface', 'source_bus', 'units',
                         'site_interpretation', 'travel_interpretation', 'case_id'))


def _validate_observed(observed, config):
    if not isinstance(observed, Mapping) or set(observed)-_OBSERVED_KEYS:
        raise ValueError('selector accepts only the observed-history allowlist')
    for key in ('history', 'history_meter', 'future_meter', 'timestamps'):
        if key not in observed:
            raise ValueError('missing observed field '+key)
    for key in ('public_parameters', 'network'):
        if key in observed and (not isinstance(observed[key], Mapping)
                or set(observed[key])-_PUBLIC_KEYS):
            raise ValueError(key+' may contain public units/site labels, never a true physical tree')
    D = observed['history']
    base_kw = observed.get('base_kw', observed.get('public_parameters', {}).get('base_kw', config.get('base_kw')))
    if base_kw is None or not np.isfinite(base_kw) or base_kw <= 0:
        raise ValueError('a positive public three-phase base_kw is required')
    if observed.get('horizon', config['T']) != config['T']:
        raise ValueError('observed horizon differs from the configured planning horizon')
    if D.n < 2 or len(observed['timestamps']) != D.t:
        raise ValueError('matching history timestamps and at least two measured nodes required')
    return float(base_kw)


def signed_forecast(history, meter, states, timestamps, *, horizon, base_kw, forecast_configs,
                    source_p_bounds, source_q_bounds):
    """Point forecasts only; no simulator, truth, source class or future values."""
    if isinstance(forecast_configs, Mapping) and set(forecast_configs) == {'p', 'q'}:
        forecast_configs = {key.upper():value for key,value in forecast_configs.items()}
    if not isinstance(forecast_configs, Mapping) or set(forecast_configs) != {'P', 'Q'}:
        raise ValueError('separate development-frozen P and Q forecast configurations required')
    configs = {key:forecast.ForecastConfig(**forecast_configs[key]) for key in ('P', 'Q')}
    if any(c.dt_minutes != 15 or c.seasonal_slots != 96 for c in configs.values()):
        raise ValueError('this benchmark uses 15-minute clock and 96-slot daily seasonality')
    p = history.p-lk._array(meter.bias_p, (history.t, history.n), 'bias_p')
    q = history.q-lk._array(meter.bias_q, (history.t, history.n), 'bias_q')
    values = {}; records = {}
    for channel, key, observed_values in (('p', 'P', p), ('q', 'Q', q)):
        result = forecast.forecast_series(observed_values, horizon, timestamps=timestamps,
                                           config=configs[key], nonnegative=False)
        values[channel] = np.asarray(result['values'], float)
        records[key] = result['metadata']
    sources = {}; source_metadata = {}; clips = {}
    for name, state in states.items():
        sources[name] = {}; source_metadata[name] = {}; clips[name] = {}
        for channel, key, bounds in (('source_p', 'P', source_p_bounds), ('source_q', 'Q', source_q_bounds)):
            result = forecast.forecast_series(np.asarray(state[channel], float), horizon,
                timestamps=timestamps, config=configs[key], nonnegative=False)
            before = np.asarray(result['values'], float)
            after = np.clip(before, *bounds)
            sources[name][channel] = after
            source_metadata[name][channel] = result['metadata']
            clips[name][channel] = int(np.count_nonzero(after != before))
    return dict(p=values['p'], q=values['q'],
        root_v=np.full(horizon, float(history.root_v[-1])), source_by_candidate=sources,
        metadata=dict(observed_only=True, truth_used=False, future_used=False,
            future_regime_used=False, source_class_used=False,
            input_units='pu; base_kw converts EV kW to pu', base_kw=float(base_kw),
            P=records['P'], Q=records['Q'], source=source_metadata,
            forecast_configs={key:asdict(c) for key,c in configs.items()},
            signed_terminal_forecasts=True, signed_source_forecasts=True,
            source_domain_clips=clips, source_interface=1,
            root_rule='last arrived calibrated root voltage',
            covariance='point forecasts; no line or future-load covariance propagation'))


class Responses(FixedResponses):
    """Complete AC response/gradient context, with fixed historical lines."""
    def __init__(self, bank, D, dm, fm, fitted, timestamps, config, forecast_configs):
        began = perf_counter(); self.bank = tuple(bank); self.base_kw = config['base_kw']
        if not fitted.complete:
            raise ValueError('historical failure retained; every competitor is required')
        self.states = {c.name:dict(r=fitted.fits[c.name].r.copy(), x=fitted.fits[c.name].x.copy(),
            source_p=fitted.fits[c.name].source_p.copy(), source_q=fitted.fits[c.name].source_q.copy()) for c in bank}
        self.forecast = signed_forecast(D, dm, self.states, timestamps, horizon=config['T'],
            base_kw=self.base_kw, forecast_configs=forecast_configs,
            source_p_bounds=fitted.options['source_p_bounds'], source_q_bounds=fitted.options['source_q_bounds'])
        self.shape = (config['T'], D.n); self.sigma = sigmas(fm, *self.shape)
        self.voltage_bounds = np.asarray(fitted.options['voltage_bounds'])
        self.cache = {}; self.ac_calls = 0; self.ac_states = 0; self.derivative_calls = 0
        self.zero = self.responses(np.zeros(self.shape), derivatives=True)
        self.seconds = perf_counter()-began


def operations_config(config, n_sites):
    """Physical EV task, independent of true topology or realized future data."""
    values = dict(n_slots=config['T'], n_sites=n_sites, dt_hours=config['dt_hours'],
        p_charge_max_kw=11., p_discharge_max_kw=11., allow_discharge=True,
        battery_capacity_kwh=60., initial_soc=.4, terminal_energy_kwh=30.,
        eta_charge=.95, eta_discharge=.95, initial_site=0, initial_connected=True,
        mobile=True, access=True, travel_slots=1, setup_slots=1, travel_energy_kwh=.2,
        max_moves=1, max_throughput_kwh=15., travel_cost=15., setup_cost=15.,
        driver_budget=30., cost_units='worker_minutes', min_arrival_throughput_kwh=.25)
    # ``ev`` is the current public protocol field. Keep the earlier explicit
    # alias only when it agrees, so a silently ignored power/SOC change cannot
    # alter the meaning of a frozen experiment.
    legacy = dict(config.get('ev_operations', {}))
    current = dict(config.get('ev', {}))
    for key in set(legacy)&set(current):
        if not np.array_equal(np.asarray(legacy[key]), np.asarray(current[key])):
            raise ValueError('conflicting ev and ev_operations field: '+key)
    overrides = dict(legacy, **current)
    if any(key in overrides and overrides[key] != values[key] for key in ('n_slots', 'n_sites', 'dt_hours')):
        raise ValueError('EV dimensions/clock must match the public observation dimensions')
    values.update(overrides)
    return MobileServiceConfig(**values)


def select(observed, fit_options, config, forecast_configs, seed, report=lambda *x:None):
    """Return planning, candidates, historical fits, and all four nested domains."""
    began = perf_counter(); base_kw = _validate_observed(observed, config)
    local = dict(config, base_kw=base_kw)
    D, dm, fm = observed['history'], observed['history_meter'], observed['future_meter']
    # Public line limits are coefficient limits for proposals only, not extra
    # likelihood or a known physical X/R relation imposed on AC fits.
    passive_options = dict(max_candidates=12, bootstrap_replicates=3, block_length=4,
                           kappa=.5, wzzt_time_limit=5., max_repairs=12)
    passive_options.update(config.get('passive', {}))
    upper = max(float(np.max(np.asarray(fit_options[key][1], float))) for key in ('r_bounds', 'x_bounds'))
    passive_options['coefficient_bound'] = 2*upper
    bank, pre = passive.propose(D, dm, passive_options, seed=seed)
    pre['public_coefficient_bound_rule'] = '2*max(public_r_upper,public_x_upper)'
    pre['public_coefficient_bound_value'] = 2*upper
    pre['AC_ratio_inherited_from_proposal'] = False
    report('passive', dict(candidates=len(bank), failures=len(pre['failures'])))
    history = inference.fit_history(bank, D, dm, fit_options, seed=seed+1)
    report('history_fit', dict(complete=history.complete, failures=history.failures))
    template = operations_config(local, D.n)
    domains = nested.build_nested_domains(template)
    shape = np.asarray(domains['mobile_bidir'].indices.u).shape
    service = template.terminal_energy_kwh-template.initial_soc*template.battery_capacity_kwh
    if service <= 0:
        raise ValueError('CC/HL reference charging task requires positive battery service')
    constant = np.zeros(shape)
    constant[:, template.initial_site] = service/(template.eta_charge*config['T']*config['dt_hours'])
    first = config['T']//2
    if first < 1:
        raise ValueError('at least two planning slots required for the high-low baseline')
    highlow = np.zeros(shape)
    highlow[:first, template.initial_site] = service/(template.eta_charge*first*config['dt_hours'])
    sb = perf_counter(); baselines = {}
    for label, power in (('CC', constant), ('HL', highlow)):
        x = complete_curve(domains['fixed_charge'], power)
        if not domains['fixed_charge'].check(x)['feasible']:
            x, repair = solver.repair_operation(domains['fixed_charge'], x)
            if x is None:
                raise RuntimeError(label+' failed the common EV task: '+str(repair))
        baselines[label] = x
    plans = {name:_plan_record('fixed_charge', domains['fixed_charge'], x) for name,x in baselines.items()}
    zero = np.zeros(shape)
    plans['OBS'] = dict(domain='passive', power=zero, action_key=nested.action_key(zero), primary=True)
    cost = dict(passive_pre_seconds=pre['seconds'], history_seconds=history.seconds,
                baseline_completion_seconds=perf_counter()-sb)
    sr = perf_counter()
    parents = {key:nested.lift_operation(domains['fixed_charge'], domains['mobile_bidir'], x)
               for key,x in baselines.items()}
    random, random_log = _random_plan(domains['mobile_bidir'], parents, seed+2, local)
    plans['RAND'] = _plan_record('mobile_bidir', domains['mobile_bidir'], random)
    cost['random_seconds'] = perf_counter()-sr
    squares, rows, seconds = _family(domains, [precision_task(fm, shape)], baselines, local)
    families = {'P-SQ':rows}; cost['P-SQ_seconds'] = seconds
    for suffix, domain in (('', 'mobile_bidir'), ('/F±', 'fixed_bidir')):
        plans['P-SQ'+suffix] = _plan_record(domain, domains[domain], squares[domain])
    result = dict(plans=plans, passive_pre=pre, history=history.record(),
        domain_config=asdict(template), random=random_log, families=families, cost=cost,
        tasks={}, nominal_AC={}, failures={}, selector_uses_truth=False,
        selector_uses_future_noise=False, selector_uses_true_future_load=False,
        planner_lines_fixed_per_candidate=True, base_kw=base_kw,
        site_interpretation='all observed load nodes; MV sites are downstream load-equivalent connections',
        travel_interpretation='15-minute travel plus 15-minute setup scenario, not measured road times',
        forecast_acceptance='record all feasible complete-AC scores; no forced full-AC objective improvement')
    try:
        if len(bank) < 2:
            raise ValueError('only one candidate, no pairwise planning objective')
        context = Responses(bank, D, dm, fm, history, observed['timestamps'], local, forecast_configs)
        cost['response_setup_seconds'] = context.seconds
        for method, intercept in (('G-SQ', False), ('G-AQ', True)):
            tasks = build_tasks_from_responses(context.zero, context.sigma, include_intercept=intercept)
            family, logs, seconds = _family(domains, tasks, baselines, local, extra=squares)
            families[method] = logs; cost[method+'_seconds'] = seconds
            result['tasks'][method] = [task.record() for task in tasks]
            for suffix, domain in (('', 'mobile_bidir'), ('/F±', 'fixed_bidir')):
                name = method+suffix; plan = _plan_record(domain, domains[domain], family[domain])
                try:
                    context.scores(plan['power'])
                except Exception as exc:
                    result['failures'][name+'_forecast_physical'] = repr(exc)
                    plan = deepcopy(plans['P-SQ'+suffix])
                    plan['fallback_reason'] = 'complete-AC forecast physical failure'
                plans[name] = plan
            report(method, dict(seconds=seconds))
        for name, plan in plans.items():
            try:
                result['nominal_AC'][name] = context.scores(plan['power'])
            except Exception as exc:
                result['nominal_AC'][name] = dict(failure=repr(exc))
        result['context'] = context.record()
    except Exception as exc:
        result['failures']['response_or_G_planning'] = repr(exc)
    for method in ('G-SQ', 'G-AQ'):
        for suffix in ('', '/F±'):
            name = method+suffix
            if name not in plans:
                plans[name] = deepcopy(plans['P-SQ'+suffix])
                plans[name]['fallback_reason'] = 'retained same-domain P-SQ after G failure'
    for plan in plans.values():
        plan['primary'] = True
    cost['selection_total_seconds'] = perf_counter()-began
    return result, bank, history, domains


__all__ = ['Responses', 'signed_forecast', 'operations_config', 'select']
