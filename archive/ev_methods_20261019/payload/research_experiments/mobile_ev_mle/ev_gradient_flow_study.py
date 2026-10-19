"""End-to-end EV policies: fixed-surrogate MILP and AC/I2 gradient MILPs.

Truth is used only by the simulator and evaluator. The planning function takes
training observations, fixed fitted lines, training forecasts and public meters.
All final decisions retain the complete four-topology candidate bank.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import traceback

import numpy as np
import scipy

from . import ev_dynamic_study as dynamic
from . import ev_fixed_train_study as old
from . import ev_fixed_train_design as fixed
from . import ev_rx_study as rx
from . import ev_rx_design as rx_design
from . import gradient_price_study as io
from . import ev_gradient_flow_design as gradient
from .candidates import propose
from .ev_dynamic_loads import fit_forecast
from .ev_original_domain_milp import exchange_original_domain
from .ev_rx_precision import solve_squared_precision
from .information_context import meter_to_record

HERE = Path(__file__).parent
DOMAINS = ('v2g_fixed', 'v2g_mobile')
PREFIXES = ('square', 'nearest', 'gradac', 'gradi2', 'pairac')
METHODS = ('none', 'constant', 'highlow', *(f'{p}_{d}' for d in DOMAINS for p in PREFIXES))
CONFIG = deepcopy(rx.CONFIG)
CONFIG.update(version='end_to_end_gradient_EV_v2_canonical_square',
    select_modes=[m for m in METHODS if m != 'highlow'],
    mean_policy_ablation=False, diagnostics=False, domains=list(DOMAINS),
    planning_seconds=35., gradient=dict(gradient.DEFAULT_OPTIONS),
    development_seeds=list(range(993101, 993105)),
    calibration_seeds=list(range(993301, 993337)),
    evaluation_seeds=list(range(993201, 993217)),
    budget_scope='standalone soft planning budget includes initialization, dependencies and AC acceptance',
    truth_scope='simulator and final evaluator only; shortlist never receives true topology')


def validate_config(config):
    sets = [set(config[k]) for k in ('development_seeds', 'calibration_seeds', 'evaluation_seeds')]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('development/calibration/evaluation histories must be disjoint')
    if config['planning_seconds'] <= 0 or config['public_voltage_repeat_counts'] != [5]*6:
        raise ValueError('declared positive budget and uniform precision required')
    if tuple(config['domains']) != DOMAINS:
        raise ValueError('this experiment freezes two V2G domains')


def source_hashes():
    result = rx.source_hashes()
    for name in ('ev_fixed_response.py', 'ev_fixed_gradient.py', 'ev_fixed_gradient_math.md',
        'ev_gradient_flow_design.py', 'ev_gradient_flow_study.py',
        'ev_gradient_flow_math.md', 'ev_gradient_flow_protocol.md'):
        result[name] = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return result


def freeze(directory, config):
    validate_config(config)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    record = dict(config=config, source_sha256=sources, policy_id=io.identity(config, sources),
        base_commit='5c3fec5', environment=dict(python=platform.python_version(),
            numpy=np.__version__, scipy=scipy.__version__))
    path = directory/'freeze.json'
    if path.exists():
        if io.read_json(path)['policy_id'] != record['policy_id']:
            raise ValueError('frozen policy differs; preserve directory and use a new name')
        return io.read_json(path)
    io.write_json(path, record)
    snapshot = {name:base64.b64encode(((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/')
        else HERE/name).read_bytes()).decode() for name in sources}
    io.write_json(directory/'source_snapshot.json.gz', dict(source_sha256=sources,
        source_base64=snapshot), compressed=True)
    return record


def public_curves(T, n):
    constant = np.zeros((T, n))
    constant[:, 0] = (fixed.CONFIG['terminal_energy_kwh']-fixed.CONFIG['initial_energy_kwh'])/(
        fixed.CONFIG['eta_charge']*T*fixed.CONFIG['dt_hours'])
    highlow = dynamic.highlow_curve(T, n)
    # A topology-independent strong initial plan from the public energy task.
    # Its global squared-power value for this uniform six-slot domain was
    # proved in ev_rx_precision_proof.md. It is not an action-library search.
    H = fixed.CONFIG['throughput_kwh']; ec = fixed.CONFIG['eta_charge']; ed = fixed.CONFIG['eta_discharge']
    delta = fixed.CONFIG['terminal_energy_kwh']-fixed.CONFIG['initial_energy_kwh']
    C = (H+ed*delta)/(1+ec*ed); D = H-C
    dt = fixed.CONFIG['dt_hours']; pmax = fixed.CONFIG['command_power_kw']
    square = np.zeros((T, n)); slot = 0
    for energy, sign in ((C, 1.), (D, -1.)):
        while energy > 1e-10:
            if slot >= T:
                raise ValueError('public saturated plan needs more slots')
            power = min(pmax, energy/dt)
            square[slot, 0] = sign*power
            energy -= dt*power; slot += 1
    return dict(none=np.zeros_like(constant), constant=constant, highlow=highlow,
                analytic_square=square)


def _counts(counter):
    return {k:int(v) for k, v in counter.items() if isinstance(v, (int, np.integer))}


def _delta(counter, before):
    return {k:int(v)-before.get(k, 0) for k, v in _counts(counter).items()}


def _subset_forecast(forecast, names):
    answer = deepcopy(forecast)
    answer['source_by_candidate'] = {k:v for k, v in answer['source_by_candidate'].items() if k in names}
    answer['metadata']['planning_candidate_subset'] = sorted(names)
    return answer


def fixed_surrogate_policy(context, domain, initials, config, *, seed, seconds):
    """Original frozen-residual distance MILP, within the remaining soft budget."""
    began = perf_counter(); deadline = began+seconds
    checked, points, initial_records, failures = [], [], [], []
    engine = gradient.ProfileEngine(context, fit_options={**config['profile'], **config['source']}, seed=seed)

    def score(u):
        profile = engine.evaluate(u, 'ac', deadline=deadline)
        value = gradient._aggregate(profile['pairs'], 'nearest') if profile['success'] else {'score':None}
        return profile, value['score']

    for origin, u in initials.items():
        if u is None:
            continue
        if perf_counter() >= deadline:
            initial_records.append(dict(origin=origin, attempted=False, failure='planning budget exhausted'))
            continue
        realized = fixed._realize(domain, u)
        row = dict(origin=origin, operation=realized['audit'], attempted=True)
        if realized['success']:
            points.append(realized['x'])
            pred, value = score(u)
            row.update(AC_score=value, prediction=pred)
            if value is not None:
                checked.append((value, origin, np.asarray(u).copy(), pred))
        initial_records.append(row)
    record = dict(initial_candidates=initial_records, failures=failures,
        aggregation='nearest', global_AC_gap=None, all_AC_tasks_required=True)
    remaining = deadline-perf_counter()
    if checked and remaining > .05:
        options = deepcopy(config['exchange'])
        options['total_seconds'] = max(.01, remaining-min(3., remaining/3))
        options['solver_seconds'] = min(options['solver_seconds'], options['total_seconds'])
        op = domain.as_dict()
        groups = {m['name']:m['a'] for m in context['models']}
        try:
            raw = exchange_original_domain(context['pairs'], op['control_lower'], op['control_upper'],
                operation=op, offsets=np.zeros(len(groups)), weights=np.full(len(groups), 1/len(groups)),
                initial_operation_points=points, aggregation='nearest', task_groups=groups, **options)
            record['raw'] = raw
            if raw.get('u') is not None and perf_counter() < deadline:
                u = np.asarray(raw['u']).reshape(context['control_shape'])
                realized = fixed._realize(domain, u); record['raw_operation'] = realized['audit']
                if realized['success']:
                    pred, value = score(u); record['raw_profile'] = pred
                    if value is not None:
                        checked.append((value, 'MILP', u.copy(), pred))
            elif raw.get('u') is None:
                failures.append('no MILP incumbent; initialized plan retained')
            else:
                failures.append('budget before raw complete-AC acceptance; initialized plan retained')
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures.append(type(exc).__name__+':'+str(exc))
    if not checked:
        record.update(success=False, seconds=perf_counter()-began, failure='no complete feasible source fit',
                      counters=engine.counters())
        return None, record
    best = max(checked, key=lambda row:row[0])
    record.update(success=True, selected_score=best[0], selected_origin=best[1],
        selected_is_new_MILP_proposal=best[1]=='MILP', selected_prediction=best[3],
        seconds=perf_counter()-began, counters=engine.counters())
    return best[2], record


def select_plans(training, bank, fitted, training_meter, future_meter, forecast, config, *, seed, counter):
    """Planning boundary: no simulator case, true family or heldout meters."""
    prepared = perf_counter()
    T, n = config['horizon'], training.n
    original_lines = {c.name:fixed._line_id(fitted[c.name].r, fitted[c.name].x) for c in bank}
    curves = public_curves(T, n)
    plans = {k:curves[k] for k in ('none', 'constant', 'highlow')}
    metadata = dict(methods={}, timing={}, failures={}, one_trained_parameter_vector=True,
        configuration=deepcopy(fixed.CONFIG), line_parameters_unchanged=True,
        uses_truth=False, uses_validation=False, uses_future_noise=False, random_used_by_primary=False)
    budget = config['planning_seconds']; domains = {d:fixed._service(T, n,
        mobile=d.endswith('mobile'), discharge=True) for d in DOMAINS}
    top2 = sorted(bank, key=lambda c:(float(fitted[c.name].loss), c.name))[:2]
    metadata['training_top2_names'] = [c.name for c in top2]
    setup_seconds = perf_counter()-prepared
    metadata['public_setup_seconds'] = setup_seconds
    for name in ('none', 'constant', 'highlow'):
        metadata['timing'][name] = dict(planning_actual_seconds=0., planning_standalone_seconds=0.,
            dependency_seconds=0., soft_budget_overrun_seconds=0., actual_calls={})
    for d, domain in domains.items():
        name = 'square_'+d; before = _counts(counter); began = perf_counter()
        initial = dict(constant=curves['constant'], highlow=curves['highlow'], analytic_square=curves['analytic_square'])
        dep = setup_seconds
        if d == 'v2g_mobile' and plans.get('square_v2g_fixed') is not None:
            initial['fixed_parent'] = plans['square_v2g_fixed']
            dep += metadata['timing']['square_v2g_fixed']['planning_actual_seconds']
        options = deepcopy(config['exchange'])
        options['total_seconds'] = max(.01, budget-dep)
        try:
            row = solve_squared_precision(domain, future_meter, initial, exchange_options=options)
            # For the declared uniform six-slot device, the public saturated
            # curve attains the independently proved global Q*. Keep it at
            # numerical Q ties, rather than choosing a temporal permutation
            # because feasibility roundoff changes Q by about 1e-7.
            canonical = curves['analytic_square'].copy()
            qstar = float(np.sum(canonical**2))
            raw_q = row.get('Q')
            row.update(raw_power_kw=row.get('power_kw'), raw_Q=raw_q,
                selected_power_kw=canonical, selected_Q=qstar,
                tie_break='public analytic saturated curve at the proved uniform-domain Q optimum',
                raw_above_proved_optimum=bool(raw_q is not None and raw_q > qstar+1e-5),
                proof_reference='ev_rx_precision_proof.md; same six-slot uniform-precision device')
            plans[name] = canonical; metadata['methods'][name] = row
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            plans[name] = curves['analytic_square'].copy()
            metadata['methods'][name] = dict(success=False, fallback='public_analytic_square', failure=str(exc))
        own = perf_counter()-began
        metadata['timing'][name] = dict(planning_actual_seconds=own, planning_standalone_seconds=own+dep,
            dependency_seconds=dep, dependencies=['square_v2g_fixed'] if d == 'v2g_mobile' else [],
            soft_budget_overrun_seconds=max(0., own+dep-budget), actual_calls=_delta(counter, before))
    contexts, context_costs, context_calls = {}, {}, {}
    for key, selected_bank in (('all', bank), ('pair', top2)):
        before = _counts(counter); began = perf_counter()
        fore = forecast if key == 'all' else _subset_forecast(forecast, [c.name for c in selected_bank])
        contexts[key] = fixed.build_fixed_contexts(training, selected_bank, fitted, training_meter, future_meter,
            -fixed.CONFIG['command_power_kw'], fixed.CONFIG['command_power_kw'], curves['constant'],
            forecast=fore, **config['source'])
        context_costs[key] = perf_counter()-began; context_calls[key] = _delta(counter, before)
    metadata['contexts'] = {k:v['diagnostics'] for k, v in contexts.items()}
    for prefix in ('nearest', 'gradac', 'gradi2', 'pairac'):
        for d, domain in domains.items():
            name = prefix+'_'+d; ctx_key = 'pair' if prefix == 'pairac' else 'all'
            context = contexts[ctx_key]; before = _counts(counter); began = perf_counter()
            initials = dict(constant=curves['constant'], highlow=curves['highlow'], square=plans.get('square_'+d))
            dependencies = {'square_'+d}
            if d == 'v2g_mobile' and plans.get(prefix+'_v2g_fixed') is not None:
                initials['fixed_parent'] = plans[prefix+'_v2g_fixed']
                dependencies.add(prefix+'_v2g_fixed')
            # Charge shared ancestors once, even when both the direct square
            # initial and a fixed parent depend on the fixed square plan.
            paid = set()
            def dependency_cost(dep_name):
                if dep_name in paid:
                    return 0.
                paid.add(dep_name)
                timing = metadata['timing'][dep_name]
                return timing['planning_actual_seconds']+sum(dependency_cost(parent)
                    for parent in timing.get('dependencies', []))
            dep = sum(dependency_cost(parent) for parent in sorted(dependencies)) + context_costs[ctx_key] + setup_seconds
            # Parent methods paid their own context; same context can be built
            # once by a standalone mobile pipeline, so it is charged once here.
            remaining = budget-dep
            try:
                if not context['success']:
                    raise RuntimeError('fixed-line context failed')
                if remaining <= 0:
                    parent = initials.get('fixed_parent', initials.get('square'))
                    power = None if parent is None else np.asarray(parent).copy()
                    row = dict(success=power is not None, selected_origin='dependency_budget_fallback',
                        failures=['no remaining standalone planning budget'])
                elif prefix == 'nearest':
                    power, row = fixed_surrogate_policy(context, domain, initials, config,
                        seed=seed, seconds=remaining)
                else:
                    options = deepcopy(config['gradient']); options['total_seconds'] = remaining
                    power, row = gradient.select_gradient_controls(context, domain, initials,
                        method='i2' if prefix == 'gradi2' else 'ac',
                        aggregation='binarymean' if prefix == 'pairac' else 'nearest',
                        fit_options={**config['profile'], **config['source']}, options=options,
                        seed=seed, cache=None)
                plans[name] = power; metadata['methods'][name] = row
                if power is None:
                    fallback = row.get('fallback_power')
                    plans[name] = np.asarray(fallback).copy() if fallback is not None else plans.get('square_'+d)
                    row['executed_fallback'] = 'first_feasible' if fallback is not None else 'square'
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                plans[name] = plans.get('square_'+d)
                metadata['methods'][name] = dict(success=False, fallback='square', failure=type(exc).__name__+':'+str(exc))
            own = perf_counter()-began
            calls = _delta(counter, before)
            charged_calls = dict(calls)
            for charged in paid:
                for k, v in metadata['timing'][charged].get('actual_calls', {}).items():
                    charged_calls[k] = charged_calls.get(k, 0)+v
            for k, v in context_calls[ctx_key].items():
                charged_calls[k] = charged_calls.get(k, 0)+v
            metadata['timing'][name] = dict(planning_actual_seconds=own, planning_standalone_seconds=own+dep,
                dependency_seconds=dep, dependencies=sorted(dependencies), context_seconds=context_costs[ctx_key],
                context_calls=context_calls[ctx_key], soft_budget_overrun_seconds=max(0., own+dep-budget),
                actual_calls=calls, standalone_calls=charged_calls, charged_ancestors=sorted(paid))
    metadata['line_parameters_unchanged'] = all(original_lines[c.name] ==
        fixed._line_id(fitted[c.name].r, fitted[c.name].x) for c in bank)
    return plans, metadata


def _run_trial(seed, family_index, config, counter):
    began = perf_counter(); hashes = source_hashes()
    case = dynamic.make_case(seed, family_index, config)
    tr, tm, T = case['train'], case['tm'], config['horizon']
    started = perf_counter(); before = _counts(counter)
    bank, proposal = propose(tr, run_wzzt=True, wzzt_time_limit=.3)
    fitted, training = rx.train_bank(bank, tr, tm, config, case['assets']['seeds']['policy'])
    common_train_seconds = perf_counter()-started; common_train_calls = _delta(counter, before)
    true_name = next((b.name for b in bank if b.supports == case['assets']['candidate'].supports), None)
    record = dict(schema='ev_gradient_end_to_end_v1', seed=seed, family_index=family_index,
        config=config, source_sha256=hashes, policy_id=io.identity(config, hashes),
        training=io.observations_record(tr), training_meter=meter_to_record(tm),
        future_meter=meter_to_record(case['fm']), validation=io.observations_record(case['validation']),
        validation_meter=meter_to_record(case['vm']), training_result=training,
        candidates=[dict(name=b.name, supports=b.supports) for b in bank], proposal=proposal,
        true_name=true_name, candidate_miss=true_name is None, methods=[],
        truth=dict(r=case['assets']['r'], x=case['assets']['x'], supports=case['assets']['candidate'].supports),
        background=case['background'], seeds=case['assets']['seeds'])
    if not training['all_complete']:
        record.update(status='training_failure', seconds=perf_counter()-began)
        return record
    started = perf_counter(); before = _counts(counter)
    _, validated = old.fit_bank(bank, case['validation'], case['vm'], trained=fitted,
        options={**config['profile'], **config['source']}, seed=seed+100)
    validation_seconds = perf_counter()-started; validation_calls = _delta(counter, before)
    started = perf_counter()
    forecast = fit_forecast(tr, tm, fitted, horizon=T, gap=config['forecast_gap'], base_kw=20., kind='ar1')
    forecast_seconds = perf_counter()-started
    record.update(forecast=forecast, validation_result=validated)
    started = perf_counter()
    plans, selection = select_plans(tr, bank, fitted, tm, case['fm'], forecast, config,
        seed=seed+200, counter=counter)
    record['selection_seconds'] = perf_counter()-started
    names = selection['training_top2_names']
    record.update(selection=selection, training_top2=dict(names=names, miss_true=true_name not in names))
    cache = {}
    common_seconds = common_train_seconds+validation_seconds+forecast_seconds
    for name in METHODS:
        power = plans.get(name); row = dict(name=name, power_kw=power, status='selection_failure')
        row['planning_success'] = selection['methods'].get(name, {}).get('success', power is not None)
        if power is None:
            record['methods'].append(row); continue
        mobile = name.endswith('mobile')
        operation_began = perf_counter()
        operation = (dict(success=True, battery_service=False) if name == 'none'
            else dynamic.operation_record(power, mobile=mobile, discharge=name not in ('constant', 'highlow')))
        operation_seconds = perf_counter()-operation_began
        row['operation'] = operation
        if not operation['success']:
            row['status'] = 'operation_failure'; record['methods'].append(row); continue
        key = fixed._control_key(power)
        hit = key in cache
        if not hit:
            started = perf_counter(); before = _counts(counter)
            try:
                value = dynamic.actual_curve(case, bank, fitted, validated, power, config, seed=seed+500)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                value = dict(status='physical_or_fit_failure', M=None, failure=str(exc))
            value['actual_evaluation_seconds'] = perf_counter()-started
            value['actual_evaluation_calls'] = _delta(counter, before)
            cache[key] = value
        row.update(cache[key]); row['actual_fit_cache_hit'] = hit
        timing = selection['timing'][name]
        future_fit = float(row.get('noisy', {}).get('seconds', 0.))
        row['runtime'] = dict(common_history_pipeline_seconds=common_seconds,
            operational_future_fit_seconds=future_fit, **timing,
            final_operation_audit_seconds=operation_seconds,
            end_to_end_standalone_seconds=common_seconds+timing['planning_standalone_seconds']+future_fit+operation_seconds,
            standalone_AC_batches=common_train_calls.get('AC_batches', 0)+validation_calls.get('AC_batches', 0)
                +timing.get('standalone_calls', timing.get('actual_calls', {})).get('AC_batches', 0)
                +row.get('noisy', {}).get('cost', {}).get('ac_calls', 0),
            simulation_and_clean_diagnostic_seconds=max(0., row['actual_evaluation_seconds']-future_fit))
        record['methods'].append(row)
    record['common_runtime'] = dict(training_and_candidates_seconds=common_train_seconds,
        validation_seconds=validation_seconds, forecast_seconds=forecast_seconds,
        training_calls=common_train_calls, validation_calls=validation_calls)
    if true_name is not None:
        f = fitted[true_name]; a = case['assets']
        record['training_parameter_error'] = dict(R=float(np.linalg.norm(f.r-a['r'])/np.linalg.norm(a['r'])),
            X=float(np.linalg.norm(f.x-a['x'])/np.linalg.norm(a['x'])))
    record.update(status='complete' if validated['all_complete'] and all(r['status']=='complete' for r in record['methods'])
        else 'partial_failure', seconds=perf_counter()-began, unique_actual_curves=len(cache),
        one_trained_parameter_vector=True, validation_used_by_selector=False,
        future_noise_used_by_selector=False)
    return record


def run_trial(seed, family_index, config=None):
    cfg = deepcopy(CONFIG if config is None else config); validate_config(cfg)
    before = source_hashes(); began = perf_counter()
    with dynamic.count_physics_calls() as counter:
        try:
            record = _run_trial(seed, family_index, cfg, counter)
        except Exception as exc:
            record = dict(schema='ev_gradient_end_to_end_v1', seed=seed, family_index=family_index,
                config=cfg, source_sha256=before, policy_id=io.identity(cfg, before), methods=[],
                status='driver_failure', failure=type(exc).__name__+':'+str(exc), traceback=traceback.format_exc(),
                seconds=perf_counter()-began)
    record.update(actual_solver_calls=counter, source_unchanged=before==source_hashes())
    return io.serial(record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--family-offset', type=int, default=0)
    args = parser.parse_args(); cfg = deepcopy(CONFIG)
    frozen = freeze(args.output, cfg)
    for index, seed in enumerate(args.seeds):
        path = args.output/f'trial_{seed}.json.gz'
        if path.exists():
            raise ValueError('existing trial retained; use a new output directory')
        record = run_trial(seed, (index+args.family_offset)%4, cfg)
        if record['policy_id'] != frozen['policy_id'] or not record['source_unchanged']:
            raise RuntimeError('source/config changed; preserve records and refreeze')
        io.write_json(path, record, compressed=True)
        planned = next((cfg[k] for k in ('development_seeds', 'calibration_seeds', 'evaluation_seeds')
                        if set(args.seeds) <= set(cfg[k])), args.seeds)
        io.write_json(args.output/'checkpoint.json', dict(completed=len(list(args.output.glob('trial_*.json.gz'))),
            planned=len(planned), batch_completed=index+1, batch_planned=len(args.seeds), last_seed=seed,
            policy_id=frozen['policy_id']))
        print(json.dumps(dict(seed=seed, status=record['status'], seconds=record['seconds'],
            M={r['name']:r.get('M') for r in record['methods']})), flush=True)
        if record['status'] == 'driver_failure':
            raise RuntimeError('driver failure saved; review before continuation')


if __name__ == '__main__':
    main()
