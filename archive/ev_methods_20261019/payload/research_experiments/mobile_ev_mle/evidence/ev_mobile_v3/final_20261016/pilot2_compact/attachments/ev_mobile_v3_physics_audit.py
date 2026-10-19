"""Read-only saved-state AC and EV-operation replay for the v3 pilot.

Uses the older independent branch-current replay, never an optimizer, simulator
generator, production AC predictor, or policy selector.  Method slots remain in
the denominator when actions share a physical replay or recorded fits fail.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import ev_mobile_v2_review as old
from . import ev_mobile_v3_cases as cases
from . import gradient_price_study as io

HERE = Path(__file__).resolve().parent
METHODS = ('constant', 'highlow', 'square_fixed', 'square_mobile',
           'scalar_fixed', 'scalar_mobile', 'projection_fixed', 'projection_mobile')


def _identity(value):
    return sha256(json.dumps(io.serial(value), sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def _bank(record):
    key = record['spec']['bank']
    if key not in ('n3', 'n4'):
        raise ValueError('unknown physical bank')
    return {b.name: b for b in cases.old.candidate_bank('remote8' if key == 'n3' else 'twinblock12')}


def _training_predictions(row, observation, meter, actual, metrics):
    white = actual['blocks']['training']['residual']
    T, n = np.asarray(observation['p']).shape
    for key in ('p', 'q'):
        corrected = np.asarray(observation[key]) - old._array(meter['bias_' + key], (T, n))
        old.close(row[key], corrected, key + ' fixed input', atol=0., rtol=0.)
        old.close(row['predicted'][key], observation[key], key + ' predicted input', atol=2e-10, rtol=0.)
    start = 0
    for key, count in (('v', n), ('master_p', 1), ('master_q', 1)):
        shape = (T, n) if key == 'v' else (T,)
        residual = white[:, start:start + count].reshape(shape); start += count
        predicted = np.asarray(observation[key]) + residual * old._array(meter[key + '_sigma'], shape)
        saved = np.asarray(row['predicted'][key])
        old.close(saved, predicted, key + ' saved measurement prediction', atol=2e-10, rtol=0.)
        metrics['max_measurement_prediction_error'] = max(metrics['max_measurement_prediction_error'],
            float(np.max(abs(saved - predicted), initial=0.)))


def _operation(domain, method, row, planning, metrics):
    """Reconstruct a saved route from its arc records, without a MILP solve."""
    saved = row['operation']; audit = saved['audit']
    replay = old._operation(domain, audit)
    old.close(row['power_kw'], replay['u_kw'], method + ': executed power', atol=1e-7, rtol=0.)
    old.close(saved['energy_kwh'], replay['energy_kwh'], method + ': energy', atol=1e-7, rtol=0.)
    old.close(saved['driver_cost'], replay['driver_cost'], method + ': driver cost', atol=1e-7, rtol=0.)
    old.close(saved['charge_kw'], np.maximum(row['power_kw'], 0.), method + ': charge', atol=1e-7, rtol=0.)
    old.close(saved['discharge_kw'], np.maximum(-np.asarray(row['power_kw']), 0.), method + ': discharge', atol=1e-7, rtol=0.)
    fields = ('kind', 'departure', 'arrival', 'origin', 'destination', 'driving_energy_kwh')
    slim = [{k: arc[k] for k in fields} for arc in replay['route']]
    old.require(Counter(_identity(a) for a in saved['arcs']) == Counter(_identity(a) for a in slim),
                method + ': decoded arcs differ from saved physical path')
    raw = planning.get('raw', {})
    x = raw.get('selected_operation_x')
    if x is None: x = raw.get('operation', {}).get('x')
    if x is not None:
        checked = domain.check(np.asarray(x, float))
        old.require(checked['feasible'], method + ': saved planning x infeasible')
        for key in ('u_kw', 'energy_kwh', 'soc', 'driver_cost', 'moves'):
            old.close(checked[key], replay[key], method + ': planning x/' + key, atol=1e-7, rtol=0.)
        old.require(Counter(_identity(a) for a in checked['route']) == Counter(_identity(a) for a in replay['route']),
                    method + ': planning x route changed')
        metrics['saved_planning_x_checks'] += 1
    elif method.startswith('projection') and raw.get('success'):
        raise ValueError(method + ': successful projection has no saved selected_operation_x')
    metrics['operation_checks'] += 1
    metrics['max_operation_violation'] = max(metrics['max_operation_violation'], float(replay['max_violation']))
    return dict(success=True, moves=replay['moves'], driver_cost=replay['driver_cost'],
                max_violation=replay['max_violation'], saved_planning_x_checked=x is not None)


def audit_record(record, *, expected_methods=METHODS):
    began = perf_counter()
    errors, missing, states, methods, recorded = [], [], [], [], []
    metrics = dict(saved_states_replayed=0, AC_block_replays=0, max_whitened_residual_error=0.,
        max_block_SSE_error=0., max_total_SSE_error=0., max_AC_equation=0., max_power_balance=0.,
        max_measurement_prediction_error=0., operation_checks=0, saved_planning_x_checks=0,
        max_operation_violation=0., PQ_once_checks=0, joint_method_candidate_slots=0,
        joint_unique_action_candidate_states=0, recorded_joint_fit_failure_slots=0,
        recorded_inference_failure_method_slots=0)

    def attempt(label, function):
        try: return function()
        except (ValueError, KeyError, TypeError, IndexError, AssertionError, FloatingPointError,
                np.linalg.LinAlgError) as exc:
            errors.append(dict(check=label, reason=type(exc).__name__ + ': ' + str(exc)))
            return None

    def absent(label, reason='saved state unavailable'):
        missing.append(dict(check=label, reason=reason))

    def replay(label, candidate, saved, state, blocks):
        actual = old.replay_state(candidate, state, blocks, record['config'])
        old._compare_score(saved, actual, label, metrics, selected=bool(saved.get('success')))
        metrics['max_total_SSE_error'] = max(metrics['max_total_SSE_error'], abs(float(saved['loss']) - actual['loss']))
        states.append(dict(check=label, declared_fit_success=bool(saved.get('success')),
            replay_loss=actual['loss'], saved_loss=saved['loss'],
            line_domain_feasible=actual['line_domain_feasible'], source_box_feasible=actual['source_box_feasible'],
            physical_voltage_feasible=actual['physical_voltage_feasible']))
        if not saved.get('success'):
            recorded.append(dict(kind='fit_failure', check=label, selected_origin=saved.get('selected_origin'),
                                 optimizer_error=saved.get('optimizer_error')))
        return actual

    expected_methods = tuple(expected_methods)
    if len(set(expected_methods)) != len(expected_methods) or not expected_methods:
        raise ValueError('distinct nonempty expected methods required')
    bank = attempt('candidate bank', lambda: _bank(record))
    candidate_count = len(bank) if bank is not None else 4
    expected = dict(method_slots=len(expected_methods), training_candidate_states=candidate_count,
                    joint_method_candidate_slots=len(expected_methods) * candidate_count)
    required = ('training', 'training_meter', 'validation', 'validation_meter', 'future_meter', 'training_result', 'config')
    for key in required:
        if key not in record: absent(key, record.get('failure', record.get('status', 'unavailable')))
    if missing or bank is None:
        return dict(success=False, complete_physical_coverage=False, spec=record.get('spec'),
            recorded_status=record.get('status'), recorded_failure=record.get('failure'),
            expected=expected, metrics=metrics, errors=errors, missing=missing, states=states,
            methods=methods, recorded_failures=recorded, seconds=perf_counter() - began)
    n = len(record['training']['p'][0]); T = int(record['spec']['T'])
    attempt('candidate terminal count', lambda: old.require(all(c.n == n for c in bank.values()), 'bank/observation n differs'))
    train_blocks = (('training', record['training'], record['training_meter']),)
    history_blocks = (*train_blocks, ('validation', record['validation'], record['validation_meter']))
    trained = {r.get('candidate', r.get('name')): r for r in record['training_result'].get('fits', [])}
    for name, candidate in bank.items():
        saved = trained.get(name)
        if saved is None:
            absent('training/' + name, str(record['training_result'].get('failures', {}))); continue
        def training_work(c=candidate, r=saved):
            old.require(tuple(map(tuple, r['supports'])) == c.supports, 'saved edge order differs')
            actual = replay('training/' + c.name, c, r, r, train_blocks)
            _training_predictions(r, record['training'], record['training_meter'], actual, metrics)
        attempt('training/' + name, training_work)
        joint_pre = record.get('history_prefits', {}).get(name, {}).get('joint')
        if joint_pre is not None and joint_pre.get('state') is not None:
            attempt('joint_pre/' + name, lambda c=candidate, s=joint_pre:
                replay('joint_pre/' + c.name, c, s, s['state'], history_blocks))

    domains = {key: cases.service_domain(T, n, mobile=key == 'mobile') for key in ('fixed', 'mobile')}
    raw_methods = record.get('methods', {})
    if not isinstance(raw_methods, dict): raise ValueError('v3 methods must be a dictionary')
    attempt('method coverage', lambda: old.require(set(raw_methods) <= set(expected_methods), 'unexpected method'))
    cache, paired = {}, None
    for name in expected_methods:
        row = raw_methods.get(name)
        method = dict(name=name, recorded_success=False if row is None else row.get('success'),
                      present=row is not None, joint_candidate_slots=candidate_count)
        methods.append(method)
        metrics['joint_method_candidate_slots'] += candidate_count
        if row is None:
            absent('methods/' + name, record.get('failure', 'method unavailable')); continue
        if row.get('planning_failure') or row.get('planning_success') is False:
            recorded.append(dict(kind='planning_failure', check=name, reason=row.get('planning_failure')))
        if row.get('failure'): recorded.append(dict(kind='method_failure', check=name, reason=row['failure']))
        if not row.get('operation', {}).get('audit', {}).get('feasible', False):
            recorded.append(dict(kind='operation_failure', check=name))
        elif 'operation' in row:
            method['operation'] = attempt('operation/' + name, lambda:
                _operation(domains['mobile' if name.endswith('mobile') else 'fixed'], name, row,
                           record.get('planning', {}).get(name, {}), metrics))
        evaluation = row.get('evaluation')
        if evaluation is None:
            absent('evaluation/' + name, row.get('failure', 'no saved evaluation')); continue
        power = np.asarray(row['power_kw'], float)
        key = sha256(np.asarray(power, dtype='<f8').tobytes()).hexdigest()
        attempt('action hash/' + name, lambda: old.require(key == row['action_key'], 'action bytes/hash mismatch'))
        digest = _identity(evaluation)
        statuses = {candidate_name: evaluation.get('comparisons', {}).get(candidate_name, {}).get('joint', {}).get('post', {})
                    for candidate_name in bank}
        method['candidate_status'] = {k:dict(present=bool(s), success=s.get('success'),
            selected_origin=s.get('selected_origin'), saved_loss=s.get('loss')) for k,s in statuses.items()}
        metrics['recorded_joint_fit_failure_slots'] += sum(not s.get('success', False) for s in statuses.values())
        metrics['recorded_inference_failure_method_slots'] += not evaluation.get('complete', False)
        method.update(action_key=key, action_replay_reused=key in cache,
                      recorded_evaluation_complete=bool(evaluation.get('complete')))
        if key in cache:
            attempt('duplicate action/' + name, lambda: old.require(cache[key] == digest,
                'same action has different saved data, states or scores'))
            continue
        cache[key] = digest
        future = evaluation['future']
        baseline = dict(p=np.asarray(future['p']) - power / float(record['config']['base_kw']),
                        q=np.asarray(future['q']), root_v=np.asarray(future['root_v']))
        if paired is None: paired = baseline
        else:
            def pq_once():
                for channel in ('p', 'q', 'root_v'):
                    old.close(baseline[channel], paired[channel], 'common background/PQ once/' + channel,
                              atol=3e-15, rtol=2e-13)
                metrics['PQ_once_checks'] += 1
            attempt('PQ_once/' + name, pq_once)
        blocks = (*history_blocks, ('future', future, record['future_meter']))
        comparisons = evaluation.get('comparisons', {})
        actual_losses, complete = {}, len(comparisons) == candidate_count
        for candidate_name, candidate in bank.items():
            comparison = comparisons.get(candidate_name)
            if comparison is None:
                absent('joint_post/' + name + '/' + candidate_name,
                       str(evaluation.get('failures', {}).get(candidate_name, 'candidate missing')))
                complete = False; continue
            saved = comparison.get('joint', {}).get('post')
            if saved is None or saved.get('state') is None:
                absent('joint_post_state/' + name + '/' + candidate_name); complete = False; continue
            metrics['joint_unique_action_candidate_states'] += 1
            actual = attempt('joint_post/' + name + '/' + candidate_name,
                lambda c=candidate, s=saved: replay('joint_post/' + name + '/' + c.name, c, s, s['state'], blocks))
            if actual is not None:
                actual_losses[candidate_name] = actual['loss']
                attempt('joint total/' + name + '/' + candidate_name, lambda:
                    old.close(comparison['joint']['joint_total'], actual['loss'], 'joint_total/replay'))
                attempt('saved loss/' + name + '/' + candidate_name, lambda:
                    old.close(evaluation['losses'][candidate_name], actual['loss'], 'evaluation loss/replay'))
            complete = complete and bool(saved.get('success'))
        attempt('evaluation completeness/' + name, lambda: old.require(bool(evaluation['complete']) == complete,
            'complete flag differs from all required candidate statuses'))
        if len(actual_losses) == candidate_count and complete and evaluation.get('M') is not None:
            truth = record['true_name']  # Post-hoc saved loss arithmetic only.
            margin = min(v for k, v in actual_losses.items() if k != truth) - actual_losses[truth]
            attempt('saved M/' + name, lambda: old.close(evaluation['M'], margin, 'saved margin/replay'))
        method['replayed_action'] = True
    metrics['unique_action_replays'] = len(cache)
    return dict(success=not errors and not missing, complete_physical_coverage=not missing,
        spec=record.get('spec'), recorded_status=record.get('status'), expected=expected,
        metrics=metrics, errors=errors, missing=missing, states=states, methods=methods,
        recorded_failures=recorded, seconds=perf_counter() - began,
        independent_layer='old independent branch-current sweep and metering/SSE arithmetic; saved operation route replay',
        optimizer_calls=0, production_AC_calls=0, new_simulation_calls=0,
        scope='saved retained training and final joint states; no local/global optimality claim or price-gradient check')


def run(directory, output, *, phase='development'):
    directory, output = Path(directory), Path(output)
    if output.exists(): raise FileExistsError(output)
    frozen = old.read(directory / 'freeze.json')
    expected = frozen['specifications'][phase]
    files = sorted((directory / phase).glob('*.json.gz'))
    rows = []
    for file in files:
        record = old.read(file)
        try:
            old.require(record.get('policy_id') == frozen['policy_id'], 'case/freeze policy differs')
            audit = audit_record(record, expected_methods=frozen['methods'])
        except (ValueError, KeyError, TypeError, IndexError, AssertionError, FloatingPointError, np.linalg.LinAlgError) as exc:
            audit = dict(success=False, spec=record.get('spec'), recorded_status=record.get('status'),
                         errors=[dict(check='record', reason=type(exc).__name__ + ': ' + str(exc))],
                         missing=[], metrics={}, recorded_failures=[])
        audit.update(file=str(file.resolve()), input_sha256=sha256(file.read_bytes()).hexdigest())
        rows.append(audit)
        print('REPLAY', file.name, audit['success'], flush=True)
    expected_ids = {_identity(s) for s in expected}
    actual_ids = [_identity(r.get('spec')) for r in rows]
    coverage = len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == expected_ids
    totals = Counter(); maxima = {}
    for row in rows:
        for key, value in row.get('metrics', {}).items():
            if key.startswith('max_'): maxima[key] = max(maxima.get(key, 0.), value)
            else: totals[key] += value
    result = dict(success=coverage and all(r['success'] for r in rows), phase=phase,
        policy_id=frozen['policy_id'], expected_case_slots=len(expected), saved_case_slots=len(rows),
        exact_case_coverage=coverage, case_results=rows, totals=dict(totals), maxima=maxima,
        recorded_failure_counts=dict(Counter(r['kind'] for row in rows for r in row.get('recorded_failures', []))),
        input_freeze_sha256=sha256((directory / 'freeze.json').read_bytes()).hexdigest(),
        original_source_sha256=frozen['source_sha256'],
        reviewer_source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), HERE/'ev_mobile_v2_review.py', HERE/'ev_original_review.py',
             HERE/'ev_mobile_v3_cases.py', HERE/'ev_mobile_service_domain.py')},
        optimizer_calls=0, production_AC_calls=0, new_simulation_calls=0,
        denominator='saved experimental case and method slots; action deduplication only reduces repeated physical work')
    old.write(output, io.serial(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=('development', 'evaluation'), default='development')
    args = parser.parse_args()
    result = run(args.directory, args.output, phase=args.phase)
    print(json.dumps({k:result[k] for k in ('success', 'expected_case_slots', 'saved_case_slots', 'totals', 'maxima')}))


if __name__ == '__main__': main()
