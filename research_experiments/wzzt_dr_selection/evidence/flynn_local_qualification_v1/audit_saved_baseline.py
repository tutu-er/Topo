"""Independent saved-input replay; no project selection/contract/solver calls."""
from pathlib import Path
import gzip
import hashlib
import json
import pickle
import sys
import traceback
from datetime import datetime, timezone
from itertools import combinations
import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent
OLD = RUN.parents[1] / 'wzzt_dr_scenario_review_v1' / 'candidate_extract.json.gz'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(supports):
    return frozenset(tuple(sorted(s)) for s in supports)


def key_point(point):
    return (tuple(tuple(s) for s in point['supports']), tuple(point['r']), tuple(point['x']))


def main():
    receipt_path = ROOT / 'receipt.json'
    if receipt_path.exists() or (ROOT / 'failure.json').exists():
        raise FileExistsError('retain existing independent audit attempt')
    inputs = [RUN / 'baseline_request' / 'result.json', RUN / 'history_and_forecast.pkl', OLD]
    protocol = {'status': 'frozen_before_replay', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': sha(Path(__file__)), 'inputs': {str(p): sha(p) for p in inputs},
        'scope': 'saved residuals, original selection/contraction, accepted/rejected scans, historical candidate scope',
        'new_solver_calls': 0, 'true_topology_or_future_allowed': False}
    protocol_path = ROOT / 'protocol.json'
    if protocol_path.exists():
        raise FileExistsError('protocol retained')
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    checks = []
    def check(name, passed, details=None):
        checks.append({'name': name, 'passed': bool(passed), 'details': details})
        if not passed:
            raise AssertionError(name + ': ' + str(details))
    try:
        fit = json.loads(inputs[0].read_text(encoding='utf-8'))
        packet = pickle.loads(inputs[1].read_bytes())
        check('packet_observed_history_and_forecast_only', set(packet) == {'history', 'forecast'})
        history = packet['history']
        labels = history['labels']
        n = len(labels)
        check('six_complete_past_days', len(history['training']) == len(history['validation']) == 3)
        observations = {}
        for role in ('training', 'validation'):
            frames = history[role]
            p = np.concatenate([v['P_terminal'].to_numpy(dtype=float) for v in frames])
            q = np.concatenate([v['Q_terminal'].to_numpy(dtype=float) for v in frames])
            volt = np.concatenate([v['V_terminal'].to_numpy(dtype=float) for v in frames])
            root = np.concatenate([v['V_root_observed'].to_numpy(dtype=float) for v in frames])
            y = root[:, None] ** 2 - volt ** 2
            check(role + '_shape', p.shape == q.shape == volt.shape == y.shape == (288, n))
            check(role + '_finite', all(np.isfinite(a).all() for a in (p, q, y)))
            check(role + '_labels', all(v['P_terminal'].columns.tolist() == labels for v in frames))
            observations[role] = (p, q, y)
        loss_cache = {}
        def losses(point):
            token = key_point(point)
            if token not in loss_cache:
                r = np.asarray(point['r'], float)
                x = np.asarray(point['x'], float)
                if len(r) != len(x) or len(r) != len(point['supports']):
                    raise ValueError('support and weight lengths')
                values = {}
                for role, (p, q, y) in observations.items():
                    prediction = np.zeros_like(y)
                    for s, rv, xv in zip(point['supports'], r, x):
                        signal = p[:, s].sum(axis=1) * rv + q[:, s].sum(axis=1) * xv
                        prediction[:, s] += signal[:, None]
                    values[role] = float(np.mean(np.abs(y - prediction)))
                loss_cache[token] = values
            return loss_cache[token]
        point_groups = {'selection_candidates': fit['selection_candidates'], 'bank': fit['bank'],
                        'start_preparation': fit['start_preparation']}
        point_groups.update({'path_' + name: s['path'] for name, s in fit['searches'].items()})
        point_groups['models'] = list(fit['models'].values())
        point_groups['selected_uncontracted'] = [v['selected_uncontracted'] for v in fit['models'].values()]
        errors = []
        for group, points in point_groups.items():
            for i, point in enumerate(points):
                got = losses(point)
                for field, role in [('objective', 'training'), ('train_mae', 'training'), ('validation_mae', 'validation')]:
                    if field in point:
                        error = abs(got[role] - point[field])
                        errors.append(error)
                        check(f'{group}_{i}_{field}', error <= 2e-12, error)
                check(f'{group}_{i}_nonnegative_bound', all(0 <= w <= 2 for w in point['r'] + point['x']))
        cal = fit['calibrations']['raw']
        threshold = .16 * cal['median_root_depth']
        def independently_shrink(point):
            keep = [i for i, s in enumerate(point['supports'])
                    if len(s) == 1 or cal['a'] * point['r'][i] + cal['b'] * point['x'][i] + 1e-10 >= threshold]
            return {'supports': [point['supports'][i] for i in keep],
                    'r': [point['r'][i] for i in keep], 'x': [point['x'][i] for i in keep]}
        def eligible(points):
            vals = [losses(p)['validation'] for p in points]
            floor = min(vals)
            indices = [i for i, v in enumerate(vals) if v <= 1.03 * floor + 1e-12]
            return indices, vals
        candidates = fit['selection_candidates']
        model_lists = {'rnj_bank': fit['bank'],
                       'baseline25': candidates[:len(fit['bank']) + len(fit['searches']['primary']['path'])],
                       'multistart_original_rule': candidates}
        selections = {}
        for name, points in model_lists.items():
            indices, vals = eligible(points)
            chosen = min(indices, key=lambda i: (len(points[i]['supports']), vals[i], i))
            model = fit['models'][name]
            check(name + '_selection_index', model['selection_index'] == chosen)
            check(name + '_selected_raw_weights', key_point(model['selected_uncontracted']) == key_point(points[chosen]))
            shrunk = independently_shrink(points[chosen])
            check(name + '_contraction_weights', key_point(model) == key_point(shrunk))
            check(name + '_raw_validation_score', abs(model['selection_validation_mae'] - vals[chosen]) <= 2e-12)
            check(name + '_final_validation_score', abs(losses(model)['validation'] - model['validation_mae']) <= 2e-12)
            selections[name] = {'selection_index': chosen, 'eligible_points': len(indices),
                'raw_support_count': len(points[chosen]['supports']), 'final_support_count': len(model['supports']),
                'origin': model['origin'], 'raw_validation_mae': vals[chosen],
                'final_validation_mae': losses(model)['validation'],
                'removed_by_contraction': [list(s) for s in sorted(canonical(points[chosen]['supports']) - canonical(model['supports']))]}
        pool = canonical(fit['pool_support_indices'])
        lp_by_family = {canonical(v['family']): v for v in fit['LP_solves']}
        for i, receipt in enumerate(fit['LP_solves']):
            check('LP_' + str(i) + '_saved_status', receipt['status'] == 2 and receipt['accepted'])
            check('LP_' + str(i) + '_saved_physical_objective', abs(receipt['objective'] - receipt['physical']) <= 2e-12)
            check('LP_' + str(i) + '_saved_feasibility', max(receipt['max_constraint_violation'], receipt['max_bound_violation']) <= 1e-7)
        def compatible(left, right):
            return not(set(left) & set(right)) or set(left) <= set(right) or set(right) <= set(left)
        path_records = []
        for name, run in fit['searches'].items():
            path_cursor = 0
            accepted = []
            for j, decision in enumerate(run['decisions']):
                current = run['path'][path_cursor]
                family = canonical(current['supports'])
                check(f'{name}_{j}_current', abs(decision['current'] - losses(current)['training']) <= 2e-12)
                expected = {s for s in pool if len(s) >= 2 and s not in family}
                records = decision['candidates']
                found = {tuple(v['candidate']) for v in records}
                check(f'{name}_{j}_complete_scan', decision['full_sweep'] and found == expected)
                best = None
                for k, row in enumerate(records):
                    c = tuple(row['candidate'])
                    removed = {s for s in family if not compatible(s, c)}
                    check(f'{name}_{j}_{k}_cross_removal', canonical(row['removed']) == frozenset(removed))
                    proposed = frozenset((set(family) - removed) | {c})
                    check(f'{name}_{j}_{k}_LP_receipt', proposed in lp_by_family)
                    check(f'{name}_{j}_{k}_LP_objective', abs(row['objective'] - lp_by_family[proposed]['physical']) <= 2e-12)
                    if best is None or row['objective'] < best['objective'] - 1e-12:
                        best = row
                gain = 0. if best is None else current['objective'] - best['objective']
                check(f'{name}_{j}_gain', abs(gain - decision['gain']) <= 2e-12)
                tolerance = 1e-10 + .001 * current['objective']
                should_accept = best is not None and gain > tolerance
                if should_accept:
                    path_cursor += 1
                    check(f'{name}_{j}_accepted_exists', path_cursor < len(run['path']))
                    next_point = run['path'][path_cursor]
                    check(f'{name}_{j}_accepted_objective', abs(next_point['objective'] - best['objective']) <= 2e-12)
                    proposed = (set(family) - set(canonical(best['removed']))) | {tuple(best['candidate'])}
                    check(f'{name}_{j}_accepted_supports_subset', set(canonical(next_point['supports'])) <= proposed)
                    accepted.append({'candidate_labels': [labels[k] for k in best['candidate']],
                                     'removed_labels': [[labels[k] for k in s] for s in best['removed']],
                                     'gain': gain, 'threshold': tolerance})
                else:
                    check(f'{name}_{j}_stop_is_last', j == len(run['decisions']) - 1)
                    check(f'{name}_{j}_stop_reason', run['reason'] == 'pool_local_optimum')
            check(name + '_path_count', path_cursor + 1 == len(run['path']))
            path_records.append({'name': name, 'decisions': len(run['decisions']), 'accepted': accepted,
                                 'stop_reason': run['reason'], 'final_objective': run['path'][-1]['objective']})
        indices, _ = eligible(candidates)
        raw_families = [canonical(candidates[i]['supports']) for i in indices]
        final_families = set(canonical(independently_shrink(candidates[i])['supports']) for i in indices)
        shared_raw = set.intersection(*(set(v) for v in raw_families))
        disputed = set.union(*(set(v) for v in final_families)) - set.intersection(*(set(v) for v in final_families))
        proper = sorted(s for s in disputed if 1 < len(s) < n)
        disjoint_pairs = [(c, d) for c, d in combinations(proper, 2)
                          if not(set(c) & set(d)) and any(c in f and d in f for f in final_families)]
        old_rows = json.loads(gzip.decompress(OLD.read_bytes()))['records']
        old = next(v for v in old_rows if v['job_id'] == 'main__flynn16__b7__reference__v0002__N288')
        old_points = old['base_points']
        old_floor = min(v['validation_mae'] for v in old_points)
        old_indices = [i for i, v in enumerate(old_points) if v['validation_mae'] <= 1.03 * old_floor + 1e-12]
        old_cal = old['calibration']
        old_families = set()
        old_errors = []
        for i in old_indices:
            p = old_points[i]
            old_errors.append(abs(losses(p)['validation'] - p['validation_mae']))
            keep = [j for j, s in enumerate(p['supports']) if len(s) == 1 or
                    old_cal['a'] * p['r'][j] + old_cal['b'] * p['x'][j] + 1e-10 >= .16 * old_cal['median_root_depth']]
            old_families.add(canonical([p['supports'][j] for j in keep]))
        # Original and fresh saved fits may use different solver/cache/QP paths;
        # the scope comparison below does not assume equal fitted weights.
        scope = {'current_eligible_raw_points': len(indices), 'current_eligible_raw_families': len(set(raw_families)),
            'current_eligible_final_families': len(final_families),
            'current_final_families_labels': [[[labels[k] for k in s] for s in sorted(f, key=lambda x: (len(x), x))] for f in sorted(final_families, key=lambda f: sorted(f))],
            'current_shared_raw_proper_labels': [[labels[k] for k in s] for s in sorted(shared_raw, key=lambda x: (len(x), x)) if 1 < len(s) < n],
            'current_disputed_proper_labels': [[labels[k] for k in s] for s in proper],
            'current_disjoint_disputes': [[[labels[k] for k in s] for s in pair] for pair in disjoint_pairs],
            'old_eligible_raw_points': len(old_indices), 'old_eligible_final_families': len(old_families),
            'old_validation_replay_max_absolute_error_on_current_history': max(old_errors, default=0.),
            'old_vs_current_family_sets_equal': old_families == final_families,
            'historical_comparison_only': True, 'true_topology_used': False}
        for p, digest in protocol['inputs'].items():
            check('input_unchanged_' + Path(p).name, sha(Path(p)) == digest)
        output = {'status': 'passed', 'check_count': len(checks), 'checks': checks,
            'residual_comparisons': len(errors), 'max_residual_absolute_error': max(errors, default=0.),
            'distinct_saved_coefficient_points': len(loss_cache), 'selections': selections,
            'paths': path_records, 'candidate_scope': scope, 'saved_LP_receipts': len(fit['LP_solves']),
            'new_LP_calls': 0, 'new_fits': 0, 'future_or_true_topology_read': False,
            'limitations': ['saved LP status/objective replay is not an independent optimality certificate',
                'all candidate predictions cannot be recomputed because full coefficient vectors were not saved for every LP',
                'finite candidate disagreement is not a topology-confidence or all-parameter coverage statement'],
            'source_sha256': protocol['source_sha256'], 'protocol_sha256': sha(protocol_path)}
        receipt_path.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': 'passed', 'checks': len(checks), 'residual_max_error': output['max_residual_absolute_error'],
                          'current_final_families': len(final_families), 'old_final_families': len(old_families),
                          'disputed_proper': scope['current_disputed_proper_labels'], 'receipt': str(receipt_path)}))
    except Exception as exc:
        failure = {'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc(), 'checks': checks,
                   'new_LP_calls': 0, 'new_fits': 0, 'protocol_sha256': sha(protocol_path)}
        (ROOT / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
