"""Independent saved-observation audit; refuses future reads before fit freeze.

No project solver, selection, contraction or evaluation function is imported.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import pickle
import sys
import traceback
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent
OLD = PILOT.parents[1] / 'wzzt_flynn_local_qualification_v1/run_v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
def family(supports): return frozenset(tuple(sorted(v)) for v in supports)
def compatible(a, b): return not(set(a) & set(b)) or set(a) <= set(b) or set(b) <= set(a)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'protocol.json'
    if path.exists(): raise FileExistsError(path)
    inputs = [PILOT / 'protocol.json', PILOT / 'plans.json', OLD / 'history_and_forecast.pkl', OLD / 'baseline_request/result.json']
    write(path, dict(status='frozen_before_future_audit', created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), inputs={str(p): sha(p) for p in inputs},
        mandatory_markers=['ended.json', 'fits_frozen.json'], planned=8,
        allowed='after both markers: saved observed packets, completed fit outputs, evaluator support labels; never graph or simulator',
        LP_calls=0, identifier_calls=0, AC_calls=0,
        scope='full480/480, saved L1, actual3%/.16, path/LP ledgers, final errors and where information is lost'))
    print(json.dumps(dict(status='prepared_without_future_read', source_sha256=sha(__file__))))


def run():
    # The checks must precede reading any result, physical packet or evaluator.
    if not all((PILOT / name).is_file() for name in ('ended.json', 'fits_frozen.json')):
        raise RuntimeError('pilot not ended and fits not frozen; future audit forbidden')
    if (ROOT / 'receipt.json').exists() or (ROOT / 'failure.json').exists():
        raise FileExistsError('terminal audit retained; do not rerun')
    protocol = read(ROOT / 'protocol.json')
    assert protocol['source_sha256'] == sha(__file__), 'audit source drift'
    for path, expected in protocol['inputs'].items(): assert sha(path) == expected, path
    checks, input_hashes = [], dict(protocol['inputs'])
    def check(name, ok, detail=None):
        checks.append(dict(name=name, passed=bool(ok), details=detail))
        if not ok: raise AssertionError(name + ': ' + str(detail))
    def load_json(path):
        input_hashes[str(path)] = sha(path)
        return read(path)
    def load_packet(path):
        input_hashes[str(path)] = sha(path)
        return pickle.loads(Path(path).read_bytes())
    try:
        ended = load_json(PILOT / 'ended.json')
        frozen = load_json(PILOT / 'fits_frozen.json')
        check('ended_fixed_denominator', ended['planned'] == len(ended['rows']) == 8)
        check('fits_frozen_protocol', frozen['protocol_sha256'] == sha(PILOT / 'protocol.json'))
        old_packet = load_packet(OLD / 'history_and_forecast.pkl')
        history = old_packet['history']; labels = history['labels']; n = len(labels)
        check('history_labels', labels == list(range(201, 217)))
        policy_rows = {v['policy']: v for v in ended['rows']}
        check('frozen_same_policy_rows', set(policy_rows) == {v['policy'] for v in frozen['rows']})
        outputs, truth = {}, None
        residual_errors, ledger, unknown = [], {}, []
        for folder in sorted((PILOT / 'fit_cache').glob('*')):
            if not folder.is_dir(): continue
            path = folder / 'result.json'
            if not path.exists(): path = folder / 'checkpoint.json'
            if not path.exists(): unknown.append(folder.name); continue
            data = load_json(path)
            if 'LP_solves' not in data: unknown.append(folder.name); continue
            ledger[folder.name] = dict(count=len(data['LP_solves']), status=data.get('status'),
                solver_status_counts={str(s): sum(v.get('status') == s for v in data['LP_solves']) for s in sorted({v.get('status') for v in data['LP_solves']})})
        check('all_attempted_LP_ledger', sum(v['count'] for v in ledger.values()) == ended['new_LP'])
        check('unknown_LP_preserved', sorted(unknown) == sorted(ended['LP_ledger']['unknown_cache_keys']))
        for name, row in policy_rows.items():
            if row['status'] != 'evaluated':
                outputs[name] = dict(status=row['status'], actual_payment=row.get('actual_payment'), unresolved=True)
                continue
            folder = PILOT / name
            receipt = load_json(folder / 'physical/events_receipt.json')
            result_path = Path(row['result_path'])
            check(name + '_result_hash', sha(result_path) == row['result_sha256'])
            fit = load_json(result_path)
            packet_path = folder / 'physical/packet.pkl'
            check(name + '_observed_packet_hash', sha(packet_path) == receipt['private_files']['packet.pkl']['sha256'])
            packet = load_packet(packet_path)
            truth_path = folder / 'physical/truth_evaluation.json'
            check(name + '_truth_hash', sha(truth_path) == receipt['private_files']['truth_evaluation.json']['sha256'])
            current_truth = family(load_json(truth_path)['true_supports'])
            if truth is None: truth = current_truth
            check(name + '_same_true_family', current_truth == truth)
            check(name + '_observed_only', packet['metadata']['observed_only'] is True)
            check(name + '_event_order', packet['metadata']['training_event_ids'] == [0, 1] and packet['metadata']['validation_event_ids'] == [2, 3])
            frames = {role: history[role] + packet[role] for role in ('training', 'validation')}
            arrays = {}
            for role, scenes in frames.items():
                p = np.concatenate([v['P_terminal'].to_numpy(float) for v in scenes])
                q = np.concatenate([v['Q_terminal'].to_numpy(float) for v in scenes])
                v = np.concatenate([v['V_terminal'].to_numpy(float) for v in scenes])
                root = np.concatenate([v['V_root_observed'].to_numpy(float) for v in scenes])
                y = root[:, None] ** 2 - v ** 2
                check(name + '_' + role + '_480rows', p.shape == q.shape == v.shape == y.shape == (480, n))
                check(name + '_' + role + '_labels', all(s['P_terminal'].columns.tolist() == labels for s in scenes))
                check(name + '_' + role + '_finite', all(np.isfinite(x).all() for x in (p, q, v, root)))
                check(name + '_' + role + '_measured_root_target', all(np.allclose(s['drop_target'].to_numpy(), s['V_root_observed'].to_numpy()[:, None] ** 2 - s['V_terminal'].to_numpy() ** 2, atol=1e-14, rtol=0) for s in scenes))
                concat = [pd.concat([s[key] for s in scenes]) for key in ('P_terminal', 'Q_terminal', 'V_terminal', 'V_root_observed')]
                h = hashlib.sha256()
                for value in concat: h.update(pd.util.hash_pandas_object(value, index=True).values.tobytes())
                days = [str(value) for value in concat[0].index.normalize()]
                h.update(json.dumps(list(concat[0].columns), default=str).encode()); h.update(json.dumps(days).encode())
                check(name + '_' + role + '_input_hash', h.hexdigest() == fit['input_sha256'][role])
                arrays[role] = p, q, y
            memo = {}
            def pointkey(point): return tuple(tuple(s) for s in point['supports']), tuple(point['r']), tuple(point['x'])
            def losses(point):
                key = pointkey(point)
                if key not in memo:
                    values = {}
                    for role, (p, q, y) in arrays.items():
                        predicted = np.zeros_like(y)
                        for s, rv, xv in zip(point['supports'], point['r'], point['x'], strict=True):
                            predicted[:, s] += (rv * p[:, s].sum(axis=1) + xv * q[:, s].sum(axis=1))[:, None]
                        values[role] = float(np.mean(np.abs(y - predicted)))
                    memo[key] = values
                return memo[key]
            def active(point):
                return family([[labels[i] for i in s] for s, r, x in zip(point['supports'], point['r'], point['x'], strict=True) if max(r, x) > 1e-9])
            def error(point): return len(active(point) ^ truth)
            groups = dict(selection=fit['selection_candidates'], bank=fit['bank'], starts=fit['start_preparation'], models=list(fit['models'].values()))
            groups.update({'path_' + k: v['path'] for k, v in fit['searches'].items()})
            groups['selected_raw'] = [v['selected_uncontracted'] for v in fit['models'].values()]
            for group, points in groups.items():
                for j, point in enumerate(points):
                    for field, role in [('objective', 'training'), ('train_mae', 'training'), ('validation_mae', 'validation')]:
                        if field in point:
                            discrepancy = abs(losses(point)[role] - point[field]); residual_errors.append(discrepancy)
                            check(f'{name}_{group}_{j}_{field}', discrepancy <= 2e-12, discrepancy)
                    check(f'{name}_{group}_{j}_nonnegative_bound', min(point['r'] + point['x']) >= 0 and max(point['r'] + point['x']) <= 2)
            cal = fit['calibrations']['raw']; tau = .16 * cal['median_root_depth']
            def shrink(point):
                keep = [i for i, s in enumerate(point['supports']) if len(s) == 1 or cal['a'] * point['r'][i] + cal['b'] * point['x'][i] + 1e-10 >= tau]
                return dict(supports=[point['supports'][i] for i in keep], r=[point['r'][i] for i in keep], x=[point['x'][i] for i in keep])
            candidates = fit['selection_candidates']
            model_lists = dict(rnj_bank=fit['bank'], baseline25=candidates[:len(fit['bank']) + len(fit['searches']['primary']['path'])], multistart_original_rule=candidates)
            model_results = {}
            for model_name, points in model_lists.items():
                vals = [losses(v)['validation'] for v in points]; floor = min(vals)
                eligible = [i for i, value in enumerate(vals) if value <= 1.03 * floor + 1e-12]
                chosen = min(eligible, key=lambda i: (len(points[i]['supports']), vals[i], i))
                saved = fit['models'][model_name]
                check(name + '_' + model_name + '_selection', chosen == saved['selection_index'] and pointkey(saved['selected_uncontracted']) == pointkey(points[chosen]))
                check(name + '_' + model_name + '_contraction', pointkey(saved) == pointkey(shrink(points[chosen])))
                check(name + '_' + model_name + '_reported_error', error(saved) == row['models'][model_name]['full_errors'])
                check(name + '_' + model_name + '_reported_raw_error', error(points[chosen]) == row['models'][model_name]['before_contraction_errors'])
                model_results[model_name] = dict(final_errors=error(saved), raw_errors=error(points[chosen]),
                    final_complete=active(saved) == truth, selected_index=chosen, selected_origin=saved.get('selected_origin', saved.get('origin')),
                    missed=[list(s) for s in sorted(truth - active(saved))], extra=[list(s) for s in sorted(active(saved) - truth)],
                    saved_raw_correct=sum(error(point) == 0 for point in points), eligible_raw_correct=sum(error(points[i]) == 0 for i in eligible),
                    contract_deleted_true=[list(s) for s in sorted((active(points[chosen]) - active(saved)) & truth)],
                    contract_deleted_false=[list(s) for s in sorted((active(points[chosen]) - active(saved)) - truth)])
            lp_by_family = {family(v['family']): v for v in fit['LP_solves']}
            for j, item in enumerate(fit['LP_solves']):
                check(f'{name}_LP_{j}_status', item['status'] == 2 and item['accepted'])
                check(f'{name}_LP_{j}_saved_feasibility', max(item['max_constraint_violation'], item['max_bound_violation']) <= 1e-7)
                check(f'{name}_LP_{j}_saved_objective', abs(item['objective'] - item['physical']) <= 2e-12)
            paths = []
            for path_name, path in fit['searches'].items():
                cursor = 0; accepted = 0; cross = 0
                for j, decision in enumerate(path['decisions']):
                    point = path['path'][cursor]; current_family = family(point['supports'])
                    check(f'{name}_{path_name}_{j}_current', abs(point['objective'] - decision['current']) <= 2e-12)
                    expected = {s for s in family(fit['pool_support_indices']) if len(s) >= 2 and s not in current_family}
                    found = {tuple(v['candidate']) for v in decision['candidates']}
                    check(f'{name}_{path_name}_{j}_sweep', found == expected if decision['full_sweep'] else found <= expected)
                    best = None
                    for k, candidate in enumerate(decision['candidates']):
                        c = tuple(candidate['candidate']); removed = {s for s in current_family if not compatible(s, c)}
                        check(f'{name}_{path_name}_{j}_{k}_cross', family(candidate['removed']) == frozenset(removed))
                        proposal = frozenset((set(current_family) - removed) | {c})
                        check(f'{name}_{path_name}_{j}_{k}_LP', proposal in lp_by_family and abs(candidate['objective'] - lp_by_family[proposal]['physical']) <= 2e-12)
                        if best is None or candidate['objective'] < best['objective'] - 1e-12: best = candidate
                    gain = 0. if best is None else point['objective'] - best['objective']
                    check(f'{name}_{path_name}_{j}_gain', abs(gain - decision['gain']) <= 2e-12)
                    if best is not None and gain > 1e-10 + .001 * point['objective']:
                        cursor += 1; accepted += 1; cross += bool(best['removed'])
                        check(f'{name}_{path_name}_{j}_accept', cursor < len(path['path']) and abs(path['path'][cursor]['objective'] - best['objective']) <= 2e-12)
                    else:
                        check(f'{name}_{path_name}_{j}_stop', j == len(path['decisions']) - 1)
                check(name + '_' + path_name + '_length', cursor + 1 == len(path['path']))
                if path['reason'] == 'pool_local_optimum': check(name + '_' + path_name + '_local_full_sweep', path['decisions'][-1]['full_sweep'])
                paths.append(dict(name=path_name, reason=path['reason'], accepted=accepted, cross=cross, saved_points=len(path['path'])))
            outputs[name] = dict(status='audited', models=model_results, paths=paths, LP_count=len(fit['LP_solves']),
                actual_payment=row.get('actual_payment'), committed_payment=row.get('committed_payment'), distinct_coefficient_points=len(memo))
        stop = None
        if truth is not None:
            historical = load_json(OLD / 'baseline_request/result.json')
            stop = {}
            for name, point in historical['models'].items():
                active_old = family([[labels[i] for i in s] for s, r, x in zip(point['supports'], point['r'], point['x'], strict=True) if max(r, x) > 1e-9])
                stop[name] = dict(full_errors=len(active_old ^ truth), complete=active_old == truth)
                check('historic_stop_' + name, stop[name] == ended['historical_stop'][name])
        comparisons = []
        zero = outputs.get('zero', {})
        if zero.get('status') == 'audited':
            for policy, result in outputs.items():
                if result.get('status') != 'audited': continue
                for model_name, value in result['models'].items():
                    comparisons.append(dict(policy=policy, model=model_name,
                        error_delta_vs_zero=value['final_errors'] - zero['models'][model_name]['final_errors'],
                        error_delta_vs_historic_stop=value['final_errors'] - stop[model_name]['full_errors'],
                        contraction_error_delta=value['final_errors'] - value['raw_errors']))
        result = dict(status='passed', checks_total=len(checks), checks_passed=sum(v['passed'] for v in checks),
            checks=checks, outputs=outputs, comparisons=comparisons, historical_stop=stop,
            LP_ledger=ledger, unknown_LP_caches=unknown, total_saved_LP=sum(v['count'] for v in ledger.values()),
            planned=8, audited=sum(v.get('status') == 'audited' for v in outputs.values()),
            residual_comparisons=len(residual_errors), max_residual_error=max(residual_errors, default=0.),
            independent_natural_paths=1, price_optimizer=False, new_LP_calls=0, new_fits=0, new_AC_calls=0,
            limitations=['Saved optimal statuses are not independent optimization proofs.', 'Failure/unknown records retained; one shared path is not a recovery probability estimate.'],
            input_sha256=input_hashes, audit_source_sha256=sha(__file__), audit_protocol_sha256=sha(ROOT / 'protocol.json'))
        write(ROOT / 'receipt.json', result)
        summary = {k: v for k, v in result.items() if k not in ('checks', 'input_sha256')}
        write(ROOT / 'summary.json', summary)
        print(json.dumps({k: result[k] for k in ('status', 'checks_total', 'audited', 'total_saved_LP', 'max_residual_error')}))
    except BaseException as exc:
        write(ROOT / 'failure.json', dict(status='failed', error_type=type(exc).__name__, message=str(exc),
            traceback=traceback.format_exc(), checks=checks, input_sha256=input_hashes, source_sha256=sha(__file__)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['prepare', 'run'])
    args = parser.parse_args(); (prepare if args.mode == 'prepare' else run)()
