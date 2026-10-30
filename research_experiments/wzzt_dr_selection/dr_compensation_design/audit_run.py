"""Independent post-run audit; no project optimizer or simulator is imported.

Reads private observed packets and receipts after the bounded run. Reconstructs
prediction, selection, structure errors, controls, payments and observations
with standard-library/NumPy arithmetic. It never fits or changes a policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import pickle

import numpy as np

METHODS = ('current_rebuilt_rnj', 'current_pool_wzzt', 'current_root_augmented_rnj')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def normalize(packet, role):
    """Directly reconstruct trusted observed y; do not use saved drop_target."""
    p = []; q = []; y = []; indices = []; labels = None
    for scene in packet[role]:
        current = list(scene['P_terminal'].columns)
        if labels is None:
            labels = current
        if current != labels or list(scene['Q_terminal'].columns) != labels or list(scene['V_terminal'].columns) != labels:
            raise ValueError('inconsistent observed terminal columns')
        pp = scene['P_terminal'].to_numpy(dtype=float)
        qq = scene['Q_terminal'].to_numpy(dtype=float)
        vv = scene['V_terminal'].to_numpy(dtype=float)
        rr = scene['V_root_observed'].to_numpy(dtype=float)
        if pp.shape != qq.shape or pp.shape != vv.shape or rr.shape != (len(pp),):
            raise ValueError('misaligned observed arrays')
        p.append(pp); q.append(qq); y.append(rr[:, None]**2 - vv**2)
        indices.extend(str(v) for v in scene['P_terminal'].index)
    return np.concatenate(p), np.concatenate(q), np.concatenate(y), labels, indices


def family(supports):
    return tuple(tuple(int(i) for i in s) for s in supports)


def compatible(a, b):
    a = set(a); b = set(b)
    return not a & b or a <= b or b <= a


def effective(point):
    return {tuple(s) for s, r, x in zip(point['supports'], point['r'], point['x']) if max(r, x) > 1e-9}


def matrices(point, n):
    supports = family(point['supports'])
    if len(set(supports)) != len(supports) or any(not s or len(set(s)) != len(s) or min(s) < 0 or max(s) >= n for s in supports):
        raise ValueError('invalid selected supports')
    if any(not compatible(a, b) for i, a in enumerate(supports) for b in supports[:i]):
        raise ValueError('nonlaminar selected family')
    rr = np.asarray(point['r'], float); xx = np.asarray(point['x'], float)
    if len(rr) != len(supports) or len(xx) != len(supports):
        raise ValueError('coefficient/support mismatch')
    if min(rr.min(initial=0), xx.min(initial=0)) < -1e-8 or max(rr.max(initial=0), xx.max(initial=0)) > 2 + 1e-8:
        raise ValueError('coefficient outside unchanged [0,2] domain')
    r = np.zeros((n, n)); x = np.zeros_like(r)
    for support, rv, xv in zip(supports, rr, xx):
        r[np.ix_(support, support)] += rv
        x[np.ix_(support, support)] += xv
    return r, x


def losses(point, train, valid):
    r, x = matrices(point, train[0].shape[1])
    return tuple(float(np.mean(abs(p @ r.T + q @ x.T - y))) for p, q, y in (train[:3], valid[:3]))


def truth(cfg, c, pair):
    """Evaluator-only construction, independent of original.truth()."""
    n = 3; b = float(cfg['b']); z = np.zeros(n); z[pair] = 1.
    d = float(cfg['L']) - 3*b - 2*float(c)*z
    K = np.diag(d) + b*np.ones((n, n)) + float(c)*np.outer(z, z)
    R = K / (1 + float(cfg['kappa'])); X = R.copy()
    target = {(i,) for i, value in enumerate(d) if value > 0}
    if b > 0:
        target.add(tuple(range(n)))
    if c > 0:
        target.add(tuple(pair))
    return R, X, target


def replay_controls(receipt, menu):
    prices = np.asarray(menu['price_rows'], int)[np.arange(4) % 2]
    pairs = np.asarray(menu['pair_rows'], int)[np.arange(4) % 2]
    z = np.zeros((4, 3), bool); orientation = np.empty(4, int)
    probability = np.array([0., .5, .75])
    seed = int(receipt['behavior_seed'])
    for event in range(4):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 76001, event]))
        u = rng.random(3) if receipt['dependence'] == 'local' else np.repeat(rng.random(), 3)
        z[event] = u < probability[prices[event]]
        rng_s = np.random.default_rng(np.random.SeedSequence([seed, 76002, event]))
        orientation[event] = 1 if rng_s.integers(0, 2) else -1
    wave = np.r_[np.full(4, -.2), np.full(4, .2)]
    dp = np.zeros((4, 8, 3))
    for event, (a, b) in enumerate(pairs):
        dp[event, :, a] = orientation[event] * wave * z[event, a]
        dp[event, :, b] = -orientation[event] * wave * z[event, b]
    payment = .2 * np.sum(prices*z, axis=1)
    return prices, pairs, z, orientation, dp, payment


def replay_observed_future(cfg, R, X, receipt, dp):
    result = []
    t = np.arange(96) / 96
    regime = receipt['noise_regime']
    for event in range(4):
        day = 6 + event
        rng = np.random.default_rng(np.random.SeedSequence([884001, day]))
        scalar_kw = 1. + .15*np.sin(2*np.pi*(t-.12)) + .03*rng.uniform(-1, 1, 96)
        p = np.repeat((scalar_kw / 200.)[:, None], 3, axis=1)
        q = .35*p
        delta_p = np.zeros_like(p); delta_p[76:84] = dp[event] / 200.
        delta_q = .35*delta_p
        drop = float(cfg['L'])*p[:, [0]]*np.ones((1, 3)) + delta_p @ R.T + delta_q @ X.T
        root = 1. + .003*np.sin(2*np.pi*(t+.07*day))
        v = np.sqrt(root[:, None]**2-drop)
        p = p + delta_p; q = q + delta_q
        rng_m = np.random.default_rng(np.random.SeedSequence([int(receipt['measurement_seed']), day]))
        ep, eq, ev, er = (rng_m.standard_normal(shape) for shape in (p.shape, q.shape, v.shape, root.shape))
        result.append((p*(1+regime['pq_relative']*ep), q*(1+regime['pq_relative']*eq),
                       v*(1+regime['voltage_relative']*ev), root*(1+regime['voltage_relative']*er)))
    return result


def audit(root, outpath=None):
    root = Path(root).resolve()
    cfg = read(root/'protocol.json'); rows = read(root/'evaluations.json')
    freeze = read(root/'actual_actions_frozen.json')
    checks = []; errors = []; metrics = []; source_hashes = {}

    def check(name, ok, details=None):
        entry = dict(check=name, passed=bool(ok))
        if details is not None:
            entry['details'] = details
        checks.append(entry)
        if not ok:
            errors.append(entry)

    check('protocol_sha256', sha(root/'protocol.json') == read(root/'protocol.sha256.json')['sha256'])
    for p, expected in cfg['source_sha256'].items():
        actual = sha(p); source_hashes[p] = actual
        check('source_hash', actual == expected, p)
    for entry in (cfg['history'], cfg['history_state']):
        check('history_hash', sha(entry['path']) == entry['sha256'], entry['path'])
    check('freeze_before_generation_declaration', freeze.get('actual_future_generated') is False)
    check('price_design_frozen_hash', freeze['planning_source'] == sha(root/'planning/price_design.json'))
    check('freeze_source_binding', freeze['source_sha256'] == cfg['source_sha256'])
    frozen_time = freeze.get('created_utc', freeze.get('frozen_utc'))
    check('explicit_price_freeze_timestamp_available', frozen_time is not None)
    freeze_hash = sha(root/'actual_actions_frozen.json')
    actions = {a['flow']['id']: a for a in freeze['actions']}
    flows = {f['id']: f for f in cfg['actual_flows']}
    history = pickle.loads(Path(cfg['history']['path']).read_bytes())
    ht = normalize(history, 'training'); hv = normalize(history, 'validation')
    check('all_old_rows_retained', len(ht[0]) == 288 and len(hv[0]) == 288)
    check('historical_labels', ht[3] == cfg['labels'] and hv[3] == cfg['labels'])
    row_keys = [(r['flow'], r['case'], r['policy']) for r in rows]
    check('mapped_keys_unique', len(set(row_keys)) == len(row_keys))
    expected_case_ids = ['zero', 'weak_01', 'weak_02', 'weak_12', 'strong_01', 'strong_02', 'strong_12']
    expected = {(f, c, p) for f in flows for c in expected_case_ids for p in cfg['policies']}
    check('no_undeclared_mapping', set(row_keys) <= expected)
    completed_fit_receipts = {}; total_actual_LP = 0
    costs = defaultdict(lambda: dict(known_payment=0., known=0, unknown=0))
    for row in rows:
        key = (row['flow'], row['case'], row['policy'])
        cost = costs[row['policy']]
        if 'actual_payment' in row:
            cost['known_payment'] += row['actual_payment']; cost['known'] += 1
        else:
            cost['unknown'] += 1
        menu = actions[row['flow']]['menus'][row['policy']]
        check('frozen_menu_identity', row['menu_id'] == menu['id'], key)
        if 'execution_receipt' not in row:
            continue
        rp = Path(row['execution_receipt']); receipt = read(rp)
        check('behavior_seed_frozen', receipt['behavior_seed'] == flows[row['flow']]['behavior'], key)
        check('measurement_seed_frozen', receipt['measurement_seed'] == flows[row['flow']]['measurement'], key)
        check('natural_seed_unchanged', receipt['natural_seed'] == 884001, key)
        check('receipt_menu_frozen', receipt['menu'] == menu, key)
        if frozen_time is not None:
            check('price_frozen_before_future_receipt', datetime.fromisoformat(frozen_time) <= datetime.fromisoformat(receipt['created_utc']), key)
        if 'action_freeze_sha256' in receipt:
            check('receipt_freeze_hash', receipt['action_freeze_sha256'] == freeze_hash, key)
        prices, pairs, accepted, orientation, dp, payment = replay_controls(receipt, menu)
        saved = receipt['controls']
        check('control_prices', np.array_equal(prices, saved['prices']), key)
        check('control_pairs', np.array_equal(pairs, saved['pairs']), key)
        check('participation_stream_replay', np.array_equal(accepted, saved['accepted']), key)
        check('independent_direction_stream_replay', np.array_equal(orientation, saved['orientation']), key)
        check('whole_event_P_replay', np.max(abs(dp-np.asarray(saved['delta_p_kw']))) <= 1e-15, key)
        check('whole_event_Q_replay', np.max(abs(.35*dp-np.asarray(saved['delta_q_kvar']))) <= 1e-15, key)
        check('task_power_bounds', np.min(.2+dp) >= -1e-12 and np.max(.2+dp) <= .4+1e-12, key)
        check('terminal_task_energy', np.allclose(np.sum(.2+dp, axis=1)*.25, .4, atol=1e-12, rtol=0), key)
        check('four_event_worst_budget', .2*float(prices.sum()) <= 1.6+1e-12, key)
        check('actual_payment_replay', abs(float(payment.sum())-receipt['actual_payment']) <= 1e-12 and abs(receipt['actual_payment']-row['actual_payment']) <= 1e-12, key)
        h = receipt['packet_sha256']; fid = 's_'+h[:20]
        check('fit_identity', row['fit_id'] == fid, key)
        packet_path = root/'private_observations'/'packets'/(h+'.pkl')
        check('future_packet_hash', sha(packet_path) == h, key)
        packet = pickle.loads(packet_path.read_bytes())
        ft = normalize(packet, 'training'); fv = normalize(packet, 'validation')
        R, X, target = truth(cfg, row['c'], row['pair'])
        check('simulator_parameter_hash', sha256(R.tobytes()+X.tobytes()).hexdigest() == receipt['simulator_parameter_sha256'], key)
        replay = replay_observed_future(cfg, R, X, receipt, dp)
        observation_diff = 0.
        for scene, values in zip(packet['training']+packet['validation'], replay):
            stored = [scene[k].to_numpy(dtype=float) for k in ('P_terminal', 'Q_terminal', 'V_terminal', 'V_root_observed')]
            observation_diff = max(observation_diff, *(float(np.max(abs(a-b))) for a, b in zip(stored, values)))
        check('actual_observation_replay', observation_diff <= 2e-14, dict(key=key, max_abs=observation_diff))
        train = tuple(np.concatenate([ht[i], ft[i]]) for i in range(3))
        valid = tuple(np.concatenate([hv[i], fv[i]]) for i in range(3))
        check('all_actual_rows_retained', len(train[0]) == 480 and len(valid[0]) == 480, key)
        check('actual_labels', ft[3] == cfg['labels'] and fv[3] == cfg['labels'], key)
        check('train_validation_disjoint', not set(ht[4]+ft[4]) & set(hv[4]+fv[4]), key)
        result_path = root/'records'/(fid+'.json')
        result = read(result_path)
        check('identifier_no_truth', result.get('truth_input_allowed') is False, key)
        current = read(result['current_state_path']) if Path(result['current_state_path']).exists() else None
        if current is not None and current['status'] == 'complete':
            check('one_current_calibration', result.get('applied_calibration') == current['calibration'], key)
        if fid not in completed_fit_receipts:
            completed_fit_receipts[fid] = result
            total_actual_LP += len(result.get('shared_preprocessing_LP_solves', []))
            total_actual_LP += sum(len(m.get('LP_solves', [])) for m in result.get('methods', {}).values())
        for method in METHODS:
            m = result.get('methods', {}).get(method, {})
            evaluation = row.get('evaluated', {}).get(method, {})
            if m.get('status') != 'complete':
                check('unresolved_not_reported_complete', evaluation.get('status') != 'complete', (key, method))
                continue
            points = m['points']; selected = m['selected']; raw = selected['selected_uncontracted']
            final = dict(supports=selected['support_indices'], r=selected['r_values'], x=selected['x_values'])
            ftr, fva = losses(final, train, valid)
            rtr, rva = losses(raw, train, valid)
            check('final_train_L1', abs(ftr-selected['train_mae']) <= 2e-12, (key, method))
            check('final_validation_L1', abs(fva-selected['validation_mae']) <= 2e-12, (key, method))
            check('raw_train_L1', abs(rtr-raw['objective']) <= 2e-12, (key, method))
            check('raw_validation_L1', abs(rva-raw['validation_mae']) <= 2e-12, (key, method))
            independent_val = []
            for point in points:
                ptr, pva = losses(point, train, valid); independent_val.append(pva)
                check('saved_point_train_L1', abs(ptr-point['objective']) <= 2e-12, (key, method))
                check('saved_point_validation_L1', abs(pva-point['validation_mae']) <= 2e-12, (key, method))
            best = min(independent_val)
            eligible = [i for i, value in enumerate(independent_val) if value <= 1.03*best+1e-12]
            index = min(eligible, key=lambda i:(len(points[i]['supports']), independent_val[i], i))
            check('independent_3pct_selection', index == selected['selection_index'], (key, method))
            cal = current['calibration']
            keep = [i for i, support in enumerate(raw['supports']) if len(support) == 1 or cal['a']*raw['r'][i]+cal['b']*raw['x'][i]+1e-10 >= .16*cal['median_root_depth']]
            reconstructed = dict(supports=[raw['supports'][i] for i in keep], r=[raw['r'][i] for i in keep], x=[raw['x'][i] for i in keep])
            check('original_contraction_replay', reconstructed == final, (key, method))
            expected_final = len(effective(final)^target); expected_raw = len(effective(raw)^target)
            best_saved = min(len(effective(point)^target) for point in points)
            check('final_structure_errors', evaluation.get('final_errors') == expected_final, (key, method))
            check('raw_structure_errors', evaluation.get('raw_errors') == expected_raw, (key, method))
            check('best_saved_structure_errors', evaluation.get('best_saved_errors') == best_saved, (key, method))
            check('evaluation_target', {tuple(s) for s in evaluation.get('target', [])} == target, (key, method))
            check('final_supports_report', {tuple(s) for s in evaluation.get('final', [])} == effective(final), (key, method))
            check('raw_supports_report', {tuple(s) for s in evaluation.get('raw', [])} == effective(raw), (key, method))
            check('selected_origin', evaluation.get('origin') == selected.get('selected_origin'), (key, method))
            if m.get('paths'):
                all_scans = all(path['reason'] == 'pool_local_optimum' for path in m['paths'].values())
                check('full_scan_status', all_scans == m['all_scans_complete'] == evaluation['all_scans_complete'], (key, method))
            for lp in m.get('LP_solves', []):
                check('completed_method_LP_status', lp.get('status') == 2 and lp.get('accepted') is True, (key, method))
            metrics.append(dict(flow=row['flow'], case=row['case'], policy=row['policy'], method=method,
                                final_errors=expected_final, raw_errors=expected_raw, best_saved_errors=best_saved,
                                train_L1=ftr, validation_L1=fva))
    summary_path = root/'summary.json'
    if summary_path.exists():
        summary = read(summary_path)
        check('summary_recorded_maps', summary['recorded_maps'] == len(rows))
        check('summary_unique_fit_count', summary['unique_fits'] == len({r['fit_id'] for r in rows if 'fit_id' in r}))
        check('global_LP_budget', summary['LP_calls'] <= cfg['computation']['max_LP_calls'])
        check('logged_actual_LP_within_total', total_actual_LP <= summary['LP_calls'])
        for item in summary['summary']:
            values = [m for m in metrics if m['policy'] == item['policy'] and m['method'] == item['method']]
            check('summary_complete_denominator', item['complete'] == len(values), (item['policy'], item['method']))
            check('summary_final_errors', item['final_errors'] == sum(v['final_errors'] for v in values), (item['policy'], item['method']))
            check('summary_raw_errors', item['raw_errors'] == sum(v['raw_errors'] for v in values), (item['policy'], item['method']))
            check('summary_full_recovery', item['full'] == sum(v['final_errors'] == 0 for v in values), (item['policy'], item['method']))
    result = dict(status='passed' if not errors else 'failed', created_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha(__file__), protocol_sha256=sha(root/'protocol.json'),
                  frozen_actions_sha256=freeze_hash, checks=len(checks), failures=errors,
                  recorded_maps=len(rows), planned_maps=len(expected), missing_maps=sorted(expected-set(row_keys)),
                  unique_fit_receipts=len(completed_fit_receipts), independently_counted_actual_LP_calls=total_actual_LP,
                  methods_by_status=dict(Counter(m.get('status', 'missing') for r in completed_fit_receipts.values() for m in r.get('methods', {}).values())),
                  known_costs_including_failed_fits=dict(costs), metrics=metrics,
                  no_optimizer_calls=True, no_policy_changes=True, audit_truth_used_only_after_selection=True,
                  all_checks=checks)
    target_path = Path(outpath) if outpath is not None else root/'independent_audit.json'
    if target_path.exists():
        raise FileExistsError('existing independent audit retained')
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out')
    args = parser.parse_args()
    report = audit(args.root, args.out)
    print(json.dumps({k: report[k] for k in ('status', 'checks', 'recorded_maps', 'unique_fit_receipts', 'failures')}, ensure_ascii=False))
