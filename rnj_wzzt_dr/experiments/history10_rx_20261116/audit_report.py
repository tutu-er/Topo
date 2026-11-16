"""Independent, read-only audit and summary of the paired 2/8-day RX study.

Usage: python -m ...history10_rx_20261116.audit_report --run RUN --out NEW
Writes NEW/report/{audit.json,report.json,REPORT.md}. No fit, optimization or AC
simulation is called. All model predictions and fixed-QP gradients are rebuilt
from coefficients and saved observations rather than production metric code.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...dr.selection_mse import choose_mse

ROOT = Path(__file__).resolve().parents[3]
CASES = ('soumalas11', 'pengwah18')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def near(a, b, message, *, rtol=2e-8, atol=1e-14):
    require(np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol), message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class Inputs:
    def __init__(self):
        self.checked = {}

    def path(self, value):
        item = ref(value) if isinstance(value, (str, Path)) else value
        path = Path(item['path']).resolve()
        key = str(path)
        if key in self.checked:
            require(self.checked[key]['sha256'] == item['sha256'], 'Conflicting source hash: '+key)
        else:
            require(sha(path) == item['sha256'], 'Source hash mismatch: '+key)
            self.checked[key] = dict(path=key, sha256=item['sha256'])
        return path

    def json(self, value):
        return read(self.path(value))

    def npz(self, value):
        with np.load(self.path(value), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    def final_check(self):
        for item in self.checked.values():
            require(sha(item['path']) == item['sha256'], 'Input changed during audit: '+item['path'])


def response(point, n):
    """Independent support-wise outer-product accumulation."""
    result = np.zeros((n, 2*n))
    require(len(point['supports']) == len(point['r']) == len(point['x']), 'Coefficient lengths differ')
    for support, r, x in zip(point['supports'], point['r'], point['x']):
        ix = np.asarray(support, int)
        require(len(ix) > 0 and len(set(ix)) == len(ix) and min(ix) >= 0 and max(ix) < n, 'Invalid support')
        require(np.isfinite([r, x]).all(), 'Nonfinite coefficients')
        result[np.ix_(ix, ix)] += r
        result[np.ix_(ix, ix+n)] += x
    return result


def topology(point, truth, n, zero):
    predicted = {tuple(sorted(c)) for c, r, x in zip(point['supports'], point['r'], point['x']) if max(r, x) > zero}
    physical = {tuple(sorted(c)) for c in truth}
    pi, ti = ({c for c in supports if 1 < len(c) < n} for supports in (predicted, physical))
    return dict(full_exact=predicted == physical, internal_exact=pi == ti,
                full_missing=len(physical-predicted), full_extra=len(predicted-physical),
                internal_missing=len(ti-pi), internal_extra=len(pi-ti))


def loss(H, observation):
    error = observation['Y']-np.c_[observation['P'], observation['Q']]@H.T
    return dict(sse=float(np.sum(error.astype(np.longdouble)**2)),
                mse=float(np.mean(error.astype(np.longdouble)**2)),
                mae=float(np.mean(np.abs(error))), cells=int(error.size))


def reconstruct_law(clean, noise, law, Hstar):
    value = {key: np.asarray(clean[key], float).copy() for key in ('P', 'Q', 'V', 'root')}
    require(law in ('cleanlinear', 'cleanAC', 'Vonly', 'all'), 'Unknown observation law')
    if law == 'cleanlinear':
        y = np.c_[value['P'], value['Q']]@Hstar.T
        value['V'] = np.sqrt(value['root'][:, None]**2-y)
    else:
        for key in value:
            sd = (.005 if key in ('P', 'Q') else .0005) if law == 'all' else (.0005 if law == 'Vonly' and key in ('V', 'root') else 0.)
            value[key] *= 1+sd*np.asarray(noise[key])
        y = value['root'][:, None]**2-value['V']**2
    value['Y'] = y
    return value


def slices(observed, world, days):
    return {name: {key: values[np.asarray(ix, int)] for key, values in observed.items()}
            for name, ix in [('train', world['train_indices'][str(days)]),
                             ('validation', world['validation_indices']), ('new_day', world['new_day_indices'])]}


def fixed_qp_check(point, observation, record):
    """Independent supporting-plane lower bound of the exact bounded LS QP."""
    p, q, y = (np.asarray(observation[k], float) for k in ('P', 'Q', 'Y'))
    t, n = p.shape
    columns = []
    for values in (p, q):
        for support in point['supports']:
            col = np.zeros((t, n))
            col[:, support] = values[:, support].sum(axis=1)[:, None]
            columns.append(col.reshape(-1))
    design = np.column_stack(columns) if columns else np.empty((t*n, 0))
    theta = np.r_[point['r'], point['x']].astype(np.longdouble)
    reference = float(point['reference_mse']); bound = float(record['coefficient_bound'])
    require(reference > 0 and bound > 0, 'Invalid QP scale/bound')
    require(np.all(theta >= -1e-12) and np.all(theta <= bound+1e-12), 'QP coefficient outside original domain')
    d = design.astype(np.longdouble)
    error = d@theta-y.reshape(-1).astype(np.longdouble)
    objective = float(np.mean(error**2)/reference)
    gradient = 2*d.T@error/(error.size*reference)
    lower = max(0., objective+float(np.minimum(-gradient*theta, gradient*(bound-theta)).sum()))
    near(objective, record['objective'], 'Fixed QP objective mismatch', atol=2e-10)
    near(lower, record['lower_bound'], 'Fixed QP supporting-plane bound mismatch', atol=2e-8)
    near(objective-lower, record['certificate_gap'], 'Fixed QP gap mismatch', atol=2e-8)
    near(objective*reference, point['mse'], 'Fixed QP physical MSE mismatch', atol=1e-16)
    require(lower <= objective+1e-10, 'QP lower bound exceeds objective')
    require(objective-lower <= 1e-6*(1+objective)+2e-8, 'Fixed QP not certified to existing acceptance tolerance')
    return dict(objective=objective, lower_bound=lower, certificate_gap=objective-lower,
                coefficient_min=float(theta.min()) if len(theta) else None,
                coefficient_max=float(theta.max()) if len(theta) else None,
                parameter_identifiability_not_proved=True)


def response_stats(H, star, P, Q):
    n = P.shape[1]
    result = {}
    for name, a, b in [('R', H[:, :n], star[:, :n]), ('X', H[:, n:], star[:, n:])]:
        denominator = np.linalg.norm(b)
        result[name] = dict(error_norm=float(np.linalg.norm(a-b)),
                            relative_error=float(np.linalg.norm(a-b)/denominator) if denominator else None)
    er, ex = P@(H[:, :n]-star[:, :n]).T, Q@(H[:, n:]-star[:, n:]).T
    result['prediction'] = dict(R_error_RMS=float(np.sqrt(np.mean(er**2))),
        X_error_RMS=float(np.sqrt(np.mean(ex**2))), total_error_RMS=float(np.sqrt(np.mean((er+ex)**2))),
        cross_R_X_MSE=float(2*np.mean(er*ex)))
    return result


def check_recorded_metrics(record, H, star, P, Q):
    independent = response_stats(H, star, P, Q)
    for axis in ('R', 'X'):
        for key in ('error_norm', 'relative_error'):
            near(independent[axis][key], record[axis][key], 'Recorded response metric differs')
    for key, value in independent['prediction'].items():
        near(value, record['prediction'][key], 'Recorded prediction-error component differs')
    for col in record['columns']:
        i = int(col['input_index']); n = P.shape[1]
        for axis, k in [('R', i), ('X', i+n)]:
            near(np.linalg.norm(H[:, k]-star[:, k]), col[axis]['error_norm'], 'Input-column response error differs')
    # Use least squares rather than the production SVD-projection routine.
    residual = Q-P@np.linalg.lstsq(P, Q, rcond=None)[0]
    near(np.sum(residual**2), record['historical_design']['uncentered']['Q_given_P']['residual_energy'],
         'Conditional Q energy differs', atol=2e-14)
    return independent


def _svd_design(P, Q, supports):
    n, t = P.shape[1], len(P)
    blocks = []
    for values in (P, Q):
        block = []
        for support in supports:
            feature = np.zeros((t, n))
            feature[:, support] = values[:, support].sum(axis=1)[:, None]
            block.append(feature.reshape(-1))
        blocks.append(np.column_stack(block) if block else np.empty((t*n, 0)))
    return blocks


def check_design(record, P, Q, supports):
    fr, fx = _svd_design(P, Q, supports)
    f = np.c_[fr, fx]
    singular = np.linalg.svd(f, compute_uv=False)
    near(singular, record['design']['raw']['singular_values'], 'Fixed-support singular values differ', atol=1e-13)
    leftover = fx-fr@np.linalg.lstsq(fr, fx, rcond=None)[0]
    near(np.sum(leftover**2), record['X_given_R']['residual_energy'], 'Fixed-support conditional X energy differs', atol=2e-13)


def _network_reference(case, labels, inputs):
    folder = ROOT/'topology_identification_standalone/cases'/case
    meta = inputs.json(folder/'metadata.json')
    with inputs.path(folder/'branches.csv').open(encoding='utf-8-sig', newline='') as stream:
        branches = [row for row in csv.DictReader(stream) if row['status'].lower() in ('true', '1')]
    parent = {int(b['to_bus']): int(b['from_bus']) for b in branches}
    edges = {int(b['to_bus']): b for b in branches}
    root = int(meta['root_bus']); paths = []
    for node in labels:
        visited = set(); current = int(node)
        while current != root:
            require(current in parent and current not in visited, 'Physical network is not rooted radial')
            visited.add(current); current = parent[current]
        paths.append(visited)
    n = len(labels); star = np.zeros((n, 2*n))
    zbase = float(meta['base_kv'])**2/float(meta['base_mva'])
    for i in range(n):
        for j in range(n):
            shared = paths[i] & paths[j]
            star[i, j] = 2*sum(float(edges[v]['r_ohm']) for v in shared)/zbase
            star[i, n+j] = 2*sum(float(edges[v]['x_ohm']) for v in shared)/zbase
    supports = sorted({tuple(i for i, path in enumerate(paths) if edge in path) for edge in edges})
    supports = [list(s) for s in supports if s]
    return star, supports, meta, edges


def check_ac(state, clean, meta, edges):
    base = float(meta['base_mva'])*1000
    ids = list(map(int, state['bus_ids'])); pos = {node: i for i, node in enumerate(ids)}
    voltage = state['voltage_complex_pu']; current = state['branch_current_complex_pu']
    from_ids, to_ids = map(lambda a: list(map(int, a)), (state['branch_from'], state['branch_to']))
    zbase = float(meta['base_kv'])**2/float(meta['base_mva'])
    z = np.array([complex(float(edges[node]['r_ohm']), float(edges[node]['x_ohm']))/zbase for node in to_ids])
    f, t = [np.array([pos[node] for node in seq]) for seq in (from_ids, to_ids)]
    near((voltage[:, f]-voltage[:, t])/z, current, 'AC Ohm identity differs', atol=1e-10)
    loads = (state['specified_load_kw']+1j*state['specified_load_kvar'])/base
    terminal = np.array([pos[int(node)] for node in clean['labels']])
    near(loads[:, terminal], clean['P']+1j*clean['Q'], 'AC load mapping differs', atol=1e-13)
    near(np.abs(voltage[:, terminal]), clean['V'], 'Clean terminal voltage differs', atol=1e-12)
    near(voltage[:, pos[int(meta['root_bus'])]], clean['root'], 'Root voltage differs', atol=1e-13)
    balance = np.zeros_like(voltage)
    for k, (a, b) in enumerate(zip(f, t)):
        balance[:, b] += current[:, k]; balance[:, a] -= current[:, k]
    nonroot = [i for i in range(len(ids)) if ids[i] != int(meta['root_bus'])]
    mismatch = balance[:, nonroot]-np.conj(loads[:, nonroot]/voltage[:, nonroot])
    maximum = float(np.max(np.abs(mismatch)))
    require(maximum < 1e-7, 'AC KCL residual exceeds existing tolerance')
    loss_pq = np.sum(z*np.abs(current)**2, axis=1)
    rootp = (state['root_supply_kw']+1j*state['root_supply_kvar'])/base
    power = float(np.max(np.abs(rootp-loads.sum(axis=1)-loss_pq))*base)
    require(power < 1e-5, 'AC power balance exceeds existing tolerance')
    return dict(max_KCL_residual_pu=maximum, max_power_balance_kVA=power,
                voltage_min=float(clean['V'].min()), voltage_max=float(clean['V'].max()),
                voltage_outside_095_105_cells=int(np.sum((clean['V'] < .95)|(clean['V'] > 1.05))),
                thermal_limits_available=False, no_new_AC_performed=True)


def _stats(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return dict(n=len(values), mean=float(np.mean(values)), minimum=min(values), maximum=max(values)) if values else dict(n=0, mean=None, minimum=None, maximum=None)


def summarize_main(tasks, rows):
    by_id = {row['id']: row for row in rows}
    retained = [by_id.get(task['id'], dict(**task, status='missing')) for task in tasks]
    groups = []
    for case in sorted({t['case'] for t in tasks}):
        for days in sorted({t['train_days'] for t in tasks if t['case'] == case}):
            subset = [r for r in retained if r['case'] == case and r['train_days'] == days]
            good = [r for r in subset if r['status'] == 'complete']
            row = dict(case=case, train_days=days, planned=len(subset), checked_complete=len(good),
                       failed=sum(r['status'] == 'failed' for r in subset), missing=sum(r['status'] == 'missing' for r in subset))
            for key in ('bank_has_internal_truth', 'bank_has_full_truth', 'history_internal_correct', 'cumulative_internal_correct',
                        'history_full_correct', 'cumulative_full_correct'):
                row[key] = sum(bool(r[key]) for r in good)
            for axis in ('R', 'X'):
                row[axis+'_selected_relative_error'] = _stats(r['history_matrix'][axis]['relative_error'] for r in good)
            row['common_new_linear_response_RMS'] = _stats(r['common_new_response']['prediction']['total_error_RMS'] for r in good)
            row['clean_AC_new_MSE'] = _stats(r['clean_AC_new_MSE'] for r in good)
            row['total_seconds'] = _stats(r.get('seconds') for r in subset)
            row['timed_out_MIQP_count'] = sum(r.get('timed_out_MIQP_count', 0) for r in good)
            row['nonclosed_MIQP_count'] = sum(r.get('nonclosed_MIQP_count', 0) for r in good)
            groups.append(row)
    paired = []
    for case in sorted({t['case'] for t in tasks}):
        for h in sorted({t['history_repeat'] for t in tasks if t['case'] == case}):
            pair = {r['train_days']: r for r in retained if r['case'] == case and r['history_repeat'] == h}
            if not all(d in pair and pair[d]['status'] == 'complete' for d in (2, 8)):
                paired.append(dict(case=case, history_repeat=h, status='unavailable'))
                continue
            a, b = pair[2], pair[8]
            paired.append(dict(case=case, history_repeat=h, status='complete',
                R_error_change_8_minus_2=b['history_matrix']['R']['relative_error']-a['history_matrix']['R']['relative_error'],
                X_error_change_8_minus_2=b['history_matrix']['X']['relative_error']-a['history_matrix']['X']['relative_error'],
                cumulative_restored=not a['cumulative_internal_correct'] and b['cumulative_internal_correct'],
                cumulative_lost=a['cumulative_internal_correct'] and not b['cumulative_internal_correct']))
    return dict(planned=len(tasks), statuses=dict(Counter(r['status'] for r in retained)), groups=groups, paired=paired)


def _solver_rows(value):
    found = []
    if isinstance(value, dict):
        if value.get('solver') == 'Gurobi convex MIQP':
            found.append(value)
        else:
            for sub in value.values(): found.extend(_solver_rows(sub))
    elif isinstance(value, list):
        for sub in value: found.extend(_solver_rows(sub))
    return found


def support_key(supports):
    return tuple(sorted(tuple(sorted(map(int, c))) for c in supports))


def check_offline_coverage(rows, families, days, laws):
    """Derive the denominator from source support families, not result rows."""
    expected = {}
    for case, grouped in families.items():
        for supports, aliases in grouped.items():
            for h in range(4):
                for day in days:
                    for law in laws:
                        expected[(case, h, day, law, aliases[0])] = (supports, aliases)
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in ('case', 'history_repeat', 'train_days', 'law', 'support_role'))
        require(key in expected and key not in seen, 'Unexpected or duplicated offline coordinate: '+str(key))
        seen.add(key)
        supports, aliases = expected[key]
        require(row['support_aliases'] == aliases, 'Offline support aliases differ: '+str(key))
        if row['status'] == 'complete':
            require(support_key(row['point']['supports']) == supports, 'Offline support family differs: '+str(key))
    require(seen == set(expected), 'Offline rows missing from independent planned denominator')
    return len(expected)


class Audit:
    def __init__(self, run):
        self.run = Path(run).resolve(); self.inputs = Inputs(); self.errors = []
        self.protocol = self.inputs.json(self.run/'protocol.json')
        self.world_records = []; self.contexts = {}; self.main = []; self.offline = []

    def sources(self):
        for item in self.protocol['sources']:
            live = self.inputs.path(item)
            snapshot = self.run/'source_snapshot'/live.relative_to(ROOT)
            require(sha(snapshot) == item['sha256'], 'Run snapshot differs: '+str(live))
            self.inputs.path(snapshot)
        for key in ('archive', 'old_inputs', 'source_document'): self.inputs.path(self.protocol[key])

    def world(self, world, archive):
        case, labels = world['case'], world['labels']; n = len(labels)
        clean = self.inputs.npz(world['files']['clean.npz'])
        physical = self.inputs.npz(world['files']['Hstar.npz'])
        context = self.inputs.json(world['old_entry']['resource_context'])
        equipment = self.inputs.json(world['equipment_spec'])
        require(world['equipment_spec'] == world['old_entry']['equipment_spec'], 'Equipment was reconfigured')
        require(np.array_equal(clean['labels'], labels), 'Terminal order differs')
        stamps = pd.DatetimeIndex(pd.to_datetime(clean['timestamps'], utc=True))
        require(len(stamps) == 1056 and not stamps.has_duplicates, 'Invalid eleven-day time axis')
        require(np.all((stamps[1:]-stamps[:-1]) == pd.Timedelta(minutes=15)), 'Noncontiguous time axis')
        local = stamps.tz_convert('Europe/Berlin')
        require(str(local[0]) == '2024-01-21 08:00:00+01:00', 'Wrong locked start')
        require(len(set(t.utcoffset() for t in local.to_pydatetime())) == 1, 'DST transition')
        fulltime = pd.DatetimeIndex(pd.to_datetime(archive['timestamps_utc'], utc=True)); ix = fulltime.get_indexer(stamps)
        require(np.all(ix >= 0), 'Selected dates absent from archive')
        mapping = context['node_mapping']; scales = np.array([r['scale'] for r in mapping])
        require([r['label'] for r in mapping] == labels, 'Old scale terminal mapping changed')
        require(list(map(int, archive['feeder_ids'][:n])) == [r['feeder_id'] for r in mapping], 'Source feeder mapping changed')
        require(np.all(archive['valid_minutes'][ix, :n] >= 12), 'Missing selected source bins')
        for key, source in [('P', 'p_kw'), ('Q', 'q_kvar')]:
            expected = archive[source][ix, :n]*scales/world['base_kw']
            near(clean[key], expected, 'Old common P/Q scale changed', rtol=0., atol=0.)
        for days, begin in [(2, 576), (4, 384), (8, 0)]:
            require(world['train_indices'][str(days)] == list(range(begin, 768)), 'Training nesting changed')
        require(world['validation_indices'] == list(range(768, 960)), 'Validation was not fixed across training lengths')
        require(world['new_day_indices'] == list(range(960, 1056)), 'Evaluation window changed')
        star, truth, meta, edges = _network_reference(case, labels, self.inputs)
        near(physical['Hstar'], star, 'Physical linear response units or branch aggregation differ', atol=1e-14)
        require({tuple(c) for c in json.loads(str(physical['support_json']))} == {tuple(c) for c in truth}, 'Physical support set differs')
        rng = np.random.default_rng(np.random.SeedSequence([self.protocol['noise_seed'], 1611, CASES.index(case), 1]))
        hours = local.hour.to_numpy()+local.minute.to_numpy()/60
        expected_root = 1.02+.003*(.7*np.sin(2*np.pi*hours/24+.3)+.3*rng.normal(size=1056))
        near(clean['root'], expected_root, 'Physical root process changed', rtol=0., atol=0.)
        noises = {}
        for h in range(4):
            noise = self.inputs.npz(world['noise_by_repeat'][str(h)])
            for key, component in [('P', 2), ('Q', 3), ('V', 4), ('root', 5)]:
                rng = np.random.default_rng(np.random.SeedSequence([self.protocol['noise_seed'], 1610, CASES.index(case), h, component]))
                expected = rng.normal(size=(1056,) if key == 'root' else (1056, n))
                require(np.array_equal(noise[key], expected), 'Noise stream differs: '+key)
            noises[h] = noise
        ac = check_ac(self.inputs.npz(world['physical_cache']['cache']), clean, meta, edges)
        old = self.inputs.json(world['old_entry']['native_copy'])
        old_val = self.inputs.npz(world['old_entry']['files']['validation.npz'])
        old_points = [r['point'] for r in old['frozen_candidates']]
        old_true = [i for i, point in enumerate(old_points)
                    if topology(point, truth, n, old['config']['zero_tol'])['internal_exact']]
        family_sources = [('oracle_physical_support', truth),
                          ('old_historical_winner', old_points[old['selected']['index']]['supports'])]
        if old_true:
            best = min(old_true, key=lambda i: (loss(response(old_points[i], n), old_val)['mse'], i))
            family_sources.append(('old_correct_internal', old_points[best]['supports']))
        families = {}
        for role, supports in family_sources:
            families.setdefault(support_key(supports), []).append(role)
        resources = {'P': sorted({int(d['node_label']) for d in equipment['devices']}),
                     'Q': sorted({int(d['node_label']) for d in equipment['devices'] if d['kind'] in ('pv', 'battery')})}
        self.contexts[case] = dict(world=world, clean=clean, Hstar=star, truth=truth,
                                   noises=noises, resources=resources, offline_families=families)
        self.world_records.append(dict(case=case, points=1056, train_rows=[192, 384, 768], validation_rows=192,
            evaluation_rows=96, unchanged_old_scales=True, source_minute_count_min=int(archive['valid_minutes'][ix, :n].min()),
            root_process_sigma=.003, voltage_meter_sigma=.0005, PQ_meter_sigma=.005, AC=ac))

    def native(self, native, splits_, truth):
        points = [r['point'] for r in native['frozen_candidates']]; n = splits_['train']['P'].shape[1]
        H = [response(point, n) for point in points]
        values = [{name: loss(h, obs) for name, obs in splits_.items()} for h in H]
        for row, h, v in zip(native['frozen_candidates'], H, values):
            near(h, np.c_[row['R'], row['X']], 'Stored H differs from support coefficients', atol=1e-15)
            near(v['validation']['mse'], row['validation_mse'], 'Native validation score differs')
        internal = [sum(1 < len(c) < n for c in p['supports']) for p in points]
        kwargs = dict(K=1/float(native['reference_mse']), epsilon_scaled=native['config']['validation_tie'])
        before = choose_mse([v['validation']['mse'] for v in values], internal, **kwargs)['selected']
        cumulative = [(v['validation']['sse']+v['new_day']['sse'])/(v['validation']['cells']+v['new_day']['cells']) for v in values]
        after = choose_mse(cumulative, internal, **kwargs)['selected']
        require(before == native['selected']['index'], 'Native strict historical selection differs')
        tops = [topology(p, truth, n, native['config']['zero_tol']) for p in points]
        return H, values, before, after, tops

    def fit(self, task):
        folder = self.run/'fits'/task['id']
        done = self.inputs.json(folder/'complete.json'); require(done['task'] == task, 'Completed task coordinates differ')
        result = self.inputs.json(done['result'])
        require(result['task'] == task and result['status'] == done['status'], 'Result coordinates/status differ')
        if result['status'] != 'complete':
            self.main.append(dict(**task, status='failed', seconds=result.get('seconds'), stage=result.get('stage'), error=result.get('error')))
            return
        ctx = self.contexts[task['case']]; world, clean = ctx['world'], ctx['clean']
        whole = reconstruct_law(clean, ctx['noises'][task['history_repeat']], 'all', ctx['Hstar'])
        split = slices(whole, world, task['train_days'])
        for key in ('train', 'validation', 'new_day'):
            stored = self.inputs.npz(result[key])
            for field in ('P', 'Q', 'V', 'root', 'Y'):
                near(stored[field], split[key][field], 'Sealed observed law/slice mismatch', rtol=0., atol=0.)
        seal = self.inputs.json(result['candidates_seal'])
        require(not seal['truth_used'] and not seal['new_day_used'], 'Candidate seal violates information boundary')
        native, original, extra = [self.inputs.json(seal[key]) for key in ('native', 'original', 'additional')]
        require(native['frozen_candidates'][:len(original['frozen_candidates'])] == original['frozen_candidates'], 'Original candidate prefix changed')
        expected_added = 0 if extra.get('closed_point') is None else 1
        require(len(native['frozen_candidates']) == len(original['frozen_candidates'])+expected_added, 'Unexpected candidate injection')
        if expected_added:
            require(native['frozen_candidates'][-1]['point'] == extra['closed_point'], 'Added point is not original E search output')
        enriched = self.native(native, split, ctx['truth'])
        old = self.native(original, split, ctx['truth'])
        for label, checked_bank in [('evaluation', enriched), ('original_evaluation', old)]:
            H, values, before, after, tops = checked_bank; ev = result[label]
            require(before == ev['history_selected'] and after == ev['cumulative_selected'], 'Recorded final selector differs')
            for name, index in [('history', before), ('cumulative', after)]:
                for metric, value in tops[index].items(): require(ev[name+'_topology'][metric] == value, 'Topology outcome differs')
            require(ev['bank_has_internal_truth'] == any(t['internal_exact'] for t in tops), 'Internal coverage differs')
            require(ev['bank_has_full_truth'] == any(t['full_exact'] for t in tops), 'Full support coverage differs')
        H, values, before, after, tops = enriched
        ix = world['train_indices'][str(task['train_days'])]
        for row in result['matrix_rows']:
            i = row['native_index']; point = native['frozen_candidates'][i]['point']
            check_recorded_metrics(row['metrics'], H[i], ctx['Hstar'], clean['P'][ix], clean['Q'][ix])
            check_design(row['design'], split['train']['P'], split['train']['Q'], point['supports'])
            for name in split:
                for metric in ('mse', 'mae'): near(values[i][name][metric], row['losses'][name][metric], 'Model loss differs')
        history_row = next(r for r in result['matrix_rows'] if r['role'] == 'history_selected')
        require(history_row['native_index'] == before, 'Historical matrix row is an oracle replacement')
        miqp = _solver_rows(original.get('MIQP', []))+_solver_rows(extra)
        j = world['new_day_indices']
        clean_y = clean['root'][j, None]**2-clean['V'][j]**2
        clean_mse = float(np.mean((clean_y-np.c_[clean['P'][j], clean['Q'][j]]@H[before].T)**2))
        self.main.append(dict(**task, status='complete', seconds=result['seconds'],
            bank_size=len(H), history_selected=before, cumulative_selected=after,
            bank_has_internal_truth=any(t['internal_exact'] for t in tops), bank_has_full_truth=any(t['full_exact'] for t in tops),
            history_internal_correct=tops[before]['internal_exact'], cumulative_internal_correct=tops[after]['internal_exact'],
            history_full_correct=tops[before]['full_exact'], cumulative_full_correct=tops[after]['full_exact'],
            history_matrix=history_row['metrics'],
            common_new_response=response_stats(H[before], ctx['Hstar'], clean['P'][j], clean['Q'][j]),
            clean_AC_new_MSE=clean_mse, observed_history_selected_losses=values[before],
            original_evaluation=result['original_evaluation'], enrichment_status=extra['status'],
            stop_reason=original['stop_reason'], total_cap_exceeded=original['total_cap_exceeded'],
            timed_out_MIQP_count=sum(r.get('status') == 9 for r in miqp),
            nonclosed_MIQP_count=sum(r.get('gap') is not None and r['gap'] > r.get('requested_gap', 0) for r in miqp),
            MIQP_records=miqp, bound_scope='Individual original finite-pool MIQP subproblems, not global topology or parameter truth'))

    def offline_row(self, row):
        if row['status'] != 'complete':
            self.offline.append(row); return
        ctx = self.contexts[row['case']]; world, clean = ctx['world'], ctx['clean']
        observed = reconstruct_law(clean, ctx['noises'][row['history_repeat']], row['law'], ctx['Hstar'])
        split = slices(observed, world, row['train_days']); point = row['point']
        H = response(point, len(world['labels']))
        qp = fixed_qp_check(point, split['train'], row['qp_record'])
        for name, obs in split.items():
            actual = loss(H, obs)
            for metric in ('mse', 'mae'): near(actual[metric], row['losses'][name][metric], 'Offline held-out loss differs')
        ix = world['train_indices'][str(row['train_days'])]
        check_recorded_metrics(row['metrics'], H, ctx['Hstar'], clean['P'][ix], clean['Q'][ix])
        check_design(row['design'], split['train']['P'], split['train']['Q'], point['supports'])
        expected_topology = topology(point, ctx['truth'], len(world['labels']), 1e-9)
        for name, value in expected_topology.items(): require(row['topology'][name] == value, 'Offline support outcome differs')
        j = world['new_day_indices']
        clean_y = clean['root'][j, None]**2-clean['V'][j]**2
        self.offline.append(dict(**{k: row[k] for k in ('case', 'history_repeat', 'train_days', 'law', 'support_role',
                    'support_aliases', 'status', 'deterministic_across_repeats')},
            metrics=row['metrics'], design=row['design'], losses=row['losses'], independent_QP=qp,
            qp_seconds=row['qp_record']['seconds'],
            common_new_response=response_stats(H, ctx['Hstar'], clean['P'][j], clean['Q'][j]),
            clean_AC_new_MSE=float(np.mean((clean_y-np.c_[clean['P'][j], clean['Q'][j]]@H.T)**2))))

    def execute(self):
        self.sources()
        archive = self.inputs.npz(self.protocol['archive'])
        worlds = self.inputs.json(self.run/'worlds.json')
        for world in worlds:
            try: self.world(world, archive)
            except Exception as exc: self.errors.append(dict(stage='world', id=world['case'], error=repr(exc)))
        fit_done = self.inputs.json(self.run/'fits_complete.json')
        require(fit_done['planned'] == len(self.protocol['tasks']), 'Fit denominator changed')
        for task in self.protocol['tasks']:
            try: self.fit(task)
            except Exception as exc: self.errors.append(dict(stage='fit', id=task['id'], error=repr(exc)))
        offline_done = self.inputs.json(self.run/'offline/complete.json')
        diagnostics = self.inputs.json(offline_done['diagnostics'])
        require(offline_done['rows'] == len(diagnostics['rows']), 'Offline denominator changed')
        planned_offline = check_offline_coverage(diagnostics['rows'],
            {case: ctx['offline_families'] for case, ctx in self.contexts.items()},
            self.protocol['diagnostic_training_days'], self.protocol['diagnostic_laws'])
        for index, row in enumerate(diagnostics['rows']):
            try: self.offline_row(row)
            except Exception as exc: self.errors.append(dict(stage='offline', id=index, coordinates={k: row[k] for k in ('case', 'law', 'train_days', 'history_repeat', 'support_role')}, error=repr(exc)))
        self.inputs.final_check()
        return dict(status='passed' if not self.errors else 'failed', errors=self.errors,
            source_protocol=ref(self.run/'protocol.json'), source_inputs=list(self.inputs.checked.values()),
            planned_worlds=2, audited_worlds=len(self.world_records),
            planned_fits=len(self.protocol['tasks']), audited_fits=len(self.main),
            planned_offline_rows=planned_offline, audited_offline_rows=len(self.offline),
            worlds=self.world_records, no_new_fit_or_AC=True)


def summarize_offline(rows):
    groups = defaultdict(list)
    for row in rows:
        for role in row.get('support_aliases', [row['support_role']]):
            groups[(row['case'], role, row['law'], row['train_days'])].append(row)
    result = []
    for (case, role, law, days), group in sorted(groups.items()):
        good = [r for r in group if r['status'] == 'complete']
        # Repeated clean fits share identical observations: retain counts, but
        # explicitly do not treat the four aliases as independent evidence.
        result.append(dict(case=case, support_role=role, law=law, train_days=days,
            planned=len(group), completed=len(good), failed=len(group)-len(good),
            independent_noise_repetitions=0 if law in ('cleanlinear', 'cleanAC') else len(good),
            deterministic_repeats=law in ('cleanlinear', 'cleanAC'),
            R_relative_error=_stats(r['metrics']['R']['relative_error'] for r in good),
            X_relative_error=_stats(r['metrics']['X']['relative_error'] for r in good),
            clean_AC_new_MSE=_stats(r['clean_AC_new_MSE'] for r in good),
            conditional_X_energy_fraction=_stats(r['design']['X_given_R']['residual_energy_fraction'] for r in good),
            QP_gap=_stats(r['independent_QP']['certificate_gap'] for r in good)))
    return result


def _show(value, percent=False):
    if value is None: return '—'
    return f'{100*value:.3f}%' if percent else f'{value:.6g}'


def markdown(report):
    lines = ['# 2天与8天训练：独立核验与R/X诊断', '',
        f"独立审计：**{report['audit_status']}**。保留{report['main']['planned']}个计划辨识格；状态为{report['main']['statuses']}。", '',
        '共同窗口从2024-01-21当地08:00开始。2天训练取第7–8天，8天训练取第1–8天；验证共同使用第9–10天192点，评价共同使用第11天96点。旧节点缩放、原物理线路及四类设备配置保持。没有灵活性采购。', '',
        '这是两个固定测试网络、同一个物理时间窗口和4次配对量测误差，不是16个独立物理日期。验证192点与过去96点的历史权重不同，不能把本轮权重或后续门槛直接视为旧阈值下的数据量收益。', '',
        '|网络|训练天数|已核/计划|真内部族覆盖|历史内部正确|累计内部正确|累计全支持正确|R矩阵误差均值|X矩阵误差均值|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['main']['groups']:
        den = row['planned']
        lines.append(f"|{row['case']}|{row['train_days']}|{row['checked_complete']}/{den}|{row['bank_has_internal_truth']}/{den}|{row['history_internal_correct']}/{den}|{row['cumulative_internal_correct']}/{den}|{row['cumulative_full_correct']}/{den}|{_show(row['R_selected_relative_error']['mean'], True)}|{_show(row['X_selected_relative_error']['mean'], True)}|")
    lines += ['', '上表矩阵误差使用各格实际历史赢家，没有换成库内最佳真族。内部族与完整支持分别统计；矩阵误差不是逐线路欧姆误差。共同第11天的线性响应误差、干净AC残差、各资源节点P/Q列误差、训练设计条件数和原始求解状态见report.json。', '',
              '|网络|噪声重复|R误差变化（8−2）|X误差变化（8−2）|新增内部恢复|丢失内部恢复|', '|---|---:|---:|---:|---:|---:|']
    for row in report['main']['paired']:
        if row['status'] != 'complete':
            lines.append(f"|{row['case']}|{row['history_repeat']}|未完成|未完成|—|—|")
        else:
            lines.append(f"|{row['case']}|{row['history_repeat']}|{_show(row['R_error_change_8_minus_2'], True)}|{_show(row['X_error_change_8_minus_2'], True)}|{int(row['cumulative_restored'])}|{int(row['cumulative_lost'])}|")
    lines += ['', '## 固定支持诊断', '',
              '以下四种观测分别检验：完全线性正确模型、无额外表计误差的AC、只加V/根表计误差、再加入P/Q表计误差。固定支持QP只作离线诊断，未注入101候选。干净两组在4次重复中数据相同，不把重复拟合当成4份独立证据。', '',
              '|网络|支持来源|观测|天数|完成/计划|R误差均值|X误差均值|条件X能量比例|', '|---|---|---|---:|---:|---:|---:|---:|']
    for row in report['offline_groups']:
        lines.append(f"|{row['case']}|{row['support_role']}|{row['law']}|{row['train_days']}|{row['completed']}/{row['planned']}|{_show(row['R_relative_error']['mean'], True)}|{_show(row['X_relative_error']['mean'], True)}|{_show(row['conditional_X_energy_fraction']['mean'], True)}|")
    lines += ['', '## 求解与解释范围', '',
              '101按原120秒加原E池60秒运行；每个MIQP状态、有限池界和实际用时保留。有限池间隙闭合不等于全局正确拓扑证明。固定支持QP的梯度支持平面界由审计独立重算；QP数值收敛也不等于R/X已被历史数据唯一辨识。', '',
              '增加训练天数同时增加了历史工况覆盖。由于两组训练截止时间、验证和评价相同，本轮避免了验证量和资料新旧程度的直接混杂，但不能把4倍点数当成4倍独立信息。P/Q输入误差造成的估计偏差、AC线性近似误差和结构选择误差须由四种观测及固定支持对照区分。', '',
              f"主辨识失败：{len(report['failed_main'])}；固定支持失败：{len(report['failed_offline'])}；审计错误：{len(report['audit_errors'])}。所有条目见report.json，不从分母剔除。", '']
    return '\n'.join(lines)


def run(run_directory, out):
    destination = Path(out).resolve()/'report'
    if destination.exists(): raise FileExistsError('Preserve previous report: '+str(destination))
    audit = Audit(run_directory)
    checked = audit.execute()
    main = summarize_main(audit.protocol['tasks'], audit.main)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), audit_status=checked['status'],
        source_run=str(audit.run), source_protocol=checked['source_protocol'], main=main,
        main_rows=audit.main, offline_groups=summarize_offline(audit.offline), offline_rows=audit.offline,
        failed_main=[r for r in audit.main if r['status'] != 'complete'],
        failed_offline=[r for r in audit.offline if r['status'] != 'complete'], audit_errors=checked['errors'],
        statistical_scope='2 physical networks, 1 common 11-day window, 4 paired noise draws; no new procurement',
        schema_version=1, source_postprocessor=ref(__file__))
    destination.mkdir(parents=True)
    write(destination/'audit.json', checked); write(destination/'report.json', result)
    (destination/'REPORT.md').write_text(markdown(result), encoding='utf-8')
    write(destination/'complete.json', dict(status=checked['status'], audit=ref(destination/'audit.json'),
        report=ref(destination/'report.json'), markdown=ref(destination/'REPORT.md')))
    return dict(status=checked['status'], planned_fits=checked['planned_fits'], audited_fits=checked['audited_fits'],
                offline_rows=checked['audited_offline_rows'], errors=len(checked['errors']), output=str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run, args.out)
    print(json.dumps(result, ensure_ascii=False))
    if result['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__':
    main()
