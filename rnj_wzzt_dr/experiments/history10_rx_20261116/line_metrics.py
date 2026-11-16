"""Recompute line-impedance errors from sealed offline fits; never fit or run AC."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[3]
DEFAULT_RUN = ROOT / 'outputs/history10_rx_20261116_v1'
DEFAULT_AUDIT = ROOT / 'outputs/history10_rx_audit_20261116_v2/report/audit.json'


def canonical(s):
    return tuple(sorted(int(v) for v in s))


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'minimum': float(a.min()), 'maximum': float(a.max())}


def paired_line_metrics(point, truth, zbase, matrix_relative_errors=None):
    """Match canonical supports before converting response coefficients back to ohms."""
    supports = [canonical(s) for s in point['supports']]
    assert len(supports) == len(set(supports)) and set(supports) == set(truth)
    result = {}
    for letter in ('r', 'x'):
        actual = np.array([truth[s][letter + '_ohm'] for s in supports])
        fitted = np.array(point[letter], dtype=float) * zbase / 2.
        errors = np.abs(fitted - actual)
        positive = actual > 0
        result[letter] = {
            'mape_percent': float(100. * np.mean(errors[positive] / actual[positive])) if np.any(positive) else None,
            'mean_absolute_error_ohm': float(errors.mean()),
            'maximum_absolute_error_ohm': float(errors.max()),
            'positive_truth_count': int(positive.sum()),
            'zero_truth_count': int((actual == 0).sum()),
            'zero_truth_mean_absolute_error_ohm': float(errors[actual == 0].mean()) if np.any(actual == 0) else None,
            'fitted_zero_count_at_1e_minus9_response_units': int(np.count_nonzero(np.asarray(point[letter]) <= 1e-9)),
        }
        if matrix_relative_errors is not None:
            result[letter]['response_matrix_relative_F_error_percent'] = float(100. * matrix_relative_errors[letter.upper()]['relative_error'])
    r_mape, x_mape = result['r']['mape_percent'], result['x']['mape_percent']
    result['equal_weight_R_X_MAPE_percent'] = (r_mape + x_mape) / 2. if r_mape is not None and x_mape is not None else None
    return result


def compute(run, audit_path, out):
    """Read validated prior results and write a new diagnostic directory once."""
    RUN, AUDIT, OUT = Path(run).resolve(), Path(audit_path).resolve(), Path(out).resolve()
    if OUT.exists():
        raise FileExistsError(f'Fresh output directory required: {OUT}')
    sources = {}

    def ref(path, expected=None):
        path = Path(path).resolve()
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None:
            assert value == expected, (str(path), value, expected)
        record = {'path': str(path), 'sha256': value}
        sources[str(path)] = record
        return record


    def read_json(path, expected=None):
        ref(path, expected)
        return json.loads(Path(path).read_text(encoding='utf-8'))


    protocol = read_json(RUN / 'protocol.json')
    expected_sources = {str(Path(v['path']).resolve()): v['sha256'] for v in protocol['sources']}
    worlds = read_json(RUN / 'worlds.json')
    completion = read_json(RUN / 'offline/complete.json')
    diagnostics = read_json(completion['diagnostics']['path'], completion['diagnostics']['sha256'])
    audit = read_json(AUDIT)
    assert audit['status'] == 'passed' and not audit['errors']
    assert audit['audited_offline_rows'] == 240
    assert completion['successful'] == completion['rows'] == 240
    assert diagnostics['source_unchanged'] is True
    truths, network_checks = {}, []

    for world in worlds:
        case = world['case']
        case_dir = ROOT / 'topology_identification_standalone/cases' / case
        paths = {name: case_dir / name for name in ('metadata.json', 'buses.csv', 'branches.csv')}
        for p in paths.values():
            ref(p, expected_sources[str(p.resolve())])
        metadata = json.loads(paths['metadata.json'].read_text(encoding='utf-8'))
        with paths['buses.csv'].open(encoding='utf-8-sig', newline='') as f:
            buses = list(csv.DictReader(f))
        with paths['branches.csv'].open(encoding='utf-8-sig', newline='') as f:
            edges = [r for r in csv.DictReader(f) if r['status'].lower() in ('true', '1')]
        root = int(metadata['root_bus'])
        labels = [int(x) for x in world['labels']]
        nodes = {int(r['bus_id']): r for r in buses}
        children = defaultdict(list)
        degree = defaultdict(int)
        parent = {}
        for edge in edges:
            a, b = int(edge['from_bus']), int(edge['to_bus'])
            assert b not in parent and b != root
            parent[b] = a
            children[a].append(b)
            degree[a] += 1
            degree[b] += 1
        assert len(edges) == len(nodes) - 1
        order = [root]
        for node in order:
            order.extend(children[node])
        assert len(order) == len(nodes) == len(set(order))
        descendants = {b: ({labels.index(b)} if b in labels else set()) for b in order}
        for node in reversed(order[1:]):
            descendants[parent[node]].update(descendants[node])
        aggregates = {}
        for edge in edges:
            support = canonical(descendants[int(edge['to_bus'])])
            assert support, ('unobserved edge', case, edge)
            a = aggregates.setdefault(support, {'r_ohm': 0., 'x_ohm': 0., 'physical_edges': []})
            a['r_ohm'] += float(edge['r_ohm'])
            a['x_ohm'] += float(edge['x_ohm'])
            a['physical_edges'].append([int(edge['from_bus']), int(edge['to_bus'])])
        zbase = float(metadata['base_kv']) ** 2 / float(metadata['base_mva'])
        assert abs(float(world['base_kw']) - float(metadata['base_mva']) * 1000.) < 1e-12
        H = np.zeros((len(labels), 2 * len(labels)))
        for support, item in aggregates.items():
            z = np.zeros(len(labels)); z[list(support)] = 1.
            block = np.outer(z, z)
            H[:, :len(labels)] += 2. * item['r_ohm'] / zbase * block
            H[:, len(labels):] += 2. * item['x_ohm'] / zbase * block
        href = world['files']['Hstar.npz']
        ref(href['path'], href['sha256'])
        with np.load(href['path'], allow_pickle=False) as a:
            Hstar = a['Hstar']
            saved_supports = {canonical(s) for s in json.loads(str(a['support_json']))}
            if 'labels' in a:
                assert np.array_equal(a['labels'], labels)
        assert saved_supports == set(aggregates)
        h_error = float(np.max(np.abs(H - Hstar)))
        assert h_error < 1e-12
        hidden_degree2 = [b for b, r in nodes.items() if b != root and b not in labels and degree[b] == 2]
        repeated = [list(s) for s, r in aggregates.items() if len(r['physical_edges']) > 1]
        positivity = {v: {'positive': sum(r[v] > 0 for r in aggregates.values()),
                          'zero': sum(r[v] == 0 for r in aggregates.values()),
                          'negative': sum(r[v] < 0 for r in aggregates.values()),
                          'range_ohm': [min(r[v] for r in aggregates.values()), max(r[v] for r in aggregates.values())]}
                      for v in ('r_ohm', 'x_ohm')}
        assert all(p['negative'] == 0 for p in positivity.values())
        network_checks.append({'case': case, 'terminals': len(labels), 'physical_closed_edges': len(edges),
            'canonical_supports': len(aggregates), 'hidden_degree2_nodes': hidden_degree2,
            'repeated_edge_supports': repeated, 'per_support_equals_per_physical_line': not repeated,
            'Zbase_ohm': zbase, 'coefficient_conversion': 'response coefficient=2*line_ohm/Zbase',
            'Hstar_max_absolute_reconstruction_difference': h_error, 'true_coefficients': positivity})
        truths[case] = (aggregates, zbase)

    rows = []
    for row in diagnostics['rows']:
        if 'oracle_physical_support' not in row['support_aliases']:
            continue
        if row['train_days'] not in (2, 8) or row['law'] not in ('cleanAC', 'Vonly', 'all'):
            continue
        assert row['status'] == 'complete'
        truth, zbase = truths[row['case']]
        point = row['point']
        supports = [canonical(s) for s in point['supports']]
        assert len(supports) == len(set(supports)) and set(supports) == set(truth)
        result = {k: row[k] for k in ('case', 'history_repeat', 'train_days', 'law')}
        result.update({'support_count': len(supports), 'deterministic_across_repeats': row['deterministic_across_repeats']})
        result.update(paired_line_metrics(point, truth, zbase, row['metrics']))
        rows.append(result)
    assert len(rows) == 48
    groups = []
    for case in (w['case'] for w in worlds):
        for law in ('cleanAC', 'Vonly', 'all'):
            for days in (2, 8):
                subset = [r for r in rows if (r['case'], r['law'], r['train_days']) == (case, law, days)]
                assert sorted(r['history_repeat'] for r in subset) == [0, 1, 2, 3]
                group = {'case': case, 'law': law, 'train_days': days, 'training_points': days * 96,
                         'repeats': 4, 'support_count': subset[0]['support_count'],
                         'deterministic_across_repeats': law == 'cleanAC',
                         'equal_weight_R_X_MAPE_percent': stats([r['equal_weight_R_X_MAPE_percent'] for r in subset])}
                for letter in ('r', 'x'):
                    group[letter] = {metric: stats([r[letter][metric] for r in subset]) for metric in
                        ('mape_percent', 'mean_absolute_error_ohm', 'maximum_absolute_error_ohm',
                         'response_matrix_relative_F_error_percent', 'fitted_zero_count_at_1e_minus9_response_units')}
                groups.append(group)

    OUT.mkdir(parents=True, exist_ok=False)
    source_copy = OUT / 'source_snapshot.py'
    source_copy.write_bytes(Path(__file__).read_bytes())
    ref(__file__)
    ref(source_copy)
    report = {'status': 'complete', 'scope': 'Postprocessing existing sealed oracle_physical_support fits only; no fit, AC, or optimization invoked',
        'interpretation': ['Known full physical support is supplied only for this offline parameter diagnostic, not to the identification candidate bank.',
            'MAPE is unweighted over positive physical line/support coefficients, separately for R and X; matrix Frobenius errors are separate quantities.',
            'The four cleanAC repeats are numerically identical because that law has no meter noise; they are not four independent physical datasets.',
            'Noise repeats share one fixed eleven-day physical window. Two-day and eight-day fits use nested training rows and identical validation/new day.',
            'All original physical edges are aggregated by their terminal-descendant support before matching; conversion back to ohms uses Zbase/2.'],
        'formulas': {'MAPE_percent': '100*mean_e(abs(theta_hat_e-theta_e)/theta_e) over theta_e>0',
                     'mean_absolute_error_ohm': 'mean_e(abs(theta_hat_e-theta_e))',
                     'joint_MAPE_percent': '(R_MAPE_percent+X_MAPE_percent)/2'},
        'audit_status': audit['status'], 'audited_offline_rows': audit['audited_offline_rows'],
        'selected_rows': len(rows), 'groups': groups, 'rows': rows, 'networks': network_checks,
        'source_hashes': list(sources.values()), 'no_new_fit_or_AC': True}
    target = OUT / 'report.json'
    assert not target.exists(), 'Do not overwrite an already completed diagnostic'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--audit', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--out', type=Path, required=True, help='New directory; existing paths are rejected')
    args = parser.parse_args(argv)
    result = compute(args.run, args.audit, args.out)
    report_path = args.out.resolve() / 'report.json'
    print(json.dumps({'report': str(report_path),
                      'sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                      'selected_rows': result['selected_rows'], 'groups': len(result['groups'])}))


if __name__ == '__main__':
    main()
