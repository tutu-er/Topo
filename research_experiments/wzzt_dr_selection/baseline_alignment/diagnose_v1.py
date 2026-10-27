"""Read-only diagnosis of an existing WZZT run; no fitting or optimization.

Loads trusted local run pickles. Exports only aggregate measurements and support
sets, never household identifiers or per-household/per-time trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def serialize(family):
    return [sorted(s) for s in sorted(family, key=lambda s: (len(s), tuple(sorted(s))))]


def crossing(a, b):
    return bool(a & b) and not (a <= b or b <= a)


def effective_family(point, tolerance):
    return {frozenset(s) for s, r, x in zip(point['support_labels'], point['r_values'], point['x_values'], strict=True)
            if r > tolerance or x > tolerance}


def path_diagnosis(fit, truth):
    tolerance = fit['config']['zero_tolerance']
    terminal_count = len(fit['terminal_labels'])
    internal_truth = {s for s in truth if 1 < len(s) < terminal_count}
    previous, rows = set(), []
    first_wrong, first_crossing = None, None
    for index, point in enumerate(fit['path']):
        family = effective_family(point, tolerance)
        added, removed = family - previous, previous - family
        blocking = {s: {t for t in internal_truth if crossing(s, t)} for s in family}
        blocked = set().union(*blocking.values()) if blocking else set()
        for support in sorted(added, key=lambda s: (len(s), tuple(sorted(s)))):
            item = {'path_index': index, 'support': sorted(support),
                    'crossed_true_internal_supports': serialize(blocking[support])}
            if support not in truth and first_wrong is None:
                first_wrong = item
            if blocking[support] and first_crossing is None:
                first_crossing = item
        rows.append({'path_index': index, 'train_mae_pu2': point['train_mae'],
            'validation_mae_pu2': point['validation_mae'], 'validation_se_pu2': point['validation_se'],
            'effective_support_count': len(family), 'true_internal_matched_count': len(family & internal_truth),
            'true_internal_denominator': len(internal_truth), 'matched_true_internal_supports': serialize(family & internal_truth),
            'blocked_true_internal_supports': serialize(blocked), 'added_effective_supports': serialize(added),
            'removed_effective_supports': serialize(removed),
            'stored_pruning_removals': point['removed_support_indices_since_previous_path_point']})
        previous = family
    best = min(range(len(rows)), key=lambda i: rows[i]['validation_mae_pu2'])
    threshold = rows[best]['validation_mae_pu2'] + (rows[best]['validation_se_pu2'] or 0.)
    eligible = [i for i, p in enumerate(fit['path']) if p['validation_mae'] <= threshold]
    expected = min(eligible, key=lambda i: (len(fit['path'][i]['support_indices']), i))
    chosen = fit['summary']['selected_path_index']
    crossing_weights = []
    if first_crossing is not None:
        target = first_crossing['support']
        for i, p in enumerate(fit['path']):
            if target in p['support_labels']:
                k = p['support_labels'].index(target)
                crossing_weights.append({'path_index': i, 'r_weight': p['r_values'][k], 'x_weight': p['x_values'][k]})
    return {'path': rows, 'first_incorrect_added_support': first_wrong,
        'first_added_crossing_support': first_crossing, 'crossing_support_weight_history': crossing_weights,
        'max_true_internal_matched_on_any_path': max(p['true_internal_matched_count'] for p in rows),
        'minimum_validation_path_index': best, 'selected_path_index': chosen,
        'one_se_threshold_pu2': threshold, 'one_se_recomputed_selected_index': expected,
        'one_se_selection_matches': expected == chosen}


def oracle_multiplication(simulations, r_matrix, x_matrix, kind):
    ps, qs, targets, errors = [], [], [], []
    for simulation in simulations:
        data = simulation['truth'] if kind == 'physical_truth' else simulation['observed']
        if kind == 'physical_truth':
            p, q = data['P_true'].to_numpy(), data['Q_true'].to_numpy()
            target = data['root_voltage_true'].to_numpy()[:, None] ** 2 - data['V_terminal_true'].to_numpy() ** 2
        else:
            p, q = data['P_terminal'].to_numpy(), data['Q_terminal'].to_numpy()
            target = data['drop_target'].to_numpy()
        ps.append(p); qs.append(q); targets.append(target)
        errors.append(target - p @ r_matrix.T - q @ x_matrix.T)
    p, q, target, error = map(np.vstack, (ps, qs, targets, errors))
    design = np.column_stack((p, q))
    singular = np.linalg.svd(design, compute_uv=False)
    correlations = np.array([np.corrcoef(p[:, j], q[:, j])[0, 1] for j in range(p.shape[1])])
    return {'time_rows': len(p), 'mae_pu2': float(np.abs(error).mean()),
        'max_absolute_error_pu2': float(np.abs(error).max()), 'mean_absolute_target_pu2': float(np.abs(target).mean()),
        'input_rank': int(np.linalg.matrix_rank(design)), 'input_column_count': design.shape[1],
        'raw_input_condition_number': float(singular[0] / singular[-1]),
        'raw_input_smallest_singular_value': float(singular[-1]),
        'node_PQ_correlation_min': float(correlations.min()),
        'node_PQ_correlation_median': float(np.median(correlations)),
        'node_PQ_correlation_max': float(correlations.max()),
        'interpretation': 'Evaluation with physical truth parameters only; no fitted parameters or optimizer, not a deployed estimator or identifiability certificate'}


def diagnose(repo, run_dir, output):
    sys.path.insert(0, str(repo))
    from research_experiments.wzzt_dr_selection.end_to_end import physical_case
    from rnj_wzzt.models import lin_distflow
    inputs = {}

    def load(path, binary=False):
        inputs[str(path)] = sha256(path)
        if binary:
            with path.open('rb') as stream:
                return pickle.load(stream)
        return json.loads(path.read_text(encoding='utf-8'))

    summary = load(run_dir / 'summary.json'); state = load(run_dir / 'run_state.json')
    if summary['status'] != 'complete' or state['status'] != 'complete':
        raise ValueError('Only inspect a completed run')
    cfg = state['provenance']['config']
    net = physical_case.build_reference_case(cfg['case_key'])
    r_matrix, x_matrix = lin_distflow.build_reduced_sensitivity_matrices(net, voltage_model='squared-voltage')
    r_weights = [2 * lin_distflow.ohm_to_pu(net, e['r_ohm']) for _, _, e in net.to_networkx_graph().edges(data=True)]
    x_weights = [2 * lin_distflow.ohm_to_pu(net, e['x_ohm']) for _, _, e in net.to_networkx_graph().edges(data=True)]
    history = [load(run_dir / 'history' / (d + '.pkl'), binary=True) for d in summary['wzzt_training_dates']]
    records, simulations = {}, {}
    for item in summary['strategies']:
        name = item['strategy']; folder = run_dir / 'strategies' / name
        fit = load(folder / 'wzzt.json')
        periods = {role: load(folder / (role + '.pkl'), binary=True)['simulation'] for role in ('action', 'validation')}
        simulations[name] = periods
        truth = {frozenset(s) for s in periods['action']['truth']['truth_supports']}
        labels = list(periods['action']['observed']['P_terminal'].columns)
        if labels != fit['terminal_labels']:
            raise ValueError('Observed/fitted terminal ordering mismatch')
        attempts = fit.get('attempted_extensions', [])
        first = attempts[0] if attempts else None
        first_summary = None if first is None else {
            'support_returned': first['support'], 'objective': first['objective'], **first['diagnostics'],
            'absolute_gap': first['diagnostics']['objective'] - first['diagnostics']['dual_bound'],
            'support_none_interpretation': 'Core intentionally withholds a support when numerical optimality is not established; an incumbent objective may still exist'}
        records[name] = {'status': fit['status'], 'stop_reason': fit['summary']['stop_reason'],
            'first_extension': first_summary, 'path_diagnosis': path_diagnosis(fit, truth),
            'oracle_true_weights_evaluation': {
                split: {kind: oracle_multiplication(sims, r_matrix, x_matrix, kind)
                    for kind in ('physical_truth', 'observed')}
                for split, sims in [('training', history + [periods['action']]), ('validation', [periods['validation']])]},
            'physical_weight_domain': {'minimum_r_line_weight': min(r_weights), 'maximum_r_line_weight': max(r_weights),
                'minimum_x_line_weight': min(x_weights), 'maximum_x_line_weight': max(x_weights),
                'initial_coefficient_bound': fit['config']['coefficient_bound'],
                'all_true_line_weights_in_initial_domain': min(r_weights + x_weights) >= 0 and max(r_weights + x_weights) <= fit['config']['coefficient_bound']}}
    baseline = simulations['no_reward']['action']['observed']
    for name, periods in simulations.items():
        current = periods['action']['observed']
        dp = current['P_terminal'].to_numpy() - baseline['P_terminal'].to_numpy()
        dy = current['drop_target'].to_numpy() - baseline['drop_target'].to_numpy()
        records[name]['paired_action_difference_from_no_reward'] = {
            'maximum_absolute_p_change_kw': float(np.abs(dp).max() * net.base_mva * 1000),
            'relative_frobenius_p_change': float(np.linalg.norm(dp) / np.linalg.norm(baseline['P_terminal'].to_numpy())),
            'mean_absolute_target_change_pu2': float(np.abs(dy).mean()),
            'maximum_absolute_target_change_pu2': float(np.abs(dy).max())}
    code_sources = {str(Path(__file__).resolve()): sha256(Path(__file__))}
    for module in (physical_case, lin_distflow):
        path = Path(module.__file__).resolve(); code_sources[str(path)] = sha256(path)
    comparisons = {}
    for path, current_hash in code_sources.items():
        saved = state['provenance']['source_sha256'].get(path)
        if saved is not None:
            comparisons[path] = {'saved_sha256': saved, 'current_sha256': current_hash, 'matches': saved == current_hash}
            if saved != current_hash:
                raise ValueError('Diagnostic source changed from recorded run: ' + path)
    result = {'scope': 'Post hoc read-only diagnosis of an existing run; no optimization, no new realization, no raw household trajectories exported',
        'run_fingerprint': summary['run_fingerprint'], 'run_directory': str(run_dir),
        'strategy_denominator': len(records), 'training_dates': summary['wzzt_training_dates'],
        'input_sha256': inputs, 'diagnostic_source_sha256': code_sources,
        'source_comparison_with_run': comparisons, 'new_optimizer_calls': 0, 'strategies': records,
        'limitations': ['Raw input rank/condition numbers are descriptive, not topology reliability.',
            'Physical true R/X losses evaluate an oracle feasible model; truth must not enter price or identification.',
            'An unproved first extension cannot rank price effectiveness.',
            'Observed persistent positive weights explain this recorded path, not a theorem that all later reweighting must fail.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8', newline='\n')
    print(json.dumps({'output': str(output), 'strategies': len(records), 'optimizer_calls': 0,
        'no_reward_max_true_internal_match': records['no_reward']['path_diagnosis']['max_true_internal_matched_on_any_path']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.repo.resolve(), args.run_dir.resolve(), args.output.resolve())
