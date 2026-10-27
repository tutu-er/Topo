"""Bounded, post hoc LP audit of wrong blocks in the completed default paths.

Truth is deliberately used to name wrong blocks and alternative true blocks in
this diagnostic. These LPs are not a deployable selector or a new algorithm.
No RNJ, bootstrap, free-support MILP, data generation or price design is run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
from time import perf_counter

import numpy as np
import scipy


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(family):
    return [sorted(s) for s in sorted(set(map(frozenset, family)),
                                    key=lambda s: (len(s), tuple(sorted(s))))]


def crossing(a, b):
    a, b = frozenset(a), frozenset(b)
    return bool(a & b) and not (a <= b or b <= a)


def trace(record, truth):
    rows = []
    initial = set(map(frozenset, record['path'][0]['supports']))
    previous = set()
    first_seen = {}
    for point in record['path']:
        family = set(map(frozenset, point['supports']))
        wrong = []
        for support, r, x in zip(point['supports'], point['r_values'], point['x_values'], strict=True):
            s = frozenset(support)
            if s in truth:
                continue
            first_seen.setdefault(s, point['index'])
            wrong.append({'support': sorted(s), 'r': r, 'x': x,
                          'first_path_index': first_seen[s], 'initial_frozen': s in initial,
                          'active': r > 1e-9 or x > 1e-9,
                          'crossing_true_blocks': serial(t for t in truth if crossing(s, t))})
        rows.append({'path_index': point['index'], 'new_supports': serial(family - previous),
                     'removed_supports': serial(previous - family),
                     'train_mae': point['train_mae'], 'validation_mae': point['validation_mae'],
                     'wrong_blocks': wrong})
        previous = family
    return rows


def declare_plan(record, truth):
    """Freeze every LP before execution, keeping at most twelve per condition."""
    trajectory = trace(record, truth)
    first_wrong = next(row['path_index'] for row in trajectory if row['wrong_blocks'])
    first_cross = next(row['path_index'] for row in trajectory
                       if any(w['crossing_true_blocks'] for w in row['wrong_blocks']))
    points = record['path']
    before = points[first_cross - 1]
    chosen = points[record['selected_path_index']]
    last = points[-1]
    first_cross_support = next(w['support'] for w in trajectory[first_cross]['wrong_blocks']
                               if w['crossing_true_blocks'])
    plan = []

    def add(kind, family, reference, **extra):
        family = serial(family)
        assert all(not crossing(a, b) for i, a in enumerate(family) for b in family[i+1:])
        plan.append({'id': f"{record['condition']}_{len(plan)+1:02d}_{kind}", 'kind': kind,
                     'supports': family, 'reference_path_index': reference['index'],
                     'reference_train_mae': reference['train_mae'],
                     'reference_validation_mae': reference['validation_mae'], **extra})

    # Every missing true block compatible BEFORE the first crossing error is
    # compared against the actual winning MILP objective at that step.
    before_family = set(map(frozenset, before['supports']))
    for support in serial(truth - before_family):
        if all(not crossing(support, s) for s in before_family):
            add('first_cross_true_alternative', list(before_family) + [support],
                points[first_cross], alternative_true_block=support,
                predecessor_train_mae=before['train_mae'])

    chosen_family = set(map(frozenset, chosen['supports']))
    for support in serial(chosen_family - truth):
        add('selected_wrong_deletion', chosen_family - {frozenset(support)}, chosen,
            deleted_blocks=[support])
    for support in serial(truth - chosen_family):
        removed = {s for s in chosen_family if crossing(s, support)}
        add('selected_true_replacement', list(chosen_family - removed) + [support], chosen,
            added_true_block=support, deleted_blocks=serial(removed))
    add('last_first_cross_deletion',
        set(map(frozenset, last['supports'])) - {frozenset(first_cross_support)}, last,
        deleted_blocks=[first_cross_support])
    assert len(plan) <= 12, f"Diagnostic cap exceeded: {len(plan)}"
    return {'condition': record['condition'], 'first_wrong_path_index': first_wrong,
            'first_crossing_wrong_path_index': first_cross,
            'first_crossing_wrong_support': first_cross_support,
            'selected_path_index': record['selected_path_index'],
            'trajectory': trajectory, 'planned_lp_count': len(plan), 'lp_plan': plan,
            'first_wrong_deletion_reuses_previous_certified_path': {
                'previous_path_index': first_wrong - 1,
                'train_mae': points[first_wrong - 1]['train_mae'],
                'increase_over_first_wrong_point': points[first_wrong - 1]['train_mae'] - points[first_wrong]['train_mae']}}


def main(args):
    repo, runs, source, output = map(Path.resolve, (args.repo, args.runs, args.source_run, args.output))
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'rnj_wzzt_core'))
    from research_experiments.wzzt_dr_selection.baseline_alignment import run_observed_baseline as loader
    from research_experiments.wzzt_dr_selection.end_to_end import wzzt_adapter as observed
    from rnj_wzzt.estimation import laminar_l1_milp as core

    evidence = repo / 'research_experiments/wzzt_dr_selection/evidence/baseline_alignment_v1'
    aligned_file = evidence / 'alignment_results.json'
    aligned = json.loads(aligned_file.read_text(encoding='utf-8'))
    with (runs / 'N0_default_reference/prepared_case.pkl').open('rb') as stream:
        case, _ = pickle.load(stream)
    n1_train, n1_valid, n1_truth, n1_hashes = loader.load_inputs(source)
    datasets = {'N0': (case.training, case.validation, list(case.terminals),
                       set(case.truth_clades) | {frozenset([t]) for t in case.terminals}),
                'N1': (n1_train, n1_valid, list(n1_train[0]['P_terminal'].columns),
                       set(map(frozenset, n1_truth)))}
    conditions = [declare_plan(r, datasets[r['condition']][3]) for r in aligned['records']]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Existing diagnostic evidence; refusing repeated LPs')
    lock = output.with_suffix('.lock')
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(now())
    options = {'solver': 'highs', 'time_limit': 1800., 'r_upper_bound': 2.,
               'x_upper_bound': 2., 'presolve': True}
    result = {'status': 'planned', 'created_at': now(), 'scope': __doc__,
              'truth_policy': 'Post hoc explanatory audit; true alternatives do not enter deployed selection',
              'conditions': conditions, 'planned_lp_count': sum(c['planned_lp_count'] for c in conditions),
              'attempted_lp_count': 0, 'completed_lp_count': 0, 'failed_lp_count': 0,
              'milp_calls': 0, 'options': options, 'records': [],
              'source_sha256': {str(p): sha(p) for p in [Path(__file__).resolve(), Path(core.__file__).resolve(),
                                                       Path(loader.__file__).resolve()]},
              'input_sha256': {str(aligned_file): sha(aligned_file),
                  str(runs / 'N0_default_reference/prepared_case.pkl'): sha(runs / 'N0_default_reference/prepared_case.pkl'),
                  **n1_hashes},
              'environment': {'python': sys.version, 'executable': sys.executable,
                              'platform': platform.platform(), 'numpy': np.__version__, 'scipy': scipy.__version__}}

    def save():
        observed._write_json(output, result)

    # The separate immutable protocol records the entire support-level LP list
    # and hashes before the first call; results are checkpointed after each LP.
    protocol = output.with_name(output.stem + '_protocol.json')
    if protocol.exists():
        lock.unlink()
        raise FileExistsError('Existing diagnostic protocol; no duplicate run')
    observed._write_json(protocol, result)
    result['protocol_sha256'] = sha(protocol)
    save()
    try:
        result['status'] = 'running'
        for condition in conditions:
            training, validation, terminals, truth = datasets[condition['condition']]
            positions = {label: i for i, label in enumerate(terminals)}
            for task in condition['lp_plan']:
                rec = {'id': task['id'], 'condition': condition['condition'], 'status': 'running', 'started_at': now()}
                result['records'].append(rec)
                result['attempted_lp_count'] += 1
                save()
                start = perf_counter()
                try:
                    supports = [tuple(positions[label] for label in s) for s in task['supports']]
                    fit = core.solve_fixed_support_l1(training, supports, **options)
                    r, x = core.build_matrices_from_atoms(len(terminals), fit.supports, fit.r_values, fit.x_values)
                    train_mae, _, _ = core.evaluate_l1_matrices(training, r, x)
                    val_mae, val_se, _ = core.evaluate_l1_matrices(validation, r, x, blocks_per_scenario=4)
                    if not np.isclose(train_mae, fit.objective, rtol=1e-7, atol=1e-10):
                        raise ValueError('Independent objective recomputation failed')
                    if fit.diagnostics.status != 0:
                        raise ValueError('Fixed-support LP did not return optimal status')
                    rec.update({'status': 'complete', 'solver': observed._diagnostic(fit.diagnostics),
                                'supports': [[terminals[i] for i in s] for s in fit.supports],
                                'r_values': fit.r_values.tolist(), 'x_values': fit.x_values.tolist(),
                                'train_mae': train_mae, 'validation_mae': val_mae, 'validation_se': val_se,
                                'delta_train_vs_reference': train_mae - task['reference_train_mae'],
                                'delta_validation_vs_reference': val_mae - task['reference_validation_mae'],
                                'reference_gain_tolerance': 1e-10 + 1e-8 * abs(task['reference_train_mae'])})
                    result['completed_lp_count'] += 1
                except Exception as exc:
                    rec.update({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
                    result['failed_lp_count'] += 1
                rec.update({'finished_at': now(), 'elapsed_seconds': perf_counter() - start})
                save()
        result.update({'status': 'complete' if result['failed_lp_count'] == 0 else 'complete_with_failures',
                       'finished_at': now(), 'inputs_unchanged': all(sha(p) == h for p, h in result['input_sha256'].items())})
        save()
        print(json.dumps({'status': result['status'], 'planned': result['planned_lp_count'],
                          'completed': result['completed_lp_count'], 'failed': result['failed_lp_count'],
                          'records': [{k: r.get(k) for k in ('id', 'train_mae', 'delta_train_vs_reference', 'status')}
                                      for r in result['records']]}, indent=2))
    finally:
        lock.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
