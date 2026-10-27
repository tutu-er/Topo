"""Two post hoc fixed-RNJ-family LP diagnostics; never launch a MILP.

Support sets come only from the previously generated complete RNJ clades plus
terminal singletons. Truth is read only for subsequent recovery evaluation.
This is a diagnostic after observing RNJ6/6, not a preregistered new algorithm.
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


def family_json(family):
    return [sorted(s) for s in sorted(family, key=lambda s: (len(s), tuple(sorted(s))))]


def recovery(support_labels, r_values, x_values, true_internal, terminals):
    active = {frozenset(s) for s, r, x in zip(support_labels, r_values, x_values, strict=True) if r > 1e-9 or x > 1e-9}
    true_all = set(true_internal) | {frozenset([t]) for t in terminals}
    internal = {s for s in active if 1 < len(s) < len(terminals)}
    return {'effective_supports': family_json(active), 'effective_support_count': len(active),
        'internal_matched_count': len(internal & true_internal), 'true_internal_count': len(true_internal),
        'internal_predicted_count': len(internal), 'missing_true_internal': family_json(true_internal - internal),
        'extra_internal': family_json(internal - true_internal),
        'complete_effective_family': active == true_all, 'zero_tolerance': 1e-9}


def main(args):
    repo, root, source, output = args.repo.resolve(), args.runs.resolve(), args.source_run.resolve(), args.output.resolve()
    sys.path.insert(0, str(repo))
    from research_experiments.wzzt_dr_selection.end_to_end import wzzt_adapter as observed
    from research_experiments.wzzt_dr_selection.baseline_alignment import run_observed_baseline as loader
    from rnj_wzzt.estimation import laminar_l1_milp as core
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Diagnostic already recorded; refusing duplicate LPs: ' + str(output))
    lock = output.with_suffix('.lock')
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(now())
    records = []
    result = {'status': 'running', 'created_at': now(), 'scope': __doc__, 'planned_lp_count': 2,
        'attempted_lp_count': 0, 'milp_calls': 0,
        'options': {'solver': 'highs', 'time_limit': 1800., 'r_upper_bound': 2., 'x_upper_bound': 2., 'presolve': True},
        'support_policy': 'All already generated complete RNJ clades plus singletons; no truth-based support filtering',
        'truth_policy': 'Recovery only, after support selection and parameter optimization',
        'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'numpy': np.__version__, 'scipy': scipy.__version__},
        'source_sha256': {str(Path(__file__).resolve()): sha(__file__), str(Path(core.__file__).resolve()): sha(core.__file__),
            str(Path(loader.__file__).resolve()): sha(loader.__file__)}, 'records': records}

    def save():
        observed._write_json(output, result)

    save()
    try:
        for name in ('N0_default_reference', 'N1_iflex_no_reward'):
            folder = root / name
            selection_path = folder / ('selection.json' if name.startswith('N0') else 'rnj.json')
            selection = json.loads(selection_path.read_text(encoding='utf-8'))
            hashes = {str(selection_path): sha(selection_path)}
            if name.startswith('N0'):
                prepared_path = folder / 'prepared_case.pkl'; hashes[str(prepared_path)] = sha(prepared_path)
                with prepared_path.open('rb') as stream:
                    case, manifest = pickle.load(stream)
                training, validation, terminals = case.training, case.validation, list(case.terminals)
                true_internal = set(case.truth_clades)
                snapshot_path = folder / 'fit_progress.json'
            else:
                training, validation, truth, data_hashes = loader.load_inputs(source)
                hashes.update(data_hashes)
                terminals = list(training[0]['P_terminal'].columns)
                true_internal = {frozenset(s) for s in truth if 1 < len(s) < len(terminals)}
                snapshot_path = folder / 'wzzt.json'
            # The selection is wholly algorithm-generated; do not read truth
            # when constructing supports or deciding which weights to retain.
            positions = {t: i for i, t in enumerate(terminals)}
            support_labels = [[t] for t in terminals] + selection['full_clades']
            supports = [tuple(positions[t] for t in clade) for clade in support_labels]
            snapshot_bytes = snapshot_path.read_bytes()
            snapshot = json.loads(snapshot_bytes)
            snapshot_hash = hashlib.sha256(snapshot_bytes).hexdigest()
            points = snapshot.get('path', snapshot.get('last_certified_path', []))
            path_comparison = [{'path_index': i, 'train_mae_pu2': p['train_mae'],
                'validation_mae_pu2': p['validation_mae'], 'validation_se_pu2': p['validation_se'],
                'recovery': recovery(p['support_labels'], p['r_values'], p['x_values'], true_internal, terminals)}
                for i, p in enumerate(points)]
            record = {'case': name, 'status': 'running', 'started_at': now(), 'input_sha256': hashes,
                'training_time_rows': sum(len(s['P_terminal']) for s in training),
                'validation_time_rows': sum(len(s['P_terminal']) for s in validation),
                'requested_rnj_support_labels': support_labels, 'requested_support_indices': supports,
                'algorithm_path_snapshot': {'file': str(snapshot_path), 'sha256': snapshot_hash,
                    'captured_at': now(), 'status': snapshot.get('status', snapshot.get('event')),
                    'not_final_if_running': True, 'path': path_comparison}}
            records.append(record); result['attempted_lp_count'] += 1; save()
            start = perf_counter()
            try:
                # Exactly one fixed-support LP per dataset. This function does
                # not call the free-support extension or RNJ/bootstrap again.
                fitted = core.solve_fixed_support_l1(training, supports, **result['options'])
                r, x = core.build_matrices_from_atoms(len(terminals), fitted.supports, fitted.r_values, fitted.x_values)
                train_mae, _, _ = core.evaluate_l1_matrices(training, r, x)
                valid_mae, valid_se, valid_blocks = core.evaluate_l1_matrices(validation, r, x, blocks_per_scenario=4)
                if not np.isclose(train_mae, fitted.objective, atol=1e-10, rtol=1e-7):
                    raise ValueError('LP objective does not reproduce from fixed-support matrix multiplication')
                optimized_labels = [[terminals[i] for i in s] for s in fitted.supports]
                record.update({'status': 'complete', 'solver': observed._diagnostic(fitted.diagnostics),
                    'elapsed_seconds': perf_counter() - start, 'train_mae_pu2': train_mae,
                    'validation_mae_pu2': valid_mae, 'validation_se_pu2': valid_se,
                    'validation_block_means_pu2': valid_blocks.tolist(),
                    'optimized_support_labels': optimized_labels,
                    'r_values': fitted.r_values.tolist(), 'x_values': fitted.x_values.tolist(),
                    'recovery': recovery(optimized_labels, fitted.r_values, fitted.x_values, true_internal, terminals),
                    'minimum_recorded_path_train_mae_pu2': min(p['train_mae_pu2'] for p in path_comparison),
                    'minimum_recorded_path_validation_mae_pu2': min(p['validation_mae_pu2'] for p in path_comparison)})
            except Exception as exc:
                record.update({'status': 'failed', 'elapsed_seconds': perf_counter() - start,
                    'exception_type': type(exc).__name__, 'exception': str(exc)})
            record['finished_at'] = now(); save()
        result.update({'status': 'complete', 'finished_at': now(),
            'completed_lp_count': sum(r['status'] == 'complete' for r in records),
            'failed_lp_count': sum(r['status'] != 'complete' for r in records)})
        save()
        print(json.dumps({'status': result['status'], 'attempted_lp_count': result['attempted_lp_count'],
            'completed_lp_count': result['completed_lp_count'], 'milp_calls': 0,
            'records': [{k: r.get(k) for k in ('case', 'status', 'train_mae_pu2', 'validation_mae_pu2', 'elapsed_seconds', 'recovery', 'exception')} for r in records]}, indent=2))
    finally:
        lock.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True); p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--source-run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
