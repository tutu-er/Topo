"""Predeclared geometry, synthetic-observation and AC paired stress study.

Run as python -m research_experiments.rnj.run_advantage_stress from the repository
root. Output directories must be new; raw observations,
failures, budgets, source hashes and every comparison are retained. Search uses
the existing bounded experiment adapter; production defaults are not changed.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "rnj_wzzt_core"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CORE))
import numpy as np
import pandas as pd
import scipy
from rnj_wzzt.estimation.laminar_l1_milp import (
    is_admissible_extension, solver_diagnostics_prove_optimality,
)
from rnj_wzzt.estimation.multiscenario import align_scenarios, fit_projected_sensitivity
from research_experiments.rnj.sensitivity_geometry import sensitivity_geometry
from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
from rnj_wzzt.graph.bootstrap import _boundary_cherries
from rnj_wzzt.pipeline import _select_boundary_blocks
from research_experiments.rnj.rooted_aggregation import PseudoCluster, aggregate_rooted_scenarios, _expand_pseudo_result_clades
from research_experiments.rnj.rooted_ablation_support import (BoundedPath, rooted_scores, serialized,
    _rnj_reduced_candidate_pool, _map_clade_to_reduced_support)
from research_experiments.rnj.rooted_study_baselines import infer_classical_nj
from research_experiments.rnj.run_rooted_ablation import fixed_fit, prediction_metrics
from research_experiments.rnj.stress_support import (generate_tree, perturb_matrices, synthetic_splits,
                            plain, write_json, source_fingerprint, paired_summary)

GEOMETRY_STRESSES = ('exact', 'weak_internal', 'noise_02', 'noise_10', 'x_noisy', 'r_noisy', 'diagonal_only')
MEASUREMENT_STRESSES = ('reference', 'low_sample', 'high_noise', 'pq_collinear',
                       'terminal_correlated', 'outliers', 'weak_internal', 'common_noise')
AC_STRESSES = ('reference', 'low_sample', 'high_noise')
GEOMETRY_METHODS = ('rnj_R', 'rnj_X', 'rnj_RX75', 'rnj_RX75_zero_tau',
                    'rnj_RX50', 'classical_nj_zero', 'classical_nj_016')
MEASUREMENT_METHODS = ('rnj_R', 'rnj_X', 'rnj_RX75', 'rnj_fixed_tree_lp',
                       'classical_nj_validation', 'wzzt_rnj_pool', 'wzzt_rnj_pool_native', 'wzzt_one_edit',
                       'wzzt_unrestricted', 'wzzt_frozen', 'wzzt_contracted',
                       'validation_selected_hybrid')


def validation_mae(scenarios, r, x):
    """Observed-root prediction loss; no fitted bias or truth/test input exists."""
    residual = np.concatenate([
        (s['drop_target'].to_numpy() - s['P_terminal'].to_numpy() @ r.T
         - s['Q_terminal'].to_numpy() @ x.T).ravel()
        for s in scenarios
    ])
    return float(np.mean(np.abs(residual)))



def candidate_coverage(pool, initial, labels, members, truth_clades):
    """Report raw coverage and compatibility with the initial fixed family.

    These are recall upper bounds, not a certificate that greedy search can
    recover the whole true family. None denotes unrestricted, unenumerated
    coverage; the legacy candidate_truth_recall aliases admissible coverage.
    """
    if pool is None:
        raw_recall = admissible_recall = None
    else:
        def expanded(supports):
            return {frozenset().union(*(members[int(labels[i])] for i in support))
                    for support in supports}
        raw = expanded((*initial, *pool))
        compatible = tuple(s for s in pool if is_admissible_extension(s, initial, len(labels)))
        admissible = expanded((*initial, *compatible))
        denominator = len(truth_clades)
        raw_recall = len(raw & truth_clades) / denominator if denominator else 1.0
        admissible_recall = len(admissible & truth_clades) / denominator if denominator else 1.0
    return {'candidate_raw_truth_recall': raw_recall,
            'candidate_admissible_truth_recall': admissible_recall,
            'candidate_truth_recall': admissible_recall}


def infer(r, x, terminals, *, mode='RX_75R_25X', tau=.16, root=0):
    geo = sensitivity_geometry(r, x, mode)
    tree = rooted_neighbor_joining(geo.shared_paths, geo.root_depths, terminals, root,
                                  group_tolerance=tau * max(float(np.median(geo.root_depths)), 1e-12))
    return rooted_clades(tree.edges, root, terminals)


def geometry_job(job):
    truth = generate_tree(job['shape'], job['n'], job['seed'], .03 if job['stress'] == 'weak_internal' else 1.)
    r, x = perturb_matrices(truth, job['stress'], job['seed'] + 700_001)
    variants = [('rnj_R', 'R', .16), ('rnj_X', 'X', .16),
                ('rnj_RX75', 'RX_75R_25X', .16), ('rnj_RX75_zero_tau', 'RX_75R_25X', 0.),
                ('rnj_RX50', 'RX_equal_normalized', .16),
                ('classical_nj_zero', None, 0.), ('classical_nj_016', None, .16)]
    rows = []
    for name, mode, factor in variants:
        row = {k: job[k] for k in ('tier', 'stress', 'shape', 'n', 'repeat', 'job_id', 'cluster_id')}
        started = time.perf_counter()
        try:
            if mode is None:
                tree = infer_classical_nj(r, x, truth.terminals, 0, collapse_factor=factor)
                clades = rooted_clades(tree.edges, 0, truth.terminals)
            else:
                clades = infer(r, x, truth.terminals, mode=mode, tau=factor)
            row.update(rooted_scores(clades, truth.clades, truth.terminals), status='complete')
        except Exception as exc:
            row.update(status='failed', error=repr(exc), clade_f1=None)
        row.update(method=name, elapsed_seconds=time.perf_counter() - started)
        rows.append(row)
    return rows


def ac_splits(job):
    from rnj_wzzt.scenario.simulation import _simulate_pool, _terminal_buses
    from rnj_wzzt.models.lin_distflow import build_reduced_sensitivity_matrices
    from rnj_wzzt.reporting import _truth_nontrivial_clades
    sets = {}
    noise = .001 if job['stress'] == 'high_noise' else .0002
    for split_id, split in enumerate(('train', 'validation', 'test')):
        net, raw = _simulate_pool(
            case_key=job['shape'], t_count=96, replicate=3*job['repeat'] + split_id,
            maximum_scenarios=3, pq_noise_rel=.005, v_noise_rel=noise,
            scenario_suite='reference', root_observation='noisy', root_meter_noise_rel=.0002)
        count = 96 if split == 'test' else (8 if job['stress'] == 'low_sample' else job['samples'])
        selected = np.linspace(0, 95, count, dtype=int)
        sets[split] = []
        for idx, scenario in enumerate(raw):
            item = {key: value.iloc[selected].copy() if isinstance(value, (pd.DataFrame, pd.Series)) and len(value) == 96 else value
                    for key, value in scenario.items()}
            item['name'] = f'scenario_{idx}'
            item['source_sample_indices'] = selected.tolist()
            item['scenario_settings'] = {**scenario['scenario_settings'],
                                         'observed_sample_count': count,
                                         'observation_interval_hours': 24.0/count}
            item['diagnostics'] = {**scenario['diagnostics'], 'observed_sample_count': count,
                                   'selected_sample_count': count,
                                   'sample_basis': 'full_physical_grid; observed count reflects selected rows'}
            sets[split].append(item)
    terminals = _terminal_buses(net)
    truth = tuple(build_reduced_sensitivity_matrices(net, terminals, voltage_model='squared-voltage'))
    return sets, terminals, truth, _truth_nontrivial_clades(net, terminals), {
        'root': int(net.root_bus), 'edges': net.branches.to_dict(orient='records'),
        'scenario_diagnostics': {k: [s['diagnostics'] for s in v] for k, v in sets.items()},
        'scenario_settings': {k: [s['scenario_settings'] for s in v] for k, v in sets.items()},
        'terminal_meter_noise_rel': noise, 'pq_meter_noise_rel': .005,
        'root_observation': 'noisy', 'root_meter_noise_rel': .0002,
    }


def measurement_job(job):
    started = time.perf_counter()
    out = Path(job['output']) / 'jobs' / job['job_id']
    out.mkdir(parents=True, exist_ok=True)
    if job['tier'] == 'ac':
        sets, terminals, truth, true_clades, metadata = ac_splits(job)
    else:
        tree = generate_tree(job['shape'], job['n'], job['seed'], .03 if job['stress'] == 'weak_internal' else 1.)
        sets, truth = synthetic_splits(tree, job['stress'], job['seed'] + 900_001, job['samples'])
        terminals, true_clades = tree.terminals, tree.clades
        metadata = {'root': 0, 'edge_supports': tree.supports, 'r_weights': .001*tree.r_weights, 'x_weights': .001*tree.x_weights}
    n = len(terminals)
    root = metadata['root']
    packed = {f'{split}_{day}_{key}': s[key].to_numpy()
              for split, scenarios in sets.items() for day, s in enumerate(scenarios)
              for key in ('P_terminal', 'Q_terminal', 'drop_target', 'V_terminal', 'root_voltage',
                          'P_true', 'Q_true', 'V_terminal_true', 'root_voltage_true')
              if isinstance(s.get(key), (pd.DataFrame, pd.Series))}
    packed.update(R_true=truth[0], X_true=truth[1], terminals=terminals)
    np.savez_compressed(out/'inputs.npz', **packed)
    metadata.update(job=job, truth_clades=serialized(true_clades),
                    snapshots={split: sum(len(s['P_terminal']) for s in raw) for split, raw in sets.items()},
                    source_sample_indices={split: [s.get('source_sample_indices') for s in raw]
                                           for split, raw in sets.items()})
    write_json(out/'metadata.json', metadata)
    rows, details = [], {}
    base = {k: job[k] for k in ('tier', 'stress', 'shape', 'repeat', 'job_id', 'cluster_id')}
    base.update(n=n, bootstrap_replicates=job['bootstrap'], search_budget_seconds=job['search_seconds'])
    models = {}

    def record(name, clades, rr=None, xx=None, *, elapsed=0., status='complete', extra=None):
        row = dict(base, method=name, elapsed_seconds=elapsed, status=status, clade_f1=None)
        if clades is not None:
            row.update(rooted_scores(clades, true_clades, terminals))
        if rr is not None:
            row.update(prediction_metrics(sets, rr, xx, truth))
            models[name] = (clades, rr, xx)
        if extra:
            row.update({k: v for k, v in extra.items() if v is None or isinstance(v, (str, int, float, bool))})
        rows.append(row)
        details[name] = {'row': row, 'clades': None if clades is None else serialized(clades), 'details': extra}
        return row

    pre = time.perf_counter()
    training = align_scenarios(sets['train'])
    diagnostics = {}
    fitted = fit_projected_sensitivity(training, diagnostics=diagnostics)
    r, x = fitted[:2]
    qp_seconds = time.perf_counter() - pre
    clades, rnj_seconds = {}, {}
    for name, mode in [('rnj_R', 'R'), ('rnj_X', 'X'), ('rnj_RX75', 'RX_75R_25X')]:
        start = time.perf_counter()
        clades[name] = infer(r, x, terminals, mode=mode, root=root)
        rnj_seconds[name] = time.perf_counter()-start
        record(name, clades[name], r, x,
               elapsed=qp_seconds+time.perf_counter()-start, extra={'regression_diagnostics': diagnostics})
    rnj = clades['rnj_RX75']

    def refit(name, family, required=qp_seconds+rnj_seconds['rnj_RX75'], extra=None):
        start = time.perf_counter()
        try:
            sol, rr, xx, active = fixed_fit(sets['train'], family, terminals, job['solve_seconds'])
            record(name, family, rr, xx, elapsed=required+time.perf_counter()-start,
                   extra={'fixed_solver': sol.diagnostics.to_dict(),
                          'fixed_refit_certified': solver_diagnostics_prove_optimality(sol.diagnostics),
                          'active_weight_clades': serialized(active), **(extra or {})})
        except Exception as exc:
            record(name, family, elapsed=required+time.perf_counter()-start, status='refit_failed',
                   extra={'error': repr(exc), **(extra or {})})

    refit('rnj_fixed_tree_lp', rnj)
    # Both methods see the root anchor and identical RX75 geometry. NJ's four
    # declared thresholds are selected only by held-out validation prediction.
    start = time.perf_counter()
    nj_models, nj_errors = [], []
    for factor in (0., .04, .08, .16):
        try:
            nt = infer_classical_nj(r, x, terminals, root, collapse_factor=factor)
            nc = rooted_clades(nt.edges, root, terminals)
            sol, rr, xx, active = fixed_fit(sets['train'], nc, terminals, job['solve_seconds'])
            # No test-set or truth metrics are computed before threshold choice.
            val = validation_mae(sets['validation'], rr, xx)
            nj_models.append((val, factor, nc, rr, xx))
        except Exception as exc:
            nj_errors.append({'factor': factor, 'error': repr(exc)})
    if nj_models:
        best = min(nj_models, key=lambda m: (m[0], m[1]))
        record('classical_nj_validation', best[2], best[3], best[4],
               elapsed=qp_seconds+time.perf_counter()-start,
               extra={'selected_collapse_factor': best[1], 'candidate_errors': nj_errors})
    else:
        record('classical_nj_validation', None, status='failed', extra={'errors': nj_errors})

    start = time.perf_counter()
    _, confidence, selected, _, _ = _select_boundary_blocks(
        training, terminals, root, bootstrap_replicates=job['bootstrap'], block_length=4,
        confidence_threshold=.75, maximum_candidate_count=2, tolerance_factor=.16,
        seed=job['seed']+2_000_003, sensitivity_fit=fitted)
    bootstrap_seconds = time.perf_counter()-start
    identity = {t: frozenset({t}) for t in terminals}
    position = {t: i for i, t in enumerate(terminals)}
    pool_started = time.perf_counter()
    pools = {name: _rnj_reduced_candidate_pool(rnj, terminals, identity, include_one_edit=edit)
             for name, edit in [('rnj', False), ('one_edit', True)]}
    pool_seconds = time.perf_counter()-pool_started
    true_cherries = _boundary_cherries(true_clades)
    write_json(out/'candidates.json', {
        'pools': pools, 'selected': serialized(selected),
        'confidence': [{'clade': sorted(c), 'frequency': value} for c, value in confidence.items()],
        'truth_used_for_selection': False,
    })
    variants = ('wzzt_rnj_pool', 'wzzt_rnj_pool_native', 'wzzt_one_edit', 'wzzt_unrestricted', 'wzzt_frozen', 'wzzt_contracted')
    for name in variants:
        raw_train, raw_validation, members = sets['train'], sets['validation'], identity
        initial = [(i,) for i in range(n)]
        pool = None if name == 'wzzt_unrestricted' else pools['one_edit' if name == 'wzzt_one_edit' else 'rnj']
        required = 0. if pool is None else qp_seconds+rnj_seconds['rnj_RX75']+pool_seconds
        extra = {'selected_blocks': len(selected), 'selected_false_blocks': len(set(selected)-true_cherries),
                 'selected_false_clades': len(set(selected)-true_clades),
                 'selected_nonboundary_true_clades': len((set(selected)&true_clades)-true_cherries),
                 'search_adapter': 'bounded_native_milp' if pool is None or name == 'wzzt_rnj_pool_native' else 'bounded_exact_lp_enumeration'}
        start = time.perf_counter()
        try:
            if name in ('wzzt_frozen', 'wzzt_contracted'):
                required += bootstrap_seconds
            if name == 'wzzt_frozen':
                initial += [tuple(sorted(position[t] for t in c)) for c in selected]
            if name == 'wzzt_contracted':
                clusters = [PseudoCluster(900000+i, c, confidence[c], (), ()) for i, c in enumerate(selected)]
                raw_train, members = aggregate_rooted_scenarios(sets['train'], terminals, r, x, clusters,
                    voltage_mode='deembedded_vsq', deembedding_weight=.5)
                raw_validation, vm = aggregate_rooted_scenarios(sets['validation'], terminals, r, x, clusters,
                    voltage_mode='deembedded_vsq', deembedding_weight=.5)
                if members != vm:
                    raise RuntimeError('contraction maps differ')
                labels = list(raw_train[0]['P_terminal'].columns)
                mapped = [_map_clade_to_reduced_support(frozenset(terminals[i] for i in c), labels, members) for c in pool]
                pool = tuple(sorted({c for c in mapped if c is not None and len(c)>1}, key=lambda c: (len(c), c)))
                initial = [(i,) for i in range(len(labels))]
            labels = list(raw_train[0]['P_terminal'].columns)
            extra.update(candidate_coverage(pool, initial, labels, members, true_clades))
            extra['candidate_coverage_scope'] = 'raw pool versus compatibility with initial fixed family; neither certifies greedy recovery'
            extra['candidate_count'] = None if pool is None else len(pool)
            with BoundedPath(total_seconds=job['search_seconds'], solve_seconds=job['solve_seconds'],
                             enumerate_pool=pool is not None and name != 'wzzt_rnj_pool_native') as adapter:
                result = adapter.fit(raw_train, validation_scenarios=raw_validation,
                                     initial_supports=initial, candidate_supports=pool)
            recovered = _expand_pseudo_result_clades(result, members, n)
            extra.update(stop_reason=result.stop_reason, selected_path_index=result.selected_path_index,
                         selected_iteration=result.path[result.selected_path_index].iteration,
                         attempted_extensions=len(result.attempted_extensions), solver_audit=adapter.records(),
                         path=[{'iteration': p.iteration, 'train_mae': p.train_mae,
                                'validation_mae': p.validation_mae, 'validation_se': p.validation_se}
                               for p in result.path])
            extra['search_incomplete'] = result.stop_reason in (
                'experiment_total_budget', 'extension_not_proven_optimal',
                'bound_expansion_limit', 'extension_without_incumbent',
                'repeated_family_after_zero_pruning') or any(
                    a.diagnostics.status not in (0, 2) for a in result.attempted_extensions)
            extra['extensions_certified_optimal'] = sum(
                solver_diagnostics_prove_optimality(a.diagnostics) for a in result.attempted_extensions)
            extra['all_attempted_extensions_certified'] = all(
                a.diagnostics.status == 2 or solver_diagnostics_prove_optimality(a.diagnostics)
                for a in result.attempted_extensions)
            extra['structural_search_complete'] = not extra['search_incomplete']
            # All structured methods receive the same original-terminal L1
            # parameter refit. Native path matrices and diagnostics are saved.
            np.savez_compressed(out/f'{name}_native.npz', R=result.r_matrix, X=result.x_matrix)
            refit(name, recovered, required=required+time.perf_counter()-start, extra=extra)
        except Exception as exc:
            record(name, None, elapsed=required+time.perf_counter()-start, status='failed',
                   extra={**extra, 'error': repr(exc), 'traceback': traceback.format_exc()})
    # A declared deployable selector; topology truth and test observations do
    # not reach selection. Ties favor the faster RNJ fixed-tree reference.
    choices = [name for name in ('rnj_fixed_tree_lp', 'wzzt_rnj_pool') if name in models]
    if choices:
        selected_name = min(choices, key=lambda name: next(row['validation_mae'] for row in rows if row['method'] == name))
        picked = models[selected_name]
        record('validation_selected_hybrid', *picked,
               elapsed=sum(row['elapsed_seconds'] for row in rows if row['method'] in choices)
                       - (qp_seconds+rnj_seconds['rnj_RX75'] if len(choices)>1 else 0.),
               extra={'selected_method': selected_name})
    else:
        record('validation_selected_hybrid', None, status='failed',
               extra={'error': 'no available fitted candidate model'})
    write_json(out/'methods.json', details)
    write_json(out/'rows.json', rows)
    write_json(out/'completion.json', {'wall_seconds': time.perf_counter()-started})
    return rows


def safe_job(job):
    started = time.perf_counter()
    try:
        return geometry_job(job) if job['tier'] == 'geometry' else measurement_job(job)
    except Exception as exc:
        write_json(Path(job['output'])/'job_failures'/f"{job['job_id']}.json",
                   {'job': job, 'error': repr(exc), 'traceback': traceback.format_exc()})
        methods = GEOMETRY_METHODS if job['tier'] == 'geometry' else MEASUREMENT_METHODS
        elapsed = time.perf_counter()-started
        # Preserve every predeclared method in failure/missingness denominators.
        return [{**{key: job.get(key) for key in ('tier', 'stress', 'shape', 'n', 'repeat', 'job_id', 'cluster_id')},
                 'method': method, 'status': 'job_failed', 'clade_f1': None,
                 'clade_exact': None, 'elapsed_seconds': None, 'job_wall_seconds': elapsed,
                 'error': repr(exc)} for method in methods]


def build_jobs(args):
    jobs = []
    for tier in args.tiers:
        shapes = ('balanced', 'caterpillar', 'random', 'multifurcating', 'star') if tier == 'geometry' else (
            ('paper15', 'soumalas11', 'flynn16', 'pengwah18') if tier == 'ac' else ('balanced', 'caterpillar', 'random'))
        sizes = args.geometry_sizes if tier == 'geometry' else ([0] if tier == 'ac' else args.measurement_sizes)
        stresses = GEOMETRY_STRESSES if tier == 'geometry' else (AC_STRESSES if tier == 'ac' else MEASUREMENT_STRESSES)
        repeats = args.geometry_repeats if tier == 'geometry' else args.repeats
        for shape_id, shape in enumerate(shapes):
            for n in sizes:
                for repeat in range(repeats):
                    # Reuse independent base tree/AC replicate across stress levels.
                    seed = 20260910 + 100_003*shape_id + 1009*n + 37*repeat
                    cluster = f'{tier}_{shape}_n{n}_r{repeat}'
                    for stress in stresses:
                        jobs.append(dict(tier=tier, shape=shape, n=n, repeat=repeat, stress=stress,
                                         seed=seed, cluster_id=cluster, job_id=f'{cluster}_{stress}',
                                         output=str(args.output.resolve()), samples=args.samples,
                                         bootstrap=args.bootstrap, search_seconds=args.search_seconds,
                                         solve_seconds=args.solve_seconds))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tiers', nargs='+', choices=('geometry', 'synthetic', 'ac'), default=['geometry', 'synthetic', 'ac'])
    parser.add_argument('--geometry-sizes', nargs='+', type=int, default=[8, 16, 32, 64])
    parser.add_argument('--measurement-sizes', nargs='+', type=int, default=[8])
    parser.add_argument('--geometry-repeats', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--bootstrap', type=int, default=20)
    parser.add_argument('--search-seconds', type=float, default=4.)
    parser.add_argument('--solve-seconds', type=float, default=1.5)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if min(args.geometry_repeats, args.repeats, args.bootstrap, args.workers) < 1:
        parser.error('repeat/bootstrap/worker counts must be positive')
    if not 2 <= args.samples <= 96 or min(*args.geometry_sizes, *args.measurement_sizes) < 2:
        parser.error('samples must be in [2,96] and terminal counts >= 2')
    if any(not np.isfinite(v) or v <= 0 for v in (args.search_seconds, args.solve_seconds)):
        parser.error('solver budgets must be positive and finite')
    if args.output.exists():
        parser.error('output directory already exists; use a new path to preserve previous results')
    args.output.mkdir(parents=True)
    jobs = build_jobs(args)
    write_json(args.output/'protocol.json', {
        'schema_version': 3, 'jobs': jobs, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'observation_model': 'Observed-root raw drop = PR + QX, without fitted terminal/scenario bias. Synthetic injected offsets remain deliberate model mismatch; old intercept results require artifacts/observed_root_model_20260928/before.',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__},
        'source_sha256': source_fingerprint(CORE),
        'primary_reference': 'rnj_RX75', 'primary_metric': 'nontrivial rooted clade F1 of structural support family, including zero-weight fixed supports',
        'scope': 'reduced terminal tree; stem and hidden degree-two subdivisions excluded',
        'candidate_coverage': 'candidate_raw_truth_recall counts pool plus initial supports; candidate_admissible_truth_recall filters pool compatibility with the initial fixed family; candidate_truth_recall aliases admissible coverage. None denotes unenumerated unrestricted coverage. Schema 1 used the legacy field for raw coverage.',
        'geometry_scope': 'known matrices, not end-to-end meter accuracy; fixed NJ thresholds are not tuned on truth',
        'split_policy': 'independent train/validation/test replicas; all methods share arrays; test never selects',
        'sample_budget': '3*N training + 3*N validation, independent 288-point test',
        'synthetic_noise': 'noise relative to population squared-drop RMS, not meter percentage',
        'ac_noise': 'P/Q relative meter SD .005; terminal voltage SD .0002 or .001; noisy root .0002',
        'search_budget_scope': 'per structural model search, not entire runtime; preprocessing/bootstrap/refit separately included in elapsed',
        'solver_adapter': 'finite pools: bounded exact LP enumeration; unrestricted uses native MILP; paths are greedy. Historical wzzt_rnj_pool_native is unsupported by the current core and records an explicit failure; replay it with artifacts/core_audit_20260927/before.',
        'production_relation': 'component ablations, 20 bootstrap by default versus production 100; original-terminal L1 refit after contraction; production defaults unchanged',
        'statistics': 'paired by job, cluster bootstrap by independent tree/AC replicate; descriptive 95% intervals for this bank; all failed rows retained',
        'hypotheses': ['No universal winner on exact identifiable trees.',
                       'Fixed depth-based RNJ tolerance can erase weak internal branches.',
                       'RX75 advantage depends on channel error and branch separation.',
                       'L1 search can prune false candidate clades, but cannot recover clades missing from its pool.',
                       'Wrong frozen blocks and weak excitation can remove any topology advantage.'],
    })
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(safe_job, job): job for job in jobs}
        for completed, future in enumerate(as_completed(pending), 1):
            batch = future.result()
            rows.extend(batch)
            with (args.output/'rows.jsonl').open('a', encoding='utf-8') as stream:
                for row in batch:
                    stream.write(json.dumps(plain(row), ensure_ascii=False, allow_nan=False)+'\n')
            if completed % 20 == 0 or completed == len(jobs):
                print(f'{completed}/{len(jobs)} jobs; {len(rows)} method rows', flush=True)
    frame = pd.DataFrame(rows).sort_values(['tier', 'job_id', 'method'])
    frame.to_csv(args.output/'results.csv', index=False)
    frame.groupby(['tier', 'stress', 'method'], dropna=False).agg(
        attempted=('status', 'size'), available=('clade_f1', 'count'),
        mean_f1=('clade_f1', 'mean'), exact_rate=('clade_exact', 'mean'),
        mean_seconds=('elapsed_seconds', 'mean')).reset_index().to_csv(args.output/'summary.csv', index=False)
    paired = paired_summary(rows, 'rnj_RX75')
    write_json(args.output/'paired_comparisons.json', paired)
    pd.DataFrame(paired).to_csv(args.output/'paired_comparisons.csv', index=False)
    write_json(args.output/'completion.json', {'jobs': len(jobs), 'method_rows': len(rows),
               'job_failures': int(frame.loc[frame.status == 'job_failed', 'job_id'].nunique()),
               'source_unchanged': source_fingerprint(CORE) == json.loads((args.output/'protocol.json').read_text(encoding='utf-8'))['source_sha256']})
    print(f'Completed: {args.output}', flush=True)


if __name__ == '__main__':
    main()
