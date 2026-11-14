"""Bounded, isolated inspection of Pengwah autumn's original finite E pool.

The two search runs start independently from the original returned incumbent.
Offline truth and held-out observations are never supplied to StructuralSolver.
This diagnostic does not amend the original candidate bank or DFS inputs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from ... import io
from ...observations import load, fingerprint
from ...identification import algorithm, optimization, trees
from ...identification.algorithm import SearchConfig
from ...identification.optimization import QPFitter, StructuralSolver, criterion, errors


def topology(point, truth, n):
    active = {tuple(c) for c, r, x in zip(point['supports'], point['r'], point['x'])
              if max(r, x) > 1e-9}
    actual = set(map(tuple, truth))
    interior = lambda s: {c for c in s if 1 < len(c) < n}
    return dict(internal_exact=interior(active) == interior(actual), full_exact=active == actual,
                missing=sorted(actual-active, key=lambda x: (len(x), x)),
                extra=sorted(active-actual, key=lambda x: (len(x), x)))


def evaluate(point, observations, reference, alpha, truth):
    losses = {name: errors(data, point) for name, data in observations.items()}
    # Independent direct sum over supports checks the fixed-family matrix evaluator.
    max_error = 0.
    for name, data in observations.items():
        prediction = np.zeros_like(data.y)
        for c, r, x in zip(point['supports'], point['r'], point['x']):
            signal = r*data.p[:, c].sum(axis=1)+x*data.q[:, c].sum(axis=1)
            prediction[:, c] += signal[:, None]
        direct_mse = float(np.mean((data.y-prediction)**2))
        max_error = max(max_error, abs(direct_mse-losses[name]['mse']))
    if max_error > 1e-15:
        raise ArithmeticError('Independent support-sum replay failed')
    n = observations['train'].n
    total = sum(observations[k].y.size for k in ('validation', 'new_day'))
    cumulative = sum(losses[k]['mse']*observations[k].y.size for k in ('validation', 'new_day'))/total
    return dict(point=point, losses=losses,
                regularized_training_objective=losses['train']['mse']/reference+
                alpha*sum(1 < len(c) < n for c in point['supports']),
                cumulative_validation_new_mse=cumulative, topology=topology(point, truth, n),
                independent_loss_replay_max_error=max_error)


def run_search(train, pool, keep, original_warm, config, reference, seconds):
    """Search-only boundary: no truth, validation, or new-day arguments."""
    fitter = QPFitter(train, reference_mse=reference, bound=config.response_bound,
                      qp_seconds=config.qp_seconds)
    solver = StructuralSolver(fitter, config)
    started = perf_counter()
    try:
        raw, record = solver.select(deepcopy(pool), deepcopy(keep), deepcopy(original_warm),
                                    deadline=started+seconds)
        search_wall = perf_counter()-started
        closed = None
        if raw is not None:
            closed = fitter.fit(raw['supports'])
            score = criterion(closed, train.n, config.alpha)
            if record.get('bound') is not None and record['bound'] > score+1e-7*(1+abs(score)):
                raise ArithmeticError('Native lower bound exceeds feasible fixed-family objective')
        return dict(budget_seconds=seconds, search_wall_seconds=search_wall,
                    total_seconds=perf_counter()-started, native=record, raw_point=raw,
                    closed_point=closed, QP=fitter.records,
                    warm_start='original 30-second E incumbent; no truth or prior diagnostic result',
                    scope='same finite E pool; not a whole-tree global search')
    finally:
        solver.close()


def run(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    original = io.read(source/'101_native.json')
    last = next(e for e in reversed(original['events']) if e['stage'] == 'E')
    original_warm = deepcopy(last['point'])
    config = SearchConfig(**original['config'])
    observations = {name: load(source/(name+'.npz')) for name in ('train', 'validation', 'new_day')}
    reference, alpha, n = original['reference_mse'], config.alpha, observations['train'].n
    oracle = io.read(source/'offline_true_structure_qp.json')
    truth = oracle['point']['supports']
    true_internal = {tuple(c) for c in truth if 1 < len(c) < n}
    extras = {tuple(c) for c in original_warm['supports'] if 1 < len(c) < n}-true_internal
    if extras != {(0, 2, 3, 4)}:
        raise ValueError('The predeclared one-extra-support diagnostic no longer matches its input')
    source_paths = [Path(__file__), Path(algorithm.__file__), Path(optimization.__file__), Path(trees.__file__)]
    refs = [io.ref(p) for p in source_paths]
    for p in source_paths:
        target = out/'source'/p.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(p, target)
    inputs = {name: io.ref(source/name) for name in ('101_native.json', 'train.npz', 'validation.npz',
              'new_day.npz', 'offline_true_structure_qp.json')}
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), input=inputs, sources=refs,
                    environment=io.environment(), training_fingerprint=fingerprint(observations['train']),
                    independent_search_budgets_seconds=[60., 180.], second_search_trigger='first native solve not optimal to configured tolerance',
                    config=original['config'], original_pool=last['pool'], original_keep=last['keep'],
                    original_warm=original_warm,
                    scoring='192 training; held-out 96 validation and 96 new-day; no new physical simulation',
                    truth_use='offline topology comparison and explicitly labeled deletion QPs only; never solver inputs',
                    no_original_changes=True)
    io.write(out/'protocol.json', protocol)
    baseline = evaluate(original_warm, observations, reference, alpha, truth)
    fitter = QPFitter(observations['train'], reference_mse=reference,
                      bound=config.response_bound, qp_seconds=config.qp_seconds)
    fixed = {}
    for name, delete_extra, delete_common in [('original', False, False), ('delete_extra', True, False),
                                             ('delete_common', False, True), ('delete_both', True, True)]:
        family = [c for c in original_warm['supports']
                  if not (delete_extra and tuple(c) in extras) and not (delete_common and len(c) == n)]
        point = fitter.fit(family)
        fixed[name] = evaluate(point, observations, reference, alpha, truth)
    io.write(out/'fixed_structure_qp.json', dict(baseline=baseline, rows=fixed, QP=fitter.records,
             original_true_structure_qp=io.ref(source/'offline_true_structure_qp.json')))
    print('Fixed-family QPs saved', str(out/'fixed_structure_qp.json'), flush=True)
    searches = []
    for seconds in (60., 180.):
        receipt = run_search(observations['train'], last['pool'], last['keep'], original_warm,
                             config, reference, seconds)
        if receipt['closed_point'] is not None:
            receipt['evaluation'] = evaluate(receipt['closed_point'], observations, reference, alpha, truth)
        io.write(out/f'search_{int(seconds)}s.json', receipt)
        searches.append(receipt)
        print('Search saved', seconds, receipt['native'].get('status'),
              receipt.get('evaluation', {}).get('regularized_training_objective'), flush=True)
        if receipt['native'].get('status') == 2:
            break
    unchanged = all(io.ref(r['path'])['sha256'] == r['sha256'] for r in [*refs, *inputs.values()])
    if not unchanged:
        raise RuntimeError('Source or original input changed during diagnostic')
    report = dict(input=inputs, protocol=io.ref(out/'protocol.json'), original=baseline,
                  original_E_native=last['solver'], fixed_structure=fixed,
                  searches=searches, source_unchanged=unchanged,
                  interpretation='Finite-pool budget and four explicitly labeled fixed-family comparisons; no original candidate replacement or new independent day')
    io.write(out/'report.json', report)
    lines = ['# Pengwah autumn finite-pool diagnostic', '',
             'Original observations and candidate bank unchanged. All R/X fits remain free within their original bounds.', '',
             '| Fixed family | Train MSE | Validation MSE | New-day MSE | Regularized train objective | Internal exact |',
             '|---|---:|---:|---:|---:|---:|']
    for name, row in fixed.items():
        losses = row['losses']
        lines.append(f"| {name} | {losses['train']['mse']:.10g} | {losses['validation']['mse']:.10g} | {losses['new_day']['mse']:.10g} | {row['regularized_training_objective']:.10g} | {row['topology']['internal_exact']} |")
    lines += ['', '| Independent budget | Native status | Feasible objective | Lower bound | Gap | Internal exact |',
              '|---|---:|---:|---:|---:|---:|']
    for receipt in searches:
        native = receipt['native']; evaluation = receipt.get('evaluation', {})
        lines.append(f"| {receipt['budget_seconds']:g}s | {native.get('status')} | {evaluation.get('regularized_training_objective')} | {native.get('bound')} | {native.get('gap')} | {evaluation.get('topology', {}).get('internal_exact')} |")
    lines += ['', 'Every search uses only the original train data, original E pool, original incumbent, and original numerical settings. A longer run does not inherit the shorter run. Held-out scores and truth are evaluated after the returned point is frozen. Finite-pool bounds do not prove whole-tree optimality.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    io.write(out/'complete.json', dict(report=io.ref(out/'report.json'), source_unchanged=unchanged))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
