"""A1 counterexamples and A2 paired price ranking, with a strict design/evaluation split."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from .core import (all_blocks, block_features, enumerate_laminar_families,
                   fit_l1, select_next_block)

ROOT = Path(__file__).resolve().parent
TOL = 1e-7


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True).encode()).hexdigest()


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT / 'core.py', ROOT / 'experiment.py', ROOT / 'protocol.json']}


def current_environment():
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'platform': platform.platform()}


def validate_resume_rows(rows, designs, cases_list, config):
    """Reject incompatible checkpoints rather than silently mixing experiments."""
    expected = config['a2']['test_draws']
    latents = {name: np.random.default_rng(config['a2']['seed_test']+index).random((expected, 3))
               for index, (name, *_rest) in enumerate(cases_list)}
    fingerprints = {name: digest(value) for name, value in designs.items()}
    menus = {m['name'] for m in MENUS}
    done = set()
    for row in rows:
        name, replicate, menu = row.get('case'), row.get('replicate'), row.get('menu')
        if name not in latents or menu not in menus or type(replicate) is not int or not 0 <= replicate < expected:
            raise ValueError('Checkpoint row has an invalid case/replicate/menu key')
        key = (name, replicate, menu)
        if key in done:
            raise ValueError('Duplicate checkpoint row key')
        if row.get('design_fingerprint') != fingerprints[name]:
            raise ValueError('Checkpoint row design fingerprint differs from the locked design')
        if not np.array_equal(np.asarray(row.get('latent')), latents[name][replicate]):
            raise ValueError('Checkpoint row test latent differs from its indexed random stream')
        if row.get('outcome') not in ('correct', 'wrong', 'stopped', 'zero_weight', 'failed'):
            raise ValueError('Checkpoint row has an invalid outcome')
        if type(row.get('correct')) is not bool or type(row.get('strict_correct')) is not bool:
            raise ValueError('Checkpoint row correctness flags must be Boolean')
        if row['correct'] != (row['outcome'] == 'correct') or (row['strict_correct'] and not row['correct']):
            raise ValueError('Checkpoint row outcome and correctness flags contradict each other')
        done.add(key)
    return done


def jblock(n, block):
    z = np.zeros(n)
    z[list(block)] = 1
    return np.outer(z, z)


def checked_fit(P, Q, Y, blocks):
    fit = fit_l1(P, Q, Y, blocks)
    if not fit.success:
        raise RuntimeError(f'LP failed: {fit.status}: {fit.message}')
    return fit


def omission_loss(P, Q, Y, block):
    """Exact minimum over all n<=4 families omitting block, not a pool certificate."""
    fits = [checked_fit(P, Q, Y, f) for f in enumerate_laminar_families(P.shape[0])
            if block not in f]
    if not fits:
        raise ValueError('Empty null family must never be treated as confirmation')
    return min(f.loss for f in fits)


def run_a1():
    P = np.array([[1., 0.], [1., 1.], [1., 3.]])
    R = np.array([[3., 2., 1.], [2., 3., 1.], [1., 1., 2.]])
    Q, Y = np.zeros_like(P), R @ P
    B = ((0,), (1,), (2,))
    step = select_next_block(P, Q, Y, B)
    losses = {str(b): f.loss for b, f in step.fits.items()}
    assert abs(losses['(0, 2)']) < TOL
    assert abs(losses['(0, 1, 2)'] - 5/3) < TOL
    assert abs(losses['(0, 1)'] - 20/3) < TOL
    assert step.best_candidates == ((0, 2),)
    old_null = {str(c): omission_loss(P, Q, Y, c) for c in ((0, 1), (0, 2), (0, 1, 2))}
    assert max(old_null.values()) < TOL
    P_full = np.concatenate([P, np.eye(3)], axis=1)
    Y_full = R @ P_full
    new_null = {str(c): omission_loss(P_full, np.zeros_like(P_full), Y_full, c)
                for c in ((0, 1), (0, 2), (0, 1, 2))}
    assert new_null['(0, 1)'] > TOL and new_null['(0, 1, 2)'] > TOL
    assert new_null['(0, 2)'] < TOL

    weak = []
    P2 = np.column_stack([np.ones(2), np.eye(2)])
    for a in (0., 1e-6, 1e-3, .03, .3):
        Ra = np.array([[2-a, a], [a, 2-a]])
        null = omission_loss(P2, np.zeros_like(P2), Ra @ P2, (0, 1))
        assert abs(null - 4*a) < TOL
        common = np.array([[1., 2., 3.], [1., 2., 3.]])
        common_null = omission_loss(common, np.zeros_like(common), Ra @ common, (0, 1))
        assert common_null < TOL
        weak.append({'weight': a, 'null_loss': null, 'expected': 4*a,
                     'common_mode_null_loss': common_null})

    K = np.eye(3) + jblock(3, (0, 1))
    I = np.eye(3)
    pq_null = omission_loss(I, I, K, (0, 1))
    assert pq_null > TOL
    # R/X are not individually identified, while their common support is.
    assert np.allclose(K @ I + 0*I, 0*I + K @ I)
    H = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
    assert np.allclose(H @ I + (-H) @ I, 0)

    # Nonunique historical weights give different future predictions.
    Ph = np.ones((2, 1))
    diag_R, shared_R = 2*np.eye(2), np.ones((2, 2))
    assert np.allclose(diag_R @ Ph, shared_R @ Ph)
    prediction_gap = float(np.abs(diag_R - shared_R).sum())
    assert prediction_gap == 4

    # E[min L1] != min E[L1] even with a nonnegative scalar coefficient.
    individual = [checked_fit(np.array([[p]]), np.zeros((1, 1)), np.ones((1, 1)), ((0,),)).loss
                  for p in (1., 2.)]
    together = checked_fit(np.array([[1., 2.]]), np.zeros((1, 2)), np.ones((1, 2)), ((0,),)).loss / 2
    assert max(individual) < TOL and abs(together - .25) < TOL
    return {
        'status': 'passed', 'checks': 6,
        'omitted_lines': {'losses': losses, 'selected_false_block': step.best_candidates,
                          'historical_omission_losses': old_null, 'after_identity_inputs': new_null},
        'zero_weak_and_common': weak,
        'pq_confounding': {'R_plus_X_identified_block_null_loss': pq_null,
                           'R_alone_design_can_be_zero': True},
        'nonunique_fit': {'same_history': True, 'future_prediction_L1_difference': prediction_gap},
        'random_refit_order': {'mean_of_minima': np.mean(individual), 'minimum_mean': together},
        'family_counts': {str(n): len(enumerate_laminar_families(n)) for n in range(1, 5)},
        'claim': 'deterministic exact-model checks only; no statistical confidence claim'
    }


@dataclass(frozen=True)
class History:
    P: np.ndarray
    Q: np.ndarray
    Y: np.ndarray
    selected: tuple


@dataclass(frozen=True)
class Evaluator:
    """Never passed to the design function."""
    R: np.ndarray
    true_blocks: frozenset
    response: str


def cases():
    P = np.array([[1., 0.], [1., 1.], [1., 3.]])
    R = np.eye(3) + jblock(3, (0, 1, 2)) + jblock(3, (0, 1))
    common = np.array([[1., 2.], [1., 2.], [1., 2.]])
    weak = np.eye(3) + .4*jblock(3, (0, 1, 2)) + .03*jblock(3, (0, 1))
    B = ((0,), (1,), (2,))
    truth = frozenset(B + ((0, 1), (0, 1, 2)))
    specifications = [('omitted_lines', P, R, 'independent', 'independent'),
                      ('weak_line', common, weak, 'independent', 'independent'),
                      ('common_mode', common, R, 'common', 'common'),
                      ('forecast_mismatch', P, R, 'independent', 'ignores_price_common')]
    return [(name, History(p, np.zeros_like(p), r @ p, B), forecast,
             Evaluator(r, truth, actual)) for name, p, r, forecast, actual in specifications]


MENUS = (
    {'name': 'no_reward', 'preferred': None, 'q': [.5, .5, .5]},
    {'name': 'uniform_early', 'preferred': [1, 1, 1], 'q': [.9, .9, .9]},
    {'name': 'node0_early', 'preferred': [1, 0, 0], 'q': [.9, .1, .1]},
    {'name': 'node1_early', 'preferred': [0, 1, 0], 'q': [.1, .9, .1]},
    {'name': 'node2_early', 'preferred': [0, 0, 1], 'q': [.1, .1, .9]},
    {'name': 'uniform_late', 'preferred': [0, 0, 0], 'q': [.1, .1, .1]},
)


def response(menu, latent, mode):
    q = np.asarray(menu['q'])
    if mode == 'independent':
        early = latent < q
    elif mode in ('common', 'ignores_price_common'):
        probability = float(q.mean()) if mode == 'common' else .5
        early = np.full(3, latent[0] < probability)
    else:
        raise ValueError(mode)
    flex = np.column_stack([early, ~early]).astype(float)
    assert np.allclose(flex.sum(axis=1), 1.) and np.max(flex) <= 1
    cost = 0. if menu['preferred'] is None else .1*float(np.sum(early == menu['preferred']))
    assert cost <= .3 + 1e-12
    P = .2 + flex  # fixed background plus the feasible one-unit session
    return P, np.zeros_like(P), cost, tuple(int(x) for x in early)


def build_hypotheses(history):
    fits = [checked_fit(history.P, history.Q, history.Y, f)
            for f in enumerate_laminar_families(history.P.shape[0])]
    best = min(f.loss for f in fits)
    compatible = [f for f in fits if f.loss <= best + TOL]
    # Deduplicate identical full prediction matrices, not names of parameterizations.
    unique = {}
    I, Z = np.eye(history.P.shape[0]), np.zeros((history.P.shape[0],)*2)
    for fit in compatible:
        key = tuple(np.round(np.concatenate([fit.predict(I, Z).ravel(), fit.predict(Z, I).ravel()]), 10))
        unique.setdefault(key, fit)
    if not unique:
        raise RuntimeError('No compatible hypotheses')
    return list(unique.values()), {'all_families': len(fits), 'compatible_families': len(compatible),
                                  'unique_representatives': len(unique), 'best_history_loss': best,
                                  'parameter_coverage': False}


def utilities(base_loss, candidate_losses, good):
    if not candidate_losses:
        return 0., None
    best = min(candidate_losses.values())
    if base_loss - best <= 1e-8:
        return 0., None
    g = [loss for block, loss in candidate_losses.items() if block in good]
    b = [loss for block, loss in candidate_losses.items() if block not in good]
    if not g:
        return 0., None
    margin = (min(b) - min(g)) if b else float('inf')
    return float(margin > TOL), margin


def design(history, forecast_mode, latent):
    """Contains no truth, true parameters, actual-response mode or test RNG argument."""
    hypotheses, pool = build_hypotheses(history)
    historical = select_next_block(history.P, history.Q, history.Y, history.selected)
    if historical.stopped is None:
        raise RuntimeError('Historical candidate fit failed')
    cache, records = {}, []
    Fh = block_features(history.P, history.Q, all_blocks(3))
    for menu in MENUS:
        scores, screens, infos, costs = [], [], [], []
        for draw in latent:
            P, Q, cost, pattern = response(menu, draw, forecast_mode)
            if pattern not in cache:
                h_scores, h_screens, h_margins = [], [], []
                for h in hypotheses:
                    predicted = h.predict(P, Q)
                    Pa = np.concatenate([history.P, P], axis=1)
                    Qa = np.concatenate([history.Q, Q], axis=1)
                    Ya = np.concatenate([history.Y, predicted], axis=1)
                    result = select_next_block(Pa, Qa, Ya, history.selected, tie_tol=TOL)
                    if result.stopped is None:
                        raise RuntimeError('Design reference LP incomplete')
                    good = set(h.effective_blocks()) - set(history.selected)
                    score, margin = utilities(result.base_fit.loss,
                                             {c: f.loss for c, f in result.fits.items()}, good)
                    frozen = {c: f.loss + float(np.abs(predicted - f.predict(P, Q)).sum())
                              for c, f in historical.fits.items()}
                    base_frozen = historical.base_fit.loss + float(np.abs(predicted - historical.base_fit.predict(P, Q)).sum())
                    screen, _ = utilities(base_frozen, frozen, good)
                    h_scores.append(score)
                    h_screens.append(screen)
                    h_margins.append(margin)
                F = block_features(P, Q, all_blocks(3))
                info = np.linalg.slogdet(np.eye(F.shape[1]) + Fh.T @ Fh + F.T @ F)[1]
                cache[pattern] = (float(np.mean(h_scores)), float(np.mean(h_screens)),
                                  float(info), h_margins)
            score, screen, info, _ = cache[pattern]
            scores.append(score); screens.append(screen); infos.append(info); costs.append(cost)
        records.append({'menu': menu['name'], 'predicted_strict_selection': np.mean(scores),
                        'frozen_screen': np.mean(screens), 'd_opt': np.mean(infos),
                        'predicted_cost': np.mean(costs)})
    choices = {}
    for metric in ('predicted_strict_selection', 'frozen_screen', 'd_opt'):
        maximum = max(row[metric] for row in records)
        eligible = [i for i, row in enumerate(records) if row[metric] >= maximum - TOL]
        # Cost only breaks a declared numerical score tie; no test information used.
        choices[metric] = min(eligible, key=lambda i: (records[i]['predicted_cost'], i))
    return {'scores': records, 'choices': choices, 'pool': pool,
            'hypotheses': [{'blocks': h.blocks, 'beta': h.beta, 'effective': h.effective_blocks(),
                            'loss': h.loss} for h in hypotheses],
            'cached_response_patterns': len(cache),
            'scope': 'finite fitted representatives; exhaustive supports but NOT exhaustive weights',
            'screen_scope': 'one historical LP optimizer; feasible loss upper bounds, no candidate-order guarantee; refuted by exact-pattern diagnostic'}


def evaluate(history, evaluator, menu, latent):
    P, Q, cost, pattern = response(menu, latent, evaluator.response)
    result = select_next_block(np.concatenate([history.P, P], axis=1),
                               np.concatenate([history.Q, Q], axis=1),
                               np.concatenate([history.Y, evaluator.R @ P], axis=1),
                               history.selected, tie_tol=TOL)
    common = {'cost': cost, 'flex_energy': 3., 'flex_squared_norm': 3., 'pattern': pattern,
              'best_candidates': result.best_candidates, 'status': result.status,
              'improvement': result.improvement, 'objective': result.best_loss,
              'candidate_gaps': {str(c): f.gap for c, f in result.fits.items()},
              'P': P, 'Q': Q, 'Y': evaluator.R @ P}
    if result.stopped is None:
        return common | {'outcome': 'failed', 'correct': False, 'strict_correct': False}
    if result.stopped:
        return common | {'outcome': 'stopped', 'correct': False, 'strict_correct': False}
    # Fixed lexical tie rule is for an unconfirmed proposal only, never a certificate.
    chosen = min(result.best_candidates)
    effective = set(result.fits[chosen].effective_blocks())
    if chosen not in effective:
        return common | {'outcome': 'zero_weight', 'correct': False, 'strict_correct': False}
    correct = chosen in evaluator.true_blocks
    strict = all(c in evaluator.true_blocks for c in result.best_candidates)
    return common | {'chosen': chosen, 'outcome': 'correct' if correct else 'wrong',
                     'correct': correct, 'strict_correct': strict,
                     'tie': len(result.best_candidates) > 1}


def summarize(rows, designs, expected):
    summary = {}
    for name, d in designs.items():
        by_menu = []
        for menu in MENUS:
            selected = [r for r in rows if r['case'] == name and r['menu'] == menu['name']]
            counts = {outcome: sum(r['outcome'] == outcome for r in selected)
                      for outcome in ('correct', 'wrong', 'stopped', 'zero_weight', 'failed')}
            known_costs = [r['cost'] for r in selected if r.get('cost') is not None]
            by_menu.append({'menu': menu['name'], 'completed': len(selected), 'denominator': expected,
                            'counts': counts, 'strict_correct': sum(r['strict_correct'] for r in selected),
                            'ties': sum(r.get('tie', False) for r in selected),
                            'mean_cost': np.mean(known_costs) if known_costs else None,
                            'cost_observations': len(known_costs)})
        case_rows = [r for r in rows if r['case'] == name]
        replicas = {r['replicate'] for r in case_rows if 'replicate' in r}
        all_menus = {m['name'] for m in MENUS}
        complete_paired = sum({r['menu'] for r in case_rows if r.get('replicate') == rep} == all_menus
                              for rep in replicas)
        summary[name] = {'menus': by_menu, 'design_choices': {k: MENUS[v]['name'] for k, v in d['choices'].items()},
                         'pool': d['pool'], 'design_scores': d['scores'],
                         'observed_replicates': len(replicas),
                         'complete_paired_replicates': complete_paired}
    return {'completed_updates': len(rows), 'planned_updates': len(designs)*len(MENUS)*expected,
            'planned_independent_test_realizations_per_case': expected,
            'cases': summary, 'not_a_confidence_audit': True,
            'parameter_coverage': False, 'production_WZZT': False}


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'run.lock'
    try:
        handle = lock.open('x', encoding='utf-8')
    except FileExistsError:
        raise RuntimeError(f'Existing run lock: {lock}; inspect recorded PID before resuming')
    try:
        handle.write(json.dumps({'pid': os.getpid(), 'created_unix': time.time()}))
        handle.close()
        yield
    finally:
        handle.close()
        lock.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['a1', 'a2', 'all'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'a1_a2_final_v4')
    parser.add_argument('--limit', type=int, default=None, help='maximum new A2 rows for this invocation')
    args = parser.parse_args()
    with run_lock(args.output):
        run(args)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    hashes = source_hashes()
    fingerprint = digest({'config': config, 'sources': hashes})
    manifest_file = args.output / 'manifest.json'
    if manifest_file.exists():
        previous = json.loads(manifest_file.read_text(encoding='utf-8'))
        if previous['fingerprint'] != fingerprint:
            raise RuntimeError('Source/config changed: choose a new output directory; existing results retained')
        if previous['environment'] != current_environment():
            raise RuntimeError('Execution environment changed: use a new output directory')
    else:
        save(manifest_file, {'fingerprint': fingerprint, 'sources': hashes, 'config': config,
                             'environment': current_environment()})
    a1 = run_a1()
    save(args.output / 'a1.json', a1)
    print('A1 passed: 6 diagnostic groups', flush=True)
    if args.stage == 'a1':
        return
    designs = {}
    cases_list = cases()
    # Lock all designs before constructing or reading any future test realization.
    for index, (name, history, forecast, evaluator) in enumerate(cases_list):
        latent = np.random.default_rng(config['a2']['seed_design'] + index).random((config['a2']['design_draws'], 3))
        designs[name] = design(history, forecast, latent)
        print(f'Design locked: {name}; representatives={designs[name]["pool"]["unique_representatives"]}', flush=True)
    design_file = args.output / 'designs.json'
    if design_file.exists() and digest(json.loads(design_file.read_text(encoding='utf-8'))) != digest(designs):
        raise RuntimeError('Recomputed design differs from the previously locked design')
    path = args.output / 'rows.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    done = validate_resume_rows(rows, designs, cases_list, config)
    save(design_file, designs)
    expected = config['a2']['test_draws']
    new_count = 0
    with path.open('a', encoding='utf-8') as handle:
        for index, (name, history, forecast, evaluator) in enumerate(cases_list):
            latent = np.random.default_rng(config['a2']['seed_test'] + index).random((expected, 3))
            for replicate, draw in enumerate(latent):
                for menu in MENUS:
                    key = (name, replicate, menu['name'])
                    if key in done:
                        continue
                    if args.limit is not None and new_count >= args.limit:
                        save(args.output / 'summary.json', summarize(rows, designs, expected))
                        print(f'Checkpoint: {len(rows)}/{len(cases_list)*len(MENUS)*expected}', flush=True)
                        return
                    try:
                        result = evaluate(history, evaluator, menu, draw)
                    except Exception as exc:
                        result = {'outcome': 'failed', 'correct': False, 'strict_correct': False,
                                  'cost': None, 'error': repr(exc)}
                    row = clean({'case': name, 'replicate': replicate, 'menu': menu['name'],
                                 'latent': draw, 'design_fingerprint': digest(designs[name]), **result})
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                    handle.flush()
                    rows.append(row); done.add(key); new_count += 1
            save(args.output / 'summary.json', summarize(rows, designs, expected))
            print(f'Completed {name}: {len(rows)} rows', flush=True)
    summary = summarize(rows, designs, expected)
    summary['invocation_wall_seconds'] = time.perf_counter() - started
    save(args.output / 'summary.json', summary)
    print(f'Complete: {len(rows)}/{summary["planned_updates"]}; seconds={summary["invocation_wall_seconds"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
