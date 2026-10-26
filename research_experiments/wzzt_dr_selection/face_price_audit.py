"""A2b: optimal-face pricing and finite-batch counterexample audit.

The future-loss weight is a diagnostic, never a tunable price parameter.
All designs are locked before evaluator use. Finite hypothesis representatives
and floating-point dual intervals do not provide topology coverage.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import time
import json

import numpy as np

from .core import block_features, select_next_block
from .experiment import (ROOT, MENUS, History, TOL, build_hypotheses, cases,
                         checked_fit, clean, current_environment, digest,
                         jblock, run_lock, save, utilities)
from .exact_patterns import PATTERNS, pattern_cost, probability
from .optimal_face import fit_future_on_history_optimal_face

WEIGHTS = (.05, .25, 1., 4.)
METHODS = ('frozen', 'optimal_face', 'exact_refit', 'dual_lower_score', 'd_opt')
STOP = 1e-8


class DesignAuditFailure(RuntimeError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def validate_design_lock(path, designs):
    fingerprint = digest(designs)
    if path.exists() and json.loads(path.read_text())['fingerprint'] != fingerprint:
        raise RuntimeError('Locked design changed; existing evidence is retained')
    return fingerprint


def rank_pairs(left, right):
    out = {'pairs': 0, 'strict_reversals': [], 'left_ties_right_strict': [],
           'left_strict_right_ties': [], 'both_ties': 0}
    def sign(value):
        return 0 if abs(value) <= TOL else (1 if value > 0 else -1)
    for i in range(len(MENUS)):
        for j in range(i+1, len(MENUS)):
            a, b = sign(left[i]-left[j]), sign(right[i]-right[j])
            names = [MENUS[i]['name'], MENUS[j]['name']]
            out['pairs'] += 1
            if a*b == -1:
                out['strict_reversals'].append(names)
            elif a == 0 and b != 0:
                out['left_ties_right_strict'].append(names)
            elif a != 0 and b == 0:
                out['left_strict_right_ties'].append(names)
            elif a == 0 and b == 0:
                out['both_ties'] += 1
    return out


def matrix_for_pattern(pattern):
    early = np.asarray(pattern, dtype=float)
    P = .2 + np.column_stack([early, 1-early])
    return P, np.zeros_like(P)


def select(history, P=None, Q=None, Y=None, weight=1.):
    if P is None:
        return select_next_block(history.P, history.Q, history.Y, history.selected, tie_tol=TOL)
    return select_next_block(np.column_stack([history.P, weight*P]),
                             np.column_stack([history.Q, weight*Q]),
                             np.column_stack([history.Y, weight*Y]),
                             history.selected, tie_tol=TOL)


def audit_cases():
    specs = list(cases())
    name, h, forecast, evaluator = specs[1]
    metadata = {s[0]: {'prefix': None, 'selected': s[1].selected} for s in specs}
    for i in range(3):
        pattern = tuple(int(j == i) for j in range(3))
        P, Q = matrix_for_pattern(pattern)
        augmented = History(np.column_stack([h.P, P]), np.column_stack([h.Q, Q]),
                            np.column_stack([h.Y, evaluator.R @ P]), h.selected)
        step = select(augmented)
        if step.stopped is None:
            raise RuntimeError('Prefix WZZT proposal failed')
        B = augmented.selected
        if not step.stopped:
            B = B + (min(step.best_candidates),)
        observed = History(augmented.P, augmented.Q, augmented.Y, B)
        case_name = f'weak_after_prefix_{i}'
        specs.append((case_name, observed, forecast, evaluator))
        metadata[case_name] = {'prefix': pattern, 'selected': B,
                              'prefix_best_candidates': step.best_candidates,
                              'prefix_proposal_stopped': step.stopped,
                              'rule': 'append lexical WZZT proposal from observed data; no truth check'}
    return specs, metadata


def decision(base, losses):
    if not losses:
        return {'stopped': True, 'best': (), 'improvement': 0.}
    best = min(losses.values())
    return {'stopped': base-best <= STOP,
            'best': tuple(c for c, loss in losses.items() if loss <= best+TOL),
            'improvement': base-best}


def dual_interval(hist_fit, face, Fh, Ff, yh, yf, weight):
    alpha = float(face['dual_history_cap_multiplier'])
    eta = weight if alpha <= 0 else min(weight, 1/alpha)
    u = (1-eta*alpha)*hist_fit.dual + eta*np.asarray(face['dual_history'])
    v = eta*np.asarray(face['dual_future'])
    violation = float(max(0., np.max(np.abs(u)-1, initial=0.),
                          np.max(np.abs(v)-weight, initial=0.),
                          np.max(Fh.T@u + Ff.T@v, initial=0.)))
    combined = float(yh@u + yf@v)
    lower = max(hist_fit.dual_bound, combined, 0.)
    upper = face['history_loss'] + weight*face['future_loss']
    return {'lower': lower, 'upper': upper, 'dual_violation': violation,
            'alpha': alpha, 'eta': eta, 'local_range': weight*alpha <= 1+1e-10,
            'face_gap': face['gap'], 'history_gap': hist_fit.gap,
            'numerically_feasible': violation <= TOL}


def numerical_winner(intervals):
    candidate = {c: x for c, x in intervals.items() if c is not None}
    if not candidate or not all(x['numerically_feasible'] for x in intervals.values()):
        return ()
    c = min(candidate, key=lambda k: candidate[k]['upper'])
    competing = [v['lower'] for k, v in candidate.items() if k != c]
    if (candidate[c]['upper'] + TOL < min(competing, default=float('inf'))
            and candidate[c]['upper'] + STOP < intervals[None]['lower']):
        return (c,)
    return ()


def bound_good_score(intervals, good):
    if not all(x['numerically_feasible'] for x in intervals.values()):
        return 0.
    upper = [v['upper'] for c, v in intervals.items() if c in good and c is not None]
    lower = [v['lower'] for c, v in intervals.items() if c not in good and c is not None]
    if not upper:
        return 0.
    u = min(upper)
    return float(u+TOL < min(lower, default=float('inf')) and u+STOP < intervals[None]['lower'])


def design_case(history, forecast):
    """Only historical observations and the forecast law enter this function."""
    clocks, calls = defaultdict(float), defaultdict(int)
    t0 = time.perf_counter()
    hs, pool = build_hypotheses(history)
    clocks['hypothesis_setup'] += time.perf_counter()-t0
    calls['hypothesis_setup'] += pool['all_families']
    t0 = time.perf_counter()
    historical = select(history)
    clocks['historical_candidates'] += time.perf_counter()-t0
    calls['historical_candidates'] += 1+len(historical.fits)
    if historical.stopped is None:
        raise RuntimeError('Historical fits failed')
    fits = {None: historical.base_fit, **historical.fits}
    support_features = {c: block_features(history.P, history.Q, f.blocks) for c, f in fits.items()}
    all_supports = tuple((i,) for i in range(3)) + ((0, 1), (0, 2), (1, 2), (0, 1, 2))
    Fh_all = block_features(history.P, history.Q, all_supports)
    rows, failures = [], []
    pattern_scores = {str(w): {} for w in WEIGHTS}
    for pattern in PATTERNS:
        P, Q = matrix_for_pattern(pattern)
        Ff_all = block_features(P, Q, all_supports)
        info = {}
        for w in WEIGHTS:
            info[w] = float(np.linalg.slogdet(np.eye(Fh_all.shape[1]) +
                        Fh_all.T@Fh_all + w*w*(Ff_all.T@Ff_all))[1])
        accum = {w: defaultdict(list) for w in WEIGHTS}
        for hi, h in enumerate(hs):
            Yf = h.predict(P, Q)
            good = set(h.effective_blocks())-set(history.selected)
            faces, frozen_future = {}, {}
            for c, fitted in fits.items():
                t0 = time.perf_counter()
                face = fit_future_on_history_optimal_face(
                    history.P, history.Q, history.Y, P, Q, Yf, fitted.blocks)
                clocks['optimal_face'] += time.perf_counter()-t0
                calls['optimal_face'] += 1 if face['status'] == 'history_fit_failed' else 2
                faces[c] = face
                frozen_future[c] = float(np.abs(Yf-fitted.predict(P, Q)).sum())
            if not all(f['success'] for f in faces.values()):
                failures.append({'hypothesis': hi, 'pattern': pattern,
                                 'face_statuses': {str(c): f['status'] for c, f in faces.items()}})
                continue
            for w in WEIGHTS:
                losses = {method: {} for method in ('frozen', 'optimal_face', 'exact_refit')}
                intervals = {}
                for c, fitted in fits.items():
                    losses['frozen'][c] = fitted.loss+w*frozen_future[c]
                    losses['optimal_face'][c] = faces[c]['history_loss']+w*faces[c]['future_loss']
                    t0 = time.perf_counter()
                    exact = checked_fit(np.column_stack([history.P, w*P]),
                                        np.column_stack([history.Q, w*Q]),
                                        np.column_stack([history.Y, w*Yf]), fitted.blocks)
                    clocks[f'exact_refit_{w}'] += time.perf_counter()-t0
                    calls[f'exact_refit_{w}'] += 1
                    losses['exact_refit'][c] = exact.loss
                    interval = dual_interval(fitted, faces[c], support_features[c],
                                             block_features(P, Q, fitted.blocks),
                                             history.Y.ravel(), Yf.ravel(), w)
                    interval['exact'] = exact.loss
                    interval['face_minus_exact'] = losses['optimal_face'][c]-exact.loss
                    interval['brackets_exact'] = (interval['lower']-TOL <= exact.loss <= interval['upper']+TOL)
                    intervals[c] = interval
                decisions, scores = {}, {}
                for method, values in losses.items():
                    cand = {c: v for c, v in values.items() if c is not None}
                    decisions[method] = decision(values[None], cand)
                    score, _ = utilities(values[None], cand, good)
                    scores[method] = score
                    accum[w][method].append(score)
                scores['dual_lower_score'] = bound_good_score(intervals, good)
                accum[w]['dual_lower_score'].append(scores['dual_lower_score'])
                winner = numerical_winner(intervals)
                exact_decision = decisions['exact_refit']
                rows.append({'hypothesis': hi, 'pattern': pattern, 'weight': w,
                    'scores': scores, 'losses': {m: {str(c): v for c, v in ls.items()} for m, ls in losses.items()},
                    'decisions': decisions, 'intervals': {str(c): v for c, v in intervals.items()},
                    'numerical_winner': winner,
                    'numerical_winner_matches_exact': not winner or
                       (not exact_decision['stopped'] and exact_decision['best'] == winner),
                    'all_local_ranges_hold': all(x['local_range'] for x in intervals.values())})
        for w in WEIGHTS:
            if any(len(accum[w][m]) != len(hs) for m in METHODS[:-1]):
                pattern_scores[str(w)][str(pattern)] = None
            else:
                pattern_scores[str(w)][str(pattern)] = {
                    **{m: float(np.mean(accum[w][m])) for m in METHODS[:-1]}, 'd_opt': info[w]}
    menus, choices = {}, {}
    for w in WEIGHTS:
        menu_rows = []
        for menu in MENUS:
            weights = [probability(p, menu, forecast) for p in PATTERNS]
            ps = [pattern_scores[str(w)][str(p)] for p in PATTERNS]
            if any(s is None and pr > 0 for s, pr in zip(ps, weights)):
                raise DesignAuditFailure('A failed design pattern cannot be removed from its probability denominator',
                    {'failed_patterns': failures, 'partial_rows': rows, 'lp_calls': dict(calls)})
            menu_rows.append({'menu': menu['name'],
                'predicted_cost': sum(pr*pattern_cost(p, menu) for pr, p in zip(weights, PATTERNS)),
                **{m: sum(pr*s[m] for pr, s in zip(weights, ps) if pr > 0) for m in METHODS}})
        menus[str(w)] = menu_rows
        choices[str(w)] = {}
        for method in METHODS:
            best = max(row[method] for row in menu_rows)
            ties = [i for i, row in enumerate(menu_rows) if row[method] >= best-TOL]
            chosen = min(ties, key=lambda i: (round(menu_rows[i]['predicted_cost'], 12), i))
            choices[str(w)][method] = {'index': chosen, 'menu': MENUS[chosen]['name'],
                                      'ties': [MENUS[i]['name'] for i in ties]}
    return {'pool': pool, 'hypotheses': [{'blocks': h.blocks, 'beta': h.beta} for h in hs],
            'menus': menus, 'choices': choices, 'rows': rows, 'failures': failures,
            'lp_calls': dict(calls), 'seconds': dict(clocks),
            'cost_scope': 'face solves shared across four diagnostic weights; two LPs per face call; not an online speedup comparison',
            'parameter_coverage': False}


def observed_path(history, evaluator, P=None, Q=None, weight=1.):
    """Evaluation only; truth is used for labels after every design is locked."""
    if P is None:
        H = history
    else:
        H = History(np.column_stack([history.P, weight*P]), np.column_stack([history.Q, weight*Q]),
                    np.column_stack([history.Y, weight*(evaluator.R@P)]), history.selected)
    trace, B = [], H.selected
    targets = set(evaluator.true_blocks)-set(history.selected)
    final = None
    for _ in range(8):  # all seven nonempty supports bound the count, not true line count
        s = select(History(H.P, H.Q, H.Y, B))
        if s.stopped is None:
            return {'outcome': 'failed', 'trace': trace, 'target_blocks': targets,
                    'next_correct': False, 'full_recovery': False}
        final = s.base_fit
        if s.stopped:
            break
        c = min(s.best_candidates)
        trace.append({'chosen': c, 'all_best': s.best_candidates,
                      'strict_correct': all(b in evaluator.true_blocks for b in s.best_candidates),
                      'improvement': s.improvement})
        B = B+(c,)
    else:
        raise RuntimeError('Selection did not stop within support-count bound')
    effective = set(final.effective_blocks())
    recovered = targets & effective
    return {'outcome': 'correct' if trace and trace[0]['strict_correct'] else ('stopped' if not trace else 'wrong'),
            'next_correct': bool(trace and trace[0]['strict_correct']),
            'trace': trace, 'final_effective_blocks': effective, 'target_blocks': targets,
            'recovered_target_blocks': recovered, 'target_total': len(targets),
            'target_recovery_fraction': len(recovered)/len(targets) if targets else None,
            'full_recovery': effective == set(evaluator.true_blocks), 'final_loss': final.loss}


def evaluator_case(history, evaluator):
    no_new = observed_path(history, evaluator)
    all_rows, menus = [], {}
    for w in WEIGHTS:
        outcomes = []
        for pattern in PATTERNS:
            P, Q = matrix_for_pattern(pattern)
            result = observed_path(history, evaluator, P, Q, w)
            outcomes.append(result)
            all_rows.append({'weight': w, 'pattern': pattern, **result})
        menu_rows = []
        for menu in MENUS:
            probs = [probability(p, menu, evaluator.response) for p in PATTERNS]
            menu_rows.append({'menu': menu['name'],
                'next_correct': sum(pr*o['next_correct'] for pr, o in zip(probs, outcomes)),
                'full_recovery': sum(pr*o['full_recovery'] for pr, o in zip(probs, outcomes)),
                'target_recovery_fraction': (sum(pr*o['target_recovery_fraction'] for pr, o in zip(probs, outcomes))
                    if all(o.get('target_recovery_fraction') is not None for o in outcomes) else None),
                'cost': sum(pr*pattern_cost(p, menu) for pr, p in zip(probs, PATTERNS)),
                'failed_probability': sum(pr*(o['outcome']=='failed') for pr, o in zip(probs, outcomes))})
        menus[str(w)] = menu_rows
    return {'no_new_data': no_new, 'patterns': all_rows, 'menus': menus}


def summary(designs, evaluated):
    report = {}
    for name, d in designs.items():
        e = evaluated[name]
        by_weight = {}
        for w in WEIGHTS:
            rows = [r for r in d['rows'] if r['weight'] == w]
            menus = e['menus'][str(w)]
            designed_menus = d['menus'][str(w)]
            rankings = {
                'face_vs_exact_score': rank_pairs([m['optimal_face'] for m in designed_menus], [m['exact_refit'] for m in designed_menus]),
                'face_vs_actual_next': rank_pairs([m['optimal_face'] for m in designed_menus], [m['next_correct'] for m in menus]),
                'face_vs_actual_full': rank_pairs([m['optimal_face'] for m in designed_menus], [m['full_recovery'] for m in menus])}
            selections = {}
            for method, choice in d['choices'][str(w)].items():
                actual = menus[choice['index']]
                selections[method] = actual | {
                    'menu_regret_full_recovery': max(m['full_recovery'] for m in menus)-actual['full_recovery'],
                    'menu_regret_next_correct': max(m['next_correct'] for m in menus)-actual['next_correct']}
            by_weight[str(w)] = {'design_pattern_hypothesis_rows': len(rows),
                'face_stop_disagreements': sum(r['decisions']['optimal_face']['stopped'] != r['decisions']['exact_refit']['stopped'] for r in rows),
                'face_winner_disagreements': sum(
                    (r['decisions']['optimal_face']['stopped'], r['decisions']['optimal_face']['best']) !=
                    (r['decisions']['exact_refit']['stopped'], r['decisions']['exact_refit']['best']) for r in rows),
                'numerical_winners': sum(bool(r['numerical_winner']) for r in rows),
                'wrong_numerical_winners': sum(not r['numerical_winner_matches_exact'] for r in rows),
                'all_local_ranges_hold': sum(r['all_local_ranges_hold'] for r in rows),
                'interval_bracket_failures': sum(not v['brackets_exact'] for r in rows for v in r['intervals'].values()),
                'selection': selections, 'no_reward': menus[0], 'menu_rankings': rankings}
        report[name] = {'no_new_data': e['no_new_data'], 'weights': by_weight,
                        'pool': d['pool'], 'lp_calls': d['lp_calls'], 'seconds': d['seconds']}
    return clean_sets({'cases': report, 'price_problems': len(designs)*len(WEIGHTS),
                       'actual_pattern_evaluations': len(evaluated)*len(WEIGHTS)*len(PATTERNS),
                       'primary_weight': 1, 'scope': 'finite synthetic exact integration; no statistical confidence coverage'})


def clean_sets(value):
    if isinstance(value, (set, frozenset)):
        return [clean_sets(v) for v in sorted(value)]
    if isinstance(value, dict):
        return {str(k): clean_sets(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_sets(v) for v in value]
    return clean(value)


def sources():
    names = ('face_price_audit.py','protocol_a2b.json','core.py','experiment.py',
             'exact_patterns.py','optimal_face.py')
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'outputs'/'a2b_face_final_v2')
    args = parser.parse_args()
    with run_lock(args.output):
        protocol = json.loads((ROOT/'protocol_a2b.json').read_text())
        manifest = {'protocol': protocol, 'sources': sources(), 'environment': current_environment()}
        manifest['fingerprint'] = digest(manifest)
        mf = args.output/'manifest.json'
        if mf.exists() and json.loads(mf.read_text()) != manifest:
            raise RuntimeError('Source, protocol or environment changed; use a new output directory')
        save(mf, manifest)
        specs, prefix_metadata = audit_cases()
        assert [s[0] for s in specs] == protocol['cases']
        assert tuple(protocol['diagnostic_future_loss_weights']) == WEIGHTS
        assert protocol['menus'] == len(MENUS) and protocol['response_patterns'] == len(PATTERNS)
        save(args.output/'prefix_metadata.json', prefix_metadata)
        designs = {}
        for name, h, forecast, _ in specs:
            path = args.output/('design_'+name+'.json')
            if (args.output/('failure_design_'+name+'.json')).exists():
                raise RuntimeError('A retained design failure requires review and a new run directory')
            if path.exists():
                designs[name] = json.loads(path.read_text())
            else:
                try:
                    designs[name] = clean_sets(design_case(h, forecast))
                except Exception as exc:
                    save(args.output/('failure_design_'+name+'.json'), {'phase':'design','case':name,'error':repr(exc),'details':clean_sets(getattr(exc,'details',None))})
                    raise
                save(path, designs[name])
            print('Design locked:', name, flush=True)
        design_hash = validate_design_lock(args.output/'design_lock.json', designs)
        save(args.output/'design_lock.json', {'fingerprint': design_hash})
        evaluated = {}
        for name, h, _, ev in specs:
            path = args.output/('evaluation_'+name+'.json')
            if path.exists():
                data = json.loads(path.read_text())
                if data['design_fingerprint'] != design_hash:
                    raise RuntimeError('Evaluation checkpoint used a different locked design')
                evaluated[name] = data['result']
            else:
                evaluated[name] = clean_sets(evaluator_case(h, ev))
                save(path, {'design_fingerprint': design_hash, 'result': evaluated[name]})
            print('Evaluated:', name, flush=True)
        out = summary(designs, evaluated)
        save(args.output/'summary.json', out)
        print('Complete:', out['price_problems'], 'price problems;', out['actual_pattern_evaluations'], 'actual patterns')
        for name, c in out['cases'].items():
            x = c['weights']['1.0']
            print(name, 'no-new full', c['no_new_data']['full_recovery'],
                  'face mismatches', x['face_winner_disagreements'], '/', x['design_pattern_hypothesis_rows'],
                  {m: [v['menu'], round(v['next_correct'],4), round(v['full_recovery'],4)] for m,v in x['selection'].items()})


if __name__ == '__main__':
    main()
