"""Finite-response diagnostic: integrate all eight outcomes, with no Monte Carlo error.

This is a verification reference, not the proposed non-nested pricing algorithm.
Historical fitted representatives are unchanged and do not cover their optimal faces.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
from pathlib import Path

import numpy as np

from .experiment import (ROOT, MENUS, TOL, cases, clean, design, digest, evaluate,
                         save, source_hashes)


PATTERNS = tuple(product((0, 1), repeat=3))


def probability(pattern, menu, mode):
    b = np.asarray(pattern)
    q = np.asarray(menu['q'])
    if mode == 'independent':
        return float(np.prod(np.where(b, q, 1-q)))
    if mode in ('common', 'ignores_price_common'):
        if not (np.all(b == 0) or np.all(b == 1)):
            return 0.
        p = float(q.mean()) if mode == 'common' else .5
        return p if np.all(b == 1) else 1-p
    raise ValueError(mode)


def pattern_cost(pattern, menu):
    return 0. if menu['preferred'] is None else .1 * sum(
        a == b for a, b in zip(pattern, menu['preferred']))


def choose(rows, metric):
    best = max(row[metric] for row in rows)
    tied = [i for i, row in enumerate(rows) if row[metric] >= best-TOL]
    return min(tied, key=lambda i: (round(rows[i]['predicted_cost'], 12), i)), tied


def run():
    specifications = cases()
    locked = {}
    # First obtain every design score, without calling an evaluator or using truth.
    for name, history, forecast, _ in specifications:
        pattern_scores = []
        for pattern in PATTERNS:
            # Endpoints 0/1 yield the same forced pattern under all six interior q's.
            latent = np.array([[0. if bit else 1. for bit in pattern]])
            d = design(history, 'independent', latent)
            scores = d['scores'][0]
            for metric in ('predicted_strict_selection', 'frozen_screen', 'd_opt'):
                assert max(abs(row[metric]-scores[metric]) for row in d['scores']) < TOL
            pattern_scores.append({key: scores[key] for key in
                                   ('predicted_strict_selection', 'frozen_screen', 'd_opt')})
        menus = []
        for menu in MENUS:
            weights = [probability(p, menu, forecast) for p in PATTERNS]
            assert abs(sum(weights)-1) < 1e-12
            row = {'menu': menu['name'], 'forecast_pattern_probabilities': weights,
                   'predicted_cost': sum(w*pattern_cost(p, menu) for p, w in zip(PATTERNS, weights))}
            for metric in ('predicted_strict_selection', 'frozen_screen', 'd_opt'):
                row[metric] = sum(w*s[metric] for w, s in zip(weights, pattern_scores))
            menus.append(row)
        choices = {metric: choose(menus, metric) for metric in
                   ('predicted_strict_selection', 'frozen_screen', 'd_opt')}
        locked[name] = {'menus': menus, 'choices': choices, 'patterns': PATTERNS,
                        'pattern_scores': pattern_scores}
    design_fingerprint = digest(locked)

    # Truth is used only now, to score the already locked menus.
    result = {}
    for name, history, forecast, evaluator in specifications:
        outcomes = []
        for pattern in PATTERNS:
            latent = np.array([0. if bit else 1. for bit in pattern])
            # Force a deterministic pattern in an evaluation-only independent response.
            independent_evaluator = type(evaluator)(evaluator.R, evaluator.true_blocks, 'independent')
            outcome = evaluate(history, independent_evaluator, MENUS[0], latent)
            assert outcome['pattern'] == pattern
            outcomes.append(outcome)
        rows = []
        for i, menu in enumerate(MENUS):
            weights = [probability(p, menu, evaluator.response) for p in PATTERNS]
            assert abs(sum(weights)-1) < 1e-12
            rows.append(locked[name]['menus'][i] | {
                'actual_pattern_probabilities': weights,
                'actual_strict_correct_probability': sum(w*o['strict_correct'] for w, o in zip(weights, outcomes)),
                'actual_outcome_probabilities': {key: sum(w*(o['outcome'] == key) for w, o in zip(weights, outcomes))
                    for key in ('correct', 'wrong', 'stopped', 'zero_weight', 'failed')},
                'actual_expected_cost': sum(w*pattern_cost(p, menu) for p, w in zip(PATTERNS, weights))})
        result[name] = {'menus': rows,
            'choices': {k: {'chosen': MENUS[v[0]]['name'],
                            'score_ties': [MENUS[i]['name'] for i in v[1]]}
                        for k, v in locked[name]['choices'].items()},
            'pattern_outcomes': outcomes, 'pattern_scores': locked[name]['pattern_scores']}
    assert digest(locked) == design_fingerprint
    return clean({'status': 'complete', 'response_patterns_per_case': 8,
                  'independent_mc_samples': 0, 'integration': 'exact finite distribution',
                  'design_fingerprint': design_fingerprint,
                  'scope': 'specified synthetic model only; finite fitted representatives, no parameter coverage',
                  'cases': result})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'outputs'/'exact_patterns_v1.json')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a new output filename; prior evidence is retained')
    result = run()
    result['source_hashes'] = source_hashes() | {
        'exact_patterns.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(args.output, result)
    for name, case in result['cases'].items():
        print(name, {k: v['chosen'] for k, v in case['choices'].items()})
        for row in case['menus']:
            print(row['menu'], 'pred', round(row['predicted_strict_selection'], 6),
                  'actual', round(row['actual_strict_correct_probability'], 6),
                  'cost', round(row['actual_expected_cost'], 6))


if __name__ == '__main__':
    main()
