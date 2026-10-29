"""Frozen synthetic stochastic-DR mechanism model; no identifier imports.

The behavioral probabilities and response sizes are declared synthetic controls,
not estimates from iFlex or an EV behavioral calibration. Planning sees only
historical/forecast P and Q, prices, and the declared participation law.
"""
from __future__ import annotations

from itertools import product
import math
import numpy as np

LABELS = (101, 102, 103, 104, 105)
ROOT_LABEL = 1
BASE_KW = 100.0
POWER_FACTOR_Q_OVER_P = 0.35
SHAPE = np.array([0.9, 1.0, 1.1, 1.1, 1.0, 0.9])
RESPONSE_SHAPE = np.array([0.0, -1.0, 1.0, 0.0, 0.0, 0.0])
RESPONSE_KW = 0.2
PRICE_PROBABILITY = np.array([0.0, 0.5, 0.75])
HISTORY_MULTIPLIERS = (0.95, 1.0, 1.05)
ACTUAL_MULTIPLIERS = (0.98, 1.02, 1.03, 0.97)
HISTORY_DATES = ('2021-02-03', '2021-02-04', '2021-02-05')
EVENT_DATES = ('2021-02-09', '2021-02-10', '2021-02-11', '2021-02-12')
RIDGE = 1e-6
TIE_TOLERANCE = 1e-6
TOTAL_BUDGET = 1.6
CASE_B_R = {'normal': 0.06, 'weak': 0.006, 'zero': 0.0}


def constants():
    return {'labels': list(LABELS), 'root_label': ROOT_LABEL, 'base_kw': BASE_KW,
            'q_over_p': POWER_FACTOR_Q_OVER_P, 'baseline_shape': SHAPE.tolist(),
            'response_shape': RESPONSE_SHAPE.tolist(), 'response_kw': RESPONSE_KW,
            'price_probability': PRICE_PROBABILITY.tolist(),
            'history_multipliers': list(HISTORY_MULTIPLIERS),
            'actual_multipliers': list(ACTUAL_MULTIPLIERS),
            'history_dates': list(HISTORY_DATES), 'event_dates': list(EVENT_DATES),
            'ridge': RIDGE, 'tie_tolerance': TIE_TOLERANCE, 'total_budget_currency': TOTAL_BUDGET,
            'cases_B_r': dict(CASE_B_R), 'power_bounds_kw': [0.0, 1.5],
            'training_events': 2, 'validation_events': 2,
            'event_hours': 6, 'price_choices_currency_per_kwh': [0, 1, 2],
            'settlement': 'verified reduction in the designated reduction hour only; rebound unpaid',
            'behavioral_scope': 'synthetic declared law; no iFlex or EV empirical calibration'}


def _plan(plan):
    values = np.asarray(plan)
    if values.size != 10 or not np.isfinite(values).all():
        raise ValueError('A plan contains ten finite round-major menu choices')
    if not np.isin(values, (0, 1, 2)).all():
        raise ValueError('Allowed prices are 0, 1 and 2 currency/kWh')
    values = values.astype(int).reshape(2, 5)
    if values.sum() > 4:
        raise ValueError('Plan exceeds the frozen total worst-case budget')
    return values


def enumerate_plans():
    """All feasible plans, including concentrated and zero allocations."""
    return [p for p in product((0, 1, 2), repeat=10) if sum(p) <= 4]


def classify(plan):
    p = _plan(plan)
    return {'single': int(np.any(p > 0, axis=0).sum()) <= 1,
            'sync': bool(np.array_equal(p[0], p[1])), 'full': True}


def budget_bound(plan):
    """Two training rounds plus the same two validation price rows."""
    return float(2 * RESPONSE_KW * _plan(plan).sum())


def expected_payment(plan):
    p = _plan(plan)
    return float(2 * RESPONSE_KW * np.sum(p * PRICE_PROBABILITY[p]))


def _event_patterns(prices, law):
    probs = PRICE_PROBABILITY[np.asarray(prices, dtype=int)]
    if law == 'local':
        active = np.flatnonzero(probs > 0)
        patterns = []
        for bits in product((0, 1), repeat=len(active)):
            z = np.zeros(5, dtype=int)
            z[active] = bits
            probability = math.prod(float(probs[i] if z[i] else 1 - probs[i])
                                    for i in active)
            patterns.append((probability, z))
        return patterns
    if law == 'common':
        # One shared U in each event. Different prices keep their marginals.
        cuts = sorted({0.0, 1.0, *probs.tolist()})
        return [(hi - lo, (((lo + hi) / 2) < probs).astype(int))
                for lo, hi in zip(cuts[:-1], cuts[1:]) if hi > lo]
    raise ValueError("law must be 'local' or 'common'")


def participation_patterns(plan, law):
    """Exact two-event joint law; events independent in both controls."""
    p = _plan(plan)
    return [(a * b, np.stack((za, zb)))
            for a, za in _event_patterns(p[0], law)
            for b, zb in _event_patterns(p[1], law)]


def four_event_patterns(plan, law):
    """Independent training/validation pairs; no evaluator RNG is required."""
    two = participation_patterns(plan, law)
    return [(a * b, np.vstack((za, zb))) for a, za in two for b, zb in two]


def _convert(p, unit):
    if unit not in ('kw', 'pu'):
        raise ValueError("unit must be 'kw' or 'pu'")
    scale = 1.0 if unit == 'kw' else BASE_KW
    return p / scale, POWER_FACTOR_Q_OVER_P * p / scale


def history_arrays(unit='kw'):
    p = np.concatenate([np.repeat((m * SHAPE)[:, None], 5, axis=1)
                        for m in HISTORY_MULTIPLIERS])
    return _convert(p, unit)


def forecast_arrays(plan, participation=None, unit='kw'):
    p = _plan(plan)
    z = PRICE_PROBABILITY[p] if participation is None else np.asarray(participation, dtype=float)
    if z.shape != (2, 5) or not np.isfinite(z).all() or np.any((z < 0) | (z > 1)):
        raise ValueError('Participation must be a 2 by 5 array in [0,1]')
    if np.any(z[p == 0] != 0):
        raise ValueError('A zero-price node cannot respond in this synthetic model')
    rows = [np.repeat(SHAPE[:, None], 5, axis=1)
            + RESPONSE_KW * RESPONSE_SHAPE[:, None] * zk[None, :]
            for zk in z]
    return _convert(np.concatenate(rows), unit)


def planning_information():
    hp, hq = history_arrays()
    design = np.column_stack((hp, hq))
    scales = np.sqrt(np.mean(design ** 2, axis=0))
    scales = np.where(scales > 0, scales, 1.0)
    normalized = design / scales
    return normalized.T @ normalized + RIDGE * np.eye(10), scales


def design_score(plan, participation=None):
    """Original ten-column [P,Q] historical-RMS D-opt baseline."""
    information, scales = planning_information()
    p, q = forecast_arrays(plan, participation)
    design = np.column_stack((p, q)) / scales
    sign, value = np.linalg.slogdet(information + design.T @ design)
    if sign <= 0 or not np.isfinite(value):
        raise ArithmeticError('Regularized information matrix is invalid')
    return float(value)


def stochastic_score(plan, law):
    """E[logdet], exactly integrated; never logdet of expected information."""
    return float(sum(prob * design_score(plan, z)
                     for prob, z in participation_patterns(plan, law)))


def _winner(records, key, plan_class):
    eligible = [r for r in records if r['classes'][plan_class]]
    best = max(r[key] for r in eligible)
    ties = [r for r in eligible if best - r[key] <= TIE_TOLERANCE]
    # Cost is a total worst-case reservation over all four events.
    winner = min(ties, key=lambda r: (r['budget_bound'], tuple(r['plan'])))
    return {'plan': winner['plan'], 'score': winner[key], 'maximum_score': best,
            'within_score_tolerance': best - winner[key], 'tied_plans': len(ties),
            'class_count': len(eligible), 'budget_bound': winner['budget_bound'],
            'expected_payment': winner['expected_payment']}


def plan_all():
    """JSON-friendly planning receipt. This function never constructs Y/truth."""
    records = []
    for plan in enumerate_plans():
        row = {'plan': list(plan), 'classes': classify(plan),
               'budget_bound': budget_bound(plan), 'expected_payment': expected_payment(plan),
               'mean_score': design_score(plan)}
        for law in ('local', 'common'):
            patterns = participation_patterns(plan, law)
            row[law + '_score'] = float(sum(prob * design_score(plan, z) for prob, z in patterns))
            row[law + '_pattern_count'] = len(patterns)
        records.append(row)
    selections = {objective: {cls: _winner(records, key, cls)
                             for cls in ('single', 'sync', 'full')}
                  for objective, key in (('local', 'local_score'), ('common', 'common_score'),
                                         ('mean', 'mean_score'))}
    for objective, classes in selections.items():
        for result in classes.values():
            result['participation_patterns'] = {
                law: [{'probability': prob, 'participation': z.tolist()}
                      for prob, z in participation_patterns(result['plan'], law)]
                for law in ('local', 'common')}
    return {'plans': records, 'selected': selections,
            'counts': {cls: sum(r['classes'][cls] for r in records)
                       for cls in ('single', 'sync', 'full')},
            'objective': 'expected ten-column historical-RMS logdet under declared law',
            'mean_ablation': 'logdet at expected response; identical across dependence controls',
            'tie_rule': 'within 1e-6 of maximum, lowest total worst-case payment, then lexicographic plan',
            'total_budget': TOTAL_BUDGET, 'ridge': RIDGE,
            'truth_or_future_Y_used': False,
            'scope': 'Ordinary stochastic input design baseline, not a new WZZT price rule',
            'node_symmetry': 'Forecast/history are node-permutation symmetric; tie location is not learned topology',
            'behavioral_calibration': 'synthetic declared probabilities, not empirical calibration'}


def truth_matrices(case='normal'):
    if case not in CASE_B_R:
        raise ValueError('Unknown frozen physical case')
    supports = [(i,) for i in range(5)] + [(0, 1, 2, 3), (0, 1), (2, 3)]
    r = [0.08] * 5 + [0.10, 0.06, CASE_B_R[case]]
    x = [0.5 * value for value in r]
    matrices = []
    for weights in (r, x):
        matrix = np.zeros((5, 5))
        for support, weight in zip(supports, weights):
            matrix[np.ix_(support, support)] += weight
        matrices.append(matrix)
    truth = {'case': case, 'support_labels': [[LABELS[i] for i in s] for s in supports],
             'r': r, 'x': x, 'R': matrices[0].tolist(), 'X': matrices[1].tolist(),
             'effective_support_labels': [[LABELS[i] for i in s]
                                         for s, ri, xi in zip(supports, r, x) if max(ri, xi) > 1e-9]}
    truth['supports'] = truth['effective_support_labels']
    truth['parameters'] = {'configured_supports': truth['support_labels'], 'r': r, 'x': x}
    return matrices[0], matrices[1], truth


def make_data(plan, participation, case='normal'):
    """Evaluator-only observations and truth, with four independent event slots.

    The caller separates truth before sending observations to the identifier.
    Returned arrays have no measurement noise and are not an AC simulation.
    """
    p = _plan(plan)
    prices = np.vstack((p, p))
    z = np.asarray(participation)
    if z.shape != (4, 5) or not np.isin(z, (0, 1)).all() or np.any(z[prices == 0] != 0):
        raise ValueError('Evaluator participation must be binary 4 by 5 and zero at no-offer nodes')
    r, x, truth = truth_matrices(case)

    def day(date, powers):
        pu = powers / BASE_KW
        qu = POWER_FACTOR_Q_OVER_P * pu
        y = pu @ r.T + qu @ x.T
        root = np.ones(len(powers))
        if np.any(1 - y <= 0):
            raise ValueError('Linear synthetic voltage squared is nonpositive')
        voltage = np.sqrt(1 - y)
        return {'date': date, 'p_kw': powers, 'q_kvar': POWER_FACTOR_Q_OVER_P * powers,
                'p_pu': pu, 'q_pu': qu, 'v_pu': voltage, 'root_pu': root, 'drop_pu2': y}

    history = [day(date, np.repeat((m * SHAPE)[:, None], 5, axis=1))
               for date, m in zip(HISTORY_DATES, HISTORY_MULTIPLIERS)]
    events = []; energy = []; event_payments = []
    for date, m, zk, price in zip(EVENT_DATES, ACTUAL_MULTIPLIERS, z, prices):
        baseline = np.repeat((m * SHAPE)[:, None], 5, axis=1)
        delta = RESPONSE_KW * RESPONSE_SHAPE[:, None] * zk[None, :]
        events.append(day(date, baseline + delta))
        energy.append(delta.sum(axis=0).tolist())
        event_payments.append(float(RESPONSE_KW * np.sum(price * zk)))
    all_power = np.concatenate([d['p_kw'] for d in history + events])
    all_voltage = np.concatenate([d['v_pu'] for d in history + events])
    audit = {'payment': sum(event_payments), 'event_payments': event_payments,
             'budget_bound': budget_bound(plan), 'budget_limit': TOTAL_BUDGET,
             'budget_pass': sum(event_payments) <= TOTAL_BUDGET + 1e-12,
             'per_event_net_energy_kwh': energy,
             'energy_pass': bool(np.max(np.abs(energy)) <= 1e-12),
             'power_min_kw': float(all_power.min()), 'power_max_kw': float(all_power.max()),
             'power_pass': bool(all_power.min() >= 0 and all_power.max() <= 1.5),
             'voltage_min_pu': float(all_voltage.min()), 'voltage_max_pu': float(all_voltage.max()),
             'voltage_screen_pass': bool(all_voltage.min() >= 0.9 and all_voltage.max() <= 1.1),
             'linear_reconstruction_max_error': max(float(np.max(abs(d['root_pu'][:, None] ** 2 - d['v_pu'] ** 2 - d['drop_pu2']))) for d in history + events),
             'AC_run': False, 'scope': 'synthetic linear response mechanism, no network loss/thermal validation'}
    audit.update(payment_currency=audit['payment'], upper_payment_currency=audit['budget_bound'],
                 max_energy_residual=float(np.max(np.abs(energy))),
                 valid=all(audit[k] for k in ('budget_pass', 'energy_pass', 'power_pass', 'voltage_screen_pass')))
    return {'training_days': history + events[:2], 'validation_days': events[2:],
            'truth': truth, 'physical_audit': audit}


def self_check():
    """Small pure-numerical invariants; no optimization solver or identifier."""
    plans = enumerate_plans()
    counts = {c: sum(classify(p)[c] for p in plans) for c in ('single', 'sync', 'full')}
    assert counts == {'single': 41, 'sync': 21, 'full': 891}, counts
    probability_error = marginal_error = permutation_error = objective_error = 0.0
    samples = [plans[0], (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
               (0, 0, 1, 1, 0, 0, 0, 1, 1, 0), (0, 0, 0, 1, 0, 0, 0, 1, 0, 2)]
    for p in samples:
        for law in ('local', 'common'):
            patterns = participation_patterns(p, law)
            probability_error = max(probability_error, abs(sum(w for w, _ in patterns) - 1))
            marginal = sum(w * z for w, z in patterns)
            marginal_error = max(marginal_error, float(np.max(abs(marginal - PRICE_PROBABILITY[_plan(p)]))))
            permuted = tuple(_plan(p)[:, [4, 2, 0, 3, 1]].ravel())
            permutation_error = max(permutation_error, abs(stochastic_score(p, law) - stochastic_score(permuted, law)))
            for _, z in patterns:
                data = make_data(p, np.vstack((z, z)))
                audit = data['physical_audit']
                assert all(audit[k] for k in ('budget_pass', 'energy_pass', 'power_pass', 'voltage_screen_pass'))
        hp, hq = history_arrays(); fp, fq = forecast_arrays(p)
        h = np.column_stack((hp, hq)); scales = np.sqrt(np.mean(h * h, axis=0))
        combined = np.vstack((h, np.column_stack((fp, fq)))) / scales
        direct = np.linalg.slogdet(combined.T @ combined + RIDGE * np.eye(10))[1]
        objective_error = max(objective_error, abs(design_score(p) - direct))
    assert probability_error < 1e-12
    assert marginal_error < 1e-12
    # The 10D information matrix retains exact P/Q collinearity and a small ridge.
    assert permutation_error < TIE_TOLERANCE, permutation_error
    assert objective_error < TIE_TOLERANCE, objective_error
    assert all(budget_bound(p) <= TOTAL_BUDGET + 1e-12 for p in plans)
    return {'counts': counts, 'probability_error': probability_error, 'marginal_error': marginal_error,
            'node_permutation_score_error': permutation_error, 'ten_column_score_error': objective_error,
            'checked_plans': len(samples), 'all_passed': True, 'solver_calls': 0, 'identifier_calls': 0}


if __name__ == '__main__':
    import json
    print(json.dumps(self_check(), indent=2))
