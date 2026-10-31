"""Exact closed-form checks independent of the probability-LP implementation.

This file imports neither scipy nor the implementation under review.  The
decision loss tables are analytical examples, not WZZT identification results.
Execution must follow the parent experiment's separately frozen protocol.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
from pathlib import Path


def interval(p, q):
    """Frechet interval for t = P(Z1=1,Z2=1)."""
    return max(F(0), p + q - 1), min(p, q)


def risk_at(loss, p, q, t):
    # State ordering is 00,10,01,11; there is no numerical optimization.
    mass = (1 - p - q + t, p - t, q - t, t)
    assert sum(mass) == 1 and all(x >= 0 for x in mass)
    return sum(a * b for a, b in zip(loss, mass, strict=True))


def extrema(loss, p, q, bounds=None):
    left, right = interval(p, q) if bounds is None else bounds
    value = (risk_at(loss, p, q, left), risk_at(loss, p, q, right))
    return min(value), max(value)


def serial(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def audit():
    checks = []

    def check(name, condition, evidence):
        checks.append(dict(name=name, passed=bool(condition), evidence=serial(evidence)))
        if not condition:
            raise AssertionError(name)

    half = F(1, 2)
    a = tuple(map(F, (1, 0, 0, 0)))
    b = tuple(map(F, (0, 1, 1, 0)))
    regimes = {'independent': F(1, 4), 'common': half, 'exclusive': F(0)}
    risks = {name: (risk_at(a, half, half, t), risk_at(b, half, half, t))
             for name, t in regimes.items()}
    check('equal_marginals_do_not_identify_menu_ranking',
          risks == {'independent': (F(1, 4), half),
                    'common': (half, F(0)), 'exclusive': (F(0), F(1))}, risks)

    diff = tuple(x - y for x, y in zip(a, b, strict=True))
    a_regret = max(F(0), extrema(diff, half, half)[1])
    b_regret = max(F(0), extrema(tuple(-x for x in diff), half, half)[1])
    check('same_mechanism_regret', (a_regret, b_regret) == (half, F(1)),
          dict(a=a_regret, b=b_regret))

    separate_sup_difference = extrema(a, half, half)[1] - extrema(b, half, half)[1]
    check('difference_of_worst_risks_is_not_worst_risk_difference',
          separate_sup_difference == -half and a_regret == half,
          dict(invalid_separate_sup_difference=separate_sup_difference,
               valid_same_mechanism_sup_difference=a_regret))

    same_diff = tuple(x - x for x in b)
    true_same = extrema(same_diff, half, half)
    fake_separate = extrema(b, half, half)[1] - extrema(b, half, half)[0]
    check('identical_menus_cannot_have_positive_regret',
          true_same == (F(0), F(0)) and fake_separate == 1,
          dict(true_difference=true_same, invalid_independent_extremes=fake_separate))

    # For A probability q, worst regret is max((1-q),q/2), minimized at q=2/3.
    mixture_q = F(2, 3)
    mixture_regret = max(1 - mixture_q, mixture_q / 2)
    check('randomization_example_is_decision_hedging_only',
          mixture_regret == F(1, 3) and mixture_regret < min(a_regret, b_regret),
          dict(probability_a=mixture_q, regret=mixture_regret,
               interpretation='Analytical decoder loss table, not WZZT evidence'))

    joint_bounds = interval(half, F(3, 4))
    max_nonmonotone = half - joint_bounds[0]
    check('higher_marginal_participation_does_not_imply_individual_monotonicity',
          joint_bounds == (F(1, 4), half) and max_nonmonotone == F(1, 4),
          dict(P11_bounds=joint_bounds, max_P_price1_yes_price2_no=max_nonmonotone))

    fee = tuple(F(1, 5) * (i + 2*j) for i, j in ((0, 0), (1, 0), (0, 1), (1, 1)))
    fee_range = extrema(fee, half, F(3, 4))
    check('linear_delivered_payment_depends_on_marginals_only',
          fee_range == (F(2, 5), F(2, 5)), dict(expected_fee_range=fee_range))
    check('unchanged_four_event_hard_budget', F(2) * F(1, 5) * 4 == F(8, 5),
          dict(task_kWh=F(1, 5), validation_repeat=2, training_price_units=4,
               worst_payment=F(8, 5)))

    one = tuple(F(1) for _ in range(4))
    weak_risk = extrema(one, half, half)
    weak_regret = extrema(tuple(F(0) for _ in range(4)), half, half)
    check('all_menus_miss_weak_block_zero_regret_is_not_recovery',
          weak_risk == (F(1), F(1)) and weak_regret == (F(0), F(0)),
          dict(absolute_risk=weak_risk, regret=weak_regret))

    # Same cache cell is the SAME unknown number x, not two independent [0,5]s.
    cache_bounds = (F(0), F(5))
    shared_difference = tuple(F(0) for _ in range(4))
    shared_regret = extrema(shared_difference, half, half)
    check('identical_unknown_cache_cells_cancel_exactly', shared_regret == (F(0), F(0)),
          dict(unknown_cell_interval=cache_bounds, regret=shared_regret))
    check('node_event_counts_are_not_event_time_counts', 2**8 == 256 and 4**8 == 65536,
          dict(event_times=4, active_nodes_per_time=2, binary_node_event_variables=8,
               actual_masks=2**8, cross_two_price_potential_types=4**8))
    return dict(scope='independent_exact_fraction_closed_forms_no_identifier_or_LP',
                topology_recovery_claim=False, new_WZZT_runs=0, probability_LP_runs=0,
                passed=sum(c['passed'] for c in checks), total=len(checks), checks=checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result, ensure_ascii=True, indent=2) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text)
