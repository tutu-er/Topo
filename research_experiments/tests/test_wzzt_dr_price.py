"""Behavioral contracts for the research price/DR stage; no raw data needed."""
import inspect
import pathlib
import sys

import numpy as np
import pytest

from research_experiments.wzzt_dr_selection.end_to_end.price_response import (MENUS, MenuModel, draw_per_menu_banks, plan_prices,
                            predefined_candidates, realize_response, settlement)


@pytest.fixture
def model():
    factors = {}
    prices = {}
    for i, menu in enumerate(MENUS):
        factors[menu] = np.ones(24)
        factors[menu][16:18] = 1 - 0.01 * (i + 1)
        prices[menu] = np.ones(24)
        prices[menu][16:18] = [2, 5, 10, 15][i]
    return MenuModel(factors, prices)


def test_candidate_count_and_no_topology_inputs():
    assert len(predefined_candidates(15)) == 65
    args = inspect.signature(plan_prices).parameters
    assert not set(args).intersection({'true_topology', 'true_response', 'actual_p_kw', 'future_voltage'})


def test_no_reward_is_exact_realized_identity(model):
    p = np.arange(48, dtype=float).reshape(24, 2) / 10
    q = 0.3 * p
    result = realize_response(p, q, {'menus': ['no_reward'] * 2}, model, {}, settlement_baseline_kw=p)
    assert np.array_equal(result['p_kw'], p)
    assert np.array_equal(result['q_kvar'], q)
    assert result['total_payment_nok'] == 0
    assert result['audit']['clipped_samples'] == 0


def test_signed_credits_are_accumulated_before_flooring():
    baseline = np.ones((24, 1))
    actual = baseline.copy()
    actual[0] = 0
    actual[1] = 2
    credits, payment, bound = settlement(np.ones_like(baseline), baseline, actual)
    assert credits[0] == 0 and payment[0] == 0
    assert bound == 24
    assert np.maximum(baseline - actual, 0).sum() == 1  # Incorrect hourly-floor alternative differs.


def test_conservative_budget_uses_entire_price_vector(model):
    p = np.ones((24, 1))
    candidates = [
        {'name': 'none', 'menus': ['no_reward']},
        {'name': 'p2', 'menus': ['P_2']},
    ]
    plan = plan_prices(p, p * 0.3, model, objective='peak', budget_nok=25,
                       candidates=candidates)
    paid = plan['candidates'][1]
    assert paid['conservative_payment_bound_nok'] == 26  # Includes 22 non-event hours.
    assert not paid['feasible']
    assert paid['operating']['valid']
    assert not paid['budget_feasible']  # A fixed candidate receives the same audit.
    assert plan['selected_assignment']['name'] == 'none'
    assert plan['forecast_baseline_operating']['valid']
    assert 'provisional' in plan['mean_response_status']


def test_d_opt_score_uses_fixed_history_scales(model):
    hp = np.arange(1, 13, dtype=float).reshape(6, 2)
    hq = hp[:, ::-1] * 0.2
    p = np.tile([4.0, 6.0], (24, 1))
    q = p * 0.3
    ridge = 1e-4
    plan = plan_prices(p, q, model, history_p_kw=hp, history_q_kvar=hq,
                       budget_nok=1e5, ridge=ridge)
    h = np.column_stack((hp, hq))
    scale = np.sqrt(np.mean(h * h, axis=0))
    assert np.allclose(plan['history_column_scales'], scale)
    for audit in plan['candidates']:
        factor = np.column_stack([np.ones(24) if m == 'no_reward' else model.mean_factors[m]
                                  for m in audit['menus']])
        full = np.concatenate([h / scale, np.column_stack((p * factor, q * factor)) / scale])
        expected = np.linalg.slogdet(full.T @ full + ridge * np.eye(4))[1]
        assert audit['score'] == pytest.approx(expected, abs=1e-8)
    feasible = [a['score'] for a in plan['candidates'] if a['feasible']]
    assert plan['selected_score'] == max(feasible)


def test_draw_preserves_one_whole_joint_trajectory():
    banks = {m: np.arange(3 * 24 * 2, dtype=float).reshape(3, 24, 2) / 100 + 0.5 for m in MENUS}
    factors, meta = draw_per_menu_banks(banks, seed=20261027, n_nodes=2)
    factors2, meta2 = draw_per_menu_banks(banks, seed=20261027, n_nodes=2)
    assert meta == meta2
    for m in MENUS:
        ix = meta['draws'][m]['index']
        assert np.array_equal(factors[m], banks[m][ix])
        assert np.array_equal(factors[m], factors2[m])


def test_realized_factors_are_not_replaced_by_planning_means(model):
    p = np.ones((24, 2))
    observed = np.tile([1.2, 0.9], (24, 1))
    result = realize_response(p, p * 0.4, {'menus': ['P_2', 'no_reward']}, model,
                              {'P_2': observed}, settlement_baseline_kw=p,
                              power_limits_kw=1.1, max_daily_energy_change_fraction=0.1)
    assert np.array_equal(result['p_kw'][:, 0], np.full(24, 1.2))
    assert np.array_equal(result['p_kw'][:, 1], p[:, 1])
    assert not result['audit']['valid']
    assert {x['kind'] for x in result['audit']['violations']} == {'power', 'daily_energy_change'}
    assert result['audit']['clipped_samples'] == 0


def test_negative_net_generation_is_not_scaled_as_household_response(model):
    p = np.ones((24, 1))
    p[0] = -0.1
    with pytest.raises(ValueError, match='nonnegative'):
        plan_prices(p, p * 0.3, model, objective='peak', budget_nok=100)


def test_actual_payment_obeys_conservative_bound(model):
    forecast = np.full((24, 2), 3.0)
    actual = np.full((24, 2), 2.0)
    effects = {'P_2': np.full((24, 2), 0.5)}
    response = realize_response(actual, actual * 0.3, {'menus': ['P_2'] * 2}, model,
                                effects, settlement_baseline_kw=forecast, budget_nok=156)
    assert response['audit']['budget_valid']
    assert response['total_payment_nok'] <= response['audit']['payment_bound_nok'] == 156


def test_actual_budget_violation_is_retained_with_unchanged_response(model):
    baseline = np.ones((24, 1))
    response = realize_response(baseline, baseline * 0.3, {'menus': ['P_2']}, model,
                                {'P_2': np.full((24, 1), 0.5)},
                                settlement_baseline_kw=baseline, budget_nok=1.0)
    assert np.array_equal(response['p_kw'], baseline * 0.5)
    assert response['total_payment_nok'] == 13.0
    assert not response['audit']['valid']
    assert not response['audit']['budget_valid']
    assert response['audit']['violations'][0]['kind'] == 'settlement_budget'
    assert response['audit']['clipped_samples'] == 0
