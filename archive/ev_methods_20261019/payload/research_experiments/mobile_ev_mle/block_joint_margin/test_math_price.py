"""Analytic and physical-domain checks for the local profile-margin prices.

These are formula/solver checks, not new scientific evaluation histories.
"""
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import lsq_linear

from ..ev_mobile_service_domain import MobileServiceConfig, build_mobile_service_domain
from .price_milp import MarginPlane, solve_local_margin_milp


def small_config(**overrides):
    settings = dict(
        n_slots=6, n_sites=2, dt_hours=1., p_charge_max_kw=1.,
        p_discharge_max_kw=1., allow_discharge=False,
        battery_capacity_kwh=10., initial_soc=.5, soc_min=0., soc_max=1.,
        terminal_energy_kwh=7., eta_charge=1., eta_discharge=1.,
        travel_energy_kwh=0., setup_slots=1, mobile=True,
        max_moves=1, min_arrival_throughput_kwh=.5)
    settings.update(overrides)
    return MobileServiceConfig(**settings)


def parent_constant(config):
    u = np.zeros((config.n_slots, config.n_sites))
    u[:, config.initial_site] = (
        config.terminal_energy_kwh
        - config.initial_soc * config.battery_capacity_kwh
    ) / (config.n_slots * config.dt_hours * config.eta_charge)
    return u


def checked_solve(domain, u0, **kwargs):
    answer = solve_local_margin_milp(domain, u0, time_limit=10.,
                                      mip_rel_gap=1e-9, **kwargs)
    assert answer["success"], answer
    assert answer["surrogate_feasibility"]["feasible"]
    assert domain.check(answer["x"])["feasible"]
    assert answer["x"].shape == (domain.model.n_variables,)
    assert answer["x_milp"].shape[0] > answer["x"].shape[0]
    assert answer["surrogate_best_bound"] >= answer["surrogate_incumbent"] - 1e-7
    assert answer["objective_recomputed"] == pytest.approx(
        answer["surrogate_incumbent"], abs=1e-7)
    assert "not_AC_or_probability" in answer["bound_scope"]
    return answer


def test_rx_compensation_can_cancel_a_nonzero_fixed_center_signal():
    # Historical P=Q=1 determines only R+X=2.  A unit-P EV changes the
    # voltage-drop target to 2+u.  R=X=1 fits both exactly for every u;
    # the equally historical-valid center (.5,1.5) gives a false .5*u gap.
    for power in (-.75, -.2, .5, 1.25):
        matrix = np.array([[1., 1.], [1. + power, 1.]])
        target = np.array([2., 2. + power])
        solution = lsq_linear(matrix, target, bounds=(.25, 1.75), tol=1e-14)
        assert np.linalg.norm(matrix @ solution.x - target) < 1e-10
        np.testing.assert_allclose(matrix @ np.ones(2), target)
        fixed_future_gap = target[-1] - matrix[-1] @ np.array([.5, 1.5])
        assert fixed_future_gap == pytest.approx(.5 * power)


def test_competitor_added_distance_and_two_sided_margin_rank_oppositely():
    # A: min_theta theta^2 + (u-theta*u)^2 = u^2/(1+u^2).
    # B: min_beta (beta+1)^2/9 + (u-beta)^2 = (u+1)^2/10.
    # The reference A explanation theta=1 generates y(u)=u.  Both
    # historical minima are zero; its own refitting cost cannot be dropped.
    own = lambda u: u * u / (1 + u * u)
    competitor = lambda u: (u + 1) ** 2 / 10
    assert competitor(1) > competitor(0)
    assert competitor(1) - own(1) < competitor(0) - own(0)
    assert competitor(0) - own(0) == pytest.approx(.1)
    assert competitor(1) - own(1) == pytest.approx(-.1)


@pytest.mark.parametrize("power", [-2., -.4, .2, .9, 2.])
def test_envelope_price_matches_finite_difference_with_rx_action_product(power):
    # Box activity is allowed; selected values avoid discontinuous minimizers.
    def fit_a(u):
        theta = np.clip(u * u / (1 + u * u), 0., .7)
        return theta * theta + (u - theta * u) ** 2, theta

    def fit_b(u):
        beta = np.clip((9 * u - 1) / 10, -.2, .8)
        return (beta + 1) ** 2 / 9 + (u - beta) ** 2, beta

    a_cost, theta = fit_a(power)
    b_cost, beta = fit_b(power)
    # Generator derivative is 1; the fitted A derivative is theta.  This
    # term is exactly what evaluating at a permanently frozen center misses.
    envelope = (power - beta) - (power - theta * power) * (1 - theta)
    step = 1e-5
    value = lambda u: (fit_b(u)[0] - fit_a(u)[0]) / 2
    finite_difference = (value(power + step) - value(power - step)) / (2 * step)
    assert np.isfinite([a_cost, b_cost]).all()
    assert envelope == pytest.approx(finite_difference, rel=1e-7, abs=1e-8)


def test_price_remains_first_order_valid_at_an_active_set_switch():
    # The B optimum reaches beta=.8 at u=1.  Its value is C1 but not C2;
    # the one-sided finite-difference errors are therefore O(step).
    def margin(u):
        theta = u * u / (1 + u * u)
        beta = min(.8, (9 * u - 1) / 10)
        return ((beta + 1) ** 2 / 9 + (u - beta) ** 2
                - theta * theta - (u - theta * u) ** 2) / 2

    target_price = -.05
    for step in (1e-3, 1e-4, 1e-5):
        left = (margin(1) - margin(1 - step)) / step
        right = (margin(1 + step) - margin(1)) / step
        assert abs(left - target_price) < step
        assert abs(right - target_price) < step


def test_zero_gradient_does_not_remove_a_real_quadratic_excitation_opportunity():
    quadratic = lambda u: 4 * u * u
    assert (quadratic(1e-5) - quadratic(-1e-5)) / 2e-5 == 0
    assert quadratic(1) > quadratic(0)
    cfg = small_config(mobile=False)
    domain = build_mobile_service_domain(cfg)
    answer = checked_solve(domain, parent_constant(cfg), margins=[[0.]],
                           gradients=np.zeros((1, 1, 6, 2)), tau=5., radius=1.)
    assert answer["all_gradient_zero"]
    assert answer["surrogate_incumbent"] == pytest.approx(0.)


def test_all_scenarios_constrain_pair_and_trust_region_is_respected():
    cfg = small_config(n_slots=2, n_sites=1, mobile=False,
                       terminal_energy_kwh=6.)
    domain = build_mobile_service_domain(cfg)
    u0 = np.array([[.5], [.5]])
    price = np.array([[1.], [-1.]])
    robust = checked_solve(
        domain, u0, margins=[[0.], [0.]],
        gradients=np.stack([price, -price])[:, None], tau=4., radius=.1)
    np.testing.assert_allclose(robust["u"], u0, atol=1e-7)
    single = checked_solve(domain, u0, margins=[[0.]],
                           gradients=price[None, None], tau=4., radius=.1)
    np.testing.assert_allclose(single["u"], [[.6], [.4]], atol=1e-7)
    assert single["surrogate_incumbent"] == pytest.approx(.2)
    assert np.max(abs(single["u"] - u0)) <= .1 + 1e-8


def test_sparse_reference_scenarios_keep_pair_cap_and_design_weights():
    cfg = small_config(n_slots=2, n_sites=1, mobile=False,
                       terminal_energy_kwh=6.)
    domain = build_mobile_service_domain(cfg)
    u0 = np.array([[.5], [.5]])
    price = np.array([[1.], [-1.]])
    answer = checked_solve(domain, u0, planes=[
        MarginPlane(("a", "b"), 0., price, "a-main"),
        dict(pair=("a", "b"), value=.2, gradient=price, scenario="a-flat"),
        (("b", "a"), 3., np.zeros((2, 1)), "b-main"),
    ], pair_weights={("a", "b"): 3., ("b", "a"): 1.},
        tau={("a", "b"): .1, ("b", "a"): 2.}, radius=.1)
    np.testing.assert_allclose(answer["pair_values"], [.1, 2.], atol=1e-7)
    assert answer["surrogate_incumbent"] == pytest.approx(.575)
    assert answer["n_planes"] == 3


def test_mobile_domain_keeps_movement_optional_and_can_reach_rewarding_site():
    cfg = small_config()
    u0 = parent_constant(cfg)
    remote = np.zeros((6, 2)); remote[:, 1] = 10.
    mobile = build_mobile_service_domain(cfg)
    before_variables = mobile.model.n_variables
    before_rows = len(mobile.model.rows)
    move = checked_solve(mobile, u0, planes=[("a->b", 0., remote)],
                          tau=100., radius=1.)
    fixed = checked_solve(build_mobile_service_domain(replace(cfg, mobile=False)),
                           u0, planes=[("a->b", 0., remote)], tau=100., radius=1.)
    assert move["surrogate_incumbent"] == pytest.approx(20.)
    assert fixed["surrogate_incumbent"] == pytest.approx(0.)
    assert move["feasibility"]["moves"] == 1
    assert any(a["kind"] == "setup" for a in move["feasibility"]["route"])
    local = np.zeros((6, 2)); local[:, 0] = 10.
    stay = checked_solve(mobile, u0, planes=[("a->b", 0., local)],
                          tau=100., radius=1.)
    assert stay["feasibility"]["moves"] == 0
    assert mobile.model.n_variables == before_variables
    assert len(mobile.model.rows) == before_rows


def test_real_ev_service_keeps_soc_loss_driver_access_and_bidir_constraints():
    access = np.ones((8, 3), bool); access[3:5, 0] = False
    worker = np.zeros((8, 3), bool); worker[3, 1] = True
    cfg = MobileServiceConfig(
        n_slots=8, n_sites=3, p_charge_max_kw=40 / 7,
        p_discharge_max_kw=40 / 7, terminal_energy_kwh=26.85,
        allow_discharge=True, mobile=True, access=access,
        worker_available=worker, setup_slots=1, travel_slots=1,
        travel_energy_kwh=.2, travel_cost=1., setup_cost=1.,
        driver_budget=2., max_moves=1, max_throughput_kwh=10.,
        min_arrival_throughput_kwh=.25)
    domain = build_mobile_service_domain(cfg)
    # Use a genuinely feasible parent despite the origin access outage.
    u0 = np.zeros((8, 3)); u0[[0, 1, 2, 5, 6, 7], 0] = 2.
    price = np.zeros((8, 3)); price[4:, 1] = [2., -1., 3., 1.]
    answer = checked_solve(domain, u0, planes=[("a->b", .3, price)],
                           tau=100., radius=40 / 7)
    physical = answer["feasibility"]
    assert physical["energy_kwh"][-1] == pytest.approx(26.85)
    assert physical["driver_cost"] <= 2. + 1e-7
    assert physical["throughput_kwh"] <= 10. + 1e-7
    assert np.max(abs(answer["u"][3:5, 0])) < 1e-7
    assert physical["moves"] <= 1
    assert len(physical["energy_kwh"]) == 9


def test_infeasible_operation_keeps_failure_instead_of_a_plan():
    cfg = small_config(n_slots=1, n_sites=1, mobile=False,
                       terminal_energy_kwh=7.)
    domain = build_mobile_service_domain(cfg)
    answer = solve_local_margin_milp(
        domain, [[0.]], margins=[[0.]], gradients=np.ones((1, 1, 1, 1)),
        radius=1., tau=4., time_limit=10.)
    assert not answer["success"]
    assert answer["x"] is None and answer["u"] is None
    assert answer["feasibility"]["issues"] == ["no_incumbent"]
    assert answer["surrogate_incumbent"] is None


@pytest.mark.parametrize("kwargs", [
    dict(margins=[[0.]], gradients=np.zeros((1, 1, 1, 1))),
    dict(margins=[[np.nan]], gradients=np.zeros((1, 1, 6, 2))),
    dict(planes=[]),
    dict(planes=[("a", 0., np.zeros((6, 2)))], radius=-1.),
    dict(planes=[("a", 0., np.zeros((6, 2)))], pair_weights=[0.]),
])
def test_invalid_input_rejected_before_solving(kwargs):
    domain = build_mobile_service_domain(small_config())
    with pytest.raises(ValueError):
        solve_local_margin_milp(domain, parent_constant(domain.config), **kwargs)
