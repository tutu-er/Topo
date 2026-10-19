"""Independent physical checks for the mobile same-battery-service MILP."""
from dataclasses import replace

import numpy as np
import pytest

from .ev_mobile_service_domain import MobileServiceConfig, build_mobile_service_domain


def small(**kw):
    args = dict(n_slots=6, n_sites=3, dt_hours=1., p_charge_max_kw=1.,
                p_discharge_max_kw=1., allow_discharge=False,
                battery_capacity_kwh=10., initial_soc=.5, soc_min=0., soc_max=1.,
                terminal_energy_kwh=7., eta_charge=1., eta_discharge=1.,
                travel_energy_kwh=0., setup_slots=1)
    args.update(kw)
    return MobileServiceConfig(**args)


def solve(config, prices=None):
    d = build_mobile_service_domain(config)
    if prices is None:
        prices = np.zeros((config.n_slots, config.n_sites))
    r = d.solve_prices(prices, time_limit=10, mip_rel_gap=1e-9)
    assert r.success, r.message
    a = d.check(r)
    assert a["feasible"], a
    assert r.objective_bound >= r.objective_value - 1e-7
    return d, r, a


def test_ev_and_ess_same_inputs_are_exact_same_domain():
    cfg = small(mobile=False)
    a = build_mobile_service_domain(cfg)
    b = build_mobile_service_domain(replace(cfg, device_kind="ess"))
    A, B = a.as_dict(), b.as_dict()
    for key in ("lower", "upper", "integrality", "control_indices", "u_map"):
        np.testing.assert_array_equal(A[key], B[key])
    np.testing.assert_array_equal(A["constraints"][0].A.toarray(),
                                  B["constraints"][0].A.toarray())
    np.testing.assert_array_equal(A["constraints"][0].lb, B["constraints"][0].lb)
    prices = np.arange(18).reshape(6, 3)
    _, r1, _ = solve(cfg, prices)
    _, r2, _ = solve(replace(cfg, device_kind="ess"), prices)
    assert r1.objective_value == pytest.approx(r2.objective_value)


def test_control_projection_order_zero_products_and_complete_x():
    cfg = small()
    prices = np.array([[0, 8, 0], [0, 7, 0], [0, 6, 0],
                       [0, 5, 0], [0, 4, 0], [0, 3, 0]])
    d, r, _ = solve(cfg, prices)
    op = d.as_dict()
    np.testing.assert_allclose(op["u_map"] @ r.x, d.decode(r)["u_kw"].ravel())
    np.testing.assert_allclose(r.x[op["control_indices"]], op["u_map"] @ r.x)
    assert np.any(op["integrality"])
    assert len(op["zero_product_pairs"]) == 6 * 3
    u = op["u_map"] @ r.x
    for j, k in op["zero_product_pairs"]:
        assert abs(u[j] * u[k]) < 1e-8
    assert "single_EV" in op["zero_product_proof"]


def test_mobile_reaches_rewarding_site_and_fixed_cannot():
    cfg = small(max_moves=1)
    prices = np.zeros((6, 3)); prices[:, 1] = 10
    d, r, a = solve(cfg, prices)
    _, fixed, fixed_a = solve(replace(cfg, mobile=False), prices)
    assert r.objective_value == pytest.approx(20)
    assert fixed.objective_value == pytest.approx(0)
    assert a["moves"] == 1 and fixed_a["moves"] == 0
    assert any(a.kind == "setup" for a in d.decode(r)["selected_arcs"])
    # Travel and setup slots supply no power; two sites never operate together.
    p = d.decode(r)
    for a in p["selected_arcs"]:
        if a.kind in ("travel", "setup"):
            assert np.max(abs(p["u_kw"][a.departure:a.arrival]), initial=0) < 1e-8


def test_worker_required_for_setup_not_whole_charge_period():
    worker = np.zeros((6, 3), bool); worker[2, 1] = True
    cfg = small(worker_available=worker, max_moves=1)
    prices = np.zeros((6, 3)); prices[:, 1] = 10
    d, _, a = solve(cfg, prices)
    setup = [x for x in a["route"] if x["kind"] == "setup"]
    assert len(setup) == 1 and setup[0]["departure"] == 2
    assert np.sum(np.asarray(a["u_kw"])[3:, 1]) == pytest.approx(2)
    assert np.max(abs(np.asarray(a["u_kw"])[:3, 1])) < 1e-8


def test_worker_absent_blocks_remote_connection_even_if_trip_allowed():
    worker = np.zeros((6, 3), bool)
    cfg = small(worker_available=worker, max_moves=1)
    prices = np.zeros((6, 3)); prices[:, 1] = 10
    _, r, _ = solve(cfg, prices)
    assert r.objective_value == pytest.approx(0)


def test_driver_budget_counts_trip_and_setup_separately():
    cfg = small(travel_cost=3., setup_cost=2., driver_budget=4., max_moves=1)
    prices = np.zeros((6, 3)); prices[:, 1] = 10
    _, tight, a = solve(cfg, prices)
    _, relaxed, b = solve(replace(cfg, driver_budget=5.), prices)
    assert tight.objective_value == pytest.approx(0)
    assert relaxed.objective_value == pytest.approx(20)
    assert a["driver_cost"] <= 4 and b["driver_cost"] == pytest.approx(5)


def test_v2g_equal_terminal_energy_requires_extra_grid_energy_for_losses():
    cfg = small(n_slots=2, n_sites=1, mobile=False, allow_discharge=True,
                dt_hours=1., terminal_energy_kwh=5., p_charge_max_kw=4.,
                p_discharge_max_kw=4., eta_charge=.8, eta_discharge=.8)
    d, r, a = solve(cfg, np.array([[-1.], [1.]]))
    u = d.decode(r)["u_kw"][:, 0]
    np.testing.assert_allclose(u, [-2.56, 4.], atol=1e-7)
    assert a["energy_kwh"][-1] == pytest.approx(5.)
    assert a["grid_energy_kwh"] == pytest.approx(1.44)
    assert a["conversion_loss_kwh"] == pytest.approx(1.44)
    _, charge_only, ca = solve(replace(cfg, allow_discharge=False),
                               np.array([[-1.], [1.]]))
    assert charge_only.objective_value == pytest.approx(0)
    assert ca["grid_energy_kwh"] == pytest.approx(0)


def test_driving_energy_deducted_once_at_departure_and_recharged_for_service():
    cfg = small(travel_slots=2, travel_energy_kwh=.5, max_moves=1,
                terminal_site=1, terminal_connected=True)
    d, _, a = solve(cfg, np.pad(np.ones((6, 1)), ((0, 0), (1, 1))))
    travel = [x for x in a["route"] if x["kind"] == "travel"]
    assert len(travel) == 1 and travel[0]["arrival"] - travel[0]["departure"] == 2
    t = travel[0]["departure"]
    assert a["energy_kwh"][t + 1] == pytest.approx(a["energy_kwh"][t] - .5)
    assert a["energy_kwh"][t + 2] == pytest.approx(a["energy_kwh"][t + 1])
    assert a["travel_energy_kwh"] == pytest.approx(.5)
    assert a["grid_energy_kwh"] == pytest.approx(2.5)
    assert a["energy_kwh"][-1] == pytest.approx(7.)


def test_soc_before_arrival_prevents_borrowing_later_charge():
    # A 1-kWh battery at E0=0.2 cannot pay a 0.3-kWh trip before charging.
    # Only the destination provides power: future charging cannot fund departure.
    access = np.ones((4, 2), bool); access[:, 0] = False
    cfg = small(n_slots=4, n_sites=2, battery_capacity_kwh=1., initial_soc=.2,
                terminal_energy_kwh=.5, travel_slots=2, travel_energy_kwh=.3,
                terminal_site=1, terminal_connected=True, p_charge_max_kw=1.,
                access=access)
    d = build_mobile_service_domain(cfg)
    r = d.solve_prices(np.zeros((4, 2)), time_limit=10)
    assert r.status == 2 and r.x is None
    assert d.check(r)["issues"] == ["no_incumbent"]


def test_zero_duration_setup_acyclic_and_access_still_required():
    cfg = small(n_slots=2, n_sites=1, mobile=False, initial_connected=False,
                setup_slots=0, terminal_energy_kwh=6.)
    d, _, a = solve(cfg)
    assert a["route"][0]["kind"] == "setup"
    assert a["route"][0]["arrival"] == a["route"][0]["departure"] == 0
    assert len(a["route"]) == 3
    no_worker = build_mobile_service_domain(replace(cfg, worker_available=False))
    r = no_worker.solve_prices(np.zeros((2, 1)), time_limit=10)
    assert r.status == 2


def test_charging_access_can_change_after_setup_and_limits_injection():
    access = np.ones((6, 3), bool); access[3:5, 0] = False
    cfg = small(mobile=False, access=access)
    prices = np.zeros((6, 3)); prices[3:5, 0] = 100
    d, r, _ = solve(cfg, prices)
    np.testing.assert_allclose(d.decode(r)["u_kw"][3:5, 0], 0)


def test_independent_replay_rejects_tampered_energy_and_extra_location_power():
    d, r, _ = solve(small(mobile=False))
    bad = r.x.copy(); bad[d.indices.energy[1]] += .5
    assert "energy_balance" in d.check(bad)["issues"]
    bad = r.x.copy(); bad[d.indices.charge[0, 1]] = .1
    bad[d.indices.u[0, 1]] = .1
    assert "charging_location_or_access" in d.check(bad)["issues"]


def test_throughput_budget_limits_cycle_without_changing_terminal_service():
    cfg = small(n_slots=2, n_sites=1, mobile=False, allow_discharge=True,
                terminal_energy_kwh=5., max_throughput_kwh=1.)
    d, r, a = solve(cfg, np.array([[-1.], [1.]]))
    np.testing.assert_allclose(d.decode(r)["u_kw"][:, 0], [-.5, .5], atol=1e-7)
    assert r.objective_value == pytest.approx(1.)
    assert a["throughput_kwh"] == pytest.approx(1.)
    assert a["energy_kwh"][-1] == pytest.approx(5.)


def test_fixed_device_cannot_satisfy_different_terminal_location():
    cfg = small(mobile=False, terminal_site=1)
    d = build_mobile_service_domain(cfg)
    r = d.solve_prices(np.zeros((6, 3)), time_limit=10)
    assert r.status == 2 and r.x is None


@pytest.mark.parametrize("change", [dict(travel_slots=0), dict(setup_slots=.5),
                         dict(eta_charge=1.1), dict(initial_soc=-.1),
                         dict(access=.5), dict(worker_available=np.zeros((2, 4))),
                         dict(max_moves=.5), dict(terminal_energy_kwh=20),
                         dict(p_charge_max_kw=-1), dict(driver_budget=-1)])
def test_invalid_units_durations_and_public_inputs_rejected(change):
    with pytest.raises(ValueError):
        build_mobile_service_domain(small(**change))
