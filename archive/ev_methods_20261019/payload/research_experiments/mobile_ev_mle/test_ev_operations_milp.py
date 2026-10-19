"""Independent physical and tiny exhaustive audits of the operations MILP."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import linprog

try:
    from .ev_operations_milp import (
        EVConfig, LinearMILP, check_plan, decode_plan, operation_model,
    )
except ImportError:
    from ev_operations_milp import (
        EVConfig, LinearMILP, check_plan, decode_plan, operation_model,
    )


def solve_checked(config, objective=None):
    model, indices = operation_model(config)
    result = model.solve(objective(indices) if objective else None)
    assert result.success, result.message
    plan = decode_plan(result, indices)
    report = check_plan(plan, config)
    assert report["feasible"], report
    return model, indices, result, plan, report


def test_default_charge_only_energy_and_solver_bound():
    cfg = EVConfig()
    model, idx, result, plan, audit = solve_checked(cfg)
    assert idx.u_index.shape == (8, 3)
    assert len(idx.arcs) <= cfg.n_slots * cfg.n_sites ** 2
    assert audit["grid_energy_kwh"] == pytest.approx(6)
    assert audit["energy_kwh"][-1] == pytest.approx(
        24 + .95 * 6 - audit["travel_energy_kwh"])
    assert np.max(plan.p_minus_kw) == pytest.approx(0)
    assert result.objective_value == pytest.approx(result.objective_bound)
    assert result.mip_gap <= 1e-8
    assert audit["sites"][-1] >= 0


def test_objective_pushes_site_two_with_closed_early_connection():
    access = np.ones((8, 3), bool)
    access[:2, 2] = False
    cfg = EVConfig(access=access)
    _, idx, result, plan, audit = solve_checked(
        cfg, lambda ix: {int(ix.u_index[t, 2]): 1 for t in range(8)})
    assert result.objective_value == pytest.approx(24)  # kW-sum, 6kWh at site2
    assert np.sum(plan.u_kw[:, :2]) == pytest.approx(0)
    assert np.max(plan.u_kw[:2, 2]) == pytest.approx(0)
    assert audit["travel_energy_kwh"] >= .2 - 1e-8


def test_multi_slot_trip_and_energy_charged_once_per_trip():
    cfg = EVConfig(n_slots=4, n_sites=2, net_grid_energy_kwh=1.5,
                   travel_slots=2, terminal_site=1)
    _, _, result, plan, audit = solve_checked(
        cfg, lambda ix: {int(ix.u_index[t, 1]): 1 for t in range(4)})
    trips = [a for a in plan.selected_arcs if a.is_travel]
    assert len(trips) == 1
    assert trips[0].arrival - trips[0].departure == 2
    assert audit["travel_energy_kwh"] == pytest.approx(.2)
    for t in range(trips[0].departure, trips[0].arrival):
        assert np.sum(abs(plan.u_kw[t])) == pytest.approx(0)
        assert audit["sites"][t] == -1
    assert audit["energy_kwh"][-1] == pytest.approx(24 + .95 * 1.5 - .2)


def test_arrival_soc_before_charging_cannot_borrow_later_energy():
    cfg = EVConfig(n_slots=2, n_sites=2, initial_soc=.1,
                   net_grid_energy_kwh=1.5, terminal_site=1)
    model, _ = operation_model(cfg)
    result = model.solve()
    assert result.status == 2  # travelling first would take energy below lower SOC
    assert result.x is None


def test_inaccessible_all_slots_preserves_infeasibility():
    cfg = EVConfig(access=np.zeros((8, 3), bool))
    model, _ = operation_model(cfg)
    result = model.solve()
    assert result.status == 2


def test_impossible_move_duration_preserves_infeasibility():
    cfg = EVConfig(n_slots=2, n_sites=2, travel_slots=3, terminal_site=1,
                   net_grid_energy_kwh=0)
    model, _ = operation_model(cfg)
    assert model.solve().status == 2


def test_bidirectional_power_throughput_and_efficiency():
    cfg = EVConfig(n_slots=2, n_sites=1, net_grid_energy_kwh=0,
                   allow_discharge=True, max_throughput_kwh=3)
    _, idx, result, plan, report = solve_checked(
        cfg, lambda ix: {int(ix.u_index[0, 0]): 1})
    assert plan.u_kw[:, 0] == pytest.approx([6, -6])
    assert report["throughput_kwh"] == pytest.approx(3)
    assert report["energy_kwh"][-1] == pytest.approx(24 + 1.5 * .95 - 1.5 / .95)
    assert np.max(np.minimum(plan.p_plus_kw, plan.p_minus_kw)) == pytest.approx(0)


def test_independent_checker_rejects_soc_power_and_route_mutations():
    cfg = EVConfig()
    _, _, _, plan, _ = solve_checked(cfg)
    broken = replace(plan, energy_kwh=plan.energy_kwh + 1)
    assert "energy_balance" in check_plan(broken, cfg)["issues"]
    broken = replace(plan, selected_arcs=plan.selected_arcs[:-1])
    assert not check_plan(broken, cfg)["feasible"]
    bad_power = plan.p_plus_kw.copy()
    bad_power[0] = [6, 6, 6]
    broken = replace(plan, p_plus_kw=bad_power)
    assert "charging_access_or_location" in check_plan(broken, cfg)["issues"]


def test_append_variable_rows_clone_and_objective_direction():
    model = LinearMILP()
    x = model.add_variables("x", 2, 0, 1, integer=True)
    model.add_row({int(x[0]): 1, int(x[1]): 1}, upper=1)
    copy = model.clone()
    z = copy.add_variables("z", (), -2, 2)
    copy.add_row({int(z): 1, int(x[0]): -2}, 0, 0)
    result = copy.solve({int(z): 1})
    assert result.objective_value == pytest.approx(2)
    assert result.objective_bound == pytest.approx(2)
    assert model.n_variables == 2
    result = copy.solve({int(z): 1}, cmax=False)
    assert result.objective_value == pytest.approx(0)
    assert result.objective_bound == pytest.approx(0)
    with pytest.raises(ValueError):
        model.add_variables("x")


def test_final_connection_is_a_declared_feasible_domain():
    cfg = EVConfig(n_slots=1, n_sites=2, net_grid_energy_kwh=0,
                   terminal_site=1, require_terminal_connection=True)
    model, _ = operation_model(cfg)
    assert model.solve().status == 2
    _, _, _, plan, report = solve_checked(replace(cfg, require_terminal_connection=False))
    assert report["sites"] == [-1]
    assert plan.selected_arcs[0].arrival == 1
    assert report["travel_energy_kwh"] == pytest.approx(.2)


def test_zero_time_moves_rejected_and_forbidden_moves_absent():
    with pytest.raises(ValueError, match="at least one"):
        operation_model(EVConfig(travel_slots=0))
    allowed = np.zeros((2, 2), bool)
    cfg = EVConfig(n_slots=2, n_sites=2, net_grid_energy_kwh=0,
                   travel_allowed=allowed)
    model, idx = operation_model(cfg)
    assert not any(a.is_travel for a in idx.arcs)
    assert model.solve().success


def test_tiny_exhaustive_route_lp_reference_matches_milp():
    # Independent route enumeration is confined to this tiny audit. Each
    # route then has a separate continuous LP, not an enumerated power grid.
    cfg = EVConfig(n_slots=3, n_sites=2, p_max_kw=4,
                   net_grid_energy_kwh=1.5)
    weights = np.array([[.5, 3], [1, 4], [2, .1]])
    _, idx, result, _, _ = solve_checked(
        cfg, lambda ix: {int(ix.u_index[t, i]): weights[t, i]
                         for t in range(3) for i in range(2)})
    routes = []

    def extend(t, site, connected):
        if t == 3:
            if connected[-1] >= 0:
                routes.append(connected)
            return
        extend(t + 1, site, connected + [site])
        extend(t + 1, 1 - site, connected + [-1])

    extend(0, 0, [])
    values = []
    for route in routes:
        bounds = [(0, 4 if route[t] == i else 0)
                  for t in range(3) for i in range(2)]
        reference = linprog(-weights.ravel(), A_eq=np.full((1, 6), .25),
                            b_eq=[1.5], bounds=bounds, method="highs")
        if reference.success:
            values.append(-reference.fun)
    assert result.objective_value == pytest.approx(max(values))
    assert result.objective_bound == pytest.approx(max(values))


@pytest.mark.parametrize("kwargs", [
    {"eta_charge": 0}, {"initial_soc": .95}, {"n_slots": 0},
    {"access": np.ones((2, 3), bool)}, {"p_max_kw": -1},
    {"travel_energy_kwh": -.2}, {"net_grid_energy_kwh": -1},
])
def test_invalid_config_is_not_silently_relaxed(kwargs):
    with pytest.raises(ValueError):
        operation_model(EVConfig(**kwargs))
