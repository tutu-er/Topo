"""Graph-only audits: no AC calls and no optimization solves.

Tiny complete-path enumeration is an independent test oracle, never part of
the production control selector or bound-tightening algorithm.
"""
from dataclasses import replace

import numpy as np
import pytest

from .ev_mobile_service_domain import MobileServiceConfig, build_mobile_service_domain
from .ev_mobile_v3_domain import graph_reachability, tighten_operation_bounds


def config(**kwargs):
    values = dict(n_slots=4, n_sites=2, dt_hours=1., p_charge_max_kw=1.,
                  p_discharge_max_kw=1., battery_capacity_kwh=10., initial_soc=.5,
                  soc_min=0., soc_max=1., terminal_energy_kwh=5.,
                  eta_charge=1., eta_discharge=1., travel_energy_kwh=0.,
                  travel_slots=1, setup_slots=1, max_moves=1)
    values.update(kwargs)
    return MobileServiceConfig(**values)


def enumerate_graph(domain):
    """Direct tiny path walk with independent move/cost/end tests."""
    cfg, arcs = domain.config, domain.indices.arcs
    out = {}
    for i, a in enumerate(arcs):
        out.setdefault(a.start, []).append((i, a))
    selected = np.zeros(len(arcs), bool)
    count = 0

    def visit(node, path, moves, cost):
        nonlocal count
        if cfg.max_moves is not None and moves > cfg.max_moves:
            return
        if cfg.driver_budget is not None and cost > cfg.driver_budget + 1e-12:
            return
        if node[0] == cfg.n_slots:
            if cfg.terminal_site is not None and node[1] != cfg.terminal_site:
                return
            if cfg.terminal_connected and not node[2]:
                return
            count += 1
            selected[path] = True
            return
        for i, a in out.get(node, ()):
            visit(a.end, path + [i], moves + (a.kind == "travel"), cost + a.driver_cost)

    visit((0, cfg.initial_site, cfg.initial_connected), [], 0, 0.)
    positions = {int(v): i for i, v in enumerate(domain.indices.arc)}
    connected = np.array([[selected[positions[int(v)]] for v in row]
                          for row in domain.indices.connected_arc])
    return selected, connected, count


@pytest.mark.parametrize("cfg", [
    config(), config(mobile=False), config(initial_connected=False),
    config(setup_slots=0), config(initial_connected=False, setup_slots=0),
    config(terminal_site=0, terminal_connected=True),
    config(terminal_site=1, terminal_connected=True),
    config(worker_available=False), config(max_moves=0), config(max_moves=None),
    config(travel_cost=3., setup_cost=2., driver_budget=4.),
    config(travel_cost=3., setup_cost=2., driver_budget=5.),
    config(terminal_site=1, terminal_connected=True, driver_budget=0., travel_cost=1.),
])
def test_dp_matches_independent_all_paths_on_small_graph(cfg):
    domain = build_mobile_service_domain(cfg)
    arc, connected, count = enumerate_graph(domain)
    result = graph_reachability(domain)
    np.testing.assert_array_equal(result["arc"], arc)
    np.testing.assert_array_equal(result["connected"], connected)
    assert result["graph_path_exists"] == (count > 0)


def test_initial_state_and_drive_setup_lag_are_respected():
    domain = build_mobile_service_domain(config())
    mask = graph_reachability(domain)["connected"]
    np.testing.assert_array_equal(mask[:, 0], True)
    np.testing.assert_array_equal(mask[:, 1], [False, False, True, True])


def test_terminal_requirement_removes_reachable_but_not_returnable_site():
    cfg = config(terminal_site=0, terminal_connected=True, max_moves=1)
    result = graph_reachability(build_mobile_service_domain(cfg))
    assert not result["connected"][:, 1].any()
    assert result["connected"][:, 0].all()


def test_cost_boundary_tolerance_is_explicit_and_only_retains_extra_paths():
    cfg = config(travel_cost=.1, setup_cost=.2, driver_budget=.3)
    domain = build_mobile_service_domain(cfg)
    result = graph_reachability(domain)
    assert result["connected"][2, 1]
    assert result["cost_tolerance_absolute"] > 0
    strict = graph_reachability(domain, cost_tolerance=0.)
    assert np.all(~strict["connected"] | result["connected"])


def test_access_stays_a_power_window_while_connected_state_can_wait():
    access = np.ones((4, 2), bool)
    access[2, 0] = False
    domain = build_mobile_service_domain(config(mobile=False, access=access))
    operation = tighten_operation_bounds(domain)
    # The original semantics allow staying connected across a closed power
    # window. Reachability must not silently impose disconnection/re-setup.
    assert operation["graph_reachability"]["connected"][2, 0]
    assert operation["control_upper"].reshape(4, 2)[2, 0] == 0
    assert operation["control_upper"].reshape(4, 2)[3, 0] == 1


def test_no_soc_or_task_heuristic_is_hidden_in_graph_pruning():
    # No energy can enter, but the terminal battery target is higher. A graph
    # path still exists; original energy constraints retain the infeasibility.
    cfg = config(p_charge_max_kw=0., terminal_energy_kwh=6., mobile=False,
                 min_arrival_throughput_kwh=8., max_throughput_kwh=0.)
    domain = build_mobile_service_domain(cfg)
    result = graph_reachability(domain)
    assert result["graph_path_exists"]
    assert result["connected"][:, 0].all()
    op = tighten_operation_bounds(domain)
    np.testing.assert_array_equal(op["constraints"][0].lb, domain.as_dict()["constraints"][0].lb)
    assert "SOC" in result["omitted_feasibility"]


def test_tightening_preserves_all_rows_indices_and_base_model():
    domain = build_mobile_service_domain(config(mobile=False))
    before = domain.as_dict()
    indices_before = domain.indices.u.copy()
    after = tighten_operation_bounds(domain)
    for key in ("integrality", "u_map", "control_indices"):
        np.testing.assert_array_equal(after[key], before[key])
    assert after["zero_product_pairs"] == before["zero_product_pairs"]
    for key in ("lb", "ub"):
        np.testing.assert_array_equal(getattr(after["constraints"][0], key),
                                      getattr(before["constraints"][0], key))
    assert (after["constraints"][0].A != before["constraints"][0].A).nnz == 0
    np.testing.assert_array_equal(domain.indices.u, indices_before)
    np.testing.assert_array_equal(domain.as_dict()["lower"], before["lower"])
    np.testing.assert_array_equal(domain.as_dict()["upper"], before["upper"])
    np.testing.assert_array_equal(after["control_lower"].reshape(4, 2)[:, 1], 0.)
    np.testing.assert_array_equal(after["control_upper"].reshape(4, 2)[:, 1], 0.)
    assert after["graph_reachability"]["newly_zero_control_count"] == 4
    # This full all-zero-power path is constructed without a solver. Indices,
    # decoder and original physical checker remain compatible with the new box.
    x = np.zeros(domain.model.n_variables)
    x[domain.indices.energy] = 5.
    x[domain.indices.connected_arc[:, 0]] = 1.
    assert np.all(x >= after["lower"]) and np.all(x <= after["upper"])
    assert domain.check(x)["feasible"]
    np.testing.assert_array_equal(domain.decode(x)["u_kw"], np.zeros((4, 2)))


def test_no_complete_path_reports_infeasibility_without_altering_original_rows():
    domain = build_mobile_service_domain(config(mobile=False, terminal_site=1))
    op = tighten_operation_bounds(domain)
    assert not op["graph_reachability"]["graph_path_exists"]
    assert not op["graph_reachability"]["connected"].any()
    np.testing.assert_array_equal(op["control_lower"], 0.)
    np.testing.assert_array_equal(op["control_upper"], 0.)
    assert len(op["constraints"]) == len(domain.as_dict()["constraints"])


def test_random_tiny_graphs_match_independent_enumeration():
    rng = np.random.default_rng(18125)
    total_paths = 0
    for _ in range(30):
        T, J = int(rng.integers(1, 5)), int(rng.integers(1, 4))
        cfg = config(n_slots=T, n_sites=J,
                     initial_site=int(rng.integers(J)),
                     initial_connected=bool(rng.integers(2)),
                     setup_slots=rng.integers(0, 3, J),
                     travel_slots=rng.integers(1, 3, (J, J)),
                     worker_available=rng.random((T, J)) > .25,
                     access=rng.random((T, J)) > .2,
                     travel_allowed=rng.random((J, J)) > .25,
                     setup_cost=rng.integers(0, 3, J),
                     travel_cost=rng.integers(0, 3, (J, J)),
                     driver_budget=float(rng.integers(0, 8)),
                     max_moves=int(rng.integers(0, 4)),
                     terminal_site=None if rng.integers(2) else int(rng.integers(J)),
                     terminal_connected=bool(rng.integers(2)))
        domain = build_mobile_service_domain(cfg)
        arc, mask, count = enumerate_graph(domain)
        total_paths += count
        result = graph_reachability(domain)
        np.testing.assert_array_equal(result["arc"], arc)
        np.testing.assert_array_equal(result["connected"], mask)
    assert total_paths > 0


@pytest.mark.parametrize("value", [-1., np.nan, np.inf])
def test_invalid_tolerance_rejected(value):
    with pytest.raises(ValueError, match="cost_tolerance"):
        graph_reachability(build_mobile_service_domain(config()), cost_tolerance=value)
