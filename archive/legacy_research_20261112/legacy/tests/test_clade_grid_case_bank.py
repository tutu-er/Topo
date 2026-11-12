"""Tests for the clade-grid case bank (aggregation phase-diagram case family)."""

from __future__ import annotations

import networkx as nx
import pytest

from experiments.common import simulate_case, terminal_buses
from terminal_case33.data.clade_grid_case_bank import (
    GRID_CASE_BUILDERS,
    make_clade_grid_case,
)
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.graph.rooted_hierarchy import rooted_clades

GRID_KEYS = ("grid16_k2", "grid16_k4", "grid16_k8")


@pytest.mark.parametrize("case_key", GRID_KEYS)
def test_grid_case_connected_hidden_degree_and_terminal_count(case_key: str) -> None:
    net = CASE_BUILDERS[case_key]()
    graph = net.to_networkx_graph()
    assert nx.is_connected(graph)
    assert nx.is_tree(graph)
    hidden = net.buses.loc[net.buses["bus_type"].eq("hidden_internal")]
    assert not hidden.empty
    # Degree-2 hidden nodes are unidentifiable; the factory must avoid them.
    assert (hidden["hidden_degree"] >= 3).all()
    assert net.metadata["degree_2_hidden_nodes"] == []
    assert len(terminal_buses(net)) == 16


def test_factory_rejects_unidentifiable_or_uneven_grids() -> None:
    with pytest.raises(ValueError):
        make_clade_grid_case(16, 1)  # degree-2 chain tail would be unidentifiable
    with pytest.raises(ValueError):
        make_clade_grid_case(16, 3)  # n_terminals not a multiple of clade_size
    with pytest.raises(ValueError):
        make_clade_grid_case(0, 2)


def test_two_tier_line_library_impedances() -> None:
    scale = 5.0
    net = make_clade_grid_case(16, 4, impedance_scale=scale)
    assert set(net.branches["branch_type"]) == {"backbone", "service"}
    service = net.branches.loc[net.branches["branch_type"].eq("service")].iloc[0]
    backbone = net.branches.loc[net.branches["branch_type"].eq("backbone")].iloc[0]
    assert service["r_ohm"] == pytest.approx(scale * 0.55 * service["length_m"] / 1000.0)
    assert service["x_ohm"] == pytest.approx(scale * 0.38 * service["length_m"] / 1000.0)
    assert backbone["r_ohm"] == pytest.approx(scale * 0.42 * backbone["length_m"] / 1000.0)
    assert backbone["x_ohm"] == pytest.approx(scale * 0.34 * backbone["length_m"] / 1000.0)


@pytest.mark.parametrize("clade_size", (2, 4, 8))
def test_rooted_clades_truth_contains_expected_k_clade(clade_size: int) -> None:
    n_terminals = 16
    net = make_clade_grid_case(n_terminals, clade_size)
    terminals = terminal_buses(net)
    assert len(terminals) == n_terminals
    clades = rooted_clades(net.closed_edges(), net.root_bus, terminals)
    n_attachment = n_terminals // clade_size
    # Hidden chain nodes 3..m+1 each contribute one nontrivial downstream clade.
    assert len(clades) == n_attachment - 1
    # The chain tail contributes exactly the expected size-k clade (the last
    # clade_size terminal ids in chain order).
    tail_clade = frozenset(terminals[-clade_size:])
    assert tail_clade in clades
    assert all(len(clade) % clade_size == 0 for clade in clades)


def test_grid_keys_registered_and_callable() -> None:
    for key in GRID_KEYS:
        assert key in CASE_BUILDERS
        assert CASE_BUILDERS[key] is GRID_CASE_BUILDERS[key]
        net = CASE_BUILDERS[key]()
        assert net.metadata["case_name"] == key


def test_flynn16_control_case_still_present() -> None:
    net = CASE_BUILDERS["flynn16"]()
    assert len(terminal_buses(net)) == 16


@pytest.mark.parametrize("case_key", GRID_KEYS)
def test_simulate_case_smoke(case_key: str) -> None:
    net, scenarios = simulate_case(
        case_key,
        scenario_count=1,
        t_count=24,
        pq_noise_rel=0.001,
        root_voltage_mean=1.0,
    )
    assert len(scenarios) == 1
    scenario = scenarios[0]
    n_terminals = len(terminal_buses(net))
    for field in ("P_terminal", "Q_terminal", "V_terminal", "drop_target"):
        assert scenario[field].shape == (24, n_terminals)
    assert scenario["root_voltage"].shape == (24,)
