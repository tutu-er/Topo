"""Tests for fixed-tree fitting and held-out AC candidate reranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from terminal_case33.data.small_terminal_lv import build_small_terminal_lv_case
from terminal_case33.estimation.preprocessing import squared_voltage_drop_from_observed_root
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import RootedTreeResult
from terminal_case33.models.ac_powerflow import solve_ac_power_flow_timeseries
from terminal_case33.pipeline.ac_likelihood import (
    rank_candidate_trees_ac,
    representative_validation_count,
    split_representative_scenarios,
)
from terminal_case33.pipeline.rnj_candidates import RNJCandidate


def _candidate(net, wrong: bool) -> RNJCandidate:
    oriented = net.orient_from_root()
    edges = [(int(row.parent), int(row.child), 1.0) for row in oriented.itertuples()]
    if wrong:
        edges = [edge for edge in edges if set(edge[:2]) != {2, 101}]
        edges.append((3, 101, 1.0))
    terminals = tuple(net.load_buses())
    tree = RootedTreeResult(
        root=net.root_bus,
        terminals=terminals,
        edges=tuple(edges),
        group_tolerance=0.0,
    )
    return RNJCandidate(
        tree=tree,
        distance_mode="manual_wrong" if wrong else "manual_true",
        tolerance_factor=0.0,
        clades=frozenset(rooted_clades(tree.edges, tree.root, terminals)),
    )


def _scenario(net, seed: int, sample_count: int = 128) -> dict:
    rng = np.random.default_rng(seed)
    terminals = net.load_buses()
    buses = net.buses.set_index("bus_id")
    base = np.array(
        [float(buses.loc[node, "pd_kw"]) / (1000.0 * net.base_mva) for node in terminals]
    )
    independent = rng.lognormal(mean=-0.08, sigma=0.28, size=(sample_count, len(terminals)))
    common = 0.85 + 0.20 * np.sin(np.linspace(0.0, 4.0 * np.pi, sample_count))
    p_values = base[None, :] * independent * common[:, None]
    q_ratio = np.array(
        [
            float(buses.loc[node, "qd_kvar"]) / max(float(buses.loc[node, "pd_kw"]), 1e-12)
            for node in terminals
        ]
    )
    q_values = p_values * q_ratio[None, :] * rng.uniform(
        0.92,
        1.08,
        size=p_values.shape,
    )
    index = pd.RangeIndex(sample_count)
    p_frame = pd.DataFrame(p_values, index=index, columns=terminals)
    q_frame = pd.DataFrame(q_values, index=index, columns=terminals)
    root = pd.Series(
        1.02 + 0.001 * np.sin(np.linspace(0.0, 6.0 * np.pi, sample_count)),
        index=index,
    )
    ac = solve_ac_power_flow_timeseries(net, p_frame, q_frame, v_root=root)
    voltage = ac["V_bus_mag"].loc[:, terminals]
    return {
        "name": f"scenario_{seed}",
        "P_terminal": p_frame,
        "Q_terminal": q_frame,
        "V_terminal": voltage,
        "root_voltage": root,
        "drop_target": squared_voltage_drop_from_observed_root(voltage, root),
    }


def test_heldout_ac_likelihood_prefers_true_tree() -> None:
    net = build_small_terminal_lv_case()
    scenarios = [_scenario(net, seed) for seed in (11, 22, 33)]
    true_candidate = _candidate(net, wrong=False)
    wrong_candidate = _candidate(net, wrong=True)
    result = rank_candidate_trees_ac(
        [wrong_candidate, true_candidate],
        scenarios[:2],
        scenarios[2:],
        net,
        pq_noise_relative_std=0.005,
        voltage_noise_relative_std=0.0002,
        ac_refinement_max_samples=16,
    )

    assert result.selected.candidate.distance_mode == "manual_true"
    assert result.ranked[0].voltage_rmse < result.ranked[1].voltage_rmse
    assert result.ranked[0].edge_fit.r_edge_coefficients.min() >= 0.0
    assert result.ranked[0].edge_fit.x_edge_coefficients.min() >= 0.0
    assert result.ranked[0].edge_fit.training_ac_rmse_after <= (
        result.ranked[0].edge_fit.training_ac_rmse_before
    )


def test_representative_validation_count_scales_with_case_bank() -> None:
    assert representative_validation_count(3) == 1
    assert representative_validation_count(5) == 2
    assert representative_validation_count(9) == 3


def test_representative_scenario_split_is_deterministic() -> None:
    net = build_small_terminal_lv_case()
    scenarios = [_scenario(net, seed, sample_count=32) for seed in (1, 2, 3, 4, 5)]
    training, validation, indices = split_representative_scenarios(scenarios, 2)

    assert len(training) == 3
    assert len(validation) == 2
    assert len(set(indices)) == 2
    assert [item["name"] for item in validation] == [scenarios[index]["name"] for index in indices]
