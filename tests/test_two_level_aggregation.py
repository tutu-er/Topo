"""Tests for pseudo-rooted local refitting and Ordered aggregation."""

import numpy as np
import pandas as pd

from terminal_case33.graph.rooted_hierarchy import PseudoCluster
from terminal_case33.pipeline.two_level_aggregation import (
    fit_local_cluster_sensitivities,
    region_only_clusters,
    run_ordered_two_level_aggregation,
)


def _linear_scenarios(
    terminals: list[int],
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    seeds: tuple[int, ...] = (11, 22, 33),
) -> list[dict]:
    """Generate full-rank, noiseless squared-voltage linear scenarios."""

    scenarios = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        count = 96
        p = pd.DataFrame(
            rng.uniform(0.004, 0.018, (count, len(terminals))),
            columns=terminals,
        )
        q = pd.DataFrame(
            rng.uniform(0.001, 0.008, (count, len(terminals))),
            columns=terminals,
        )
        root = pd.Series(
            1.02 + 0.001 * np.sin(np.linspace(0.0, 4.0 * np.pi, count))
        )
        drop = p.to_numpy() @ r_matrix.T + q.to_numpy() @ x_matrix.T
        voltage = pd.DataFrame(
            np.sqrt(root.to_numpy()[:, None] ** 2 - drop),
            columns=terminals,
        )
        scenarios.append(
            {
                "name": f"linear_{seed}",
                "P_terminal": p,
                "Q_terminal": q,
                "V_terminal": voltage,
                "root_voltage": root,
                "drop_target": pd.DataFrame(drop, columns=terminals),
            }
        )
    return scenarios


def test_region_only_cluster_does_not_freeze_preliminary_edges() -> None:
    cluster = PseudoCluster(
        900000,
        frozenset({1, 2, 3}),
        0.95,
        (frozenset({1, 2}),),
        ((1, 2),),
    )

    selected = region_only_clusters([cluster])[0]

    assert selected.members == cluster.members
    assert selected.frozen_clades == tuple()
    assert selected.frozen_sibling_pairs == tuple()


def test_local_refit_recovers_internal_clade_from_pseudo_root() -> None:
    terminals = [1, 2, 3]
    r_matrix = np.array(
        [
            [1.20, 0.80, 0.00],
            [0.80, 1.10, 0.00],
            [0.00, 0.00, 0.70],
        ]
    )
    x_matrix = 0.75 * r_matrix
    local = _linear_scenarios(terminals, r_matrix, x_matrix)
    pseudo_scenarios = [
        {"V_terminal": pd.DataFrame({900000: scenario["root_voltage"]})}
        for scenario in local
    ]
    cluster = PseudoCluster(
        900000,
        frozenset(terminals),
        1.0,
        tuple(),
        tuple(),
    )

    fit = fit_local_cluster_sensitivities(
        local,
        pseudo_scenarios,
        [cluster],
        {"name": "daily_demean", "kind": "demean"},
        tolerance_factor=0.10,
    )[900000]

    assert frozenset({1, 2}) in fit.clades
    assert fit.clade_stability == 1.0
    assert fit.heldout_nrmse < 1e-6
    assert fit.admissible


def test_ordered_aggregation_emits_reuse_and_local_refit_candidates() -> None:
    terminals = [1, 2, 3, 4]
    r_matrix = np.array(
        [
            [1.10, 0.90, 0.50, 0.00],
            [0.90, 1.15, 0.50, 0.00],
            [0.50, 0.50, 0.90, 0.00],
            [0.00, 0.00, 0.00, 0.70],
        ]
    )
    x_matrix = 0.75 * r_matrix
    scenarios = _linear_scenarios(terminals, r_matrix, x_matrix)
    cluster = PseudoCluster(
        900000,
        frozenset({1, 2, 3}),
        1.0,
        (frozenset({1, 2}),),
        ((1, 2),),
    )

    result = run_ordered_two_level_aggregation(
        scenarios,
        root_bus=0,
        clusters=[cluster],
        tolerance_factor=0.10,
        local_tolerance_factor=0.10,
        deembedding_weight=1.0,
    )

    assert set(result.candidate_clades_by_mode) == {
        "ordered_aggregation_reuse",
        "ordered_aggregation_hybrid_local",
        "ordered_aggregation_local_refit",
    }
    refit = result.candidate_clades_by_mode["ordered_aggregation_local_refit"]
    hybrid = result.candidate_clades_by_mode["ordered_aggregation_hybrid_local"]
    assert frozenset({1, 2, 3}) in refit
    assert frozenset({1, 2}) in refit
    assert hybrid == refit
    assert result.mean_local_nrmse < 1e-6
    assert result.all_local_fits_admissible

    rejected = run_ordered_two_level_aggregation(
        scenarios,
        root_bus=0,
        clusters=[cluster],
        tolerance_factor=0.10,
        local_tolerance_factor=0.10,
        deembedding_weight=1.0,
        local_refit_max_condition_number=0.0,
    )
    assert set(rejected.candidate_clades_by_mode) == {
        "ordered_aggregation_reuse",
        "ordered_aggregation_hybrid_local",
    }
    assert not rejected.all_local_fits_admissible
    assert (
        rejected.candidate_clades_by_mode["ordered_aggregation_hybrid_local"]
        == rejected.candidate_clades_by_mode["ordered_aggregation_reuse"]
    )
