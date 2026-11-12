"""Integration test for the named regularized joint aggregation mode."""

import numpy as np
import pandas as pd

from terminal_case33.graph.rooted_hierarchy import PseudoCluster
from terminal_case33.pipeline.pseudo_parent_voltage import (
    regularized_joint_fit_options,
)
from terminal_case33.pipeline.two_level_aggregation import (
    run_ordered_two_level_aggregation,
)


def test_regularized_joint_mode_records_effective_options() -> None:
    terminals = [1, 2, 3, 4]
    r_matrix = np.array(
        [
            [1.20, 0.80, 0.00, 0.00],
            [0.80, 1.10, 0.00, 0.00],
            [0.00, 0.00, 0.70, 0.00],
            [0.00, 0.00, 0.00, 0.90],
        ]
    )
    x_matrix = 0.75 * r_matrix
    scenarios = []
    for seed in (11, 22, 33):
        rng = np.random.default_rng(seed)
        p = pd.DataFrame(
            rng.uniform(0.004, 0.018, (96, 4)),
            columns=terminals,
        )
        q = pd.DataFrame(
            rng.uniform(0.001, 0.008, (96, 4)),
            columns=terminals,
        )
        root = pd.Series(1.02 + 0.001 * np.sin(np.linspace(0.0, 4.0 * np.pi, 96)))
        drop = p.to_numpy() @ r_matrix.T + q.to_numpy() @ x_matrix.T
        voltage = pd.DataFrame(
            np.sqrt(root.to_numpy()[:, None] ** 2 - drop),
            columns=terminals,
        )
        scenarios.append(
            {
                "name": f"regularized_{seed}",
                "P_terminal": p,
                "Q_terminal": q,
                "V_terminal": voltage,
                "root_voltage": root,
                "drop_target": pd.DataFrame(drop, columns=terminals),
            }
        )
    cluster = PseudoCluster(
        900000,
        frozenset({1, 2, 3}),
        1.0,
        tuple(),
        tuple(),
    )

    result = run_ordered_two_level_aggregation(
        scenarios,
        root_bus=0,
        clusters=[cluster],
        pseudo_voltage_mode="regularized_joint_vsq",
        tolerance_factor=0.10,
        local_tolerance_factor=0.10,
    )

    assert result.pseudo_voltage_mode == "regularized_joint_vsq"
    assert result.pseudo_voltage_fits
    for key, value in regularized_joint_fit_options().items():
        assert result.pseudo_voltage_options[key] == value
