from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from terminal_case33.estimation.laminar_l1_milp import (
    build_matrices_from_atoms,
    is_laminar_family,
)
from terminal_case33.estimation.rnj_warm_start import (
    select_stable_rnj_initial_supports,
)


def _exact_scenario() -> dict:
    rng = np.random.default_rng(211)
    n = 4
    p = rng.normal(size=(96, n))
    q = rng.normal(size=(96, n))
    supports = ((0, 1, 2, 3), (0, 1), (2, 3), (0,), (1,), (2,), (3,))
    r_matrix, x_matrix = build_matrices_from_atoms(
        n,
        supports,
        (0.2, 0.5, 0.4, 0.1, 0.12, 0.14, 0.16),
        (0.1, 0.25, 0.2, 0.05, 0.06, 0.07, 0.08),
    )
    labels = [101, 102, 103, 104]
    return {
        "name": "exact",
        "P_terminal": pd.DataFrame(p, columns=labels),
        "Q_terminal": pd.DataFrame(q, columns=labels),
        "drop_target": pd.DataFrame(p @ r_matrix.T + q @ x_matrix.T, columns=labels),
    }


def test_stable_rnj_selects_laminar_training_only_clades() -> None:
    selection = select_stable_rnj_initial_supports(
        [_exact_scenario()],
        root_bus=1,
        bootstrap_replicates=4,
        confidence_threshold=0.5,
        maximum_support_size=2,
        maximum_support_count=2,
        ridge_alpha=1e-10,
        seed=17,
    )
    assert selection.selected_support_labels
    assert set(selection.selected_support_labels) <= {(101, 102), (103, 104)}
    assert is_laminar_family(selection.selected_support_indices)
    assert all(value >= 0.5 for value in selection.selected_confidences)


def test_stable_rnj_validates_configuration() -> None:
    scenario = _exact_scenario()
    with pytest.raises(ValueError, match="bootstrap_replicates"):
        select_stable_rnj_initial_supports(
            [scenario], root_bus=1, bootstrap_replicates=0
        )
    with pytest.raises(ValueError, match="maximum_support_size"):
        select_stable_rnj_initial_supports(
            [scenario], root_bus=1, maximum_support_size=4
        )
