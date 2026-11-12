import numpy as np
import pandas as pd

from experiments.run_boundary_cherry_day96_audit import (
    _boundary_cherries,
    _moving_block_bootstrap_copy,
)


def test_boundary_cherries_are_minimal_nontrivial_clades():
    clades = {
        frozenset({1, 2, 3}),
        frozenset({4, 5}),
        frozenset({1, 2, 3, 6}),
    }

    assert _boundary_cherries(clades) == {
        frozenset({1, 2, 3}),
        frozenset({4, 5}),
    }


def test_moving_block_bootstrap_preserves_aligned_rows():
    values = np.arange(8, dtype=float)
    scenarios = [
        {
            "name": "aligned",
            "P_terminal": pd.DataFrame({1: values}),
            "Q_terminal": pd.DataFrame({1: 10.0 * values}),
            "drop_target": pd.DataFrame({1: 100.0 * values}),
        }
    ]

    sampled = _moving_block_bootstrap_copy(
        scenarios,
        np.random.default_rng(7),
        block_length=2,
    )[0]

    assert len(sampled["P_terminal"]) == 8
    np.testing.assert_allclose(
        sampled["Q_terminal"].to_numpy(),
        10.0 * sampled["P_terminal"].to_numpy(),
    )
    np.testing.assert_allclose(
        sampled["drop_target"].to_numpy(),
        100.0 * sampled["P_terminal"].to_numpy(),
    )
