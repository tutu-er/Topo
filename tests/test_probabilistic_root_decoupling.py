"""Tests for root-common-mode removal and one-level reconstruction."""

from __future__ import annotations

import numpy as np

from experiments.run_probabilistic_root_decoupling import (
    _decouple_global_root_mode,
    _root_decoupled_clades,
)


def _common_mode_example() -> tuple[np.ndarray, np.ndarray, list[int]]:
    terminals = [10, 11, 12, 13]
    true_depth = np.asarray([1.2, 1.3, 1.4, 1.5])
    true_shared = np.asarray(
        [
            [1.2, 0.8, 0.0, 0.0],
            [0.8, 1.3, 0.0, 0.0],
            [0.0, 0.0, 1.4, 0.9],
            [0.0, 0.0, 0.9, 1.5],
        ]
    )
    distance = true_depth[:, None] + true_depth[None, :] - 2.0 * true_shared
    np.fill_diagonal(distance, 0.0)
    observed_depth = true_depth + 0.35
    return distance, observed_depth, terminals


def test_root_mode_decoupling_recovers_first_level_partition() -> None:
    distance, depth, terminals = _common_mode_example()

    root_shift, _, groups, _ = _decouple_global_root_mode(
        distance,
        depth,
        terminals,
        root=1,
        rnj_tolerance_factor=1e-8,
        quantile=0.10,
    )

    assert abs(root_shift - 0.35) < 1e-10
    assert groups == {frozenset({10, 11}), frozenset({12, 13})}


def test_decoupled_local_rnj_recovers_exact_clades() -> None:
    distance, depth, terminals = _common_mode_example()
    _, _, groups, _ = _decouple_global_root_mode(
        distance,
        depth,
        terminals,
        root=1,
        rnj_tolerance_factor=1e-8,
        quantile=0.10,
    )

    predicted = _root_decoupled_clades(
        "rnj",
        distance,
        depth,
        terminals,
        groups,
        rnj_tolerance_factor=1e-8,
        rg_tolerance=1e-8,
    )

    assert predicted == {frozenset({10, 11}), frozenset({12, 13})}
