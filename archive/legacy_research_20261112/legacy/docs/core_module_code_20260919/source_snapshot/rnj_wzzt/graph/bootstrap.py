"""Circular block resampling and stable RNJ boundary-block selection."""

from __future__ import annotations

from math import ceil
from typing import Sequence

import numpy as np


def _boundary_cherries(
    clades: set[frozenset[int]],
) -> set[frozenset[int]]:
    return {
        clade
        for clade in clades
        if not any(other < clade for other in clades)
    }


def _moving_block_bootstrap_copy(
    scenarios: Sequence[dict],
    rng: np.random.Generator,
    block_length: int,
) -> list[dict]:
    sampled: list[dict] = []
    for scenario in scenarios:
        count = len(scenario["P_terminal"])
        if count < 2:
            raise ValueError("each scenario needs at least two rows")
        length = min(int(block_length), count)
        block_count = ceil(count / length)
        starts = rng.integers(0, count, size=block_count)
        indices = np.concatenate(
            [(start + np.arange(length)) % count for start in starts]
        )[:count]
        replicate = {"name": str(scenario.get("name", "scenario"))}
        for key in ("P_terminal", "Q_terminal", "drop_target"):
            replicate[key] = scenario[key].iloc[indices].reset_index(drop=True)
        sampled.append(replicate)
    return sampled


def _select_disjoint(
    candidates: set[frozenset[int]],
    confidence: dict[frozenset[int], float],
    maximum_count: int,
) -> list[frozenset[int]]:
    selected: list[frozenset[int]] = []
    occupied: set[int] = set()
    for clade in sorted(
        candidates,
        key=lambda item: (-confidence[item], -len(item), tuple(sorted(item))),
    ):
        if len(selected) >= maximum_count:
            break
        if occupied.isdisjoint(clade):
            selected.append(clade)
            occupied.update(clade)
    return selected
