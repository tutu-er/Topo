"""Tests for block-disjoint three-stage sample assignment."""

import numpy as np

from terminal_case33.pipeline.blocked_three_stage_crossfit import (
    blocked_stage_indices,
)


def test_blocked_stage_indices_are_disjoint_complete_and_rotating() -> None:
    assignments = []
    for fold in range(3):
        groups = blocked_stage_indices(96, block_length=8, fold=fold)
        sets = [set(group.tolist()) for group in groups]
        assert set().union(*sets) == set(range(96))
        assert sets[0].isdisjoint(sets[1])
        assert sets[0].isdisjoint(sets[2])
        assert sets[1].isdisjoint(sets[2])
        assert [len(group) for group in groups] == [32, 32, 32]
        assignments.append(np.concatenate(groups))
    assert all(len(np.unique(item)) == 96 for item in assignments)
