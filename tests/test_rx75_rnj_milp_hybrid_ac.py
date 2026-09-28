from types import SimpleNamespace
import sys

import pytest

import experiments.run_rx75_rnj_milp_hybrid_ac as hybrid
import rnj_wzzt.pipeline as pipeline
from research_experiments.rnj.rooted_aggregation import _expand_pseudo_result_clades
from research_experiments.rnj.rooted_ablation_support import (
    _map_clade_to_reduced_support, _rnj_reduced_candidate_pool,
)


def test_historical_hybrid_entrypoint_uses_shared_cli_and_preserves_monkeypatch(monkeypatch):
    assert hybrid is pipeline
    captured = {}
    monkeypatch.setattr(hybrid, "run", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, "argv", [
        "hybrid", "--cases", "paper15", "--selection-only", "--scenario-suite", "legacy",
    ])
    hybrid.main()
    assert captured == {
        "cases": ("paper15",), "selection_only": True, "scenario_suite": "legacy",
    }


def test_leaf_singletons_cover_every_reduced_terminal() -> None:
    assert hybrid._leaf_singletons(4) == [(0,), (1,), (2,), (3,)]
    with pytest.raises(ValueError, match="positive"):
        hybrid._leaf_singletons(0)


def test_pseudo_supports_expand_to_original_terminal_clades() -> None:
    result = SimpleNamespace(
        support_labels=((900000,), (900000, 3), (3,), (900000, 3, 4))
    )
    pseudo_members = {
        900000: frozenset({1, 2}),
        3: frozenset({3}),
        4: frozenset({4}),
    }

    expanded = _expand_pseudo_result_clades(
        result,
        pseudo_members,
        terminal_count=4,
    )

    assert expanded == {
        frozenset({1, 2}),
        frozenset({1, 2, 3}),
    }


def test_rnj_candidate_pool_maps_contracted_blocks_and_adds_one_edits() -> None:
    reduced_labels = [900000, 3, 4, 5]
    members = {
        900000: frozenset({1, 2}),
        3: frozenset({3}),
        4: frozenset({4}),
        5: frozenset({5}),
    }
    full_clades = {
        frozenset({1, 2}),
        frozenset({1, 2, 3}),
        frozenset({4, 5}),
    }

    base = _rnj_reduced_candidate_pool(
        full_clades,
        reduced_labels,
        members,
        include_one_edit=False,
    )
    augmented = _rnj_reduced_candidate_pool(
        full_clades,
        reduced_labels,
        members,
        include_one_edit=True,
    )

    assert base == ((0, 1), (2, 3))
    assert (0, 1, 2) in augmented
    assert (1, 2, 3) in augmented
    assert _map_clade_to_reduced_support(
        frozenset({1, 3}), reduced_labels, members
    ) is None
