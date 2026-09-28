from types import SimpleNamespace

import pandas as pd
import pytest

import experiments.run_rx75_rnj_milp_hybrid_ac as hybrid
from rnj_wzzt_core.experiments.rooted_ablation_support import (
    _map_clade_to_reduced_support, _rnj_reduced_candidate_pool,
)


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

    expanded = hybrid._expand_pseudo_result_clades(
        result,
        pseudo_members,
        terminal_count=4,
    )

    assert expanded == {
        frozenset({1, 2}),
        frozenset({1, 2, 3}),
    }


def test_contraction_freezes_all_reduced_leaf_singletons(monkeypatch) -> None:
    columns = [900000, 3, 4]
    training = [{"name": "train", "P_terminal": pd.DataFrame([[1.0, 2.0, 3.0]], columns=columns)}]
    validation = [{"name": "validation", "P_terminal": pd.DataFrame([[2.0, 3.0, 4.0]], columns=columns)}]
    members = {
        900000: frozenset({1, 2}),
        3: frozenset({3}),
        4: frozenset({4}),
    }

    monkeypatch.setattr(
        hybrid,
        "fit_projected_sensitivity",
        lambda *_args, **_kwargs: (None, None, 1.0, 1.0),
    )

    def fake_aggregate(scenarios, *_args, **_kwargs):
        return scenarios, members

    monkeypatch.setattr(hybrid, "aggregate_rooted_scenarios", fake_aggregate)
    monkeypatch.setattr(hybrid, "preprocess_scenarios", lambda scenarios, _recipe: scenarios)

    _, _, frozen, returned_members = hybrid._contracted_milp_inputs(
        training,
        validation,
        training,
        terminals=[1, 2, 3, 4],
        selected=[frozenset({1, 2})],
        confidence={frozenset({1, 2}): 0.9},
        deembedding_weight=0.5,
    )

    assert frozen == [(0,), (1,), (2,)]
    assert returned_members == members


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
