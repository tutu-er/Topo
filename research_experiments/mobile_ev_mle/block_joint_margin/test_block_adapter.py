"""Independent structural targets; no simulation truth or AC solver calls."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from research_experiments.mobile_ev_mle.block_joint_margin.block_adapter import (
    build_block_adapter,
)


def candidate(name, n, supports=()):
    return SimpleNamespace(name=name, n=n, supports=supports)


def test_complete_three_terminal_bank_has_no_extra_evidence():
    bank = [candidate("star", 3), candidate("01", 3, ((0, 1),)),
            candidate("02", 3, ((0, 2),)), candidate("12", 3, ((1, 2),))]
    result = build_block_adapter(bank, (2, 0))
    assert result["common_supports"] == []
    assert result["disputed_supports"] == [[0, 1], [0, 2], [1, 2]]
    assert result["disputed_nodes"] == [0, 1, 2]
    assert result["accessible_nodes"] == [0, 2]
    assert result["accessible_disputed_nodes"] == [0, 2]
    assert result["n3_complete_reduced_bank"]
    assert len(result["directed_pairs"]) == 12
    assert {(p["reference"], p["competitor"]) for p in result["directed_pairs"]} == {
        (a.name, b.name) for a in bank for b in bank if a is not b}
    for key in ("extra_likelihood", "adds_information_beyond_candidate_bank",
                "uses_measurements", "estimates_source", "restrict_actions",
                "candidate_pruning"):
        assert result[key] is False
    assert json.loads(json.dumps(result)) == result


def test_nested_structure_preserves_common_clade_and_outside_access():
    bank = [candidate("a", 5, ((0, 1, 2), (0, 1))),
            candidate("b", 5, ((0, 1, 2), (1, 2)))]
    before = deepcopy([vars(c) for c in bank])
    result = build_block_adapter(bank, [4, 0])
    assert result["common_supports"] == [[0, 1, 2]]
    assert result["disputed_supports"] == [[0, 1], [1, 2]]
    assert result["disputed_nodes"] == [0, 1, 2]
    # Site 4 stays physically available even though no disputed clade contains it.
    assert result["accessible_nodes"] == [0, 4]
    assert result["accessible_disputed_nodes"] == [0]
    assert [vars(c) for c in bank] == before
    forward, reverse = result["directed_pairs"]
    assert forward["disputed_supports"] == reverse["disputed_supports"]
    assert forward["reference"] == reverse["competitor"]


def test_actual_radial_candidate_implicit_root_and_leaves_are_removed():
    # Import the project's existing type, whose supports include root and leaves.
    from research_experiments.mobile_ev_mle.candidates import RadialCandidate
    a = RadialCandidate.from_supports("star", 3)
    b = RadialCandidate.from_supports("cherry", 3, ((0, 1),))
    assert (0, 1, 2) in a.supports and (0,) in a.supports
    result = build_block_adapter([a, b])
    assert result["candidates"][0]["supports"] == []
    assert result["disputed_supports"] == [[0, 1]]
    assert result["accessible_nodes"] == [0, 1, 2]
    assert result["disputed_nodes"] == [0, 1]


def test_identical_structures_retain_pair_without_fabricating_disagreement():
    result = build_block_adapter([candidate("a", 4, ((0, 1),)),
                                  candidate("b", 4, ((0, 1),))], [])
    assert result["common_supports"] == [[0, 1]]
    assert result["disputed_supports"] == result["disputed_nodes"] == []
    assert result["accessible_nodes"] == []
    assert len(result["directed_pairs"]) == 2
    assert all(p["structurally_identical"] for p in result["directed_pairs"])


def test_single_candidate_has_no_comparisons():
    result = build_block_adapter([candidate("only", 4, ((0, 1),))])
    assert result["common_supports"] == [[0, 1]]
    assert result["directed_pairs"] == []
    assert result["disputed_nodes"] == []


@pytest.mark.parametrize("bank,access", [
    ([], None),
    ([candidate("a", 3), candidate("b", 4)], None),
    ([candidate("a", True)], None),
    ([candidate("a", 3), candidate("a", 3)], None),
    ([candidate("", 3)], None),
    ([candidate("a", 3, ((0, 1), (1, 2)))], None),
    ([candidate("a", 3, ((0, 3),))], None),
    ([candidate("a", 3, ((),))], None),
    ([candidate("a", 3, ((0, 1.0),))], None),
    ([candidate("a", 3)], [True]),
    ([candidate("a", 3)], [-1]),
    ([candidate("a", 3)], [3]),
])
def test_invalid_structure_and_access_fail_explicitly(bank, access):
    with pytest.raises(ValueError):
        build_block_adapter(bank, access)
