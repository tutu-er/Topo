import pytest

from rnj_wzzt.pipeline import _validate_run_options
from rnj_wzzt.reporting import _rnj_candidate_rows, _rnj_summary_row


VALID_OPTIONS = {
    "scenario_suite": "reference",
    "samples_per_scenario": 96,
    "scenario_count": 3,
    "bootstrap_replicates": 100,
    "block_length": 4,
    "confidence_threshold": 0.75,
    "maximum_candidate_count": 2,
    "tolerance_factor": 0.16,
    "deembedding_weight": 0.5,
    "time_limit": 1800.0,
    "coefficient_bound": 2.0,
    "candidate_pool_mode": "rnj",
    "contract_blocks": True,
    "root_observation": "noisy",
}


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("samples_per_scenario", 96.0),
        ("bootstrap_replicates", 0),
        ("block_length", 0),
        ("confidence_threshold", 1.01),
        ("maximum_candidate_count", -1),
        ("tolerance_factor", float("nan")),
        ("deembedding_weight", 1.01),
        ("time_limit", 0.0),
        ("coefficient_bound", float("inf")),
        ("candidate_pool_mode", "unknown"),
    ),
)
def test_validate_run_options_rejects_invalid_values(name, value) -> None:
    options = {**VALID_OPTIONS, name: value}
    with pytest.raises(ValueError):
        _validate_run_options(**options)


def test_validate_run_options_rejects_unobserved_root_contraction() -> None:
    options = {
        **VALID_OPTIONS,
        "root_observation": "unobserved",
        "contract_blocks": True,
    }
    with pytest.raises(ValueError, match="unobserved root"):
        _validate_run_options(**options)


def test_rnj_reporting_keeps_summary_and_candidate_evidence_separate() -> None:
    cherry = frozenset({1, 2})
    unstable = frozenset({3, 4})
    truth_clades = {cherry, frozenset({1, 2, 3})}
    confidence = {cherry: 0.9, unstable: 0.8}

    summary = _rnj_summary_row(
        case_name="case",
        scenario_count=3,
        samples_per_scenario=96,
        r2_score=0.99,
        condition=10.0,
        selection_seconds=1.0,
        full_clades={cherry},
        full_cherries={cherry},
        selected=[cherry],
        confidence=confidence,
        truth_clades=truth_clades,
        truth_cherries={cherry},
    )
    candidates = _rnj_candidate_rows(
        case_name="case",
        confidence=confidence,
        full_cherries={cherry},
        selected=[cherry],
        truth_cherries={cherry},
        confidence_threshold=0.75,
    )

    assert summary["full_rnj_clade_recall"] == 0.5
    assert summary["selected_supports"] == "1,2"
    by_support = {row["support_labels"]: row for row in candidates}
    assert by_support["1,2"]["passes_threshold"] is True
    assert by_support["3,4"]["passes_threshold"] is False
