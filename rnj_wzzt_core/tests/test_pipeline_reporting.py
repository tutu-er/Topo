from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import rnj_wzzt.pipeline as pipeline
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
    "time_limit": 1800.0,
    "coefficient_bound": 2.0,
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
        ("time_limit", 0.0),
        ("coefficient_bound", float("inf")),
    ),
)
def test_validate_run_options_rejects_invalid_values(name, value) -> None:
    options = {**VALID_OPTIONS, name: value}
    with pytest.raises(ValueError):
        _validate_run_options(**options)


def test_pipeline_requires_observed_root_before_creating_output(tmp_path):
    output = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="root_observation"):
        pipeline.run(output, cases=(), root_observation="unobserved")
    assert not output.exists()


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


def test_pipeline_default_search_has_no_rnj_allowlist(tmp_path) -> None:
    result = pipeline.run(tmp_path, cases=())
    assert result["config"]["scenario_settings"]["scenario_suite"] == "reference"
    assert result["config"]["selection_only"] is False
    assert result["config"]["run_baseline"] is False
    assert result["config"]["support_search_mode"] == "unrestricted"
    assert result["config"]["preprocessing"] == "raw"
    assert "candidate_pool_mode" not in result["config"]
    assert "contract_blocks" not in result["config"]
    assert "deembedding_weight" not in result["config"]


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("candidate_pool_mode", "rnj"),
        ("support_search_mode", "rnj"),
        ("contract_blocks", True),
        ("deembedding_weight", 0.5),
    ],
)
def test_pipeline_rejects_removed_option(tmp_path, keyword, value):
    with pytest.raises(TypeError, match=keyword):
        pipeline.run(tmp_path, cases=(), **{keyword: value})


def test_case_preserves_observed_root_levels_and_validation_identity(monkeypatch):
    simulate_pool = pipeline._simulate_pool
    generated = []

    def capture_pool(*args, **kwargs):
        net, raw = simulate_pool(*args, **kwargs)
        generated.append(raw)
        return net, raw

    monkeypatch.setattr(pipeline, "_simulate_pool", capture_pool)
    case, _ = pipeline._prepare_case(
        "paper15", samples_per_scenario=12, scenario_count=1,
        training_replicate=0, validation_replicate=1,
        pq_noise_rel=0., voltage_noise_rel=0.,
        scenario_options={"scenario_suite": "reference", "root_observation": "exact"},
        root_observation="exact",
    )
    assert len(generated) == 2
    for raw_scenarios, fitted_scenarios in zip(
        generated, (case.training, case.validation), strict=True,
    ):
        for raw, fitted in zip(raw_scenarios, fitted_scenarios, strict=True):
            expected = raw["root_voltage"].to_numpy()[:, None]**2 - raw["V_terminal"].to_numpy()**2
            np.testing.assert_array_equal(fitted["drop_target"], expected)
            for key in ("P_terminal", "Q_terminal", "drop_target"):
                pd.testing.assert_frame_equal(fitted[key], raw[key])
            assert fitted["name"] == raw["name"]
    assert case.training[0]["name"] != case.validation[0]["name"]
    assert np.max(np.abs(case.training[0]["drop_target"].mean())) > 1e-9


@pytest.mark.parametrize("solver", ["highs", "gurobi"])
def test_hybrid_keeps_rnj_preset_and_recovers_support_outside_rnj_tree(monkeypatch, solver):
    if solver == "gurobi":
        pytest.importorskip("gurobipy")
    labels = [10, 20, 30, 40]
    fixed = frozenset({10, 20})
    novel = frozenset({10, 20, 30})
    # Independent P/Q excitations identify every matrix column; the novel
    # true support is deliberately absent from the supplied RNJ tree.
    excitation = np.vstack([np.eye(4), -np.eye(4)])
    p = np.vstack([excitation, np.zeros_like(excitation)])
    q = np.vstack([np.zeros_like(excitation), excitation])
    r, x = .2 * np.eye(4), .1 * np.eye(4)
    r[:2, :2] += .5
    x[:2, :2] += .2
    r[:3, :3] += .7
    x[:3, :3] += .3
    scenarios = [{"name": "identifiable", **{
        key: pd.DataFrame(value, columns=labels)
        for key, value in (("P_terminal", p), ("Q_terminal", q), ("drop_target", p @ r + q @ x))
    }}]
    case = SimpleNamespace(name="preset", terminals=labels, training=scenarios,
                           validation=scenarios, truth_clades={fixed, novel})
    selection = SimpleNamespace(selected=[fixed],
                                full_clades={fixed, frozenset({10, 20, 40})})
    actual = pipeline.fit_laminar_l1_sensitivity
    captured = {}

    def fit(*args, **kwargs):
        captured.update(kwargs)
        result = actual(*args, **kwargs)
        captured["result"] = result
        return result

    monkeypatch.setattr(pipeline, "fit_laminar_l1_sensitivity", fit)
    baseline, row = pipeline._fit_milp_variants(
        case, selection, run_baseline=True,
        coefficient_bound=2., time_limit=20., milp_solver=solver,
    )
    assert "allowed_supports" not in captured
    assert "candidate_supports" not in captured
    assert (0, 1) in captured["initial_supports"]
    result = captured["result"]
    assert all((10, 20) in point.support_labels for point in result.path)
    assert tuple(sorted(novel)) in result.support_labels
    assert novel not in selection.full_clades
    np.testing.assert_allclose(result.r_matrix, r, atol=1e-8)
    np.testing.assert_allclose(result.x_matrix, x, atol=1e-8)
    assert row["train_mae"] < 1e-8
    assert row["support_search_mode"] == "unrestricted"
    assert "allowed_support_count" not in row
    assert "candidate_pool_count" not in row
    assert baseline["rnj_initial_supports"] == ""
    assert baseline["frozen_rnj_block_count"] == 0
    assert row["rnj_initial_supports"] == "10,20"
    assert row["frozen_rnj_block_count"] == 1
