"""Independent linear identifiability checks for the one-day WZZT wrapper."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj import source_ac96_wzzt as experiment
from rnj_wzzt.estimation.unmetered_inputs import SourceInputs


def _linear_problem(with_source):
    rng = np.random.default_rng(20261018)
    n, count = 2, 16
    p = rng.uniform(0.05, 0.4, (count, n))
    q = rng.uniform(-0.08, 0.15, (count, n))
    source_p = rng.uniform(0.02, 0.2, count) if with_source else np.zeros(count)
    source_q = rng.uniform(-0.07, 0.08, count) if with_source else np.zeros(count)
    # Independent direct matrix construction: two leaf branches plus a
    # common upstream branch. Source is at the common hidden node.
    expected_R = np.array([[0.45, 0.25], [0.25, 0.55]])
    expected_X = np.array([[0.18, 0.10], [0.10, 0.22]])
    target = p @ expected_R.T + q @ expected_X.T
    target += 0.25 * source_p[:, None] + 0.10 * source_q[:, None]
    index = pd.Index([f"time-{k}" for k in range(count)])
    labels = ["a", "b"]
    scenarios = [{"P_terminal": pd.DataFrame(p, index=index, columns=labels),
                  "Q_terminal": pd.DataFrame(q, index=index, columns=labels),
                  "drop_target": pd.DataFrame(target, index=index, columns=labels)}]
    source_inputs = [SourceInputs(
        pd.Series(source_p, index=index), pd.Series(source_q, index=index),
        provenance="independent_linear_test",
    )]
    # Each column is the direct terminal response to one R/X branch
    # coefficient; no estimator or core feature builder constructs this.
    design = []
    for t in range(count):
        for i in range(n):
            design.append([
                p[t, 0] if i == 0 else 0.0,
                p[t, 1] if i == 1 else 0.0,
                p[t].sum() + source_p[t],
                q[t, 0] if i == 0 else 0.0,
                q[t, 1] if i == 1 else 0.0,
                q[t].sum() + source_q[t],
            ])
    assert np.linalg.matrix_rank(design) == 6
    weights, *_ = np.linalg.lstsq(design, target.ravel(), rcond=None)
    np.testing.assert_allclose(weights, [0.2, 0.3, 0.25, 0.08, 0.12, 0.1], atol=1e-12)
    return scenarios, source_inputs, expected_R, expected_X


@pytest.mark.parametrize("with_source", [False, True], ids=["no_source", "unknown_source_site"])
def test_recovers_identifiable_direct_linear_network(with_source):
    scenarios, amplitudes, expected_R, expected_X = _linear_problem(with_source)
    checkpoints = []
    result = experiment.fit_wzzt(
        scenarios, amplitudes if with_source else None,
        per_solve_seconds=15, wall_seconds=30, checkpoint=checkpoints.append,
    )
    assert result["status"] == "complete", result
    assert result["stop_reason"] == "laminar_family_limit"
    assert result["supports"] == [[0], [1], [0, 1]]
    np.testing.assert_allclose(result["R"], expected_R, atol=1e-7)
    np.testing.assert_allclose(result["X"], expected_X, atol=1e-7)
    assert result["train_mae"] < 1e-8
    assert len(result["path"]) == 2
    assert all(item["optimality_certified"] for item in result["attempts"])
    assert all(item["incumbent"]["valid"] for item in result["attempts"])
    assert all(item["incumbent"]["objective_difference"] < 1e-8 for item in result["attempts"])
    assert checkpoints[-1]["status"] == "complete"
    if with_source:
        # Initialization can only locate the source at a leaf; the extension
        # must change its entire h rather than fixing that earlier answer.
        assert result["path"][0]["h"] in ([1, 0], [0, 1])
        assert result["h"] == [0, 0, 1]
        np.testing.assert_allclose(result["source_gR"], [0.25, 0.25], atol=1e-7)
        np.testing.assert_allclose(result["source_gX"], [0.1, 0.1], atol=1e-7)
        assert not any(result["h"][:2])


def test_explicit_zero_source_matches_disabled_source_and_aligns_labels():
    scenarios, zero_amplitudes, *_ = _linear_problem(False)
    plain = experiment.fit_wzzt(scenarios)
    reordered = [SourceInputs(
        zero_amplitudes[0].p_pu.iloc[::-1], zero_amplitudes[0].q_pu.iloc[::-1], "zero_reordered",
    )]
    explicit = experiment.fit_wzzt(scenarios, reordered)
    assert explicit["source_enabled"] is False
    assert explicit["supports"] == plain["supports"]
    for name in ("R", "X", "r_weights", "x_weights"):
        np.testing.assert_allclose(explicit[name], plain[name], atol=1e-9)
    assert explicit["train_mae"] == pytest.approx(plain["train_mae"], abs=1e-10)
    assert [item["diagnostics"]["binary_variable_count"] for item in explicit["attempts"]] == [
        item["diagnostics"]["binary_variable_count"] for item in plain["attempts"]
    ]


def test_uncertified_extension_keeps_incumbent_but_retains_certified_baseline(monkeypatch):
    scenarios, amplitudes, *_ = _linear_problem(True)
    real_solve = experiment.core._run_milp
    solve_count = 0

    def return_unproved_extension(*args, **kwargs):
        nonlocal solve_count
        result, diagnostics = real_solve(*args, **kwargs)
        solve_count += 1
        if solve_count == 2:
            diagnostics = replace(diagnostics, status=1, success=False,
                                  message="test interrupted backend incumbent", mip_gap=0.1)
        return result, diagnostics

    monkeypatch.setattr(experiment.core, "_run_milp", return_unproved_extension)
    checkpoints = []
    result = experiment.fit_wzzt(scenarios, amplitudes, checkpoint=checkpoints.append)
    assert result["status"] == "incomplete"
    assert result["stop_reason"] == "extension_not_proven_optimal"
    assert result["supports"] == [[0], [1]]
    assert len(result["path"]) == 1
    failed_attempt = result["attempts"][-1]
    assert not failed_attempt["accepted"]
    assert not failed_attempt["optimality_certified"]
    assert failed_attempt["incumbent"]["valid"]
    assert failed_attempt["incumbent"]["supports"] == [[0], [1], [0, 1]]
    assert failed_attempt["incumbent"]["train_mae"] < result["train_mae"]
    assert checkpoints[-1]["stop_reason"] == "extension_not_proven_optimal"


def test_nonzero_initial_source_failure_preserves_solver_diagnostics(monkeypatch):
    scenarios, amplitudes, *_ = _linear_problem(True)
    diagnostics = experiment.core.SolverDiagnostics(
        status=1, success=False, message="test timeout without incumbent",
        objective=None, dual_bound=None, mip_gap=None, node_count=0,
        runtime_seconds=0.01, variable_count=1, binary_variable_count=2,
        constraint_count=1,
    )
    monkeypatch.setattr(experiment.core, "_run_milp",
                        lambda *args, **kwargs: (SimpleNamespace(x=None, fun=None), diagnostics))
    result = experiment.fit_wzzt(scenarios, amplitudes)
    assert result["status"] == "failed"
    assert result["R"] is None and result["X"] is None
    assert result["supports"] == [[0], [1]]
    assert result["path"] == []
    assert result["attempts"][0]["diagnostics"]["status"] == 1
    assert result["attempts"][0]["incumbent"]["has_incumbent"] is False
