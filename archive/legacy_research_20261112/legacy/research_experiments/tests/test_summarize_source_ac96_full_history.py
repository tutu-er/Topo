"""Postprocessing checks use saved synthetic records; no estimator/solver."""
import json

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj import summarize_source_ac96_full_history as report


def fixture_point(extra=None, *, val=.13, se=.01, zero_extra=False):
    supports = [[0], [1], [2], [3]]
    r = [.04, .06, .08, .1]
    x = [.02, .03, .04, .05]
    if extra is not None:
        supports += [extra]
        r += [0. if zero_extra else .1]
        x += [0. if zero_extra else .05]
    R, X = np.zeros((4, 4)), np.zeros((4, 4))
    for support, a, b in zip(supports, r, x):
        R[np.ix_(support, support)] += a
        X[np.ix_(support, support)] += b
    return {"iteration": len(supports), "support_indices": supports,
            "support_labels": [[301 + index for index in s] for s in supports],
            "r_values": r, "x_values": x, "r_matrix": R.tolist(), "x_matrix": X.tolist(),
            "train_mae": .2 if extra is None else .1, "validation_mae": val,
            "validation_se": se, "accepted_gain": .1,
            "solver": {"status": 0, "success": True, "objective": .1, "dual_bound": .1,
                       "mip_gap": 0., "binary_variable_count": 4, "runtime_seconds": .01}}


def truth():
    point = fixture_point([0, 1])
    return {"R": np.array(point["r_matrix"]), "X": np.array(point["x_matrix"]),
            "clades": {frozenset([301, 302])}}


def test_family_score_keeps_zero_clade_and_active_score_filters_it():
    point = fixture_point([0, 1])
    point["support_indices"].append([0, 1, 2])
    point["r_values"].append(0.)
    point["x_values"].append(0.)
    scored = report.score_point(point, [301, 302, 303, 304], truth())
    assert scored["family_f1"] == pytest.approx(2/3)
    assert scored["active_f1"] == 1.
    assert scored["K"] == 6 and scored["active_K"] == 5
    assert scored["r_error_percent"] == 0.


def test_saved_one_se_index_is_retained_even_when_later_truth_score_is_better():
    first = fixture_point(val=.13, se=.01)
    last = fixture_point([0, 1], val=.10, se=.04)
    # The .14 one-SE threshold permits the smaller first model. Physical
    # truth favors the last model, so replacing the saved choice is detectable.
    job = {"job_id": "WZZT_t1", "method": "WZZT", "budget_multiplier": 1}
    model = {"terminal_labels": [301, 302, 303, 304], "path": [first, last],
             "selected_path_index": 0, "r_upper_bound": 2., "x_upper_bound": 2., "bound_expansions": 1}
    old = {**model, "path": [first], "bound_expansions": 0, "r_upper_bound": 1., "x_upper_bound": 1.}
    final_rows, history, warnings = report.certified_path_rows(job, model, truth(), [old, model])
    assert not warnings
    assert len(final_rows) == 2 and len(history) == 3
    assert final_rows[0]["selected"] and not final_rows[1]["selected"]
    assert final_rows[0]["family_f1"] == 0.
    assert final_rows[1]["family_f1"] == 1.
    assert not history[0]["in_final_path"] and not history[0]["selected"]
    summary = {**job, "status": "incomplete", "stop_reason": "timeout"}
    effect = report.selection_effect_rows([summary], final_rows)[0]
    assert effect["selected_minus_last_family_f1"] == -1.
    assert effect["selected_minus_last_K"] == -1
    assert not effect["truth_used_for_selection"]


def test_budget_deltas_wait_for_both_results_and_keep_failure():
    summaries = []
    for method in ("WZZT", "RNJ+WZZT"):
        for multiplier in (1, 100):
            row = {key: None for key in report.SUMMARY_COLUMNS}
            row.update(method=method, budget_multiplier=multiplier, status="failed" if multiplier == 1 else "running",
                       stop_reason="solver_error" if multiplier == 1 else "in_progress", family_f1=.2 if multiplier == 1 else .4)
            summaries.append(row)
    comparisons = report.budget_effect_rows(summaries)
    assert len(comparisons) == 2
    assert all(not row["both_jobs_returned"] for row in comparisons)
    assert all(row["t100_minus_t1_family_f1"] is None for row in comparisons)
    assert comparisons[0]["t1_status"] == "failed"


def test_six_row_reporting_preserves_pending_and_failed_and_never_writes_input(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    jobs = []
    for method in ("NJ", "RNJ", "WZZT", "RNJ+WZZT"):
        for multiplier in ((None,) if "WZZT" not in method else (1, 100)):
            jobs.append({"job_id": method + str(multiplier), "method": method, "budget_multiplier": multiplier,
                         "case": "pengwah18", "variant": "no_source", "per_solve_seconds": 60,
                         "method_wall_seconds": 300})
    (run / "protocol.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    failed = run / "jobs" / jobs[0]["job_id"]
    failed.mkdir(parents=True)
    (failed / "result.json").write_text(json.dumps({"status": "failed", "stop_reason": "fixture_failure"}), encoding="utf-8")
    before = {str(p): report.sha256(p) for p in run.rglob("*") if p.is_file()}
    output = tmp_path / "reports"
    manifest = report.summarize_run(run, tmp_path / "dataset_unused", output, plots=False)
    summary = pd.read_csv(output / "summary.csv")
    assert len(summary) == 6
    assert manifest["status_counts"] == {"failed": 1, "not_run": 5}
    assert not manifest["estimator_or_solver_invoked"]
    assert not manifest["truth_used_for_selection"]
    assert len(pd.read_csv(output / "selection_effect.csv")) == 4
    assert len(pd.read_csv(output / "budget_comparison.csv")) == 2
    assert before == {str(p): report.sha256(p) for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="separate"):
        report.summarize_run(run, tmp_path, run, plots=False)
