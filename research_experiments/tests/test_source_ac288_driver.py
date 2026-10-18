"""Data-budget, no-validation and raw-solver evidence contracts."""
from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj import run_source_ac288 as driver


def observations():
    scenarios, sources = [], []
    for scenario in range(3):
        index = pd.Index(range(96), name="t")
        p = pd.DataFrame(np.arange(192).reshape(96, 2)+scenario*1000., index=index, columns=[301, 302])
        scenarios.append({"name": f"scenario_{scenario}", "P_terminal": p,
                          "Q_terminal": p/4., "drop_target": p/10.})
        sources.append(SimpleNamespace(p_pu=pd.Series(np.arange(96)/1000., index=index),
                                       q_pu=pd.Series(np.arange(96)/2000., index=index)))
    return scenarios, sources


def test_all_288_rows_are_used_in_original_scenario_order():
    scenarios, sources = observations()
    arrays, metadata = driver.input_arrays(scenarios, sources)
    assert arrays["P"].shape == (288, 2)
    np.testing.assert_array_equal(arrays["P"], np.vstack([s["P_terminal"] for s in scenarios]))
    np.testing.assert_array_equal(arrays["P_source"], np.concatenate([s.p_pu for s in sources]))
    assert metadata["training_samples"] == 288 and metadata["validation_samples"] == 0
    assert metadata["validation_supplied"] is False
    assert [r["row_range"] for r in metadata["scenarios"]] == [[0, 96], [96, 192], [192, 288]]


def test_source_rows_and_scenario_count_cannot_silently_change():
    scenarios, sources = observations()
    with pytest.raises(ValueError, match="three"):
        driver.input_arrays(scenarios[:2], sources[:2])
    sources[0].p_pu.index = pd.RangeIndex(1, 97)
    with pytest.raises(ValueError, match="source schedules"):
        driver.input_arrays(scenarios, sources)


def point(k, mae):
    return SimpleNamespace(support_indices=tuple((i,) for i in range(k)),
        support_labels=tuple((301+i,) for i in range(k)), r_values=np.ones(k), x_values=np.ones(k),
        r_matrix=np.eye(2)*k, x_matrix=np.eye(2)*k, source_h=None,
        source_response_r=None, source_response_x=None, train_mae=mae,
        validation_mae=None, validation_se=None)


def test_no_validation_selects_last_saved_certified_point_even_after_deadline():
    earlier, last = point(1, .2), point(2, .1)
    model = driver.select_snapshot_model({"terminal_labels": [301, 302], "path": [earlier, last]},
        solver_events=[], progress_events=[], status="incomplete", stop_reason="deadline")
    assert model["selected_path_index"] == model["last_path_index"] == 1
    assert model["train_mae"] == .1 and model["validation_mae"] is None
    last.validation_mae = .04
    with pytest.raises(ValueError, match="no-validation"):
        driver.select_snapshot_model({"path": [last]}, solver_events=[], progress_events=[],
                                     status="complete", stop_reason="max_atoms")


def test_save_recomputes_source_prediction_without_refitting(tmp_path):
    arrays = {"P": np.zeros((288, 2)), "Q": np.zeros((288, 2)),
              "P_source": np.arange(288)/1000., "Q_source": np.ones(288)*.02}
    gr, gx = np.array([.1, .2]), np.array([.3, .4])
    arrays["Y"] = arrays["P_source"][:, None]*gr+arrays["Q_source"][:, None]*gx
    saved = driver.save_model(tmp_path, {"R": np.zeros((2, 2)), "X": np.zeros((2, 2)),
        "source_g_r": gr, "source_g_x": gx}, arrays, driver.perf_counter())
    assert saved["train_mae_recomputed"] == 0.
    with np.load(tmp_path/"model.npz") as archive:
        np.testing.assert_array_equal(archive["Y"], arrays["Y"])


def test_solver_observer_retains_raw_model_and_caps_remaining_budget(tmp_path, monkeypatch):
    calls = []
    result = SimpleNamespace(x=np.array([.2]), fun=.4)
    diagnostics = SimpleNamespace(to_dict=lambda: {"status": 0, "objective": .4, "dual_bound": .4})
    def backend(variables, rows, **kwargs):
        calls.append(kwargs)
        return result, diagnostics
    core = SimpleNamespace(_run_milp=backend, solver_diagnostics_prove_optimality=lambda d: True)
    variables = SimpleNamespace(objective=[2.], lower=[0.], upper=[1.], integrality=[0], slices={"r": slice(0, 1)})
    rows = SimpleNamespace(row_index=[0], col_index=[0], data=[3.], lower=[.5], upper=[.7])
    monkeypatch.setattr(driver, "perf_counter", lambda: 105.)
    with driver.BudgetedSolver(core, tmp_path, 100., {"method_wall_seconds": 10.}) as observer:
        actual = core._run_milp(variables, rows, time_limit=6000.)
        assert actual[0] is result and calls[0]["time_limit"] == 5.
        event = observer.events[0]
        assert event["primal_objective_recomputed"] == pytest.approx(.4)
        assert event["max_constraint_violation"] == 0.
        with np.load(tmp_path/event["primal_file"]) as raw:
            assert set(raw.files) == {"primal", "objective_coefficients", "lower_bounds", "upper_bounds", "integrality",
                "constraint_row_index", "constraint_col_index", "constraint_data", "constraint_lower", "constraint_upper"}
    assert core._run_milp is backend


def test_declared_catalog_keeps_all_source_cases_and_only_long_budget():
    jobs = driver.job_catalog()
    assert len(jobs) == 16 and len({j["job_id"] for j in jobs}) == 16
    assert {(j["variant"], j["method"]) for j in jobs} == {(v, m) for v in driver.VARIANTS for m in driver.METHODS}
    assert {j["per_solve_seconds"] for j in jobs if "WZZT" in j["method"]} == {6000.}


@pytest.mark.parametrize("method", ["NJ", "RNJ", "RNJ+WZZT"])
def test_real_source_atomic_group_worker_uses_all_rows_without_validation(tmp_path, method):
    runtime = driver.ROOT/"outputs/source_ac288_runtime_20261018"
    if not (runtime/"runtime_manifest.json").exists():
        pytest.skip("freeze the declared runtime before the integration check")
    rng = np.random.default_rng(7123)
    labels, n = [301, 302, 303, 304], 4
    r, x = np.eye(n)*.04, np.eye(n)*.02
    r[:3, :3] += .12
    x[:3, :3] += .08
    gr, gx = np.array([.12, .12, .12, 0.]), np.array([.08, .08, .08, 0.])
    inputs = tmp_path/"observations"/"smooth"
    up, uq = rng.uniform(.005, .02, 96), rng.uniform(.001, .01, 96)
    for i in range(3):
        observed, source = inputs/f"scenario_{i}"/"observed", inputs/f"scenario_{i}"/"source"
        observed.mkdir(parents=True)
        source.mkdir()
        index = pd.Index(range(96), name="t")
        p, q = rng.uniform(.01, .06, (96, n)), rng.uniform(.001, .03, (96, n))
        y = p@r.T+q@x.T+up[:, None]*gr+uq[:, None]*gx
        for field, values in (("P_terminal", p), ("Q_terminal", q), ("drop_target", y),
                              ("V_terminal", np.sqrt(1.02**2-y))):
            pd.DataFrame(values, index=index, columns=labels).to_csv(observed/f"{field}.csv")
        for field in ("root_voltage", "P0_measured", "Q0_measured"):
            pd.Series(np.full(96, 1.02 if field == "root_voltage" else 0.), index=index).to_csv(observed/f"{field}.csv")
        driver.write_json(observed/"metadata.json", {"root_observation": "noisy", "scenario_settings": {}})
        pd.DataFrame({"P_pu": up, "Q_pu": uq}, index=index).to_csv(source/"amplitudes.csv")
    directory = tmp_path/"job"
    directory.mkdir()
    config = {"solver": "gurobi", "core_parameters": dict(driver.CORE_PARAMETERS, max_atoms=7),
              "screen_parameters": dict(driver.SCREEN_PARAMETERS, bootstrap_replicates=3)}
    job = {"case": "pengwah18", "variant": "smooth", "method": method, "per_solve_seconds": 30.,
           "method_wall_seconds": 180., "job_id": "independent_source_group_preflight"}
    driver.write_json(directory/"payload.json", {"job_dir": directory, "input_dir": inputs,
        "runtime_dir": runtime, "job": job, "config": config, "protocol_sha256": "preflight_only"})
    result = subprocess.run([sys.executable, "-B", "-m", "research_experiments.rnj.run_source_ac288",
        "--worker", str(directory/"payload.json")], cwd=driver.ROOT, capture_output=True, text=True, timeout=200)
    assert result.returncode == 0, result.stderr
    model = driver.read_json(directory/"model.json")
    assert model["has_model"] and model["status"] == "complete", model
    if method == "RNJ+WZZT":
        assert model["requested_terminal_groups"] == [[0, 1, 2]]
        assert [0, 1, 2] in model["terminal_groups"]
        assert model["source_h"] is not None
        assert model["selected_path_index"] == len(model["path"])-1
    else:
        assert model["requested_terminal_groups"] == [] and model["source_h"] is None
        assert model["selected_path_index"] is None
    assert model["train_mae_recomputed"] < 1e-8
    assert all(p["validation_mae"] is None and p["validation_se"] is None for p in model["path"])
    with np.load(directory/"model.npz") as data:
        assert data["P"].shape == (288, 4)
    origins = driver.read_json(directory/"runtime_import.json")["module_origins"]
    assert origins and all(Path(p).is_relative_to(runtime) for p in origins.values())
