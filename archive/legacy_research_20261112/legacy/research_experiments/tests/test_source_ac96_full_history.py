"""Checks for the unchanged core observer and complete 64/32 protocol."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj import run_source_ac96_full_history as driver
from rnj_wzzt.estimation import laminar_l1_milp as core
from rnj_wzzt.estimation.unmetered_inputs import SourceInputs


def linear_day(n=4):
    rng = np.random.default_rng(20260903)
    p = rng.uniform(.01, .05, (96, n))
    q = rng.uniform(-.01, .02, (96, n))
    r = np.diag(np.linspace(.04, .07, n))
    x = np.diag(np.linspace(.02, .05, n))
    if n >= 2:
        r[:2, :2] += .08
        x[:2, :2] += .06
    if n >= 4:
        r[2:, 2:] += .09
        x[2:, 2:] += .07
    labels = [301 + i for i in range(n)]
    index = pd.Index(range(96), name="t")
    scenario = {"name": "independent_linear96",
                "P_terminal": pd.DataFrame(p, index=index, columns=labels),
                "Q_terminal": pd.DataFrame(q, index=index, columns=labels),
                "drop_target": pd.DataFrame(p @ r.T + q @ x.T, index=index, columns=labels)}
    sources = [SourceInputs(pd.Series(np.zeros(96), index=index),
                            pd.Series(np.zeros(96), index=index), "test_zero")]
    return [scenario], sources, r, x


def test_actual_split_preserves_one_day_and_training_only_values():
    scenarios, sources, *_ = linear_day()
    train, validation, source_train, arrays, split = driver.split_day(scenarios, sources)
    assert split["training_positions"] == list(range(64))
    assert split["validation_positions"] == list(range(64, 96))
    assert list(train[0]["P_terminal"].index) == list(range(64))
    assert list(validation[0]["P_terminal"].index) == list(range(64, 96))
    assert len(source_train[0].p_pu) == 64
    for field in ("P", "Q", "Y"):
        np.testing.assert_array_equal(np.r_[arrays[f"{field}_train"], arrays[f"{field}_validation"]], arrays[f"{field}_all"])
    changed = [SourceInputs(sources[0].p_pu + .01, sources[0].q_pu, "nonzero")]
    with pytest.raises(ValueError, match="no_source"):
        driver.split_day(scenarios, changed)


def test_core_observer_default_none_and_mutating_observer_do_not_change_search():
    scenarios, sources, *_ = linear_day(2)
    train, validation, *_ = driver.split_day(scenarios, sources)
    options = dict(validation_scenarios=validation, initial_supports=[(0,), (1,)],
                   r_upper_bound=2., x_upper_bound=2., time_limit=10)
    expected = core.fit_laminar_l1_sensitivity(train, **options)
    default_none = core.fit_laminar_l1_sensitivity(train, **options, progress_callback=None)
    events = []
    def malicious_observer(state):
        events.append(state["event"])
        state["path"][0].r_values[:] = 999
        state["path"] = ()
        state["attempted_extensions"] = ()
        return "ignored"
    actual = core.fit_laminar_l1_sensitivity(train, **options, progress_callback=malicious_observer)
    for result in (default_none, actual):
        assert result.support_indices == expected.support_indices
        assert result.stop_reason == expected.stop_reason
        assert result.selected_path_index == expected.selected_path_index
        assert result.bound_expansions == expected.bound_expansions
        np.testing.assert_allclose(result.r_matrix, expected.r_matrix, atol=1e-10)
        np.testing.assert_allclose(result.x_matrix, expected.x_matrix, atol=1e-10)
        assert len(result.path) == len(expected.path)
        for a, b in zip(result.path, expected.path):
            np.testing.assert_allclose(a.r_values, b.r_values, atol=1e-10)
            assert a.validation_mae == pytest.approx(b.validation_mae, abs=1e-12)
    assert events[0] == "initialized" and events[-1] == "stopped"
    assert "extension_attempted" in events and "path_accepted" in events


def test_core_observer_retains_pre_expansion_and_restarted_domains():
    # One known branch coefficient requires three bound doublings. Independent
    # target construction does not use core atom feature/matrix helpers.
    p = np.arange(1., 7.)[:, None]
    q = np.array([0., 1., 0., 1., 0., 1.])[:, None]
    scenario = {"P_terminal": p, "Q_terminal": q, "drop_target": .9 * p + .05 * q}
    states = []
    result = core.fit_laminar_l1_sensitivity([scenario], r_upper_bound=.2, x_upper_bound=.2,
                                            time_limit=10, progress_callback=states.append)
    assert result.bound_expansions >= 1
    before = [s for s in states if s["event"] == "bounds_expansion_requested"]
    after = [s for s in states if s["event"] == "bounds_restarted"]
    assert len(before) == len(after) == result.bound_expansions
    for old, new in zip(before, after):
        assert new["bound_expansions"] == old["bound_expansions"] + 1
        assert len(new["path"]) == 1
        assert new["path"][0].support_indices == ()
        assert new["r_upper_bound"] >= old["r_upper_bound"]
        assert new["x_upper_bound"] >= old["x_upper_bound"]
    np.testing.assert_allclose(result.r_matrix, [[.9]], atol=1e-8)
    np.testing.assert_allclose(result.x_matrix, [[.05]], atol=1e-8)


def test_deadline_snapshot_uses_same_validation_one_se_selection():
    diag = core.SolverDiagnostics(0, True, "test", 0., 0., 0., 0, 0., 0, 0, 0)
    def point(supports, val, se):
        return core.LaminarL1PathPoint(0, supports, supports, np.ones(len(supports)),
            np.ones(len(supports)), np.eye(2), np.eye(2), .1, val, se, None, diag)
    # Best val .10 plus its .04 SE admits the smaller .13 model; .16 fails.
    path = (point((), .16, .01), point(((0,),), .13, .01), point(((0,), (1,)), .10, .04))
    model = driver.select_snapshot_model({"terminal_labels": [0, 1], "path": path,
        "initial_supports": (), "attempted_extensions": (), "r_upper_bound": 2., "x_upper_bound": 2.},
        solver_events=[], progress_events=[], status="incomplete", stop_reason="deadline")
    assert model["selected_path_index"] == 1
    assert model["validation_mae"] == .13
    assert model["status"] == "incomplete"


def test_solver_budget_caps_remaining_and_preserves_backend_result(tmp_path, monkeypatch):
    result = SimpleNamespace(x=np.array([.2]), fun=.3)
    diagnostics = core.SolverDiagnostics(0, True, "test", .3, .3, 0., 0, .1, 1, 0, 1)
    observed = []
    def backend(variables, rows, **kwargs):
        observed.append(kwargs)
        return result, diagnostics
    fake = SimpleNamespace(_run_milp=backend, solver_diagnostics_prove_optimality=lambda d: True)
    monkeypatch.setattr(driver, "perf_counter", lambda: 105.)
    variables = SimpleNamespace(slices={"r": slice(0, 1)})
    with driver.BudgetedSolver(fake, tmp_path, 100., {"method_wall_seconds": 10.}) as observer:
        returned = fake._run_milp(variables, None, time_limit=60., mip_rel_gap=0.)
        assert returned[0] is result and returned[1] is diagnostics
        assert observed[0]["time_limit"] == 5.
        assert observer.events[0]["requested_time_limit_seconds"] == 60.
        assert observer.events[0]["time_limit_seconds"] == 5.
    assert fake._run_milp is backend
    with driver.BudgetedSolver(fake, tmp_path, 90., {"method_wall_seconds": 10.}):
        with pytest.raises(driver.MethodBudgetExhausted):
            fake._run_milp(variables, None, time_limit=60.)


def write_linear_input(directory, scenarios, sources):
    observed = directory / "observed"
    observed.mkdir(parents=True)
    scenario = scenarios[0]
    for key in ("P_terminal", "Q_terminal", "drop_target"):
        scenario[key].to_csv(observed / f"{key}.csv", index_label="t")
    voltage = scenario["drop_target"].copy()
    voltage[:] = np.sqrt(1.02**2 - voltage.to_numpy())
    voltage.to_csv(observed / "V_terminal.csv", index_label="t")
    for key in ("root_voltage", "P0_measured", "Q0_measured"):
        pd.Series(np.full(96, 1.02 if key == "root_voltage" else 0.)).to_csv(observed / f"{key}.csv", index_label="t")
    driver.write_json(observed / "metadata.json", {"root_observation": "noisy", "scenario_settings": {}})
    source_dir = directory / "source"
    source_dir.mkdir()
    pd.DataFrame({"P_pu": np.zeros(96), "Q_pu": np.zeros(96)}).to_csv(source_dir / "amplitudes.csv", index_label="t")


def test_small_real_gurobi_worker_saves_64_32_complete_path(tmp_path):
    pytest.importorskip("gurobipy")
    scenarios, sources, expected_r, expected_x = linear_day(4)
    inputs, job_dir = tmp_path / "inputs", tmp_path / "job"
    job_dir.mkdir()
    write_linear_input(inputs, scenarios, sources)
    job = dict(driver.job_catalog()[2], per_solve_seconds=10., method_wall_seconds=45.)
    parameters = dict(driver.CORE_PARAMETERS, max_atoms=7)
    config = {"observed_root_label": 1, "core_parameters": parameters, "solver": "gurobi"}
    payload = job_dir / "payload.json"
    driver.write_json(payload, {"job": job, "config": config, "job_dir": job_dir,
                               "input_dir": inputs, "protocol_sha256": "independent_linear_fixture"})
    assert driver.worker(payload) == 0
    model = driver.read_json(job_dir / "model.json")
    result = driver.read_json(job_dir / "result.json")
    assert result["status"] == "complete", result
    assert model["has_model"]
    assert model["validation_mae_recomputed"] < 1e-8
    assert len(model["path"]) > 1
    assert model["selected_path_index"] is not None
    assert all(e["solver"] == "gurobi" for e in model["solver_events"])
    with np.load(job_dir / "model.npz") as arrays:
        assert arrays["P_train"].shape == (64, 4)
        assert arrays["P_validation"].shape == (32, 4)
        np.testing.assert_allclose(arrays["R"], expected_r, atol=1e-7)
        np.testing.assert_allclose(arrays["X"], expected_x, atol=1e-7)
    split = driver.read_json(job_dir / "split.json")
    assert split["validation_positions"] == list(range(64, 96))
    assert len(list((job_dir / "progress").glob("*.json"))) == len(model["progress_events"])
