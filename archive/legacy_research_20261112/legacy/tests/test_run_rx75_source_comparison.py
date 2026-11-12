"""Small mocked workers and persistence tests; no estimator or MIP is run."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj import run_rx75_source_comparison as driver


def test_catalog_budgets_order_seed_and_terminal_dimension():
    jobs = driver.job_catalog()
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 30
    assert sum(j["role"] == "main" for j in jobs) == 24
    assert sum(j["role"] == "initializer_ablation" for j in jobs) == 6
    assert all(j["method"] in {"NJ", "RNJ"} for j in jobs[:12])
    assert [(j["method"], j["replicate"], j["variant"]) for j in jobs[12:15]] == [
        (m, 0, "rank2_source") for m in ("RNJ+WZZT_tree_init", "RNJ+WZZT", "WZZT")]
    for job in jobs:
        assert job["screen_parameters"]["seed"] == 20260903+job["replicate"]
        assert job["method_wall_seconds"] == (30000 if "WZZT" in job["method"] else 300)
        assert job["per_solve_seconds"] == (6000 if "WZZT" in job["method"] else None)
    assert driver.core_parameters(11)["max_atoms"] == 21
    assert driver.core_parameters(18)["max_atoms"] == 35
    assert driver.CORE_PARAMETERS["max_atoms"] == 35  # frozen helper unchanged


def test_resolution_keeps_equality_and_terminal_leaves():
    tau = .1
    result = driver.resolution_tree([(1, -1, .02), (-1, -2, tau), (-2, 11, .001),
                                     (-2, 12, .01), (-1, 13, .001)], 1, [11, 12, 13], {"tau": tau})
    assert result["contracted_edges"] == [[1, -1, .02]]
    assert result["clades"] == [(11, 12)]
    assert [1, -2, tau] in result["edges"]
    assert [-2, 11, .001] in result["edges"]
    result = driver.resolution_tree([(1, -1, .2), (-1, -2, tau-2e-10), (-2, 11, .01),
                                     (-2, 12, .01), (-1, 13, .01)], 1, [11, 12, 13], {"tau": tau})
    assert result["clades"] == []


def test_atom_resolution_preserves_raw_family_and_zero_weights():
    raw = {"terminal_labels": [11, 12, 13, 14], "support_indices": [(0,), (0, 1), (2, 3)],
           "r_values": [0., 0., .1], "x_values": [0., 0., 0.], "family_clades": [[11, 12], [13, 14]]}
    result = driver.with_resolution(raw, {"a": 1., "b": 1., "tau": .1})
    assert result["resolution_clades"] == [[13, 14]]
    assert result["family_clades"] == raw["family_clades"]
    assert result["r_values"] == [0., 0., .1]
    assert result["resolution_atoms"]["keep"] == [True, False, True]


@dataclass
class Tree:
    edges: tuple
    root: int = 1
    terminals: tuple = (11, 12, 13, 14)


@dataclass
class Point:
    support_indices: tuple
    support_labels: tuple
    r_values: np.ndarray
    x_values: np.ndarray
    r_matrix: np.ndarray
    x_matrix: np.ndarray
    train_mae: float
    source_h: object = None
    source_response_r: object = None
    source_response_x: object = None
    validation_mae: object = None
    validation_se: object = None


def worker_fixture(tmp_path, monkeypatch, method, *, interrupt=False, empty_screen=False):
    job = next(j for j in driver.job_catalog() if j["method"] == method and j["replicate"] == 0 and j["variant"] == "rank2_source")
    directory, copied = tmp_path/"job", tmp_path/"inputs"
    directory.mkdir()
    copied.mkdir()
    data, sources = [], []
    labels = [11, 12, 13, 14]
    for scenario in range(3):
        index = pd.Index(range(96), name="t")
        p = pd.DataFrame(np.tile(np.arange(4)+1., (96, 1)), index=index, columns=labels)
        q = p*.3
        data.append({"name": f"scenario_{scenario}", "P_terminal": p, "Q_terminal": q, "drop_target": p*.05+q*.02})
        sources.append(SimpleNamespace(p_pu=pd.Series(.1, index=index), q_pu=pd.Series(.02, index=index)))
    loaded = {"input_file_sha256": {}}
    calls = {"context": [], "front_methods": [], "source_class": None}
    dense = np.eye(4)*.05+.01
    tree = Tree(((1, -1, .2), (-1, -2, .2), (-2, 11, .1), (-2, 12, .1), (-1, 13, .1), (1, 14, .1)))
    def front(scenarios, source_inputs, method, root_bus):
        calls["front_methods"].append(method)
        assert len(scenarios) == len(source_inputs) == 3
        assert root_bus == 1
        return {"method": method, "R": dense.copy(), "X": dense.copy(), "clades": [(11, 12)], "tree": tree,
                "source_response_g": np.zeros((2, 4)), "diagnostics": {"source_location_used": False}, "timing": {}}
    def loader(path, *, source_inputs_class):
        assert path == str(copied)
        calls["source_class"] = source_inputs_class
        return data, sources, loaded
    def selector(scenarios, terminals, root, **kwargs):
        calls["screen_parameters"] = kwargs
        kwargs["diagnostics"].update(subclade_frequency={frozenset((11, 12, 13)): 0.})
        selected = [] if empty_screen else [frozenset((11, 12, 13))]
        return [frozenset((11, 12, 13))], {frozenset((11, 12, 13)): .9}, selected, .9, 3.
    core = SimpleNamespace(_run_milp=lambda *a, **k: None)
    def normalize(groups, n):
        used = {v for group in groups for v in group}
        return SimpleNamespace(groups=tuple(map(tuple, groups))+tuple((i,) for i in range(n) if i not in used))
    core.normalize_terminal_groups = normalize
    def fit(scenarios, **kwargs):
        calls["fit"] = kwargs
        assert kwargs["validation_scenarios"] is None
        assert kwargs["max_atoms"] == 7
        assert kwargs["time_limit"] == 6000
        supports = tuple(map(tuple, kwargs["initial_supports"]))
        coefficients = np.array([.02 if len(s) == 1 else .2 for s in supports])
        r = sum(value*np.outer(np.isin(range(4), s), np.isin(range(4), s)) for value, s in zip(coefficients, supports))
        point = Point(supports, tuple(tuple(labels[i] for i in s) for s in supports), coefficients,
                      coefficients.copy(), r, r.copy(), .01,
                      source_h=np.ones(len(supports)), source_response_r=np.ones(4)*.1, source_response_x=np.ones(4)*.03)
        state = {"terminal_labels": labels, "path": (point,), "attempted_extensions": (), "initial_supports": supports,
                 "terminal_groups": normalize(kwargs["terminal_groups"] or [], 4).groups,
                 "r_upper_bound": 2., "x_upper_bound": 2., "event": "path_accepted"}
        kwargs["progress_callback"](state)
        if interrupt:
            raise TimeoutError("mock method wall limit")
        return SimpleNamespace(stop_reason="max_atoms", selected_path_index=0)
    core.fit_laminar_l1_sensitivity = fit
    source_class = type("FrozenSourceInputs", (), {})
    monkeypatch.setattr(driver, "_components", lambda runtime: (core, object(), source_class, lambda s, u: u, selector, loader, front))
    @contextmanager
    def floor(c, source, calibration):
        assert c is core
        calls["context"].append(("enter", dict(calibration)))
        try:
            yield
        finally:
            calls["context"].append(("exit", dict(calibration)))
    monkeypatch.setattr(driver, "enforce_rx75_floor", floor)
    from research_experiments.rnj import prepare_source_ac288_runtime as runtime_module
    monkeypatch.setattr(runtime_module, "verify_loaded_modules", lambda runtime: {"rnj_wzzt": str(Path(runtime)/"rnj_wzzt/__init__.py")})
    payload = {"job_dir": directory, "job": job, "runtime_dir": tmp_path/"runtime", "input_dir": copied,
               "input_sha256": {}, "protocol_sha256": "abc", "config": {"core_parameters": driver.core_parameters(4),
               "screen_parameters": driver.SCREEN_PARAMETERS, "solver": "gurobi", "timing_environment": {"exclusive": False}}}
    path = tmp_path/"payload.json"
    driver.write_json(path, payload)
    return path, directory, calls, source_class


@pytest.mark.parametrize("method", ["NJ", "RNJ", "WZZT", "RNJ+WZZT", "RNJ+WZZT_tree_init"])
def test_mock_worker_all_methods_288_no_validation_and_source_alignment(tmp_path, monkeypatch, method):
    payload, directory, calls, source_class = worker_fixture(tmp_path, monkeypatch, method)
    monkeypatch.setattr(driver, "evaluate_result", lambda *args: pytest.fail("worker read truth"))
    assert driver.worker(payload) == 0
    model = driver.read_json(directory/"model.json")
    assert model["has_model"] and model["training_rows"] == 288
    assert model["validation_supplied"] is False and model["validation_mae"] is None
    assert calls["source_class"] is source_class
    with np.load(directory/"model.npz") as arrays:
        assert arrays["P"].shape == (288, 4)
        np.testing.assert_array_equal(arrays["row_scenario"], np.repeat(range(3), 96))
        np.testing.assert_array_equal(arrays["row_position"], np.tile(range(96), 3))
        np.testing.assert_array_equal(arrays["P_source"], np.full(288, .1))
        assert arrays["calibration_R"].shape == (4, 4)
    if "WZZT" in method:
        assert [c[0] for c in calls["context"]] == ["enter", "exit"]
        assert calls["context"][0][1] == calls["context"][1][1] == model["rx75_calibration"]
        assert model["selected_path_index"] == model["last_path_index"] == 0
        assert len(model["source_h"]) == len(model["support_indices"])
        assert (directory/"progress/000001.json").exists()
        if method == "WZZT":
            assert calls["fit"]["initial_supports"] == [(0,), (1,), (2,), (3,)]
            assert calls["fit"]["terminal_groups"] is None
        else:
            assert calls["fit"]["terminal_groups"] == [(0, 1, 2)]
            assert calls["screen_parameters"]["bootstrap_replicates"] == 100
        if method.endswith("_tree_init"):
            initialization = driver.read_json(directory/"tree_initialization.json")
            assert initialization["tree_clades"] == [[11, 12, 13]]
            assert initialization["front_and_screen_clades_equal"] is False
            assert [0, 1] not in model["initial_supports"]  # no second-tree substitution
    else:
        assert not calls["context"]
        assert model["source_h"] is None
        assert model["matrix_origin"] == "source_adjusted_ordered_regression_before_tree_inference"


def test_empty_strict_groups_remain_empty_not_repaired(tmp_path, monkeypatch):
    payload, directory, calls, _ = worker_fixture(tmp_path, monkeypatch, "RNJ+WZZT", empty_screen=True)
    driver.worker(payload)
    model = driver.read_json(directory/"model.json")
    assert model["bootstrap"]["empty_selection_retained"] is True
    assert model["requested_terminal_groups"] == []
    assert calls["fit"]["initial_supports"] == [(0,), (1,), (2,), (3,)]


def test_timeout_retains_last_certified_point_and_restores_context(tmp_path, monkeypatch):
    payload, directory, calls, _ = worker_fixture(tmp_path, monkeypatch, "RNJ+WZZT", interrupt=True)
    driver.worker(payload)
    result, model = driver.read_json(directory/"result.json"), driver.read_json(directory/"model.json")
    assert result["status"] == "incomplete" and result["search_complete"] is False
    assert model["has_model"] and len(model["path"]) == 1
    assert calls["context"][-1][0] == "exit"


def test_outer_kill_retains_checkpoint_and_failure_without_model_has_null_metrics(tmp_path):
    directory = tmp_path/"job"
    directory.mkdir()
    job = driver.job_catalog()[0]
    driver.write_json(directory/"model.json", {"has_model": True, "path": [{"token": "certified"}], "timing": {"front_end_fit_seconds": 1.}})
    driver.recover_process(directory, job, "fixed", 32000., killed=True, exitcode=-9)
    assert driver.read_json(directory/"model.json")["path"] == [{"token": "certified"}]
    assert driver.read_json(directory/"result.json")["retained_last_certified_checkpoint"] is True
    failed = tmp_path/"failed"
    failed.mkdir()
    driver.recover_process(failed, job, "fixed", 3., killed=False, exitcode=1)
    result = driver.evaluate_result(failed, tmp_path/"nonexistent_truth")
    assert result["status"] == "failed"
    assert all(value is None for value in result["metrics"].values())


def test_outer_kill_restores_atomic_certified_pair_after_partial_main_write(tmp_path):
    directory = tmp_path/"job"
    directory.mkdir()
    arrays = {"P": np.zeros((288, 2)), "Q": np.zeros((288, 2)), "Y": np.zeros((288, 2)),
              "P_source": np.zeros(288), "Q_source": np.zeros(288)}
    model = {"R": np.eye(2), "X": np.eye(2)*2, "source_g_r": None,
             "path": [{"token": "last_certified"}], "selected_path_index": 0}
    driver.save_checkpoint(directory, model, arrays, driver.perf_counter(), 1)
    # Reproduce an outer kill after the ordinary NPZ replacement but before JSON.
    np.savez(directory/"model.npz", R=np.eye(2)*999, X=np.eye(2)*999)
    driver.recover_process(directory, driver.job_catalog()[0], "fixed", 30001., killed=True, exitcode=-9)
    with np.load(directory/"model.npz") as data:
        np.testing.assert_array_equal(data["R"], np.eye(2))
        assert data["P"].shape == (288, 2)
    assert driver.read_json(directory/"model.json")["path"] == [{"token": "last_certified"}]


def test_summary_retains_30_denominator_missing_failure_timeout(tmp_path):
    jobs = driver.job_catalog()
    for job, status in zip(jobs[:2], ("failed", "incomplete")):
        driver.write_json(tmp_path/"jobs"/job["job_id"]/"result.json", {"status": status, "has_model": False,
            "metrics": dict.fromkeys(driver.METRIC_KEYS)})
    rows = driver.summarize(tmp_path)
    assert len(rows) == 30
    assert driver.read_json(tmp_path/"completion.json") == {"expected": 30, "finished": 2,
        "status_counts": {"failed": 1, "incomplete": 1, "not_run": 28}, "models_available": 0}
    assert all(row.get("clade_f1") is None for row in rows)


def test_input_whitelist_and_duplicate_controller_lock(tmp_path):
    directory = tmp_path/"inputs"
    file = directory/"scenario_0/observed/P_terminal.csv"
    file.parent.mkdir(parents=True)
    file.write_text("allowed", encoding="utf-8")
    inventory = {file.relative_to(directory).as_posix(): driver.sha256(file)}
    driver.verify_inputs(directory, inventory)
    secret = directory/"truth/site.json"
    secret.parent.mkdir()
    secret.write_text("not allowed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed|unexpected"):
        driver.verify_inputs(directory, inventory)
    with driver.controller_lock(tmp_path/"run"):
        with pytest.raises(RuntimeError, match="lock exists"):
            with driver.controller_lock(tmp_path/"run"):
                pass
    assert not (tmp_path/"run/controller.lock").exists()


def test_worker_command_executes_frozen_closure_not_live_module(tmp_path):
    command = driver.worker_command(tmp_path, tmp_path/"payload.json")
    assert command[:3] == [driver.sys.executable, "-B", "-c"]
    assert command[4] == str((tmp_path/"source_snapshot").resolve())
    assert command[-2:] == ["--worker", str(tmp_path/"payload.json")]
    inventory = driver.source_closure()
    for name in ("run_rx75_source_comparison.py", "run_source_ac288.py", "rx75_edge_floor.py", "prepare_source_advantage.py",
                 "source_ac96_trees.py", "rooted_study_baselines.py", "sensitivity_geometry.py"):
        assert f"research_experiments/rnj/{name}" in inventory


def test_budget_observer_preserves_raw_coo_and_effective_limit(tmp_path):
    class Diagnostic:
        def to_dict(self):
            return {"status": 0, "objective": 1.}
    calls = []
    def solve(variables, rows, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(x=np.array([1.])), Diagnostic()
    core = SimpleNamespace(_run_milp=solve, solver_diagnostics_prove_optimality=lambda diagnostic: True)
    variables = SimpleNamespace(objective=[1.], lower=[0.], upper=[2.], integrality=[0], slices={"r": slice(0, 1)})
    rows = SimpleNamespace(row_index=[0], col_index=[0], data=[1.], lower=[1.], upper=[1.])
    with driver.BudgetedSolver(core, tmp_path, driver.perf_counter(), {"method_wall_seconds": 2.}) as observer:
        core._run_milp(variables, rows, time_limit=6000.)
        assert 0 < calls[0]["time_limit"] <= 2.
        assert observer.events[0]["max_constraint_violation"] == 0.
        with np.load(tmp_path/"solver/000001.npz") as saved:
            assert {"primal", "objective_coefficients", "constraint_row_index", "constraint_col_index", "constraint_data",
                    "constraint_lower", "constraint_upper", "integrality", "lower_bounds", "upper_bounds"} <= set(saved.files)
    assert core._run_milp is solve


def test_prepare_freezes_inputs_and_rejects_resume_changes(tmp_path, monkeypatch):
    dataset, runtime, output = tmp_path/"dataset", tmp_path/"runtime", tmp_path/"output"
    runtime.mkdir()
    driver.write_json(runtime/"runtime_manifest.json", {"frozen": True})
    inventory = {}
    for replicate in driver.REPLICATES:
        for variant in driver.VARIANTS:
            for scenario in range(3):
                for folder in ("observed", "source"):
                    path = dataset/driver.CASE/f"replicate_{replicate}"/variant/f"scenario_{scenario}"/folder/"data.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("t,value\n0,1\n", encoding="utf-8")
                    inventory[path.relative_to(dataset).as_posix()] = driver.sha256(path)
    for name in ("buses.csv", "branches.csv", "metadata.json"):
        path = dataset/driver.CASE/"network"/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evaluation truth", encoding="utf-8")
        inventory[path.relative_to(dataset).as_posix()] = driver.sha256(path)
    driver.write_json(dataset/"manifest.json", {"completed": 18, "failed": 0, "file_sha256": inventory})
    from research_experiments.rnj import prepare_source_ac288_runtime as runtime_module
    monkeypatch.setattr(runtime_module, "ensure_runtime", lambda path: None)
    frozen_name = "research_experiments/rnj/run_rx75_source_comparison.py"
    monkeypatch.setattr(driver, "source_closure", lambda: {frozen_name: driver.sha256(driver.ROOT/frozen_name)})
    args = SimpleNamespace(output_dir=output, dataset_dir=dataset, runtime_dir=runtime, solver="gurobi", workers=1)
    _, _, protocol, pending = driver.prepare(args)
    assert len(pending) == protocol["expected_jobs"] == 30
    assert protocol["core_parameters"]["max_atoms"] == 21
    assert protocol["timing_environment"]["exclusive"] is False
    assert (output/"source_snapshot"/frozen_name).read_bytes() == (driver.ROOT/frozen_name).read_bytes()
    assert driver.sha256(output/"source_snapshot"/driver.DESIGN_PROTOCOL) == protocol["design_protocol_sha256"]
    assert all("truth" not in str(path.relative_to(output/"inputs")) for path in (output/"inputs").rglob("*") if path.is_file())
    assert len(driver.prepare(args)[3]) == 30
    # Existing failed jobs remain final and are skipped on resume, not retried.
    job = driver.job_catalog()[0]
    directory = output/"jobs"/job["job_id"]
    driver.recover_process(directory, job, driver.sha256(output/"protocol.json"), 1., killed=False, exitcode=1)
    assert len(driver.prepare(args)[3]) == 29
    frozen_network = driver.verify_evaluation_network(output)
    assert frozen_network == output/"evaluation_network"
    (frozen_network/"buses.csv").write_text("changed truth", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation network changed"):
        driver.verify_evaluation_network(output)
    (frozen_network/"buses.csv").write_text("evaluation truth", encoding="utf-8")
    copied = next((output/"inputs").rglob("data.csv"))
    copied.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        driver.prepare(args)
