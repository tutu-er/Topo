"""RX75 threshold, row isolation and real joint Source fixed-family refits."""
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj import refit_rx75_resolution as post


@dataclass
class InputSource:
    p_pu: pd.Series
    q_pu: pd.Series
    provenance: str


def arrays_fixture(source=True):
    rng = np.random.default_rng(20261018)
    p, q = rng.uniform(.01, .06, (288, 4)), rng.uniform(.001, .03, (288, 4))
    r, x = np.eye(4)*.04, np.eye(4)*.02
    r[:3, :3] += .12
    x[:3, :3] += .08
    u, v = rng.uniform(.005, .02, 288), rng.uniform(.001, .01, 288)
    if not source:
        u[:], v[:] = 0., 0.
    y = p@r.T+q@x.T+u[:, None]*np.array([.12, .12, .12, 0.])+v[:, None]*np.array([.08, .08, .08, 0.])
    return {"P": p, "Q": q, "Y": y, "R": r, "X": x, "P_source": u, "Q_source": v,
            "row_scenario": np.repeat(np.arange(3), 96), "row_position": np.tile(np.arange(96), 3)}


def write_fixture(tmp_path, *, source=True):
    arrays = arrays_fixture(source)
    original, baseline = tmp_path/"raw", tmp_path/"NJ"
    for folder in (original, baseline):
        folder.mkdir()
        np.savez_compressed(folder/"model.npz", **arrays)
        post.write_json(folder/"result.json", {"status": "complete", "has_model": True,
            "variant": "source" if source else "no_source", "method": "NJ" if folder == baseline else "WZZT",
            "metrics": {"family_clade_f1": .5}})
    # The [0,1] atom initially looks strong, but fitting the actual Y gives it
    # zero weight. A second contraction/refit is therefore required.
    model = {"terminal_labels": [301, 302, 303, 304], "support_indices": [[0], [1], [2], [3], [0, 1, 2], [0, 1], [0, 1, 2, 3]],
        "r_values": [.04]*4+[.12, .06, .00001], "x_values": [.02]*4+[.08, .04, .00001],
        "has_model": True, "status": "complete", "r_upper_bound": 2., "x_upper_bound": 2.,
        "source_h": [1, 0, 0, 0, 0, 0, 0], "source_g_r": [0.]*4, "source_g_x": [0.]*4,
        "family_clades": [[301, 302, 303], [301, 302]], "active_clades": [[301, 302, 303], [301, 302]]}
    post.write_json(original/"model.json", model)
    post.write_json(baseline/"model.json", {"terminal_labels": model["terminal_labels"], "status": "complete", "has_model": True,
        "matrix_origin": "source_adjusted_ordered_regression_before_tree_inference"})
    return original, baseline, arrays


def test_threshold_equality_tolerance_singletons_and_root_stem():
    supports = ((0,), (0, 1), (1, 2), (2, 3), (0, 1, 2, 3))
    kept, value = post.contract(supports, [0., 1., 1.-5e-11, 1.-2e-10, .9], [0.]*5, {"a": 1., "b": 1., "tau": 1.})
    assert kept == supports[:3]
    assert value["contracted_indices"] == [3, 4]
    assert value["absolute_tolerance"] == 1e-10


def test_three_original_blocks_and_exact_source_class_and_bad_rows():
    arrays = arrays_fixture()
    scenarios, sources = post.restore_inputs(arrays, [301, 302, 303, 304], InputSource)
    assert len(scenarios) == len(sources) == 3
    assert all(type(s) is InputSource for s in sources)
    np.testing.assert_array_equal(np.vstack([s["P_terminal"] for s in scenarios]), arrays["P"])
    np.testing.assert_array_equal(np.concatenate([s.p_pu for s in sources]), arrays["P_source"])
    arrays["row_position"] = arrays["row_position"][::-1]
    with pytest.raises(ValueError, match="row_scenario/row_position"):
        post.restore_inputs(arrays, [301, 302, 303, 304], InputSource)


def test_finished_and_overwrite_and_same_data_guards(tmp_path):
    raw, baseline, arrays = write_fixture(tmp_path)
    result = post.read_json(raw/"result.json")
    result["status"] = "running"
    post.write_json(raw/"result.json", result)
    with pytest.raises(ValueError, match="final status"):
        post.finished_job(raw)
    result["status"] = "complete"
    post.write_json(raw/"result.json", result)
    output = tmp_path/"occupied"
    output.mkdir()
    (output/"keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        post.prepare(raw, baseline, output)
    arrays["Y"][0, 0] += .001
    np.savez_compressed(baseline/"model.npz", **arrays)
    with pytest.raises(ValueError, match="identical Y"):
        post.prepare(raw, baseline, tmp_path/"new")


def test_budget_before_next_solve_preserves_certified_checkpoint_as_incomplete(tmp_path, monkeypatch):
    """Regression: the last event is optimal, but the NEXT solve cannot start."""
    raw, _, arrays = write_fixture(tmp_path, source=False)
    output = tmp_path/"post"
    shutil.copytree(raw, output/"original")
    config = {"calibration": post.calibrate_rx75_floor(arrays["R"], arrays["X"]),
              "r_upper_bound": 2., "x_upper_bound": 2., "solver": "fake",
              "budget": {"per_solve_seconds": 6000., "method_wall_seconds": 30000.}}
    post.write_json(output/"payload.json", {"output_dir": output, "runtime_dir": tmp_path/"fake_runtime", "config": config})
    diagnostic = SimpleNamespace(to_dict=lambda: {"status": 0, "objective": 0., "dual_bound": 0., "mip_gap": 0.})
    core = ModuleType("rnj_wzzt.estimation.laminar_l1_milp")
    core._run_milp = lambda variables, rows, **kw: (SimpleNamespace(x=np.array([0.]), fun=0.), diagnostic)
    core.solver_diagnostics_prove_optimality = lambda d: True
    def matrices(n, supports, r, x):
        z = np.zeros((len(supports), n))
        for k, support in enumerate(supports):
            z[k, list(support)] = 1.
        return (z.T*np.asarray(r))@z, (z.T*np.asarray(x))@z
    core.build_matrices_from_atoms = matrices
    calls = []
    def fixed(scenarios, supports, **kwargs):
        calls.append(supports)
        if len(calls) == 2:
            raise post.MethodBudgetExhausted("method_wall_limit_before_next_solve")
        variables = SimpleNamespace(objective=[0.], lower=[0.], upper=[1.], integrality=[0], slices={})
        rows = SimpleNamespace(row_index=[], col_index=[], data=[], lower=[], upper=[])
        core._run_milp(variables, rows, time_limit=6000.)
        return SimpleNamespace(supports=supports, r_values=np.array([.04]*4+[.12, 0.]),
            x_values=np.array([.02]*4+[.08, 0.]), source_h=None, source_response_r=None,
            source_response_x=None, objective=0., diagnostics=diagnostic)
    core.solve_fixed_support_l1 = fixed
    package, estimation, inputs = ModuleType("rnj_wzzt"), ModuleType("rnj_wzzt.estimation"), ModuleType("rnj_wzzt.estimation.unmetered_inputs")
    estimation.laminar_l1_milp, inputs.SourceInputs = core, InputSource
    for name, module in (("rnj_wzzt", package), ("rnj_wzzt.estimation", estimation),
                         ("rnj_wzzt.estimation.laminar_l1_milp", core), ("rnj_wzzt.estimation.unmetered_inputs", inputs)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(post, "activate_runtime", lambda p: None)
    from research_experiments.rnj import prepare_source_ac288_runtime as runtime
    monkeypatch.setattr(runtime, "verify_loaded_modules", lambda p: {})
    assert post.worker(output/"payload.json") == 1
    result = post.read_json(output/"result.json")
    assert len(calls) == 2 and result["steps_completed"] == 1
    assert result["status"] == "incomplete" and result["has_model"] is True
    assert result["resolution_stable"] is False
    assert result["error"]["type"] == "MethodBudgetExhausted"
    previous_event = post.read_json(output/"steps/0000/solver_events.json")[-1]
    assert previous_event["optimality_certified"] is True
    assert post.read_json(output/"model.json")["status"] == "incomplete"


@pytest.mark.skipif(not (post.DEFAULT_RUNTIME/"runtime_manifest.json").exists(), reason="local frozen runtime unavailable")
@pytest.mark.parametrize("source", [False, True])
def test_real_fixed_refit_iterates_and_source_path_is_reoptimized(tmp_path, source):
    raw, baseline, arrays = write_fixture(tmp_path, source=source)
    before = {str(path): post.sha256(path) for folder in (raw, baseline) for path in folder.iterdir()}
    output = tmp_path/"post"
    result = post.run(raw, baseline, output)
    assert result["status"] == "complete", (output/"stderr.log").read_text()+json.dumps(result)
    assert result["resolution_stable"] is True and result["steps_completed"] == 2
    assert result["original_and_copied_input_files_unchanged"] is True
    assert all(post.sha256(path) == digest for path, digest in before.items())
    model = post.read_json(output/"model.json")
    assert model["support_indices"] == [[0], [1], [2], [3], [0, 1, 2]]
    assert model["positive_floor_enforced"] is False and model["source_h_refit_each_step"] is True
    assert model["train_mae_recomputed"] < 1e-9
    if source:
        assert model["source_h"] == [0, 0, 0, 0, 1]  # differs from deliberately wrong saved h
        np.testing.assert_allclose(model["source_g_r"], [.12, .12, .12, 0.], atol=1e-8)
        np.testing.assert_allclose(model["source_g_x"], [.08, .08, .08, 0.], atol=1e-8)
    else:
        assert model["source_h"] is None
    with np.load(output/"model.npz") as saved, np.load(output/"predictions.npz") as prediction:
        np.testing.assert_array_equal(saved["P"], arrays["P"])
        np.testing.assert_allclose(saved["R"], arrays["R"], atol=1e-8)
        np.testing.assert_allclose(prediction["prediction"], arrays["Y"], atol=1e-9)
    for step in range(2):
        directory = output/"steps"/f"{step:04d}"
        events = post.read_json(directory/"solver_events.json")
        assert events and events[0]["optimality_certified"]
        with np.load(directory/events[0]["primal_file"]) as primal:
            assert {"primal", "constraint_row_index", "constraint_col_index", "constraint_data"} <= set(primal.files)
    payload = (output/"payload.json").read_text()
    assert "network" not in payload and "source_bus" not in payload
    origins = post.read_json(output/"runtime_import.json")["module_origins"]
    assert origins and all(Path(p).is_relative_to(post.DEFAULT_RUNTIME) for p in origins.values())
