"""Review checks and isolated real worker smoke tests; never launch the suite."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace
import subprocess

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj import run_rnj_blocks96 as driver


class KnownSource:
    def __init__(self, p_pu, q_pu, provenance):
        self.p_pu, self.q_pu, self.provenance = p_pu, q_pu, provenance


def input_fixture(directory, active):
    rng = np.random.default_rng(85)
    labels, t = [11, 12, 13, 14], pd.Index(range(96), name="t")
    p, q = rng.uniform(.001, .025, (96, 4)), rng.uniform(.001, .015, (96, 4))
    r, x = np.eye(4)*.03, np.eye(4)*.02
    r[:2, :2] += .4; x[:2, :2] += .2
    r[2:, 2:] += .25; x[2:, 2:] += .12
    up = .025+.01*np.sin(np.arange(96)/7) if active else np.zeros(96)
    uq = .008*np.cos(np.arange(96)/9) if active else np.zeros(96)
    gr, gx = np.array([.4, .4, 0, 0]), np.array([.2, .2, 0, 0])
    y = p@r+q@x+up[:, None]*gr+uq[:, None]*gx
    directory.mkdir(parents=True, exist_ok=True)
    for key, value in {"P_terminal": p, "Q_terminal": q, "V_terminal": np.sqrt(1-y), "drop_target": y}.items():
        target = directory/"observed"/f"{key}.csv"
        target.parent.mkdir(exist_ok=True)
        pd.DataFrame(value, index=t, columns=labels).to_csv(target, float_format="%.17g")
    for key, value in {"root_voltage": np.ones(96), "P0_measured": p.sum(axis=1)+up,
                       "Q0_measured": q.sum(axis=1)+uq}.items():
        pd.Series(value, index=t, name=key).to_csv(directory/"observed"/f"{key}.csv", float_format="%.17g")
    driver.write_json(directory/"observed/metadata.json", {"root_observation": "measured", "scenario_settings": {"fixture": "linear96"}})
    (directory/"source").mkdir()
    pd.DataFrame({"P_pu": up, "Q_pu": uq}, index=t).to_csv(directory/"source/amplitudes.csv", float_format="%.17g")
    driver.write_json(directory/"source/metadata.json", {"information": "known amplitudes; unknown location"})
    hashes = {p.relative_to(directory).as_posix(): driver.sha256(p) for p in directory.rglob("*") if p.is_file()}
    return {"R": r, "X": x, "P": p, "Q": q, "Y": y, "P_source": up, "Q_source": uq,
            "gR": gr if active else np.zeros(4), "gX": gx if active else np.zeros(4)}, hashes


def test_loader_preserves_exact_96_rows_and_reads_only_permitted_channels(tmp_path):
    expected, hashes = input_fixture(tmp_path/"inputs", True)
    driver.verify_inputs(tmp_path/"inputs", hashes)
    scenarios, sources = driver.load_algorithm_inputs(tmp_path/"inputs", KnownSource)
    arrays, use = driver.input_arrays(scenarios, sources)
    for key in ("P", "Q", "Y", "P_source", "Q_source"):
        np.testing.assert_array_equal(arrays[key], expected[key])
    assert use["training_samples"] == 96 and use["validation_samples"] == 0
    np.testing.assert_array_equal(arrays["row_position"], np.arange(96))
    np.testing.assert_array_equal(arrays["row_scenario"], np.zeros(96))
    sources[0].p_pu = sources[0].p_pu.iloc[::-1]
    with pytest.raises(ValueError, match="source row alignment"):
        driver.input_arrays(scenarios, sources)
    (tmp_path/"inputs/truth").mkdir()
    (tmp_path/"inputs/truth/location.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="whitelist"):
        driver.verify_inputs(tmp_path/"inputs", hashes)


def test_all_initializer_variants_preserve_selected_pairs_and_share_only_declared_z_groups():
    labels, full, selected = [11, 12, 13, 14], [{11, 12}, {13, 14}], [{11, 12}]
    expected = [(0,), (1,), (2,), (3,), (0, 1)]
    assert driver.initialization(labels, full, selected, "RNJ+WZZT") == (expected, [(0, 1)])
    assert driver.initialization(labels, full, selected, "RNJ+WZZT_ungrouped") == (expected, [])
    assert driver.initialization(labels, full, selected, "RNJ+WZZT_tree_init") == (expected+[(2, 3)], [(0, 1)])
    assert driver.initialization(labels, full, selected, "WZZT") == (expected[:4], [])
    assert driver.core_parameters(4)["max_atoms"] == 7
    assert driver.SCREEN_PARAMETERS["minimum_block_size"] == 2
    assert driver.SCREEN_PARAMETERS["maximum_subclade_frequency"] == .15


def test_worker_command_and_closure_do_not_include_live_core(tmp_path):
    command = driver.worker_command(tmp_path, tmp_path/"payload.json")
    assert command[:3] == [driver.sys.executable, "-B", "-c"]
    assert command[4] == str((tmp_path/"source_snapshot").resolve())
    assert command[-2:] == ["--worker", str(tmp_path/"payload.json")]
    closure = driver.source_closure()
    assert not any("rnj_wzzt_core" in path or path.startswith("rnj_wzzt/") for path in closure)
    assert "research_experiments/rnj/select_rnj_blocks96.py" in closure
    assert "research_experiments/rnj/run_rnj_blocks96.py" in closure
    # Package-level ``from ...rnj import audit_source_ac288 as evidence`` is a
    # real transitive dependency of the baseline helper's pure audit imports.
    assert "research_experiments/rnj/audit_source_ac288.py" in closure
    assert "research_experiments/runtime_snapshots/source_ac288_72af_20261018.zip" in closure
    assert all(driver.sha256(driver.ROOT/path) == checksum for path, checksum in closure.items())


def test_worker_rejects_protocol_or_source_fingerprint_change_before_loading_runtime(tmp_path, monkeypatch):
    directory = tmp_path/"jobs/tiny"; directory.mkdir(parents=True)
    config = {"source_sha256": {}}
    driver.write_json(tmp_path/"protocol.json", config)
    payload = {"job_dir": directory, "job": {}, "config": config, "protocol_sha256": "incorrect"}
    driver.write_json(directory/"payload.json", payload)
    monkeypatch.setattr(driver, "activate_runtime", lambda path: pytest.fail("fingerprint failure reached runtime"))
    with pytest.raises(ValueError, match="protocol fingerprint"):
        driver.worker(directory/"payload.json")
    config["source_sha256"] = {"research_experiments/rnj/select_rnj_blocks96.py": "incorrect"}
    driver.write_json(tmp_path/"protocol.json", config)
    payload.update(config=config, protocol_sha256=driver.sha256(tmp_path/"protocol.json"))
    driver.write_json(directory/"payload.json", payload)
    with pytest.raises(ValueError, match="source closure fingerprint"):
        driver.worker(directory/"payload.json")
    assert not (directory/"execution.json").exists()


def test_subset_controller_keeps_64_denominator_and_does_not_mark_suite_complete(tmp_path, monkeypatch):
    job = driver.job_catalog()[0]
    driver.write_json(tmp_path/"jobs"/job["job_id"]/"result.json", {"status": "complete", "has_model": True,
        "metrics": {"clade_f1": .5}, "compute_seconds": .1})
    monkeypatch.setattr(driver, "controller_lock", lambda output: nullcontext())
    monkeypatch.setattr(driver, "prepare", lambda args: (tmp_path, {}, []))
    args = SimpleNamespace(output_dir=tmp_path, prepare_only=False, only_job=[job["job_id"]], workers=1)
    assert driver.run(args) == 0
    completion = driver.read_json(tmp_path/"completion.json")
    assert completion["expected"] == 64 and completion["finished"] == 1
    assert completion["status_counts"] == {"complete": 1, "not_run": 63}
    assert not (tmp_path/"execution_complete.json").exists()


def test_outer_timeout_recovers_atomic_last_certified_json_npz_pair(tmp_path):
    arrays = {"P": np.zeros((96, 2)), "Q": np.zeros((96, 2)), "Y": np.zeros((96, 2)),
              "P_source": np.zeros(96), "Q_source": np.zeros(96)}
    model = {"R": np.eye(2), "X": np.eye(2)*2, "path": [{"token": "certified"}],
             "selected_path_index": 0, "source_g_r": None, "training_rows": 96}
    driver.save_checkpoint(tmp_path, model, arrays, driver.perf_counter(), 1)
    # Simulate a kill between ordinary NPZ and JSON updates.
    np.savez(tmp_path/"model.npz", R=np.eye(2)*999)
    driver.recover_process(tmp_path, driver.job_catalog()[0], "protocol", 91., killed=True, exitcode=-9)
    result, recovered = driver.read_json(tmp_path/"result.json"), driver.read_json(tmp_path/"model.json")
    assert result["status"] == "incomplete" and not result["search_complete"]
    assert result["retained_last_certified_checkpoint"]
    assert recovered["path"] == [{"token": "certified"}] and recovered["selected_path_index"] == 0
    with np.load(tmp_path/"model.npz") as archive:
        np.testing.assert_array_equal(archive["R"], np.eye(2))
        assert archive["P"].shape == (96, 2)


@pytest.fixture(scope="module")
def frozen_snapshot(tmp_path_factory):
    runtime = driver.ROOT/"outputs/source_ac288_runtime_20261018"
    if not (runtime/"runtime_manifest.json").exists(): pytest.skip("frozen runtime not prepared")
    output = tmp_path_factory.mktemp("frozen_worker")
    inventory = driver.source_closure()
    for relative, checksum in inventory.items():
        target = output/"source_snapshot"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(driver.ROOT/relative, target)
        assert driver.sha256(target) == checksum
    return output, runtime, inventory


@pytest.mark.parametrize("active", [False, True], ids=["noSource", "Source"])
def test_real_frozen_4terminal_96_grouped_floor_greedy_worker(tmp_path, frozen_snapshot, active):
    output, runtime, inventory = frozen_snapshot
    expected, hashes = input_fixture(tmp_path/"inputs", active)
    job = deepcopy(driver.job_catalog()[0])
    job.update(case="tiny4", variant="Source" if active else "noSource", method="RNJ+WZZT",
               job_id="tiny4__"+("Source" if active else "noSource"), per_solve_seconds=30., method_wall_seconds=90.)
    # One selected pair leaves the other true pair for a real greedy extension.
    job["screen_parameters"].update(bootstrap_replicates=8, maximum_candidate_count=1)
    directory = tmp_path/"jobs"/"tiny"; directory.mkdir(parents=True)
    config = {"solver": "gurobi", "timing_environment": {"scope": "tiny smoke only"}, "source_sha256": inventory}
    driver.write_json(tmp_path/"protocol.json", config)
    payload = {"job_dir": directory, "job": job, "runtime_dir": runtime, "input_dir": tmp_path/"inputs",
               "input_sha256": hashes, "protocol_sha256": driver.sha256(tmp_path/"protocol.json"), "config": config}
    driver.write_json(directory/"payload.json", payload)
    process = subprocess.run(driver.worker_command(output, directory/"payload.json"), cwd=output/"source_snapshot",
                             capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout+process.stderr
    result, model = driver.read_json(directory/"result.json"), driver.read_json(directory/"model.json")
    assert result["status"] == "complete", result.get("error")
    assert model["has_model"] and model["training_rows"] == 96 and model["validation_supplied"] is False
    assert model["initial_supports"] == [[0], [1], [2], [3], [0, 1]]
    assert model["requested_terminal_groups"] == [[0, 1]]
    assert model["terminal_groups"] == [[0, 1], [2], [3]]
    assert len(model["path"]) >= 2 and [2, 3] in model["support_indices"]
    assert model["selected_path_index"] == model["last_path_index"] == len(model["path"])-1
    assert model["core_parameters"]["max_atoms"] == 7
    with np.load(directory/"model.npz") as arrays:
        for key in ("P", "Q", "Y", "P_source", "Q_source"): np.testing.assert_array_equal(arrays[key], expected[key])
        np.testing.assert_allclose(arrays["R"], expected["R"], atol=2e-6)
        np.testing.assert_allclose(arrays["X"], expected["X"], atol=2e-6)
    if active:
        assert len(model["source_h"]) == len(model["support_indices"])
        np.testing.assert_allclose(model["source_g_r"], expected["gR"], atol=2e-6)
        np.testing.assert_allclose(model["source_g_x"], expected["gX"], atol=2e-6)
    else: assert model["source_h"] is None
    extensions = [e for e in model["solver_events"] if "z" in e["variable_slices"]]
    assert extensions
    for event in model["solver_events"]:
        assert event["optimality_certified"] and event["requested_time_limit_seconds"] == 30.
        assert 0 < event["time_limit_seconds"] <= min(30., event["remaining_wall_before_solve"])
        assert event["old_support_indices"][:4] == [[0], [1], [2], [3]]
        assert [0, 1] in event["old_support_indices"]
    for event in extensions:
        start, end = event["variable_slices"]["z"]
        assert end-start == 3  # shared binary for pair [0,1], versus four individual terminals
        assert event["terminal_partition"] == [[0, 1], [2], [3]]
        assert event["terminal_z_columns"][0] == event["terminal_z_columns"][1]
        assert len(set(event["terminal_z_columns"])) == 3
        with np.load(directory/event["primal_file"]) as raw:
            np.testing.assert_array_equal(raw["integrality"][start:end], np.ones(3))
            assert np.all((raw["lower_bounds"][start:end] == 0) & (raw["upper_bounds"][start:end] == 1))
    # Recompute primal objective, bounds, integers and original COO rows rather
    # than accepting only the observer's boolean certificate field.
    from research_experiments.rnj.audit_rx75_source_comparison import Audit, audit_floor_domain, evidence, optimality_certificate
    audit = Audit()
    for event in model["solver_events"]:
        assert optimality_certificate(event)
        with np.load(directory/event["primal_file"]) as raw:
            evidence.audit_raw_primal(event, raw, audit, "tiny original model")
            audit_floor_domain(event, raw, 4, model["terminal_groups"], model["rx75_calibration"], audit, "tiny floor")
    assert audit.result()["passed"], [r for r in audit.checks if not r["passed"]]
    imports = driver.read_json(directory/"runtime_import.json")
    assert all(Path(path).resolve().is_relative_to(runtime) for path in imports["module_origins"].values())
    assert all(Path(path).resolve().is_relative_to(output/"source_snapshot") for path in imports["experiment_module_origins"].values())
    screen = driver.read_json(directory/"rnj_screen.json")
    assert screen["selected_support_labels"] == [[11, 12]]
    assert len(screen["diagnostics"]["bootstrap_records"]) == 8
