"""Fixed Source design and isolated loader/generator contracts; no estimator."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj import prepare_source_advantage as data


@dataclass(frozen=True)
class AlternateRuntimeSource:
    p_pu: pd.Series
    q_pu: pd.Series
    provenance: str


def _fixture(tmp_path, variant="rank2_source"):
    root = tmp_path/"replicate_2"/variant
    p_source, q_source = data.source_profile(19.1)
    index = pd.RangeIndex(96, name="t")
    for scene in range(3):
        directory = root/f"scenario_{scene}"
        observed = directory/"observed"
        observed.mkdir(parents=True)
        for key in ("P_terminal", "Q_terminal", "V_terminal", "drop_target"):
            pd.DataFrame(np.full((96, 2), .01+scene*.001), index=index, columns=[101, 102]).to_csv(observed/f"{key}.csv", float_format="%.17g")
        for key in ("root_voltage", "P0_measured", "Q0_measured"):
            pd.Series(np.ones(96), index=index).to_csv(observed/f"{key}.csv")
        data.write_json(observed/"metadata.json", {"root_observation": "noisy", "scenario_settings": {"replicate": 2, "scenario_index": scene}})
        (directory/"source").mkdir()
        pd.DataFrame({"P_pu": p_source/200, "Q_pu": q_source/200}, index=index).to_csv(directory/"source/amplitudes.csv", float_format="%.17g")
    return root


def test_exact_fixed_source_formula_positive_rank_two_and_energy():
    p, q = data.source_profile(.25*76.4)
    t = np.arange(96)
    np.testing.assert_allclose(p, 19.1*(1+.65*np.sin(2*np.pi*7*t/96)+.2*np.cos(2*np.pi*13*t/96)), rtol=1e-14)
    np.testing.assert_allclose(q, .35*19.1*(1+.7*np.cos(2*np.pi*11*t/96+.4)), rtol=1e-14)
    assert p.min() > 0 and q.min() > 0 and np.linalg.matrix_rank(np.column_stack((p, q))) == 2
    assert p.mean() == pytest.approx(19.1)
    assert p.sum()*.25*3 == pytest.approx(1375.2)
    assert q.sum()*.25*3 == pytest.approx(481.32)
    with pytest.raises(ValueError):
        data.source_profile(0)


def test_loader_uses_only_observed_amplitudes_and_exact_source_runtime_type(tmp_path):
    root = _fixture(tmp_path)
    before = {k: getattr(v, "__file__", None) for k, v in sys.modules.items() if k == "rnj_wzzt" or k.startswith("rnj_wzzt.")}
    scenarios, sources, metadata = data.load_algorithm_inputs(root, source_inputs_class=AlternateRuntimeSource)
    after = {k: getattr(v, "__file__", None) for k, v in sys.modules.items() if k == "rnj_wzzt" or k.startswith("rnj_wzzt.")}
    assert before == after
    assert len(scenarios) == len(sources) == 3
    assert all(type(s) is AlternateRuntimeSource for s in sources)
    assert metadata["replicate"] == 2 and metadata["total_training_samples"] == 288 and metadata["validation_samples"] == 0
    assert metadata["terminal_labels"] == [101, 102]
    assert [s["row_range"] for s in metadata["scenarios"]] == [[0, 96], [96, 192], [192, 288]]
    assert len(metadata["input_file_sha256"]) == 27
    assert not any(token in json.dumps(metadata) for token in ("source_bus", "truth", "network"))


def test_loader_rejects_misalignment_rank_mismatch_and_changed_curve(tmp_path):
    root = _fixture(tmp_path)
    path = root/"scenario_2/source/amplitudes.csv"
    frame = pd.read_csv(path, index_col="t", float_precision="round_trip")
    frame.iloc[0, 0] *= 1.1
    frame.to_csv(path, float_format="%.17g")
    with pytest.raises(AssertionError):
        data.load_algorithm_inputs(root, source_inputs_class=AlternateRuntimeSource)
    frame.iloc[:, :] = 0.
    frame.to_csv(path)
    with pytest.raises(ValueError, match="rank"):
        data.load_algorithm_inputs(root, source_inputs_class=AlternateRuntimeSource)
    frame = frame.iloc[:-1]
    frame.to_csv(path)
    with pytest.raises(ValueError, match="aligned"):
        data.load_algorithm_inputs(root, source_inputs_class=AlternateRuntimeSource)


def test_overwrite_guard_precedes_runtime_activation(tmp_path, monkeypatch):
    root = tmp_path/"existing"
    root.mkdir()
    sentinel = root/"keep.txt"
    sentinel.write_text("keep")
    monkeypatch.setattr(data, "activate_generation_runtime", lambda p: pytest.fail("runtime must not be activated"))
    with pytest.raises(FileExistsError):
        data.prepare_dataset(root)
    assert sentinel.read_text() == "keep"


@pytest.mark.skipif(not (data.DEFAULT_RUNTIME/"runtime_manifest.json").exists() or not data.DEFAULT_ORIGINAL.exists(),
                    reason="frozen runtime/original local network unavailable")
def test_real_frozen_runtime_generation_preserves_all_eighteen_conditions(tmp_path):
    """Only temporary AC data; a fresh subprocess prevents runtime contamination."""
    output = tmp_path/"dataset"
    before = data.inventory(data.DEFAULT_ORIGINAL)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-B", "-m", "research_experiments.rnj.prepare_source_advantage",
        "--output-dir", str(output)], cwd=data.ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr
    manifest = data.read_json(output/"manifest.json")
    assert manifest["attempted"] == manifest["completed"] == 18 and manifest["failed"] == 0
    assert manifest["settings"]["AC_snapshots_expected"] == 1728 and manifest["estimation_performed"] is False
    assert data.inventory(data.DEFAULT_ORIGINAL) == before
    assert all(data.sha256(output/path) == digest for path, digest in manifest["file_sha256"].items())
    origins = data.read_json(output/"runtime_import.json")["module_origins"]
    assert origins and all(Path(path).is_relative_to(data.DEFAULT_RUNTIME) for path in origins.values())
    original_network = pd.read_csv(data.DEFAULT_ORIGINAL/"branches.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(pd.read_csv(output/data.CASE/"network/branches.csv", float_precision="round_trip"), original_network)
    summary = pd.read_csv(output/"summary.csv")
    assert len(summary) == 18 and (summary["status"] == "completed").all()
    assert summary["AC_converged_samples"].sum() == 1728
    assert max(summary["p_balance_max_abs_pu"].max(), summary["q_balance_max_abs_pu"].max()) < 1e-7
    fixed_curve = None
    for rep in range(3):
        for variant in data.VARIANTS:
            original = output/data.CASE/f"replicate_{rep}"/variant
            worker = tmp_path/"worker"/f"replicate_{rep}"/variant
            for scene in range(3):
                source = original/f"scenario_{scene}"
                target = worker/f"scenario_{scene}"
                shutil.copytree(source/"observed", target/"observed")
                (target/"source").mkdir()
                shutil.copyfile(source/"source/amplitudes.csv", target/"source/amplitudes.csv")
                truth = data.read_json(source/"truth/metadata.json")
                assert truth["source_bus_id"] == (5 if variant == "rank2_source" else None)
                assert "source_bus" not in (source/"observed/metadata.json").read_text()
                if variant == "rank2_source":
                    curve = (source/"source/amplitudes.csv").read_bytes()
                    if fixed_curve is None:
                        fixed_curve = curve
                    assert curve == fixed_curve
            scenarios, sources, metadata = data.load_algorithm_inputs(worker, source_inputs_class=AlternateRuntimeSource)
            assert len(scenarios) == len(sources) == 3 and metadata["replicate"] == rep
            for scene, scenario in enumerate(scenarios):
                settings = scenario["scenario_settings"]
                assert settings["scenario_index"] == scene and settings["replicate"] == rep
                for key, value in data.NOISE.items():
                    assert settings[key] == value
                assert scenario["P_terminal"].shape == (96, 11)
