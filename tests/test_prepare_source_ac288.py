"""Check three-scenario inputs, exact baseline reuse and runtime source typing."""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj.prepare_source_ac288 import (
    DEFAULT_BASELINE, VARIANTS, _file_inventory, load_algorithm_inputs, prepare_dataset,
)


@dataclass(frozen=True)
class AlternateRuntimeSource:
    p_pu: pd.Series
    q_pu: pd.Series
    provenance: str


def _loader_fixture(tmp_path, *, variant="constant"):
    directory = tmp_path / variant
    columns = [301, 302]
    for scene in range(3):
        scenario = directory / f"scenario_{scene}"
        observed = scenario / "observed"
        observed.mkdir(parents=True)
        index = pd.RangeIndex(96, name="t")
        value = np.arange(192, dtype=float).reshape(96, 2) * .00001 + .01 + scene * .001
        for key, multiplier in (("P_terminal", 1.), ("Q_terminal", .4),
                                 ("V_terminal", 2.), ("drop_target", .5)):
            pd.DataFrame(value * multiplier, index=index, columns=columns).to_csv(observed / f"{key}.csv", float_format="%.17g")
        for key in ("root_voltage", "P0_measured", "Q0_measured"):
            pd.DataFrame({key: np.ones(96)}, index=index).to_csv(observed / f"{key}.csv")
        (observed / "metadata.json").write_text(json.dumps({"root_observation": "noisy",
            "scenario_settings": {"scenario_index": scene, "profile_scenario": f"profile_{scene}"}}), encoding="utf-8")
        (scenario / "source").mkdir()
        pd.DataFrame({"P_pu": np.full(96, .01), "Q_pu": np.full(96, .004)}, index=index).to_csv(
            scenario / "source/amplitudes.csv", float_format="%.17g")
    return directory


def test_loader_needs_only_observations_amplitudes_and_keeps_runtime_exact_type(tmp_path):
    variant = _loader_fixture(tmp_path)
    scenarios, sources, metadata = load_algorithm_inputs(variant, source_inputs_class=AlternateRuntimeSource)
    assert len(scenarios) == len(sources) == 3
    assert all(type(source) is AlternateRuntimeSource for source in sources)
    assert metadata["total_training_samples"] == 288 and metadata["validation_samples"] == 0
    assert metadata["scenario_order"] == [0, 1, 2]
    assert [s["concatenated_row_range"] for s in metadata["scenarios"]] == [[0, 96], [96, 192], [192, 288]]
    assert all(s["P_terminal"].shape == (96, 2) for s in scenarios)
    assert list(scenarios[0]["P_terminal"].columns) == [301, 302]
    assert len(metadata["input_file_sha256"]) == 27
    assert not any(word in json.dumps(metadata) for word in ("source_bus", "truth", "network", "diagnostics"))
    for scene, scenario in enumerate(scenarios):
        reference = pd.read_csv(variant / f"scenario_{scene}/observed/P_terminal.csv", index_col="t", float_precision="round_trip")
        np.testing.assert_array_equal(scenario["P_terminal"].to_numpy(), reference.to_numpy())
        assert sources[scene].p_pu.index.equals(scenario["P_terminal"].index)


def test_loader_rejects_unaligned_or_different_source_curves(tmp_path):
    variant = _loader_fixture(tmp_path)
    path = variant / "scenario_2/source/amplitudes.csv"
    frame = pd.read_csv(path, index_col="t", float_precision="round_trip")
    frame.iloc[0, 0] *= 2
    frame.to_csv(path, float_format="%.17g")
    with pytest.raises(ValueError, match="same known source"):
        load_algorithm_inputs(variant, source_inputs_class=AlternateRuntimeSource)
    frame = frame.iloc[:-1]
    frame.to_csv(path, float_format="%.17g")
    with pytest.raises(ValueError, match="aligned"):
        load_algorithm_inputs(variant, source_inputs_class=AlternateRuntimeSource)


def test_loader_rejects_missing_observed_point_and_unknown_variant(tmp_path):
    variant = _loader_fixture(tmp_path)
    path = variant / "scenario_1/observed/P_terminal.csv"
    frame = pd.read_csv(path, index_col="t").iloc[:-1]
    frame.to_csv(path)
    with pytest.raises(ValueError, match="96"):
        load_algorithm_inputs(variant, source_inputs_class=AlternateRuntimeSource)
    with pytest.raises(ValueError, match="variant_dir"):
        load_algorithm_inputs(tmp_path / "another_network", source_inputs_class=AlternateRuntimeSource)


def test_preparer_never_overwrites_an_existing_output(tmp_path):
    output = tmp_path / "occupied"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing result", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_dataset(output)
    assert sentinel.read_text(encoding="utf-8") == "existing result"


@pytest.mark.skipif(not DEFAULT_BASELINE.is_dir(), reason="existing local AC96 dataset unavailable")
def test_actual_pengwah_three_days_reuse_baseline_and_all_ac_converge(tmp_path):
    before = _file_inventory(DEFAULT_BASELINE)
    output = tmp_path / "ac288"
    manifest = prepare_dataset(output)
    assert manifest["attempted"] == manifest["completed"] == 12 and manifest["failed"] == 0
    assert manifest["scenario_0_reused"] == 4 and manifest["newly_simulated"] == 8
    assert manifest["all_original_AC96_files_unchanged"] is True
    assert before == _file_inventory(DEFAULT_BASELINE)
    assert sorted(path.name for path in output.iterdir() if path.is_dir()) == ["pengwah18"]
    for variant in VARIANTS:
        assert _file_inventory(output / "pengwah18" / variant / "scenario_0") == _file_inventory(DEFAULT_BASELINE / variant)
        worker_copy = tmp_path / "worker_inputs" / variant
        for scene in range(3):
            source = output / "pengwah18" / variant / f"scenario_{scene}"
            copied = worker_copy / f"scenario_{scene}"
            shutil.copytree(source / "observed", copied / "observed")
            (copied / "source").mkdir()
            shutil.copyfile(source / "source/amplitudes.csv", copied / "source/amplitudes.csv")
            assert (source / "source/amplitudes.csv").read_bytes() == (DEFAULT_BASELINE / variant / "source/amplitudes.csv").read_bytes()
        scenarios, sources, metadata = load_algorithm_inputs(worker_copy, source_inputs_class=AlternateRuntimeSource)
        assert metadata["total_training_samples"] == sum(len(s["P_terminal"]) for s in scenarios) == 288
        assert all(type(s) is AlternateRuntimeSource for s in sources)
        for scene in (1, 2):
            settings = scenarios[scene]["scenario_settings"]
            assert settings["scenario_index"] == scene and settings["replicate"] == 0
            assert settings["pq_noise_rel"] == .005 and settings["v_noise_rel"] == .0002
    summary = pd.read_csv(output / "summary.csv")
    assert len(summary) == 12 and summary["all_ac_samples_converged"].all()
    assert summary["p_balance_max_abs_pu"].max() < 1e-7
    assert summary["q_balance_max_abs_pu"].max() < 1e-7
