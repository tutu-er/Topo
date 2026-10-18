"""Saved-state audit must reject independently corrupted physical evidence."""
import json
import shutil

import pandas as pd
import pytest

from research_experiments.rnj.audit_source_ac96 import Audit, audit_variant, read_network
from research_experiments.rnj.prepare_source_ac96 import load_algorithm_inputs, prepare_dataset
from rnj_wzzt.estimation.unmetered_inputs import prepare_source_inputs


@pytest.fixture(scope="module")
def ac96_saved_data(tmp_path_factory):
    directory = tmp_path_factory.mktemp("source-ac96-input")
    # The generator makes the input only. The audit imports none of its
    # helpers, residuals, or diagnostics and works entirely from saved CSVs.
    prepare_dataset(directory, cases=("paper15",))
    return directory / "paper15"


def inspect_condition(case_directory, variant="constant"):
    audit = Audit()
    audit_variant(case_directory / variant, read_network(case_directory / "network"), variant, audit)
    return audit.result()


def test_saved_physical_day_passes_independent_audit(ac96_saved_data):
    for variant in ("no_source", "constant", "intermittent", "smooth"):
        result = inspect_condition(ac96_saved_data, variant)
        assert result["passed"], [item for item in result["checks"] if not item["passed"]]


@pytest.mark.parametrize("relative_path, failed_check", [
    ("truth/ac/branch_loss_p_pu.csv", "branch loss equals sending minus receiving"),
    ("truth/ac/V_bus_mag.csv", "complex branch voltage equation"),
    ("source/amplitudes.csv", "supplied source P matches physical truth"),
])
def test_corrupted_saved_physics_is_rejected(ac96_saved_data, tmp_path, relative_path, failed_check):
    case_copy = tmp_path / "paper15"
    shutil.copytree(ac96_saved_data / "network", case_copy / "network")
    shutil.copytree(ac96_saved_data / "constant", case_copy / "constant")
    path = case_copy / "constant" / relative_path
    table = pd.read_csv(path, index_col=0)
    table.iloc[0, 0] += .001
    table.to_csv(path, index_label="t", float_format="%.17g")
    result = inspect_condition(case_copy)
    failed = [item["check"] for item in result["checks"] if not item["passed"]]
    assert not result["passed"]
    assert any(failed_check in name for name in failed), failed


def test_source_position_leaked_into_estimator_input_is_rejected(ac96_saved_data, tmp_path):
    case_copy = tmp_path / "paper15"
    shutil.copytree(ac96_saved_data / "network", case_copy / "network")
    shutil.copytree(ac96_saved_data / "constant", case_copy / "constant")
    path = case_copy / "constant" / "source" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["source_bus_id"] = 2
    path.write_text(json.dumps(metadata), encoding="utf-8")
    result = inspect_condition(case_copy)
    failed = [item["check"] for item in result["checks"] if not item["passed"]]
    assert not result["passed"]
    assert any("estimator metadata excludes truth and source location" in name for name in failed)


def test_algorithm_loader_works_without_network_or_truth(ac96_saved_data, tmp_path):
    inputs = tmp_path / "algorithm-inputs"
    for folder in ("observed", "source"):
        shutil.copytree(ac96_saved_data / "smooth" / folder, inputs / folder)
    assert {path.name for path in inputs.iterdir()} == {"observed", "source"}
    scenarios, amplitudes = load_algorithm_inputs(inputs)
    prepared = prepare_source_inputs(scenarios, amplitudes)
    assert len(scenarios) == 1
    assert scenarios[0]["P_terminal"].shape[0] == 96
    assert scenarios[0]["root_observation"] == "noisy"
    assert scenarios[0]["scenario_settings"]["root_meter_noise_rel"] == .0002
    assert prepared is not None
    assert len(prepared.p) == 1 and prepared.p[0].shape == (96,)
    assert prepared.q[0].shape == (96,)
