"""Contracts for the measured-only entry, without changing historical tests."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from theft_wzzt.theft.identified_tree import load_identified
from theft_wzzt.theft.theft_simulation import simulate_theft_scenarios

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("simple_entry_test", ROOT / "detect_simple.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def measured_fixture():
    _, scenario = simulate_theft_scenarios("paper15", replicate=30)
    measured = {k: scenario[k] for k in (*entry.MATRIX_CHANNELS, *entry.SERIES_CHANNELS, "scenario_settings")}
    return measured, load_identified(ROOT / "outputs/theft/identified_tree.json")


def test_measured_only_prepare_aligns_all_channels_and_rejects_bad_window():
    measured, tree = measured_fixture()
    _, a, _, env, signature = entry.prepare(measured, tree, (44, 52))
    reordered = {k: v.iloc[:, ::-1] if k in entry.MATRIX_CHANNELS else v for k, v in measured.items()}
    _, b, _, _, permuted_signature = entry.prepare(reordered, tree, (44, 52))
    assert signature == permuted_signature
    for name in ("p", "q", "y", "amplitude", "extra_q", "balance"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name))
    np.testing.assert_allclose(env.point, a.amplitude)
    assert not any("true" in k or "loss" in k or "theft" in k for k in measured)
    with pytest.raises(ValueError, match="window"):
        entry.prepare(measured, tree, (52, 44))
    broken = dict(measured, drop_target=measured["drop_target"].iloc[::-1])
    with pytest.raises(ValueError, match="index"):
        entry.prepare(broken, tree, (44, 52))


def test_rank_rule_sample_resolution_and_protocol_mismatch():
    calibration = {"protocol_signature": "same", "null_gains": list(range(20))}
    # q95=18.05 would alarm at 18.5; rank=2/21 deliberately does not.
    assert entry.decision(18.5, calibration, "same")["alarm"] is False
    assert entry.decision(20., calibration, "same")["alarm"] is True
    small = dict(calibration, null_gains=list(range(5)))
    verdict = entry.decision(100., small, "same")
    assert verdict["alarm"] is None and verdict["status"] == "insufficient_calibration"
    assert entry.decision(100., None, "same")["status"] == "uncalibrated"
    with pytest.raises(ValueError, match="protocol"):
        entry.decision(20., calibration, "different")
    with pytest.raises(ValueError, match="one-dimensional"):
        entry.decision(20., dict(calibration, null_gains=[[1, 2]]), "same")


def test_missing_calibration_gates_claims_and_invalid_calibration_precedes_fit(monkeypatch):
    measured, tree = measured_fixture()
    calls = []
    def fake_compare(*args, **kwargs):
        calls.append(kwargs)
        return ({"gain": 6., "h0": {"voltage_loss": 5., "balance_loss": 8.},
                 "h1": {"voltage_loss": 4., "balance_loss": 3.}}, None,
                SimpleNamespace(selected_locations=lambda t: [t.candidates[0]] * 2))
    monkeypatch.setattr(entry, "compare_h0_h1", fake_compare)
    result = entry.detect(measured, tree, window=(44, 46))
    assert len(calls) == 1 and result["decision"]["alarm"] is None
    assert result["reported_locations_by_time"] is None
    assert result["candidate_locations_by_time"]
    assert result["gain"] == result["gain_voltage"] + result["gain_balance"]
    with pytest.raises(ValueError, match="protocol"):
        entry.detect(measured, tree, window=(44, 46), calibration={"protocol_signature": "wrong"})
    assert len(calls) == 1
