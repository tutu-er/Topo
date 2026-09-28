"""Independent contracts for physical scenarios and root-meter observations."""

import json

import numpy as np
import pandas as pd
import pytest

from rnj_wzzt.data.paper_style_case_bank import CASE_BUILDERS, build_case_bank_resource_assignment
from rnj_wzzt.data.paper_style_terminal_lv import build_paper_style_resource_assignment
from rnj_wzzt.estimation.preprocessing import squared_voltage_drop_from_observed_root
from rnj_wzzt.models.ac_powerflow import solve_ac_power_flow_timeseries
from rnj_wzzt.models.lin_distflow import build_reduced_sensitivity_matrices
from rnj_wzzt.scenario.profiles import _build_terminal_profiles
from rnj_wzzt.scenario.settings import ROOT_OBSERVATIONS, SCENARIO_SUITES, resolve_scenario_settings
from rnj_wzzt.scenario.simulation import _simulate_pool


LEGACY_SCALES = {"paper15": 5.0, "soumalas11": 5.2, "flynn16": 4.8, "pengwah18": 4.5}
ARRAY_KEYS = (
    "P_true", "Q_true", "V_terminal_true", "root_voltage_true",
    "P_terminal", "Q_terminal", "V_terminal", "drop_target",
)


def _terminals(net):
    return net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()


def _legacy_oracle(case_key, samples, replicate, pq_noise, v_noise):
    """Reproduce the pre-suite simulator directly, without the new resolver."""
    net = CASE_BUILDERS[case_key]()
    assignment = (build_paper_style_resource_assignment(net) if case_key == "paper15"
                  else build_case_bank_resource_assignment(net))
    terminals = _terminals(net)
    result = []
    for seed in [42, 7, 21]:
        shifted_seed = seed + 100_003 * replicate
        p, q, _ = _build_terminal_profiles(net, assignment, samples, shifted_seed, profile_scenario="default")
        root = pd.Series(
            1.02 + np.random.default_rng(shifted_seed + 1000).normal(0.0, 0.0008, size=samples),
            index=p.index, name="V_root_true",
        )
        ac = solve_ac_power_flow_timeseries(net, p, q, v_root=root, max_iter=100, tol=1e-10)
        assert bool(ac["converged"].all())
        voltage = ac["V_bus_mag"].loc[:, terminals]
        rng = np.random.default_rng(shifted_seed + 2000)
        measured_p = p + rng.normal(0.0, pq_noise * np.maximum(np.abs(p), 1e-12), size=p.shape)
        measured_q = q + rng.normal(0.0, pq_noise * np.maximum(np.abs(q), 1e-12), size=q.shape)
        measured_v = voltage + rng.normal(0.0, v_noise * np.maximum(np.abs(voltage), 1e-12), size=voltage.shape)
        result.append({
            "name": f"oneday_96pts_default_seed{seed}_rep{replicate}",
            "P_true": p, "Q_true": q, "V_terminal_true": voltage, "root_voltage_true": root,
            "P_terminal": measured_p, "Q_terminal": measured_q, "V_terminal": measured_v,
            "root_voltage": root,
            "drop_target": squared_voltage_drop_from_observed_root(measured_v, root),
        })
    return net, result


def test_settings_registry_defaults_and_json_roundtrip():
    assert tuple(SCENARIO_SUITES) == ("legacy", "reference", "weak_root", "strong_root", "tap_step")
    assert tuple(ROOT_OBSERVATIONS) == ("exact", "noisy")
    legacy = resolve_scenario_settings()
    reference = resolve_scenario_settings("reference")
    assert legacy["scenario_suite"] == "legacy"
    assert legacy["root_observation"] == "exact"
    assert legacy["root_process"] == "iid"
    assert legacy["root_sigma"] == 0.0008
    assert legacy["root_meter_noise_rel"] == 0.0
    assert legacy["impedance_scale"] is None
    assert legacy["profiles"] == ["default"] * 3
    assert reference["profiles"] == ["default", "cloudy_variable_pv", "evening_peak_high_load"]
    assert reference["seeds"] == [42, 7, 21]
    assert reference["root_process"] == "correlated"
    assert reference["root_sigma"] == 0.003
    assert reference["root_observation"] == "noisy"
    assert reference["root_meter_noise_rel"] == 0.0002
    assert reference["impedance_scale"] == 1.0
    assert reference["physical_sample_count"] == 96
    for suite in SCENARIO_SUITES:
        settings = resolve_scenario_settings(suite)
        assert json.loads(json.dumps(settings, allow_nan=False)) == settings
    reference["profiles"][0] = "mutated"
    assert resolve_scenario_settings("reference")["profiles"][0] == "default"


@pytest.mark.parametrize("case_key", list(LEGACY_SCALES))
@pytest.mark.parametrize("samples", [8, 11])
def test_default_legacy_reproduces_original_arrays(case_key, samples):
    net, actual = _simulate_pool(case_key, samples, 2, 3, 0.015, 0.0003)
    expected_net, expected = _legacy_oracle(case_key, samples, 2, 0.015, 0.0003)
    np.testing.assert_array_equal(net.branches[["r_ohm", "x_ohm"]], expected_net.branches[["r_ohm", "x_ohm"]])
    for item, oracle in zip(actual, expected):
        assert item["name"] == oracle["name"]
        for key in ARRAY_KEYS + ("root_voltage",):
            np.testing.assert_array_equal(item[key].to_numpy(), oracle[key].to_numpy(), err_msg=key)
        assert item["source_sample_indices"] == list(range(samples))
        assert item["scenario_settings"]["scenario_suite"] == "legacy"


def test_legacy_explicit_suite_equals_six_argument_default():
    _, default = _simulate_pool("paper15", 11, 0, 1, 0.01, 0.0002)
    _, explicit = _simulate_pool("paper15", 11, 0, 1, 0.01, 0.0002, scenario_suite="legacy")
    for key in ARRAY_KEYS + ("root_voltage",):
        np.testing.assert_array_equal(default[0][key].to_numpy(), explicit[0][key].to_numpy())


def test_root_observation_modes_preserve_paired_physics_and_terminal_noise():
    pools = {}
    for mode in ROOT_OBSERVATIONS:
        _, pools[mode] = _simulate_pool(
            "paper15", 16, 1, 3, 0.015, 0.0003,
            scenario_suite="reference", root_observation=mode,
        )
    for exact, noisy in zip(pools["exact"], pools["noisy"]):
        for key in ARRAY_KEYS[:-1]:
            np.testing.assert_array_equal(noisy[key].to_numpy(), exact[key].to_numpy(), err_msg=key)
        np.testing.assert_array_equal(exact["root_voltage"], exact["root_voltage_true"])
        assert not np.array_equal(noisy["root_voltage"], noisy["root_voltage_true"])
        for item in [exact, noisy]:
            expected = item["root_voltage"].to_numpy()[:, None]**2 - item["V_terminal"].to_numpy()**2
            np.testing.assert_array_equal(item["drop_target"], expected)


def test_core_rejects_unobserved_root_before_simulation():
    with pytest.raises(ValueError, match="root_observation"):
        resolve_scenario_settings(root_observation="unobserved")
    with pytest.raises(ValueError, match="root_observation"):
        _simulate_pool("paper15", 8, 0, 1, 0.0, 0.0, root_observation="unobserved")


@pytest.mark.parametrize("samples", [8, 16, 32])
def test_reduced_samples_are_exact_subset_of_same_noisy_96_point_day(samples):
    _, full = _simulate_pool("soumalas11", 96, 1, 3, 0.015, 0.0003, scenario_suite="reference")
    _, reduced = _simulate_pool("soumalas11", samples, 1, 3, 0.015, 0.0003, scenario_suite="reference")
    indices = list(range(0, 96, 96 // samples))
    for complete, sampled in zip(full, reduced):
        assert sampled["source_sample_indices"] == indices
        for key in ARRAY_KEYS + ("root_voltage",):
            np.testing.assert_array_equal(sampled[key].to_numpy(), complete[key].iloc[indices].to_numpy(), err_msg=key)
            assert sampled[key].index.equals(complete[key].iloc[indices].index)


@pytest.mark.parametrize("case_key,legacy_scale", LEGACY_SCALES.items())
def test_reference_scale_one_reduces_r_and_x_by_each_original_scale(case_key, legacy_scale):
    legacy_net, _ = _simulate_pool(case_key, 8, 0, 1, 0.0, 0.0, scenario_suite="legacy")
    reference_net, _ = _simulate_pool(case_key, 8, 0, 1, 0.0, 0.0, scenario_suite="reference")
    terminals = _terminals(legacy_net)
    legacy_matrices = build_reduced_sensitivity_matrices(legacy_net, terminals, terminals, voltage_model="squared-voltage")
    reference_matrices = build_reduced_sensitivity_matrices(reference_net, terminals, terminals, voltage_model="squared-voltage")
    for actual, previous in zip(reference_matrices, legacy_matrices):
        np.testing.assert_allclose(actual, previous / legacy_scale, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(reference_net.branches[["r_ohm", "x_ohm"]], legacy_net.branches[["r_ohm", "x_ohm"]] / legacy_scale)


def test_root_strength_changes_waveform_amplitude_without_changing_pq():
    _, weak = _simulate_pool("paper15", 96, 1, 3, 0.015, 0.0003, scenario_suite="weak_root")
    _, strong = _simulate_pool("paper15", 96, 1, 3, 0.015, 0.0003, scenario_suite="strong_root")
    for weak_day, strong_day in zip(weak, strong):
        for key in ("P_true", "Q_true", "P_terminal", "Q_terminal"):
            np.testing.assert_array_equal(weak_day[key], strong_day[key])
        np.testing.assert_allclose(
            strong_day["root_voltage_true"] - 1.02,
            (0.006 / 0.0008) * (weak_day["root_voltage_true"] - 1.02),
            rtol=1e-11, atol=2e-15,
        )


def test_tap_step_adds_exact_centered_midday_step_to_same_background():
    _, reference = _simulate_pool("paper15", 96, 0, 3, 0.0, 0.0, scenario_suite="reference")
    _, stepped = _simulate_pool("paper15", 96, 0, 3, 0.0, 0.0, scenario_suite="tap_step")
    expected = 0.00625 * ((np.arange(96) >= 48) & (np.arange(96) < 72))
    expected = expected - expected.mean()
    for before, after in zip(reference, stepped):
        np.testing.assert_array_equal(before["P_true"], after["P_true"])
        np.testing.assert_array_equal(before["Q_true"], after["Q_true"])
        np.testing.assert_allclose(after["root_voltage_true"] - before["root_voltage_true"], expected, atol=3e-16, rtol=1e-12)


def test_correlated_root_has_persistent_dynamics_relative_to_legacy_iid():
    correlations = {"reference": [], "legacy": []}
    for suite in correlations:
        for replicate in [0, 1, 2]:
            _, pool = _simulate_pool("paper15", 96, replicate, 3, 0.0, 0.0, scenario_suite=suite)
            for day in pool:
                values = day["root_voltage_true"].to_numpy()
                correlations[suite].append(np.corrcoef(values[:-1], values[1:])[0, 1])
    assert np.median(correlations["reference"]) > 0.5
    assert abs(np.median(correlations["legacy"])) < 0.2
    assert np.median(correlations["reference"]) - np.median(correlations["legacy"]) > 0.4


@pytest.mark.parametrize("kwargs", [
    {"scenario_suite": "missing"}, {"root_observation": "missing"},
    {"root_sigma": -0.1}, {"root_sigma": np.nan}, {"root_sigma": np.inf},
    {"root_meter_noise_rel": -0.1}, {"root_meter_noise_rel": np.nan}, {"root_meter_noise_rel": np.inf},
    {"impedance_scale": 0.0}, {"impedance_scale": -1.0}, {"impedance_scale": np.nan}, {"impedance_scale": np.inf},
])
def test_invalid_scenario_settings_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        resolve_scenario_settings(**kwargs)


@pytest.mark.parametrize("samples", [3, 5, 95, 97, 12.5, True])
def test_new_suites_reject_non_nested_sample_counts(samples):
    with pytest.raises(ValueError):
        _simulate_pool("paper15", samples, 0, 1, 0.0, 0.0, scenario_suite="reference")


@pytest.mark.parametrize("count", [0, 4])
def test_unavailable_scenario_count_fails_explicitly(count):
    with pytest.raises(ValueError):
        _simulate_pool("paper15", 8, 0, count, 0.0, 0.0)



def test_valid_overrides_reach_physical_simulation_and_saved_settings():
    settings = resolve_scenario_settings(
        "weak_root", root_observation="exact", root_sigma=0.0,
        root_meter_noise_rel=0.0, impedance_scale=2.0,
    )
    assert settings["root_sigma"] == 0.0
    assert settings["root_observation"] == "exact"
    assert settings["root_meter_noise_rel"] == 0.0
    assert settings["impedance_scale"] == 2.0
    reference_net, reference = _simulate_pool(
        "paper15", 8, 0, 1, 0.0, 0.0, scenario_suite="reference",
    )
    overridden_net, overridden = _simulate_pool(
        "paper15", 8, 0, 1, 0.0, 0.0, scenario_suite="weak_root",
        root_observation="exact", root_sigma=0.0,
        root_meter_noise_rel=0.0, impedance_scale=2.0,
    )
    day = overridden[0]
    np.testing.assert_array_equal(day["root_voltage_true"], np.full(8, 1.02))
    np.testing.assert_array_equal(day["root_voltage"], day["root_voltage_true"])
    np.testing.assert_array_equal(day["P_true"], reference[0]["P_true"])
    np.testing.assert_array_equal(day["Q_true"], reference[0]["Q_true"])
    np.testing.assert_allclose(
        overridden_net.branches[["r_ohm", "x_ohm"]],
        2.0 * reference_net.branches[["r_ohm", "x_ohm"]],
    )
    for key in ["root_sigma", "root_observation", "root_meter_noise_rel", "impedance_scale"]:
        assert day["scenario_settings"][key] == settings[key]


def test_zero_root_meter_noise_matches_exact_observation():
    _, exact = _simulate_pool(
        "paper15", 8, 0, 1, 0.01, 0.0003,
        scenario_suite="reference", root_observation="exact",
    )
    _, noiseless = _simulate_pool(
        "paper15", 8, 0, 1, 0.01, 0.0003,
        scenario_suite="reference", root_observation="noisy", root_meter_noise_rel=0.0,
    )
    for key in ARRAY_KEYS + ("root_voltage",):
        np.testing.assert_array_equal(noiseless[0][key], exact[0][key], err_msg=key)
