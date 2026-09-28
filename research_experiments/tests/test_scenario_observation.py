"""Independent paired-physics checks for the unobserved-root research case."""

import numpy as np
import pytest

from research_experiments.rnj.scenario_observation import ROOT_OBSERVATIONS, _simulate_pool


_PHYSICAL_AND_TERMINAL_KEYS = (
    "P_true", "Q_true", "V_terminal_true", "root_voltage_true",
    "P_terminal", "Q_terminal", "V_terminal",
)


@pytest.mark.parametrize("suite,samples", [("legacy", 11), ("reference", 16)])
def test_root_information_changes_preserve_paired_physics_and_terminal_noise(suite, samples):
    pools = {
        mode: _simulate_pool("paper15", samples, 1, 3, 0.015, 0.0003,
                             scenario_suite=suite, root_observation=mode)[1]
        for mode in ROOT_OBSERVATIONS
    }
    for exact, noisy, unobserved in zip(pools["exact"], pools["noisy"], pools["unobserved"]):
        for item in (noisy, unobserved):
            for key in _PHYSICAL_AND_TERMINAL_KEYS:
                np.testing.assert_array_equal(item[key], exact[key], err_msg=key)
        np.testing.assert_array_equal(exact["root_voltage"], exact["root_voltage_true"])
        assert not np.array_equal(noisy["root_voltage"], noisy["root_voltage_true"])
        assert unobserved["root_voltage"] is None
        for item in (exact, noisy):
            expected = item["root_voltage"].to_numpy()[:, None]**2 - item["V_terminal"].to_numpy()**2
            np.testing.assert_array_equal(item["drop_target"], expected)
        np.testing.assert_array_equal(unobserved["drop_target"], 1.02**2 - unobserved["V_terminal"].to_numpy()**2)


@pytest.mark.parametrize("samples", [8, 16, 32])
def test_unobserved_subsampling_keeps_full_grid_diagnostics(samples):
    options = dict(scenario_suite="reference", root_observation="unobserved", root_meter_noise_rel=0.001)
    _, full = _simulate_pool("soumalas11", 96, 2, 1, 0.015, 0.0003, **options)
    _, sampled = _simulate_pool("soumalas11", samples, 2, 1, 0.015, 0.0003, **options)
    full, sampled = full[0], sampled[0]
    indices = list(range(0, 96, 96 // samples))
    for key in _PHYSICAL_AND_TERMINAL_KEYS + ("drop_target",):
        np.testing.assert_array_equal(sampled[key], full[key].iloc[indices])
        assert sampled[key].index.equals(full[key].iloc[indices].index)
    assert sampled["source_sample_indices"] == indices
    assert sampled["diagnostics"] == {**full["diagnostics"], "observed_sample_count": samples}
    assert sampled["scenario_settings"] == {
        **full["scenario_settings"], "observed_sample_count": samples, "observation_interval_hours": 24.0 / samples,
    }
    assert sampled["scenario_settings"]["root_meter_noise_rel"] == 0.001
    assert sampled["diagnostics"]["physical_sample_count"] == 96
    assert sampled["diagnostics"]["root_observed_std_pu"] is None
    assert sampled["diagnostics"]["root_meter_noise_std_pu"] is None


def test_unobserved_root_disturbance_is_present_without_any_meter_noise():
    _, pool = _simulate_pool("paper15", 16, 0, 1, 0.0, 0.0,
                             scenario_suite="reference", root_observation="unobserved")
    diagnostics = pool[0]["diagnostics"]
    _, full = _simulate_pool("paper15", 96, 0, 1, 0.0, 0.0,
                             scenario_suite="reference", root_observation="exact")
    root_disturbance = 1.02**2 - full[0]["root_voltage_true"].to_numpy()**2
    assert diagnostics["dynamic_measurement_error_rms_pu2"] == 0.0
    assert diagnostics["dynamic_target_error_rms_pu2"] == pytest.approx(np.std(root_disturbance), rel=1e-13)
    assert diagnostics["dynamic_target_error_rms_pu2"] > 0.0


@pytest.mark.parametrize("samples", [3, 5, 95, 12.5, True])
def test_unobserved_research_mode_rejects_invalid_reference_sample_counts(samples):
    with pytest.raises(ValueError):
        _simulate_pool("paper15", samples, 0, 1, 0.0, 0.0,
                       scenario_suite="reference", root_observation="unobserved")
