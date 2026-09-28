"""Root-information experiments built on the observed-root core simulator.

An unobserved root uses a fixed nominal reference, so its target contains the
unknown physical root fluctuation. This is a model-mismatch experiment rather
than an observation mode supported by the production pipeline.
"""

from __future__ import annotations

import numpy as np

from rnj_wzzt.estimation.preprocessing import squared_voltage_drop_from_observed_root
from rnj_wzzt.scenario.simulation import (
    _dynamic_rms,
    _integer,
    _simulate_pool as _simulate_observed_pool,
)


ROOT_OBSERVATIONS = ("exact", "noisy", "unobserved")
_ARRAY_KEYS = (
    "P_terminal", "Q_terminal", "V_terminal", "root_voltage", "drop_target",
    "P_true", "Q_true", "V_terminal_true", "root_voltage_true",
)


def _simulate_pool(
    case_key: str,
    t_count: int,
    replicate: int,
    maximum_scenarios: int,
    pq_noise_rel: float,
    v_noise_rel: float,
    *,
    scenario_suite: str = "legacy",
    root_observation: str | None = None,
    root_meter_noise_rel: float | None = None,
    root_sigma: float | None = None,
    impedance_scale: float | None = None,
) -> tuple[object, list[dict]]:
    """Add the historical unobserved-root experiment outside the core package."""
    options = dict(
        scenario_suite=scenario_suite, root_observation=root_observation,
        root_meter_noise_rel=root_meter_noise_rel, root_sigma=root_sigma,
        impedance_scale=impedance_scale,
    )
    if root_observation != "unobserved":
        return _simulate_observed_pool(
            case_key, t_count, replicate, maximum_scenarios, pq_noise_rel, v_noise_rel, **options,
        )
    t_count = _integer(t_count, "t_count", 1)
    if scenario_suite != "legacy" and (t_count < 4 or 96 % t_count):
        raise ValueError("non-legacy t_count must be >= 4 and divide 96")
    physical_count = t_count if scenario_suite == "legacy" else 96
    options["root_observation"] = "exact"
    net, scenarios = _simulate_observed_pool(
        case_key, physical_count, replicate, maximum_scenarios, pq_noise_rel, v_noise_rel, **options,
    )
    indices = np.arange(0, physical_count, physical_count // t_count, dtype=int)
    for item in scenarios:
        settings = item["scenario_settings"]
        target = settings["root_mean"]**2 - item["V_terminal"].pow(2)
        # Recompute only root-dependent diagnostics on the full physical grid;
        # truth is never used to construct the unobserved regression target.
        physical_drop = squared_voltage_drop_from_observed_root(
            item["V_terminal_true"], item["root_voltage_true"],
        )
        clean_target = settings["root_mean"]**2 - item["V_terminal_true"].pow(2)
        diagnostics = item["diagnostics"]
        target_error = _dynamic_rms(target - physical_drop)
        snr = diagnostics["dynamic_drop_signal_rms_pu2"] / target_error if target_error > 0.0 else None
        diagnostics.update({
            "root_observed_std_pu": None,
            "root_meter_noise_std_pu": None,
            "dynamic_measurement_error_rms_pu2": _dynamic_rms(target - clean_target),
            "dynamic_target_error_rms_pu2": target_error,
            "dynamic_drop_snr": snr,
            "dynamic_drop_snr_db": float(20.0 * np.log10(snr)) if snr is not None and snr > 0.0 else None,
            "observed_sample_count": t_count,
        })
        item.update(root_voltage=None, root_observation="unobserved", drop_target=target)
        settings.update(root_observation="unobserved", observed_sample_count=t_count,
                        observation_interval_hours=24.0 / t_count)
        for key in _ARRAY_KEYS:
            if item[key] is not None:
                item[key] = item[key].iloc[indices].copy()
        item["source_sample_indices"] = indices.tolist()
    return net, scenarios
