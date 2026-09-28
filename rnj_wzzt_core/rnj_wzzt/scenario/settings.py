"""Named physical and measurement settings for reproducible scenario suites."""

from __future__ import annotations

import math


SCENARIO_SUITES = ("legacy", "reference", "weak_root", "strong_root", "tap_step")
ROOT_OBSERVATIONS = ("exact", "noisy")


def _finite_number(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or (number <= 0.0 if positive else number < 0.0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {bound}")
    return number


def resolve_scenario_settings(
    scenario_suite: str = "legacy",
    *,
    root_observation: str | None = None,
    root_meter_noise_rel: float | None = None,
    root_sigma: float | None = None,
    impedance_scale: float | None = None,
) -> dict:
    """Return independent JSON-compatible physical and observation settings.

    Root sigma controls physical background fluctuations. Meter noise is a
    separate relative measurement-error standard deviation and is applied only
    in noisy observation mode. Legacy leaves each case's impedance multiplier
    unchanged unless explicitly overridden.
    """

    if scenario_suite not in SCENARIO_SUITES:
        raise ValueError(f"unknown scenario_suite: {scenario_suite!r}")
    legacy = scenario_suite == "legacy"
    observation = root_observation if root_observation is not None else ("exact" if legacy else "noisy")
    if observation not in ROOT_OBSERVATIONS:
        raise ValueError(f"unknown root_observation: {observation!r}")
    default_sigma = {"legacy": 0.0008, "reference": 0.003, "weak_root": 0.0008,
                     "strong_root": 0.006, "tap_step": 0.003}[scenario_suite]
    sigma = _finite_number(default_sigma if root_sigma is None else root_sigma, "root_sigma")
    # Explicit legacy+noisy opts into the meter model; an unmodified legacy run
    # continues to use the exact root without any new random draws affecting it.
    default_noise = 0.0002 if not legacy or observation == "noisy" else 0.0
    meter_noise = _finite_number(default_noise if root_meter_noise_rel is None else root_meter_noise_rel,
                                "root_meter_noise_rel")
    scale = None if legacy else 1.0
    if impedance_scale is not None:
        scale = _finite_number(impedance_scale, "impedance_scale", positive=True)
    return {
        "scenario_suite": scenario_suite,
        "profiles": ["default", "default", "default"] if legacy else
                    ["default", "cloudy_variable_pv", "evening_peak_high_load"],
        "seeds": [42, 7, 21],
        "root_mean": 1.02,
        "root_sigma": sigma,
        "root_process": "iid" if legacy else "correlated",
        "root_observation": observation,
        "root_meter_noise_rel": meter_noise,
        "impedance_scale": scale,
        "physical_sample_count": None if legacy else 96,
        "day_hours": 24.0,
        "daily_fraction": 0.65,
        "ou_tau_hours": 1.0,
        "tap_step_amplitude": 0.00625 if scenario_suite == "tap_step" else 0.0,
        "tap_start_hour": 12.0,
        "tap_end_hour": 18.0,
        "tap_step_demeaned": True,
        "voltage_clipping": False,
    }
