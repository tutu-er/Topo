"""YAML configuration loading for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ScenarioConfig:
    """Runtime configuration for terminalized case33 experiments."""

    mode: str = "hybrid_leaf"
    service_impedance_mode: str = "scaled_original"
    service_length_m: float = 30.0
    service_length_random_range_m: tuple[float, float] | None = None
    T: int = 240
    dt_seconds: int = 60
    profile_mode: str = "step_probe"
    pf_mode: str = "original_qp_ratio"
    noise_config: dict = field(default_factory=dict)
    v0_config: dict = field(default_factory=lambda: {"sigma": 0.001})
    delay_config: dict = field(default_factory=dict)
    hidden_load_leakage: float = 0.0
    seed: int = 0
    experiments: list[str] = field(default_factory=lambda: ["raw_case33_check", "terminalize_hybrid_leaf", "ideal_reduced_sensitivity"])


def load_config(path: str | Path) -> ScenarioConfig:
    """Load a YAML config file into ``ScenarioConfig``."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if data.get("service_length_random_range_m") is not None:
        data["service_length_random_range_m"] = tuple(data["service_length_random_range_m"])
    return ScenarioConfig(**data)

