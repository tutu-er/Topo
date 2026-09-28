"""Compatibility exports for research-only temporal preprocessing."""

from terminal_case33._standalone_core import load_core_module

load_core_module("rnj_wzzt.estimation.multiscenario")

from research_experiments.rnj.temporal_preprocessing import (
    apply_preprocessing_recipe,
    daily_demean,
    preprocess_scenarios,
    rolling_highpass,
)

__all__ = [
    "daily_demean", "rolling_highpass", "apply_preprocessing_recipe",
    "preprocess_scenarios",
]
