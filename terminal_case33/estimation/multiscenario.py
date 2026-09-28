"""Compatibility imports for core R/X fitting and research-only recipes."""

from terminal_case33._standalone_core import load_core_module

_core = load_core_module("rnj_wzzt.estimation.multiscenario")
align_scenarios = _core.align_scenarios
fit_projected_sensitivity = _core.fit_projected_sensitivity

from research_experiments.rnj.sensitivity_geometry import distance_candidates
from terminal_case33.estimation.recipes import preprocess_scenarios


def __getattr__(name):
    """Keep older internal fit helpers available to parent experiments."""

    return getattr(_core, name)


__all__ = [
    "align_scenarios", "fit_projected_sensitivity", "distance_candidates",
    "preprocess_scenarios",
]
