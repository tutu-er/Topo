"""Compatibility exports for research geometry and distance diagnostics."""

from terminal_case33._standalone_core import load_core_module

load_core_module("rnj_wzzt.graph.sensitivity_geometry")

from research_experiments.rnj.sensitivity_geometry import (
    SensitivityGeometry,
    distance_candidates,
    sensitivity_geometry,
)

__all__ = ["SensitivityGeometry", "distance_candidates", "sensitivity_geometry"]
