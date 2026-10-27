"""Bound-aware decisions for approximate one-atom extensions.

These decisions certify objective intervals in the declared single-step domain,
not topology correctness. Unknown bounds remain unresolved.
"""
from __future__ import annotations

import math


def extension_decision(current, upper, lower, *, feasible, relative_gain=.001,
                       absolute_gain=1e-10, uncertainty_fraction=.25):
    if not math.isfinite(current) or current < 0:
        raise ValueError("current loss must be finite and nonnegative")
    for value in (relative_gain, absolute_gain, uncertainty_fraction):
        if not math.isfinite(value) or value < 0:
            raise ValueError("decision tolerances must be finite and nonnegative")
    tolerance = absolute_gain + relative_gain * current
    ub = upper if upper is not None and math.isfinite(upper) and feasible else None
    lb = lower if lower is not None and math.isfinite(lower) else None
    if ub is not None and lb is not None and lb > ub + 1e-8*max(1., abs(ub)):
        raise ValueError("lower bound exceeds feasible upper bound")
    guaranteed = None if ub is None else current-ub
    possible = None if lb is None else current-lb
    uncertainty = None if ub is None or lb is None else max(0., ub-lb)
    if possible is not None and possible <= tolerance:
        decision = "stop_bound_proves_no_material_gain"
    elif (guaranteed is not None and guaranteed > tolerance and uncertainty is not None
          and uncertainty <= uncertainty_fraction*guaranteed):
        decision = "accept_bounded_improvement"
    else:
        decision = "refine_or_report_unresolved"
    return {"decision": decision, "gain_tolerance": tolerance,
        "gain_guaranteed": guaranteed, "gain_possible": possible,
        "optimization_uncertainty": uncertainty,
        "uncertainty_fraction": uncertainty_fraction}
