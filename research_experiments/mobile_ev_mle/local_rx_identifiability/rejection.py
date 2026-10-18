"""Observation-only rejection diagnostics for the frozen local R/X study.

The two chi-square tests are reported separately.  A local AC fit, its loss,
and all simulation truth are deliberately absent from the certificate API.
The reused numerical bounds have floating-point guards, not interval proofs.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping

import numpy as np

from ..label_free_bounds.bounds import (
    Measurements, PhysicalDomain, TreeSpec, certify_candidate,
)


def _empty_test_row(alpha: float, status: str, reason: str) -> dict:
    """An unsupported or failed calculation never supplies a rejection."""
    return {
        "alpha": float(alpha), "status": status, "reason": reason,
        "raw": {"reject": False, "sse_lower": None, "threshold": None},
        "residual": {"reject": False, "qp_lower": None, "threshold": None},
        "certificate": None,
    }


def certify_observed_candidate(
    candidate: TreeSpec,
    p, q, v,
    sigma_p, sigma_q, sigma_v,
    domain: PhysicalDomain,
    *,
    root_v=None,
    alphas: Iterable[float] = (.05, .01),
    solver: str = "scipy",
    solver_time_limit: float = 30.,
    numerical_guard: float = 1e-10,
) -> dict:
    """Compute the two existing label-free bounds from public inputs only.

    ``root_v`` supplied: K, an independently known exact boundary voltage.
    ``root_v=None``: U, project out the common mode using node 1, the declared
    end of the common stem.  The reference root magnitude 1 is then arbitrary:
    its entire column is annihilated.  This does not add an unmetered source.

    ``power_caps`` bound each terminal's apparent power, while ``v_min`` and
    ``v_max`` apply to every bus, including the root and hidden buses.  The
    positive impedance lower bound is required by the reused certificate.
    A zero-edge sensitivity is retained as unsupported rather than silently
    replacing zero by a positive number.

    Each alpha defines its own raw and residual test.  No OR-combined test or
    topology probability is constructed.
    """
    if not isinstance(candidate, TreeSpec) or not isinstance(domain, PhysicalDomain):
        raise TypeError("candidate and domain must be TreeSpec and PhysicalDomain")
    alpha_values = tuple(float(a) for a in alphas)
    if not alpha_values or any(not 0 < a < 1 for a in alpha_values):
        raise ValueError("alphas must be a nonempty sequence in (0,1)")
    if len(set(alpha_values)) != len(alpha_values):
        raise ValueError("alphas must be distinct")
    if solver_time_limit <= 0 or numerical_guard < 0:
        raise ValueError("solver_time_limit must be positive and guard nonnegative")
    p_array = np.asarray(p, dtype=float)
    if p_array.ndim != 2 or p_array.shape[0] < 1:
        raise ValueError("p must have nonempty shape (time, terminal)")
    if p_array.shape[1] != len(candidate.observed_nodes):
        raise ValueError("p columns must match candidate.observed_nodes")
    boundary = "U" if root_v is None else "K"
    result = {
        "status": "completed", "boundary": boundary,
        "time_points": int(p_array.shape[0]),
        "observed_nodes": list(candidate.observed_nodes),
        "projection_used": boundary == "U",
        "common_stem_end": 1 if boundary == "U" else None,
        "root_reference_is_arbitrary": boundary == "U",
        "raw_noise_dimension": int(3 * p_array.size),
        "combined_reject": None,
        "tests_are_separate": True,
        "bound_validation": "floating_point_guard_not_interval_arithmetic",
        "failed_alpha_count": 0,
        "alpha_results": [],
    }
    if np.any(np.asarray(domain.z_min, dtype=float) <= 0):
        status = "unsupported_zero_impedance"
        reason = "existing_bound_requires_strictly_positive_z_min; no_epsilon_substitution"
        result["status"] = status
        result["alpha_results"] = [_empty_test_row(a, status, reason) for a in alpha_values]
        return result
    if domain.source_power_cap != 0:
        raise ValueError("the local protocol has no unmetered source")
    # This source_node field is solely the established projection mechanism.
    # There is no extra physical injection or free source power in this study.
    projected_tree = replace(candidate, source_node=1 if boundary == "U" else None)
    observed = Measurements(
        p=p_array, q=np.asarray(q, dtype=float), v=np.asarray(v, dtype=float),
        root_v=1. if boundary == "U" else root_v,
        sigma_p=sigma_p, sigma_q=sigma_q, sigma_v=sigma_v,
        sigma_root=None,
    )
    failures = 0
    for alpha in alpha_values:
        try:
            certificate = certify_candidate(
                projected_tree, observed, domain, alpha=alpha,
                source_projection=boundary == "U", solver=solver,
                solver_time_limit=solver_time_limit,
                numerical_guard=numerical_guard,
            )
            values = (
                certificate.ac_sse_lower, certificate.chi_square_threshold,
                certificate.qp_support_lower, certificate.residual_chi_threshold,
            )
            if any(not np.isfinite(x) or x < 0 for x in values):
                raise ArithmeticError("certificate contains invalid bound or threshold")
            # Decisions use the independently guarded lower bound, not a local
            # AC loss or the QP primal.  Recompute comparisons explicitly here.
            result["alpha_results"].append({
                "alpha": alpha, "status": "completed", "reason": None,
                "raw": {
                    "reject": bool(certificate.ac_sse_lower > certificate.chi_square_threshold),
                    "sse_lower": float(certificate.ac_sse_lower),
                    "threshold": float(certificate.chi_square_threshold),
                },
                "residual": {
                    "reject": bool(certificate.qp_support_lower > certificate.residual_chi_threshold),
                    "qp_lower": float(certificate.qp_support_lower),
                    "threshold": float(certificate.residual_chi_threshold),
                    "output_rank_df": int(certificate.residual_chi_df),
                    "alpha_split": list(certificate.residual_chi_alpha_split),
                },
                "certificate": certificate.as_dict(),
            })
        except Exception as exc:
            # Preserve the candidate and numerical failure in the denominator.
            failures += 1
            result["alpha_results"].append(_empty_test_row(
                alpha, "calculation_failed", f"{type(exc).__name__}: {exc}"))
    if failures:
        result["status"] = "failed" if failures == len(alpha_values) else "partial_failure"
    result["failed_alpha_count"] = failures
    return result


def bootstrap_pvalue(observed_sse: float, bootstrap_records: Iterable[Mapping]) -> dict:
    """Conservative failure accounting for an approximate plug-in bootstrap.

    A record is usable only with explicit ``success=True``, ``stationary=True``
    and a finite nonnegative ``sse``.  Every other record contributes to the
    upper tail.  A successful local fit is still not a global MLE certificate.
    """
    observed_sse = float(observed_sse)
    if not np.isfinite(observed_sse) or observed_sse < 0:
        raise ValueError("observed_sse must be finite and nonnegative")
    records = list(bootstrap_records)
    failed = valid = upper_tail = 0
    for record in records:
        usable = isinstance(record, Mapping)
        if usable:
            try:
                value = float(record.get("sse", np.nan))
                success = record.get("success", False)
                stationary = record.get("stationary", False)
                usable = (
                    isinstance(success, (bool, np.bool_)) and bool(success)
                    and isinstance(stationary, (bool, np.bool_)) and bool(stationary)
                    and np.isfinite(value) and value >= 0
                )
            except (TypeError, ValueError, OverflowError):
                usable = False
        if not usable:
            failed += 1
            upper_tail += 1
        else:
            valid += 1
            upper_tail += int(value >= observed_sse)
    count = len(records)
    return {
        "pvalue": float((1 + upper_tail) / (count + 1)),
        "replicates": count, "valid_replicates": valid,
        "failed_or_nonstationary_replicates": failed,
        "upper_tail_count_including_failures": upper_tail,
        "resolution": float(1 / (count + 1)),
        "method": "candidate_parametric_bootstrap_plugin_local_fit",
        "coverage": "approximate_not_uniform_frequentist_guarantee",
        "failure_rule": "count_as_greater_or_equal_to_observed",
    }
