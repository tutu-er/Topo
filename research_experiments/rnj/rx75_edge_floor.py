"""Isolated RX75 resolution variant; never edit the frozen solver package.

Calibrate once from the same ordered, Source-FWL dense R/X matrices used by
NJ/RNJ, and reuse that calibration across all methods, path points and bound
restarts. The floor is an RX75 geometry scale, not an Ohm lower bound.

Old non-singleton atoms may be zero or have strength >= tau. New atoms must
have strength >= tau and at least two original terminal members. Singleton
leaf atoms retain the original domain. A zero old atom still participates in
the core's unconditional laminar constraints: this variant does not permit
arbitrary structural replacement. Source h is unchanged and may traverse a
zero-weight atom, including a nonempty path with zero total response. The
certificate remains a bounded, single-new-atom certificate on a greedy path.

Only process-local builder references are patched by the context manager.
It does not fit/calibrate from truth, read files, change solver settings, or
modify Source path constraints. Use in an isolated worker, not shared threads.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from importlib import import_module
import inspect
from numbers import Real

import numpy as np


class FloorConfigurationError(ValueError):
    """The declared floor cannot fit in the current coefficient box."""


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _matrix(value, name):
    value = np.asarray(value, dtype=float)
    if (value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0
            or not np.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite nonempty square matrix")
    return .5 * value + .5 * value.T


def calibrate_rx75_floor(R_dense, X_dense, factor=.16):
    """Return a serializable calibration using the existing RX75 convention.

    Each channel scale is the mean of all strictly positive pairwise distances
    diag(M)_i + diag(M)_j - 2*M_ij, floored at 1e-12 (fallback 1 if none).
    The caller must supply its pre-tree ordered/FWL regression matrices, never
    a final WZZT matrix or simulation truth. No estimator is invoked here.
    """
    r, x = _matrix(R_dense, "R_dense"), _matrix(X_dense, "X_dense")
    if r.shape != x.shape:
        raise ValueError("R_dense and X_dense must have the same shape")
    factor = _number(factor, "factor")
    means = []
    for matrix in (r, x):
        diagonal = np.diag(matrix)
        distance = diagonal[:, None] + diagonal[None, :] - 2 * matrix
        if not np.isfinite(distance).all():
            raise ValueError("pairwise RX75 distances must be finite")
        positive = distance[distance > 0]
        means.append(max(float(np.mean(positive)) if positive.size else 1., 1e-12))
    a, b = .75 / means[0], .25 / means[1]
    median = float(np.median(np.diag(a * r + b * x)))
    tau = factor * max(median, 1e-12)
    if not np.isfinite([a, b, median, tau]).all():
        raise ValueError("RX75 calibration must be finite")
    return {"a": a, "b": b, "tau": tau, "mu_r": means[0], "mu_x": means[1],
            "factor": factor, "median_root_depth": median,
            "terminal_count": r.shape[0], "scale": "normalized_RX75_geometry",
            "calibration_contract": "once from pre-tree ordered/FWL dense R/X; fixed for all methods/path points"}


def _calibration(calibration):
    return (_number(calibration["a"], "a", positive=True),
            _number(calibration["b"], "b", positive=True),
            _number(calibration["tau"], "tau"))


def classify_rx75_atoms(supports, r_values, x_values, calibration, *, absolute_tolerance=1e-10):
    """Classify retained atoms without fitting or changing their coefficients.

    Keep singleton leaves unconditionally. Every other atom, including a full
    terminal root stem, is effective at w=a*r+b*x >= tau, with only the declared
    fixed absolute geometry tolerance. Thus exact equality is retained, matching
    the online floor; earlier NJ <=-collapse differs only at equality.
    """
    a, b, tau = _calibration(calibration)
    tolerance = _number(absolute_tolerance, "absolute_tolerance")
    family = tuple(tuple(support) for support in supports)
    if any(not support or len(set(support)) != len(support) for support in family):
        raise ValueError("supports must be nonempty distinct-member collections")
    r, x = np.asarray(r_values, dtype=float), np.asarray(x_values, dtype=float)
    if (r.shape != (len(family),) or x.shape != r.shape
            or not np.isfinite(r).all() or not np.isfinite(x).all()
            or np.any(r < 0) or np.any(x < 0)):
        raise ValueError("r_values/x_values must be finite nonnegative vectors matching supports")
    weights = a * r + b * x
    if not np.isfinite(weights).all():
        raise ValueError("RX75 atom strengths must be finite")
    keep = [len(support) == 1 or strength + tolerance >= tau
            for support, strength in zip(family, weights, strict=True)]
    return {"strengths": weights.tolist(), "keep": keep,
            "retained_indices": [i for i, retained in enumerate(keep) if retained],
            "contracted_indices": [i for i, retained in enumerate(keep) if not retained],
            "a": a, "b": b, "tau": tau, "absolute_tolerance": tolerance,
            "equality": "strength >= tau retained; fixed absolute tolerance only"}


def _add_floor(model, supports, prepared, r_bound, x_bound, signature, *, extension):
    a, b, tau = signature
    capacity = a * r_bound + b * x_bound
    if not np.isfinite(capacity) or tau > capacity:
        raise FloorConfigurationError("RX75 tau exceeds a*R_upper_bound+b*X_upper_bound")
    marker = {"calibration": signature, "supports": supports, "extension": extension,
              "r_upper_bound": r_bound, "x_upper_bound": x_bound}
    if hasattr(model, "_rx75_edge_floor"):
        if model._rx75_edge_floor != marker:
            raise ValueError("model already has a different RX75 floor")
        return
    old = [i for i, support in enumerate(supports) if len(support) > 1]
    if old:
        active = model.variables.add("edge_active", len(old), upper=1., integral=True)
        for offset, atom in enumerate(old):
            ri, xi, ei = model.r.start + atom, model.x.start + atom, active.start + offset
            model.rows.add({ri: 1., ei: -r_bound}, upper=0.)
            model.rows.add({xi: 1., ei: -x_bound}, upper=0.)
            model.rows.add({ri: a, xi: b, ei: -tau}, lower=0.)
    if extension:
        if not all((i,) in supports for i in range(prepared.n)):
            raise ValueError("RX75 floor extension requires all original singleton leaf supports")
        if model.new_r is None or model.new_x is None or model.z is None:
            raise ValueError("extension builder did not return new weight/membership slices")
        model.rows.add({model.new_r.start: a, model.new_x.start: b}, lower=tau)
        # Membership is group-indexed, but cardinality is in original terminals.
        groups = (model.partition.groups if model.partition is not None
                  else tuple((i,) for i in range(prepared.n)))
        if len(groups) != model.z.stop - model.z.start:
            raise ValueError("terminal partition and membership dimensions differ")
        model.rows.add({model.z.start + i: float(len(group))
                        for i, group in enumerate(groups)}, lower=2.)
    model._rx75_edge_floor = marker


@contextmanager
def enforce_rx75_floor(core, source, calibration):
    """Temporarily add the resolution domain to ordinary and Source builders.

    Core high-level fit/refit/search and solver certification stay unchanged.
    Source cached ordinary-builder aliases are patched too; a model marker
    prevents duplicate constraints in None/all-zero Source fallbacks. All
    references are restored on every exit. A zero tau is a complete no-op.
    """
    signature = _calibration(calibration)
    if getattr(core, "_rx75_floor_context_active", False):
        raise ValueError("RX75 floor contexts must not overlap in one core module")
    if signature[2] == 0:
        yield {"enabled": False, "a": signature[0], "b": signature[1], "tau": 0.}
        return
    if source is None:
        source = import_module(core.__package__ + ".laminar_l1_source")
    if source.L1Model is not core.L1Model:
        raise ValueError("Source and core builders must belong to the same runtime")
    saved = []

    def wrapper(original, *, extension, source_builder):
        call_signature = inspect.signature(original)

        @wraps(original)
        def build(*args, **kwargs):
            bound = call_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            scenarios = bound.arguments.get("prepared", bound.arguments.get("scenarios"))
            prepared = core._prepare_scenarios(scenarios)
            supports = core.normalize_supports(bound.arguments["supports"], prepared.n)
            bound.arguments["supports"] = supports
            r_bound = _number(bound.arguments["r_upper_bound"], "r_upper_bound", positive=True)
            x_bound = _number(bound.arguments["x_upper_bound"], "x_upper_bound", positive=True)
            if signature[2] > signature[0] * r_bound + signature[1] * x_bound:
                raise FloorConfigurationError("RX75 tau exceeds a*R_upper_bound+b*X_upper_bound")
            result = original(*bound.args, **bound.kwargs)
            model = result[0] if source_builder else result
            _add_floor(model, supports, prepared, r_bound, x_bound, signature, extension=extension)
            return result

        return build

    try:
        core._rx75_floor_context_active = True
        for module, name, extension, source_builder in (
            (core, "build_fixed_model", False, False),
            (core, "build_extension_model", True, False),
            (source, "build_fixed_model", False, False),
            (source, "build_extension_model", True, False),
            (source, "build_fixed_source_model", False, True),
            (source, "build_extension_source_model", True, True),
        ):
            original = getattr(module, name)
            saved.append((module, name, original))
            setattr(module, name, wrapper(original, extension=extension, source_builder=source_builder))
        yield {"enabled": True, "a": signature[0], "b": signature[1], "tau": signature[2]}
    finally:
        for module, name, original in reversed(saved):
            setattr(module, name, original)
        del core._rx75_floor_context_active
