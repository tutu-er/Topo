"""Symmetric distance in a declared finite, history-supported model family.

Inputs are already whitened meter means, not realized future observations.
An atom binds its line parameters and historical/future latent paths. This
module exhaustively reselects atoms; it never frees future terminal totals.
The finite minimum is not a bound for a differently constrained AC family.
"""
from __future__ import annotations

from itertools import combinations
from typing import Mapping

import numpy as np


def _finite_array(value, name, ndim):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if result.ndim != ndim or not all(result.shape):
        raise ValueError(f"{name} must have {ndim} nonempty dimensions")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains an invalid mean; no atom may be discarded")
    return result


def _squared_sum(value, axis):
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.sum(np.square(value), axis=axis)
    except FloatingPointError as error:
        raise ValueError("whitened squared distance overflowed") from error
    if not np.isfinite(result).all():
        raise ValueError("whitened squared distance is not finite")
    return result


def symmetric_scores(history_y, history_means, future_means: Mapping,
                     labels, cutoff=12.0):
    """Exhaustive symmetric future distances for a finite atom family.

    history_y: [DH]; history_means: [G,DH]; future_means[name]: [G,DN].
    Every row is one atom with the same row identity across H and every action.
    labels[G] are integer topology labels. Means/readings are already whitened.
    Within each topology retain L_atom <= min(L_atom) + cutoff. For each action
    minimize future squared distance across every retained cross-topology pair.

    Any invalid mean invalidates this call. In particular, an AC failure must
    not be encoded as a missing row and must not inflate a distance. Physics,
    action feasibility, covariance and fixed-node source construction belong
    to the caller and must be validated before constructing these arrays.
    """
    y = _finite_array(history_y, "history_y", 1)
    means = _finite_array(history_means, "history_means", 2)
    if means.shape[1] != len(y):
        raise ValueError("history_y and historical meter dimensions differ")
    raw_labels = np.asarray(labels)
    if (raw_labels.ndim != 1 or len(raw_labels) != len(means)
            or not np.issubdtype(raw_labels.dtype, np.integer)):
        raise ValueError("labels must be one integer topology label per atom")
    topology_labels = np.unique(raw_labels)
    if len(topology_labels) < 2:
        raise ValueError("symmetric discrimination requires at least two topologies")
    try:
        c = float(cutoff)
    except (TypeError, ValueError) as error:
        raise ValueError("cutoff must be finite and nonnegative") from error
    if not np.isfinite(c) or c < 0:
        raise ValueError("cutoff must be finite and nonnegative")
    if not isinstance(future_means, Mapping) or not future_means:
        raise ValueError("future_means must contain at least one named action")

    # Validate all actions before selecting anything: a bad action/atom remains
    # an explicit caller-level failure rather than a silently smaller family.
    checked_future = {}
    future_dimension = None
    for name, value in future_means.items():
        if not isinstance(name, str) or not name:
            raise ValueError("action names must be nonempty strings")
        values = _finite_array(value, f"future_means[{name}]", 2)
        if len(values) != len(means):
            raise ValueError("every action must retain every atom in row order")
        if future_dimension is None:
            future_dimension = values.shape[1]
        elif values.shape[1] != future_dimension:
            raise ValueError("actions must have the same future meter dimension")
        checked_future[name] = values

    loss = _squared_sum(means - y, axis=1)
    supports, minima = {}, {}
    for label in topology_labels:
        indices = np.flatnonzero(raw_labels == label)
        minimum = float(np.min(loss[indices]))
        supports[int(label)] = indices[loss[indices] <= minimum + c]
        minima[str(int(label))] = minimum

    actions = {}
    for name, values in checked_future.items():
        pairwise = []
        for left, right in combinations(topology_labels, 2):
            il, ir = supports[int(left)], supports[int(right)]
            distances = _squared_sum(values[il, None, :] - values[ir][None, :, :], axis=2)
            row, column = np.unravel_index(int(np.argmin(distances)), distances.shape)
            pairwise.append(dict(
                topology_pair=[int(left), int(right)],
                atom_pair=[int(il[row]), int(ir[column])],
                distance_squared=float(distances[row, column]),
                support_sizes=[len(il), len(ir)],
                evaluated_atom_pairs=int(distances.size),
            ))
        closest = min(pairwise, key=lambda item: (item["distance_squared"], item["topology_pair"], item["atom_pair"]))
        actions[name] = dict(
            score=closest["distance_squared"],
            distance_squared=closest["distance_squared"],
            closest_atom_pair=closest["atom_pair"].copy(),
            closest_topology_pair=closest["topology_pair"].copy(),
            pairwise=pairwise,
            evaluated_atom_pairs=sum(item["evaluated_atom_pairs"] for item in pairwise),
            status="finite_exhaustive",
        )
    return dict(
        criterion="symmetric_history_supported_future_mean_distance_squared",
        domain="declared_finite_atom_family",
        minimization="exhaustive_finite_pairs_with_numeric_mean_evaluation",
        score_is_topology_probability=False,
        is_continuous_ac_separation_certificate=False,
        cutoff=c,
        historical_losses=loss.tolist(),
        historical_min_losses=minima,
        supports={str(label): indices.tolist() for label, indices in supports.items()},
        support_sizes={str(label): len(indices) for label, indices in supports.items()},
        history_dimension=len(y), future_dimension=future_dimension,
        actions=actions,
    )


def solve_soft_history_qp(c, A, history_precision, *, rcond=1e-12):
    """Solve min_z ||c+A z||^2 + z' H z for H positive semidefinite.

    This is an unconstrained *local surrogate*, not the hard-history problem.
    The effective metric I-A(A'A+H)^+A' need not be an orthogonal projector.
    Augmented least squares avoids squaring the conditioning numerically.
    Returned steps must not be used as feasible nonlinear AC solutions without
    separately checking the original model, bounds and historical support.
    """
    residual = _finite_array(c, "c", 1)
    a = np.asarray(A, dtype=float)
    h = np.asarray(history_precision, dtype=float)
    if a.ndim != 2 or a.shape[0] != len(residual) or not np.isfinite(a).all():
        raise ValueError("A must be finite with one row per residual component")
    width = a.shape[1]
    if h.shape != (width, width) or not np.isfinite(h).all():
        raise ValueError("history_precision must be a finite square matrix")
    if not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError("rcond must lie strictly between zero and one")
    scale = max(1.0, float(np.max(abs(h), initial=0.0)))
    if np.max(abs(h - h.T), initial=0.0) > 1e-12 * scale:
        raise ValueError("history_precision must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh((h + h.T) / 2)
    if np.min(eigenvalues, initial=0.0) < -1e-12 * scale:
        raise ValueError("history_precision must be positive semidefinite")
    # Only roundoff-size negative eigenvalues are set to zero. Disclose this.
    clipped = int(np.count_nonzero(eigenvalues < 0))
    square_root = np.sqrt(np.maximum(eigenvalues, 0))[:, None] * eigenvectors.T
    augmented = np.vstack((a, square_root))
    inverse = np.linalg.pinv(augmented, rcond=rcond)
    step = -inverse @ np.r_[residual, np.zeros(width)]
    metric = np.eye(len(residual)) - a @ inverse[:, :len(residual)]
    metric = (metric + metric.T) / 2
    future_residual = residual + a @ step
    history_residual = square_root @ step
    objective = float(future_residual @ future_residual + history_residual @ history_residual)
    projected_objective = float(residual @ metric @ residual)
    normal = a.T @ future_residual + square_root.T @ history_residual
    return dict(
        nuisance_step=step,
        effective_metric=metric,
        objective=objective,
        projected_objective=projected_objective,
        future_residual=future_residual,
        history_penalty=float(history_residual @ history_residual),
        normal_equation_residual=float(np.max(abs(normal), initial=0.0)),
        roundoff_negative_history_eigenvalues_clipped=clipped,
        is_hard_history_distance=False,
        is_nonlinear_ac_solution=False,
    )
