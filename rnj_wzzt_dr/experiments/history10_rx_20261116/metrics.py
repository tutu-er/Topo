"""Read-only response and historical-design diagnostics for the 10-day study.

No fitting, candidate selection, random sampling, AC calculation, or optimizer
is used here. Hstar is an explicitly offline physical linear reference. A low
historical prediction error does not imply accurate response in every control
direction. Matrix errors concern squared-voltage response, not line ohms.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np


def _array(value, name, ndim=2):
    a = np.asarray(value, dtype=float)
    if a.ndim != ndim or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return a


def _power(P, Q):
    p, q = _array(P, "P"), _array(Q, "Q")
    if p.shape != q.shape or min(p.shape) <= 0:
        raise ValueError("P and Q must have the same nonempty (time, node) shape")
    return p, q


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _singular_geometry(a, absolute_rank_floor=0.):
    """Retain rank deficiency; a finite condition is for full column rank only."""
    s = np.linalg.svd(a, compute_uv=False)
    tolerance = max(float(absolute_rank_floor),
                    float(np.finfo(float).eps * max(a.shape) * s[0]) if len(s) else 0.)
    rank = int(np.sum(s > tolerance))
    columns = int(a.shape[1])
    full = bool(columns > 0 and rank == columns)
    return dict(rows=int(a.shape[0]), columns=columns, rank=rank,
                nullity=columns-rank, full_column_rank=full,
                singular_values=s.tolist(), rank_tolerance=tolerance,
                condition_number=float(s[0]/s[-1]) if full else None)


def _geometry(a):
    raw = _singular_geometry(a)
    norms = np.linalg.norm(a, axis=0)
    scaled = np.divide(a, norms, out=np.zeros_like(a), where=norms[None, :] > 0)
    return dict(raw=raw, column_normalized=_singular_geometry(scaled),
                column_norms=norms.tolist(), zero_columns=np.flatnonzero(norms == 0).tolist(),
                condition_scope="Full column rank only; null directions are not discarded")


def _conditional_block(given, block):
    """Residualize one observed design block against another; not an estimator."""
    if given.shape[0] != block.shape[0]:
        raise ValueError("Conditional design blocks must have the same row count")
    u, s, _ = np.linalg.svd(given, full_matrices=False)
    tol = float(np.finfo(float).eps*max(given.shape)*s[0]) if len(s) else 0.
    rank = int(np.sum(s > tol))
    residual = block-u[:, :rank]@(u[:, :rank].T@block)
    energy = float(np.sum(residual*residual))
    original = float(np.sum(block*block))
    columns = np.sum(block*block, axis=0)
    left = np.sum(residual*residual, axis=0)
    orthogonality = float(np.linalg.norm(given.T@residual))
    block_norm = float(np.linalg.norm(block, ord=2)) if block.size else 0.
    # Ranking a nearly-zero projected residual relative to its own norm would
    # turn roundoff into apparent independent Q information. Retain its energy,
    # but use the source-block scale for this numerical rank decision.
    projection_floor = 32*np.finfo(float).eps*max(*given.shape, *block.shape)*block_norm
    residual_geometry = _singular_geometry(residual, projection_floor)
    return dict(given_rank=rank, given_rank_tolerance=tol,
                block_energy=original, residual_energy=energy,
                residual_energy_fraction=_ratio(energy, original),
                residual_RMS=float(np.sqrt(energy/residual.size)) if residual.size else None,
                per_column_block_energy=columns.tolist(),
                per_column_residual_energy=left.tolist(),
                per_column_residual_fraction=[_ratio(a, b) for a, b in zip(left, columns)],
                residual_geometry=residual_geometry,
                projection_roundoff_rank_floor=float(projection_floor),
                projection_orthogonality_norm=orthogonality,
                projection_orthogonality_relative=_ratio(
                    orthogonality, np.linalg.norm(given)*np.linalg.norm(block)),
                interpretation="Observed energy outside the other block's column span; no confidence or bias bound")


def design_diagnostics(P, Q):
    p, q = _power(P, Q)
    pc, qc = p-p.mean(axis=0), q-q.mean(axis=0)
    return dict(points=int(p.shape[0]), nodes=int(p.shape[1]),
                uncentered=dict(design=_geometry(np.c_[p, q]),
                                Q_given_P=_conditional_block(p, q)),
                centered=dict(design=_geometry(np.c_[pc, qc]),
                              Q_given_P=_conditional_block(pc, qc)),
                interpretation=("The current QP has no intercept and uses the uncentered design. "
                                "Centered values are a descriptive variation check, not a different fit."))


def _error(fitted, reference):
    norm = float(np.linalg.norm(reference))
    fitted_norm = float(np.linalg.norm(fitted))
    absolute = float(np.linalg.norm(fitted-reference))
    cosine = (float(np.clip(np.sum(fitted*reference)/(norm*fitted_norm), -1., 1.))
              if norm > 0 and fitted_norm > 0 else None)
    return dict(reference_norm=norm, fitted_norm=fitted_norm, error_norm=absolute,
                relative_error=_ratio(absolute, norm), cosine=cosine,
                angle_degrees=float(np.degrees(np.arccos(cosine))) if cosine is not None else None)


def _labels(labels, n):
    result = list(range(n)) if labels is None else [v.item() if isinstance(v, np.generic) else v for v in labels]
    if len(result) != n or len(set(result)) != n:
        raise ValueError("Labels must identify each node exactly once")
    return result


def metrics(H, Hstar, P, Q, resource_nodes=None, *, labels=None, resource_node_mode=None):
    """Return offline matrix, input-column and prediction-error diagnostics.

    P and Q must be in the same power units used by H/Hstar, normally the common
    per-unit base. resource_nodes is optional {'P': [...], 'Q': [...]}.
    Entries are labels when labels is supplied, otherwise zero-based indices.
    resource_node_mode can explicitly select 'labels' or 'indices'.
    Each node is counted once per axis even if it has several devices. No device
    capacity, available future power, or selected action is inferred here.
    """
    p, q = _power(P, Q)
    n = p.shape[1]
    h, truth = _array(H, "H"), _array(Hstar, "Hstar")
    if h.shape != (n, 2*n) or truth.shape != h.shape:
        raise ValueError("H and Hstar must both be (nodes, 2*nodes), with P before Q")
    mode = ('labels' if labels is not None else 'indices') if resource_node_mode is None else resource_node_mode
    if mode not in ('labels', 'indices') or (mode == 'labels' and labels is None):
        raise ValueError("Resource mode must be labels with explicit labels, or indices")
    labels = _labels(labels, n)
    lookup = {label: i for i, label in enumerate(labels)}
    resources = dict(P=[], Q=[]) if resource_nodes is None else resource_nodes
    if not isinstance(resources, Mapping) or set(resources)-{'P', 'Q'}:
        raise ValueError("resource_nodes must map P/Q to observed terminal labels")
    indices = {}
    for axis in ('P', 'Q'):
        raw = list(resources.get(axis, []))
        if mode == 'indices':
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0 or v >= n for v in raw):
                raise ValueError("Resource column indices must be integers in [0,nodes)")
            requested = {labels[int(v)] for v in raw}
        else:
            requested = set(raw)
        if requested-set(labels):
            raise ValueError("Resource node absent from measured labels")
        indices[axis] = [lookup[label] for label in labels if label in requested]
    r, x = h[:, :n], h[:, n:]
    rs, xs = truth[:, :n], truth[:, n:]
    er, ex = p@(r-rs).T, q@(x-xs).T
    total = er+ex
    mse_r, mse_x = float(np.mean(er**2)), float(np.mean(ex**2))
    cross = float(2*np.mean(er*ex))
    mse_total = float(np.mean(total**2))
    predicted = p@rs.T+q@xs.T
    reference_rms = float(np.sqrt(np.mean(predicted**2)))
    columns = [dict(node_label=label, input_index=i,
                    R=_error(r[:, i], rs[:, i]), X=_error(x[:, i], xs[:, i]),
                    registered_P_resource=i in indices['P'],
                    registered_Q_resource=i in indices['Q']) for i, label in enumerate(labels)]
    resource_metrics = {}
    for axis, fitted, reference in [('P', r, rs), ('Q', x, xs)]:
        ix = indices[axis]
        resource_metrics[axis] = dict(node_count=len(ix), node_labels=[labels[i] for i in ix],
                                     response=_error(fitted[:, ix], reference[:, ix]) if ix else None)
    return dict(nodes=n, points=len(p), labels=labels,
                reference_scope="Offline physical linear Hstar; not an available planner input",
                response_units="Squared-voltage response per input power unit; not individual line ohms",
                R=_error(r, rs), X=_error(x, xs), full_response=_error(h, truth),
                columns=columns, resource_columns=resource_metrics, resource_node_mode=mode,
                prediction=dict(error_sign="frozen fitted response minus physical linear response",
                                R_error_RMS=float(np.sqrt(mse_r)), X_error_RMS=float(np.sqrt(mse_x)),
                                total_error_RMS=float(np.sqrt(mse_total)),
                                R_error_MSE=mse_r, X_error_MSE=mse_x,
                                cross_R_X_MSE=cross, total_error_MSE=mse_total,
                                identity_absolute_error=abs(mse_total-(mse_r+mse_x+cross)),
                                reference_prediction_RMS=reference_rms,
                                relative_prediction_RMS_error=_ratio(np.sqrt(mse_total), reference_rms),
                                interpretation="Linear response error on supplied P/Q; neither AC residual nor voltage-magnitude RMS"),
                historical_design=design_diagnostics(p, q))


def fixed_support_diagnostics(P, Q, supports):
    """Inspect the actual fixed-support R/X design without fitting coefficients."""
    p, q = _power(P, Q)
    t, n = p.shape
    canonical = []
    for support in supports:
        values = list(support)
        if (not values or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in values)
                or len(set(values)) != len(values) or min(values) < 0 or max(values) >= n):
            raise ValueError("Supports must contain unique integer terminal indices")
        canonical.append(tuple(sorted(map(int, values))))
    if len(set(canonical)) != len(canonical):
        raise ValueError("Duplicate supports must be aggregated before design diagnostics")
    masks = np.zeros((len(canonical), n))
    for index, support in enumerate(canonical):
        masks[index, list(support)] = 1.
    blocks = []
    for values in (p, q):
        block = np.empty((t*n, len(canonical)))
        for column, mask in enumerate(masks):
            block[:, column] = ((values@mask)[:, None]*mask[None, :]).reshape(-1)
        blocks.append(block)
    fr, fx = blocks
    return dict(points=t, nodes=n, residual_cells=t*n, support_count=len(canonical),
                supports=[list(s) for s in canonical], design=_geometry(np.c_[fr, fx]),
                X_given_R=_conditional_block(fr, fx),
                ordering="R coefficients, then X coefficients; rows are Y.ravel() in time-major order",
                estimator_scope="Unconstrained design geometry only; no bounds, active-set covariance, support-selection uncertainty, or fitting")
