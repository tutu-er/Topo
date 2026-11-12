"""Baseline A: frozen-R/X closed-form nonnegative single-source scan.

Diagnostic only (design doc: it is a baseline/shortlister, not the final model).
With the tree edge weights fixed, the residual after the no-theft prediction is
scanned over all candidate locations; each candidate adds exactly one
nonnegative amplitude per time step with a closed-form improvement

    a_hat[h,t] = [f_h' W r_t]_+ / (f_h' W f_h),   Delta[h,t] = [f_h' W r_t]_+^2 / (f_h' W f_h)

where f_h is the candidate's shared-path response column.
"""

from __future__ import annotations

import numpy as np

from theft_wzzt.theft.theft_model import TheftData, TheftTree, predict


def response_columns(tree: TheftTree, r: np.ndarray, x: np.ndarray,
                     kappa: float = 0.0) -> np.ndarray:
    """f_h[i] = sum over edges on the root-to-h path of (r_e + kappa*x_e) * z_e[i]."""

    z, c = tree.incidence()
    w = np.asarray(r, dtype=float) + float(kappa) * np.asarray(x, dtype=float)
    return z.T @ (w[:, None] * c)  # (n_observed, n_candidates)


def scan_locations(tree: TheftTree, data: TheftData, r, x, *, kappa: float = 0.0,
                   weight: float | None = None) -> dict:
    """Closed-form per-time scan. Returns per-time best candidate and profiles.

    ``weight`` is the inverse noise scale of the voltage channel; defaults to
    ``1 / data.voltage_scale`` so Delta is commensurate with the MILP objective.
    Candidates with zero response (invisible under the projection) are marked
    unidentifiable and never win.
    """

    t_count = len(data.p)
    h_count = len(tree.candidates)
    z, _ = tree.incidence()
    zero_selection = np.zeros((t_count, h_count), dtype=int)
    residual = data.y - predict(tree, data, r, x, zero_selection)
    f = response_columns(tree, r, x, kappa)  # (n, h)
    w = float(weight) if weight is not None else 1.0 / data.voltage_scale
    fw = f * w
    denom = np.einsum("ih,ih->h", f, fw)
    visible = denom > 1e-15
    scores = np.full((t_count, h_count), -np.inf)
    a_hat = np.zeros((t_count, h_count))
    proj = residual @ fw  # (t, h) = f' W r
    gain = np.where(visible[None, :], np.maximum(proj, 0.0) ** 2 / np.maximum(denom, 1e-300), -np.inf)
    a_hat = np.where(visible[None, :], np.maximum(proj, 0.0) / np.maximum(denom, 1e-300), 0.0)
    best = np.argmax(gain, axis=1)
    best_gain = gain[np.arange(t_count), best]
    best_gain[~np.isfinite(best_gain)] = 0.0
    return {
        "residual": residual,
        "response": f,
        "visible": visible,
        "gain": gain,                    # (T, H) improvement over H0
        "amplitude_hat": a_hat,          # (T, H)
        "best_index": best,              # (T,)
        "best_location": [tree.candidates[i] for i in best],
        "best_gain": best_gain,          # (T,)
        "best_amplitude": a_hat[np.arange(t_count), best],
    }
