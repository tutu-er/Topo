"""External loss estimates in pu; L1 uses measured channels only.

L0/L2/L3 and the default L4 calibration deliberately use counterfactual truth
and are simulation controls, not deployable estimators. L1 is a single forward
quadratic approximation on the supplied tree, with weights (2*r_pu, 2*x_pu).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _series(value, index, name, *, nonnegative=False):
    if isinstance(value, pd.Series) and not value.index.equals(index):
        raise ValueError(f"{name}: time index does not match terminal meters")
    a = np.asarray(value, dtype=float)
    if a.shape != (len(index),) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: require a finite time series")
    if nonnegative and np.any(a < 0):
        raise ValueError(f"{name}: require nonnegative values")
    return pd.Series(a, index=index, name=name)


def estimate_loss(scenario, weights=None, tree=None, *, model="L0", bias=0.0,
                  sigma=0.25, seed=20260919, ratio=None):
    """Return aligned P/Q loss Series. All powers and voltages are per unit.

    L3 has zero-mean Gaussian relative noise BEFORE nonnegative clipping;
    clipping means it is not exactly unbiased. Both channels share the draw.
    L4 uses (c_p, c_q) * max(sum(P_terminal), 0). Supply ratio to avoid reading
    truth at inference; ratio=None is an explicit oracle-calibrated control.
    """
    p, q = scenario["P_terminal"], scenario["Q_terminal"]
    if (not p.index.equals(q.index) or not p.columns.equals(q.columns)
            or not p.index.is_unique or not p.columns.is_unique or len(p) == 0
            or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q))):
        raise ValueError("Require finite, aligned P/Q meters with unique labels")
    if model == "L1":
        if tree is None or weights is None or len(weights) != 2:
            raise ValueError("L1 requires a tree and (r, x) weights")
        if set(p.columns) != set(tree.observed):
            raise ValueError("Terminal labels do not match the tree")
        z, _ = tree.incidence()
        w = np.asarray(weights, dtype=float)
        if w.shape != (2, len(tree.edges)) or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("Require finite nonnegative weights in tree edge order")
        voltage = _series(scenario["root_voltage"], p.index, "root_voltage")
        if np.any(voltage <= 0):
            raise ValueError("Require positive root voltage")
        fp = p.loc[:, list(tree.observed)].to_numpy() @ z.T
        fq = q.loc[:, list(tree.observed)].to_numpy() @ z.T
        current2 = (fp**2 + fq**2) / voltage.to_numpy()[:, None]**2
        losses = current2 @ (w / 2).T
    elif model == "L4":
        total = np.maximum(p.sum(axis=1).to_numpy(), 0)
        if ratio is None:
            if total.sum() <= 0:
                raise ValueError("L4 calibration requires positive net consumption")
            reference = estimate_loss(scenario, model="L0")
            ratio = [float(v.sum() / total.sum()) for v in reference]
        ratio = np.asarray(ratio, dtype=float)
        if ratio.shape != (2,) or not np.all(np.isfinite(ratio)) or np.any(ratio < 0):
            raise ValueError("L4 requires two finite nonnegative loss ratios")
        losses = total[:, None] * ratio
    elif model in ("L0", "L2", "L3"):
        losses = np.column_stack([
            _series(scenario[f"loss_{channel}_notheft"], p.index, channel,
                    nonnegative=True) for channel in ("p", "q")])
        if model == "L2":
            if not np.isfinite(bias) or bias < -1:
                raise ValueError("Require finite bias >= -1")
            losses = losses * (1 + bias)
        if model == "L3":
            if not np.isfinite(sigma) or sigma < 0:
                raise ValueError("Require finite sigma >= 0")
            factor = np.maximum(1 + np.random.default_rng(seed).normal(0, sigma, len(p)), 0)
            losses = losses * factor[:, None]
    else:
        raise ValueError(f"Unknown loss model {model}")
    return tuple(_series(losses[:, j], p.index, f"loss_{c}_estimate", nonnegative=True)
                 for j, c in enumerate(("p", "q")))


def amplitude_envelope(scenario, loss_p, *, beta_lo=-0.25, beta_hi=0.25):
    """Deterministic loss-only envelope, NOT a confidence interval for theft.

    Assumes actual technical loss is in [(1+beta_lo)*loss_p,
    (1+beta_hi)*loss_p]. Meter error and theft-induced extra loss are not bounded
    by this declaration. Relative halfwidth is undefined at zero point estimate.
    """
    if not (np.isfinite(beta_lo) and np.isfinite(beta_hi)
            and -1 <= beta_lo <= 0 <= beta_hi):
        raise ValueError("Require -1 <= beta_lo <= 0 <= beta_hi")
    p = scenario["P_terminal"]
    loss = _series(loss_p, p.index, "loss_p", nonnegative=True)
    master = _series(scenario["P0_measured"], p.index, "P0_measured")
    balance = master - p.sum(axis=1)
    result = pd.DataFrame({
        "lower": np.maximum(balance - (1 + beta_hi) * loss, 0),
        "point": np.maximum(balance - loss, 0),
        "upper": np.maximum(balance - (1 + beta_lo) * loss, 0),
    }, index=p.index)
    result["relative_halfwidth"] = ((result.upper - result.lower) / 2
                                      / result.point.where(result.point > 0))
    return result
