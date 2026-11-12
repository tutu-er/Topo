"""Module 1: unified credibility evaluation for theft-identification candidates.

Promoted from ``rnj_wzzt_core/tests/meter_theft_pilot/credibility.py`` and
generalized. Provides:

- a common two-channel data loss (voltage + master balance), fixed scales;
- fair nested H0/H1 comparison (both refit R/X on the same domain);
- full-procedure null calibration: rerun H0/H1 on clean scenarios to obtain
  the empirical null distribution of the gain, and rank an observed gain;
- a location profile: force each location at one time and refit, exposing
  which candidates remain compatible (equivalence classes stay visible).

Scores are NOT probabilities. A calibrated rank supports an alarm threshold
conditional on the declared null-generating process only.
"""

from __future__ import annotations

import numpy as np

from theft_wzzt.theft.theft_model import TheftData, TheftTree, fit_theft_milp


def evaluate(data: TheftData, prediction, balance_prediction) -> dict:
    v = np.abs(data.y - prediction)
    m = np.abs(data.balance - balance_prediction)
    lv, lm = float(v.sum() / data.voltage_scale), float(m.sum() / data.balance_scale)
    return {"loss": lv + lm, "voltage_loss": lv, "balance_loss": lm,
            "voltage_mae": float(v.mean()), "balance_mae": float(m.mean()),
            "voltage_scaled_mae": float(v.mean() / data.voltage_scale),
            "balance_scaled_mae": float(m.mean() / data.balance_scale)}


def compare_h0_h1(tree: TheftTree, data: TheftData, **fit_kwargs) -> dict:
    """Fit H0 (no theft) and H1 (at most one source per time) on equal footing."""

    h0 = fit_theft_milp(tree, data, allow_theft=False, **fit_kwargs)
    h1 = fit_theft_milp(tree, data, allow_theft=True, **fit_kwargs)
    s0 = evaluate(data, h0.prediction, h0.balance_prediction)
    s1 = evaluate(data, h1.prediction, h1.balance_prediction)
    gain = s0["loss"] - s1["loss"]
    if gain < -1e-3:
        raise RuntimeError(f"Nested H1 must include H0 (gain={gain})")
    return {"h0": s0, "h1": s1, "gain": gain,
            "h0_diagnostics": h0.diagnostics, "h1_diagnostics": h1.diagnostics}, h0, h1


def calibrated_rank(gain: float, null_gains) -> float:
    """Empirical rank of the observed gain against full-procedure null gains."""

    values = np.asarray(null_gains, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Require finite full-procedure null simulations")
    return float((1 + np.count_nonzero(values >= gain - 1e-7)) / (len(values) + 1))


def null_threshold(null_gains, alpha: float = 0.05) -> float:
    """Empirical upper-alpha quantile of the null gain distribution."""

    values = np.asarray(null_gains, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Require finite full-procedure null simulations")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return float(np.quantile(values, 1.0 - alpha))


def location_profile(tree: TheftTree, data: TheftData, time: int, *,
                     best=None, **fit_kwargs) -> list[dict]:
    """Force one time/location, refit all R/X and all other time locations.

    Rows with small loss_gap are the still-compatible locations; a wide set
    means the data cannot localize, even when the best gain is large.
    """

    best = fit_theft_milp(tree, data, **fit_kwargs) if best is None else best
    options = [-1] if data.amplitude[time] == 0 else range(-1, len(tree.candidates))
    rows = []
    for location in options:
        other = fit_theft_milp(tree, data, fixed_locations={time: location}, **fit_kwargs)
        rows.append({"location": "none" if location < 0 else tree.candidates[location],
                     "loss": other.objective, "loss_gap": other.objective - best.objective})
    return rows


def compatibility_report(comparison: dict, null_gains, *, alpha: float = 0.05) -> dict:
    """Assemble the module-1 output: gain, calibrated rank, threshold, verdicts."""

    gain = comparison["gain"]
    rank = calibrated_rank(gain, null_gains)
    threshold = null_threshold(null_gains, alpha)
    return {
        "gain": gain,
        "h0_loss": comparison["h0"]["loss"],
        "h1_loss": comparison["h1"]["loss"],
        "h1_scaled_voltage_mae": comparison["h1"]["voltage_scaled_mae"],
        "h1_scaled_balance_mae": comparison["h1"]["balance_scaled_mae"],
        "calibrated_rank": rank,
        "null_threshold": threshold,
        "null_replicates": int(len(null_gains)),
        "alpha": alpha,
        "alarm": bool(rank <= alpha),
        "residual_compatible": bool(comparison["h1"]["voltage_scaled_mae"] <= 1.0),
    }
