"""Common loss and finite-simulation diagnostics; scores are not probabilities."""
import numpy as np
from .model import fit


def evaluate(data, prediction, balance_prediction):
    v = np.abs(data.y - prediction)
    m = np.abs(data.balance - balance_prediction)
    lv, lm = float(v.sum() / data.voltage_scale), float(m.sum() / data.balance_scale)
    return {"loss": lv + lm, "voltage_loss": lv, "balance_loss": lm,
            "voltage_mae": float(v.mean()), "balance_mae": float(m.mean()),
            "voltage_scaled_mae": float(v.mean() / data.voltage_scale),
            "balance_scaled_mae": float(m.mean() / data.balance_scale)}


def compare(tree, data):
    h0 = fit(tree, data, allow_theft=False)
    h1 = fit(tree, data)
    s0 = evaluate(data, h0.prediction, h0.balance_prediction)
    s1 = evaluate(data, h1.prediction, h1.balance_prediction)
    gain = s0["loss"] - s1["loss"]
    if gain < -1e-5:
        raise RuntimeError("Nested H1 must include H0")
    return {"h0": s0, "h1": s1, "gain": gain}, h0, h1


def pilot_rank(gain, null_gains):
    values = np.asarray(null_gains, float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Require finite full-procedure null simulations")
    return float((1 + np.count_nonzero(values >= gain - 1e-7)) / (len(values) + 1))


def location_profile(tree, data, time, best=None):
    """Force one time/location, refit all R/X and all other time locations."""
    best = fit(tree, data) if best is None else best
    options = [-1] if data.amplitude[time] == 0 else range(-1, len(tree.candidates))
    rows = []
    for location in options:
        other = fit(tree, data, fixed_locations={time: location})
        rows.append({"location": "none" if location < 0 else tree.candidates[location],
                     "loss": other.objective, "loss_gap": other.objective - best.objective})
    return rows
