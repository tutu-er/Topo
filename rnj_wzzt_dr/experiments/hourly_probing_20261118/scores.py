"""BIC refit scoring (manuscript (17)), equivalence (19), posterior (20), types."""
from __future__ import annotations

import math
import numpy as np


def bic(sses, ps, N):
    sses, ps = np.asarray(sses, float), np.asarray(ps, float)
    sigma2 = float(np.min(sses/(N-ps)))
    J = sses/(2*sigma2)+.5*ps*math.log(N)
    return J, sigma2


def voltages(cand, theta, P, Q, v0sq, loss=True):
    y = cand.predict(theta, P, Q, v0sq, loss=loss)
    return np.sqrt(np.maximum(np.asarray(v0sq).reshape(-1, 1)-y, .25))


def equivalent_mask(cands, thetas, leader, P, Q, v0sq, eps, loss=True):
    vL = voltages(cands[leader], thetas[leader], P, Q, v0sq, loss)
    out = np.zeros(len(cands), bool)
    for k, (c, th) in enumerate(zip(cands, thetas)):
        if k == leader:
            out[k] = True
            continue
        out[k] = float(np.max(np.abs(voltages(c, th, P, Q, v0sq, loss)-vL))) <= eps
    return out


def posterior_ok(E, equivalent, leader, alpha):
    rivals = [k for k in range(len(E)) if k != leader and not equivalent[k]]
    total = float(np.sum(np.exp(-np.asarray(E)[rivals]))) if rivals else 0.
    return total <= alpha/(1-alpha), total


def rival_type(key, truth_key):
    """Type of a candidate relative to the truth, on non-singleton supports."""
    if truth_key is None or key == truth_key:
        return "correct" if key == truth_key else "unknown"
    if key > truth_key:
        return "extra_line"
    if key < truth_key:
        return "missing_line"
    return "wrong_placement"
