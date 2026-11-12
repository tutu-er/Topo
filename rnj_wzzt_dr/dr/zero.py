"""Existing Kronecker spectral zero-control integral and union bound."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp

def spectral_pair(data, competitor, temporal):
    """Target is data.current; return strict exclusion sufficient bounds."""
    target = data["current"]
    H1, H0 = data["H"][[target, competitor]]
    T, n = data["forecast"].shape[0], len(data["labels"])
    residual = data["history_drop"]-data["history_power"]@H1.T
    sd = np.maximum(np.sqrt(np.mean(residual**2, axis=0)),
                    2*np.sqrt(2)*data["voltage_sigma"])
    s = float(sd.max())
    if s <= 0:
        raise ValueError("Degenerate original remaining-voltage error")
    lam = .25
    t = lam*(1-2*lam*(sd/s)**2)
    sigma = data["power_relative_sd"]*np.max(np.abs(data["forecast"]), axis=0)
    dh = H1-H0
    F = (dh[:, :n]*sigma[:n]+dh[:, n:]*sigma[n:])/s
    V, singular, _ = np.linalg.svd(np.sqrt(t)[:, None]*F, full_matrices=True)
    r, U = temporal
    eigen = 1+2*r[:, None]*singular[None, :]**2
    transformed = U.T@((data["forecast"]@dh.T/s)*np.sqrt(t))@V
    q = float(np.sum(transformed**2/eigen))
    half_logdet = float(.5*np.log(eigen).sum())
    total = data["history_drop"].size+T*n
    delta = float((total*data["strict_scaled_epsilon"]/data["K"]
                   -data["history_sse"][competitor]+data["history_sse"][target])/(s*s))
    score = q+half_logdet
    log_fail = min(0., lam*delta-score)
    model, other = data["models"][target], data["models"][competitor]
    same_family = model["topology_id"] == other["topology_id"]
    equal_response = bool(np.array_equal(H1, H0))
    # Equal predictions give equal scores forever. Original fixed tie ordering
    # decides whether this declared target survives; no strict exclusion is
    # necessary when its tie priority wins.
    if equal_response:
        if (other["internal"], competitor) < (model["internal"], target):
            log_fail = 0.
        else:
            log_fail = -np.inf
    tie_win = equal_response and log_fail == -np.inf
    return dict(competitor=other["id"], same_family=bool(same_family),
        equal_response=equal_response, q=q, half_logdet=half_logdet,
        score=score, lambda_=lam, delta=delta, scale_pu2=s,
        historical_MSE_advantage=float(other["history_mse"]-model["history_mse"]),
        log_failure_upper=None if tie_win else log_fail,
        pair_success_lower=1. if tie_win else float(-np.expm1(log_fail)),
        pair_90_sufficient=bool(tie_win or log_fail <= np.log(.1)),
        strict_margin=score-lam*delta-np.log(10))


def joint(rows):
    logs = [r["log_failure_upper"] for r in rows if r["log_failure_upper"] is not None]
    log_sum = float(logsumexp(logs)) if logs else -np.inf
    return dict(competitors=len(rows),
        log_union_failure_upper=None if not logs else min(0., log_sum),
        success_lower=1. if not logs else float(-np.expm1(min(0., log_sum))),
        sufficient_90=bool(log_sum <= np.log(.1)),
        union_log_margin=None if not logs else float(np.log(.1)-log_sum),
        unproved_pair_count=sum(not r["pair_90_sufficient"] for r in rows))
