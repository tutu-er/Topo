"""Fixed historical variance: new observations add evidence, never refit."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def log_weights(sse, sigma2):
    loss = np.asarray(sse, dtype=float)
    if loss.ndim < 1 or not loss.shape[-1] or not np.isfinite(loss).all():
        raise ValueError("Finite SSE with candidates on the last axis required")
    if not np.isfinite(sigma2) or sigma2 <= 0 or np.any(loss < 0):
        raise ValueError("Nonnegative SSE and a frozen positive variance required")
    relative = -(loss-loss.min(axis=-1, keepdims=True))/(2*sigma2)
    return relative-logsumexp(relative, axis=-1, keepdims=True)


def retained(log_probability, beta):
    logp = np.asarray(log_probability, dtype=float)
    if logp.ndim != 1 or not len(logp) or not np.isfinite(logp).all() or not .5 < beta < 1:
        raise ValueError("Finite log weights and beta in (.5,1) required")
    if abs(float(logsumexp(logp))) > 1e-10:
        raise ValueError("Log weights are not normalized")
    order = np.argsort(-logp, kind="stable")
    count = int(np.searchsorted(np.cumsum(np.exp(logp[order])), beta, side="left"))+1
    count = min(count, len(order))
    boundary = logp[order[count-1]]
    return [int(i) for i in order if logp[i] >= boundary]


def add_observations(history_sse, H, power, response, sigma2):
    H, power, response = (np.asarray(x, float) for x in (H, power, response))
    if H.ndim != 3 or power.ndim != 2 or response.shape != (len(power), H.shape[1]):
        raise ValueError("H[candidate,node,channel], power[time,channel], response[time,node] required")
    fresh = np.sum((response[None]-np.einsum("tc,knc->ktn", power, H))**2, axis=(1, 2))
    total = np.asarray(history_sse, float)+fresh
    return dict(sse=total, log_probability=log_weights(total, sigma2), sigma2=float(sigma2))
