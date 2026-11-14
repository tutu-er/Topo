"""Exact expected evidence gains; no probability inequalities or new fits."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Evidence:
    # base[K,time,node], effect[K,node,resource], both divided by scales[K].
    base: np.ndarray
    effect: np.ndarray
    scales: np.ndarray
    gain_required: float
    buffer: float = 1e-5

    def __post_init__(self):
        self.base = np.asarray(self.base, float)
        self.effect = np.asarray(self.effect, float)
        self.scales = np.asarray(self.scales, float)
        if (self.base.ndim != 3 or self.effect.ndim != 3
                or self.effect.shape[:2] != (self.base.shape[0], self.base.shape[2])
                or self.scales.shape != (len(self.base),)
                or not all(np.isfinite(a).all() for a in (self.base, self.effect, self.scales))
                or np.any(self.scales <= 0) or self.gain_required <= 0):
            raise ValueError("Finite evidence arrays and a positive gain required")
        self.active = np.any(self.effect != 0, axis=1)
        self.base_norm2 = np.sum(self.base**2, axis=(1, 2))
        self.radii = np.sqrt(self.base_norm2+(self.gain_required+self.buffer)/self.scales**2)

    def contributions(self, delivery):
        return np.einsum("knr,rt->krtn", self.effect, delivery)

    def gains(self, delivery):
        shift = self.contributions(delivery).sum(axis=1)
        # This difference identity avoids cancellation against a large baseline.
        return np.sum(2*self.base*shift+shift*shift, axis=(1, 2))*self.scales**2

    def feasible(self, delivery):
        return bool(np.min(self.gains(delivery)) >= self.gain_required-1e-8)


def build(H, target, competitors, forecast, sigma2, original, base_kw, gain):
    H, forecast = np.asarray(H), np.asarray(forecast)
    difference = H[target][None]-H[np.asarray(competitors)]
    scale = np.sqrt(2*sigma2)
    raw_base = np.einsum("tc,knc->ktn", forecast, difference)/scale
    n = H.shape[1]
    raw_effect = (original.response_effect(difference, base_kw)/scale if hasattr(original, 'response_effect')
                  else (difference[:, :, original.node]
                        +difference[:, :, n+original.node]*original.qratio[None, None])/(base_kw*scale))
    scales = np.maximum(1., np.linalg.norm(raw_base, axis=(1, 2)))
    evidence = Evidence(raw_base/scales[:, None, None], raw_effect/scales[:, None, None], scales, gain)
    return evidence


def direct_gains(H, target, competitors, forecast, sigma2, original, base_kw, delivery):
    """Independent full-H computation, not the solver's resource effects."""
    dp, dq = original.aggregate(delivery)
    shift = np.column_stack((dp, dq))/base_kw
    answer = []
    for j in competitors:
        difference = np.asarray(H[target])-np.asarray(H[j])
        old, delta = forecast@difference.T, shift@difference.T
        answer.append(float(np.sum(2*old*delta+delta*delta)/(2*sigma2)))
    return np.asarray(answer)
