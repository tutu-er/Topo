"""Baselines B and C for comparison against the joint theft MILP.

- Baseline B: master-meter energy balance only. Alarm when the signed balance
  exceeds a threshold; provides detection but no location. This isolates the
  incremental value of the voltage channel.
- Baseline C: the plain no-theft wzzT fit (H0). Its residual level is the
  reference that any theft explanation must beat; comparison is done through
  ``credibility.compare_h0_h1`` so both models refit R/X on the same domain.
"""

from __future__ import annotations

import numpy as np

from theft_wzzt.theft.theft_model import TheftData


def balance_alarm(data: TheftData, threshold: float) -> np.ndarray:
    """Per-time alarm from the master-meter balance channel alone."""

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    return data.balance > threshold


def balance_alarm_rate(data: TheftData, threshold: float) -> float:
    return float(np.mean(balance_alarm(data, threshold)))
