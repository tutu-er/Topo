"""Randomness utilities."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a NumPy generator from a seed."""

    return np.random.default_rng(seed)

