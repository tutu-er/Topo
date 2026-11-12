"""Observed P/Q and squared-voltage responses; no network or truth loader."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Observation:
    """Time-by-terminal P/Q and observed squared-voltage drop in per unit."""

    p: np.ndarray
    q: np.ndarray
    y: np.ndarray
    labels: tuple[int, ...]
    voltage: np.ndarray | None = None
    root: np.ndarray | None = None

    def __post_init__(self):
        labels = tuple(map(int, self.labels))
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("Observation labels must be nonempty and unique")
        arrays = [np.asarray(value, dtype=float) for value in (self.p, self.q, self.y)]
        if any(value.ndim != 2 for value in arrays) or not all(value.shape == arrays[0].shape for value in arrays):
            raise ValueError("p, q and y must have the same time-by-terminal shape")
        if arrays[0].shape[1] != len(labels) or not arrays[0].shape[0] or not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("Observation dimensions or finite-value checks failed")
        for name, value in zip(("p", "q", "y"), arrays):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "labels", labels)
        if self.voltage is not None:
            voltage = np.asarray(self.voltage, dtype=float)
            if voltage.shape != arrays[0].shape or not np.isfinite(voltage).all():
                raise ValueError("Voltage shape or finite-value check failed")
            object.__setattr__(self, "voltage", voltage)
        if self.root is not None:
            root = np.asarray(self.root, dtype=float).reshape(-1)
            if root.shape != (arrays[0].shape[0],) or not np.isfinite(root).all():
                raise ValueError("Root-voltage shape or finite-value check failed")
            object.__setattr__(self, "root", root)

    @property
    def n(self):
        return len(self.labels)

    @property
    def points(self):
        return self.p.shape[0]

    @property
    def rows(self):
        return self.points


def load(path):
    """Read explicitly named observed arrays, never test/truth/catalog entries."""
    with np.load(Path(path), allow_pickle=False) as z:
        labels = tuple(map(int, z["labels"]))
        p, q = [np.asarray(z[k], float).reshape(-1, len(labels)) for k in ("P", "Q")]
        if "Y" in z:
            return Observation(p, q, np.asarray(z["Y"], float).reshape(p.shape), labels)
        v = np.asarray(z["V"], float).reshape(p.shape)
        root = np.asarray(z["root"], float).reshape(-1)
    if np.any(v <= 0) or np.any(root <= 0):
        raise ValueError("Observed voltages must be positive")
    return Observation(p, q, root[:, None]**2-v**2, labels, v, root)


def fingerprint(observed):
    h = hashlib.sha256(repr(tuple(observed.labels)).encode())
    for values in (observed.p, observed.q, observed.y):
        a = np.ascontiguousarray(values, dtype=np.float64)
        h.update(repr(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()
