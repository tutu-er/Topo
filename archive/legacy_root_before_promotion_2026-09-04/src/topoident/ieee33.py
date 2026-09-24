"""IEEE 33-bus data and reproducible synthetic smart-meter records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from terminal_load_only_case33.terminal_case33.data.case33bw_raw import (
    _BUS_LOADS,
    _CLOSED_BRANCHES,
    _TIE_BRANCHES,
)

from .powerflow import Branch, build_ybus, solve_power_flow


@dataclass
class MeterData:
    p: np.ndarray
    q: np.ndarray
    v: np.ndarray
    theta_true: np.ndarray
    branches: list[Branch]


_LINE_DATA = _CLOSED_BRANCHES
_TIE_DATA = _TIE_BRANCHES
_LOAD_KW_KVAR = [(pd_kw, qd_kvar) for _, pd_kw, qd_kvar in _BUS_LOADS]

def ieee33_branches(*, looped: bool = True) -> tuple[list[Branch], list[Branch]]:
    base_mva = 10.0
    base_kv = 12.66
    z_base = base_kv**2 / base_mva

    def convert(row: tuple[int, int, float, float]) -> Branch:
        u, v, r, x = row
        return Branch(u - 1, v - 1, r / z_base, x / z_base)

    closed = [convert(row) for row in _LINE_DATA]
    ties = [convert(row) for row in _TIE_DATA]
    if looped:
        # The article studies a 33-edge weakly meshed case. Its exact switch
        # trace was not released, so we explicitly close standard tie 21--8.
        closed = closed + [ties[0]]
    return closed, ties


def nominal_loads_pu() -> tuple[np.ndarray, np.ndarray]:
    loads = np.asarray(_LOAD_KW_KVAR, dtype=float)
    return loads[:, 0] / 10_000.0, loads[:, 1] / 10_000.0


def simulate_meter_data(
    *,
    samples: int = 120,
    seed: int = 7,
    p_noise: float = 0.002,
    q_noise: float = 0.002,
    v_noise: float = 0.0001,
    looped: bool = True,
) -> MeterData:
    """Simulate synchronized p/q/v records with known ground truth."""
    rng = np.random.default_rng(seed)
    branches, _ = ieee33_branches(looped=looped)
    ybus = build_ybus(33, branches)
    p0, q0 = nominal_loads_pu()
    p_all = np.zeros((samples, 33))
    q_all = np.zeros((samples, 33))
    v_all = np.zeros((samples, 33))
    theta_all = np.zeros((samples, 33))
    previous = None
    bus_shape = rng.lognormal(mean=0.0, sigma=0.12, size=33)
    bus_shape[0] = 0.0

    for t in range(samples):
        phase = 2.0 * np.pi * t / max(samples, 24)
        common = 0.72 + 0.18 * np.sin(phase - 0.8) + 0.10 * np.sin(2 * phase + 0.3)
        independent = rng.lognormal(mean=-0.5 * 0.18**2, sigma=0.18, size=33)
        scale = np.clip(common * bus_shape * independent, 0.25, 1.35)
        scale[0] = 0.0
        p_true = -p0 * scale
        q_true = -q0 * scale * rng.lognormal(mean=0.0, sigma=0.04, size=33)
        voltage = solve_power_flow(ybus, p_true, q_true, initial=previous)
        previous = voltage

        p_all[t] = p_true * (1.0 + rng.normal(0.0, p_noise, 33))
        q_all[t] = q_true * (1.0 + rng.normal(0.0, q_noise, 33))
        v_all[t] = np.abs(voltage) * (1.0 + rng.normal(0.0, v_noise, 33))
        theta_all[t] = np.angle(voltage)

    # Slack injection is unavailable from a customer meter; balance the noisy
    # non-slack injections only to keep the regression dimensions consistent.
    p_all[:, 0] = -p_all[:, 1:].sum(axis=1)
    q_all[:, 0] = -q_all[:, 1:].sum(axis=1)
    return MeterData(p_all, q_all, v_all, theta_all, branches)
