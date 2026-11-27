"""AC power flow of a radial feeder (with the unmetered load) and the meter readings.

Balanced single-phase equivalent, per unit; loads are constant power. Every quantity an identifier sees is produced
by `read`; the identifiers' linear model is never used to make data.

A copy of the case library's physics.py (rnj_wzzt_dr/experiments/case_library_20261125, registered with the generator),
with the radial ordering copied in so that nothing outside this directory is needed. Only the imports differ.
"""
from __future__ import annotations

import numpy as np


def radial_order(network):
    """Orient a connected radial graph away from its root (a copy of rnj_wzzt_dr/physical/backend._radial_order)."""
    root = network["root_bus"]
    buses = {row["bus_id"] for row in network["buses"]}
    active = [row for row in network["branches"] if row["status"]]
    if len(active) != len(buses)-1:
        raise ValueError("The bundled generator requires a connected radial case")
    adjacency = {bus: [] for bus in buses}
    for edge in active:
        a, b = edge["from_bus"], edge["to_bus"]
        adjacency[a].append((b, edge)); adjacency[b].append((a, edge))
    order, parent, incoming = [root], {root: None}, {}
    for bus in order:
        for child, edge in sorted(adjacency[bus], key=lambda value: value[0]):
            if child == parent[bus]:
                continue
            if child in parent:
                raise ValueError("The physical case contains a cycle")
            parent[child], incoming[child] = bus, edge
            order.append(child)
    if set(order) != buses:
        raise ValueError("The physical case is disconnected")
    return order, parent, incoming


def power_flow(network, labels, p, q, v0, extra=None, tolerance=1e-11, max_iterations=300):
    """Voltages (pu) of all buses and the complex power drawn at the PCC.

    p, q: slots x meters (pu). extra: None or (bus, p, q) of one unmetered load.
    Returns (order of buses, |V| slots x buses, S at the PCC, current in the line to each bus)."""
    order, parent, incoming = radial_order(network)
    at = {bus: k for k, bus in enumerate(order)}
    zbase = network["base_kv"]**2/network["base_mva"]
    load = np.zeros((len(v0), len(order)), complex)
    for k, label in enumerate(labels):
        load[:, at[label]] += p[:, k]+1j*q[:, k]
    if extra is not None:
        load[:, at[extra[0]]] += extra[1]+1j*extra[2]
    z = {b: complex(incoming[b]["r_ohm"], incoming[b]["x_ohm"])/zbase for b in order[1:]}
    v = np.repeat(np.asarray(v0, float)[:, None], len(order), axis=1).astype(complex)
    for _ in range(max_iterations):
        current = np.conj(load/v)
        for b in reversed(order[1:]):
            current[:, at[parent[b]]] += current[:, at[b]]
        new = v.copy()
        for b in order[1:]:
            new[:, at[b]] = new[:, at[parent[b]]]-z[b]*current[:, at[b]]
        error = np.max(np.abs(new-v)); v = new
        if np.min(np.abs(v)) < .2:
            raise RuntimeError("collapsed voltage")
        if error < tolerance:
            return order, np.abs(v), v[:, 0]*np.conj(current[:, 0]), np.abs(current)
    raise RuntimeError("the power flow did not converge")


class Meters:
    """Fixed errors of one set of meters (drawn once) and random reading errors.

    Voltage: gain error 0.05% and noise 0.05%. P and Q: gain error 0.3% (0.2% at the PCC) and noise 0.5%."""

    def __init__(self, n, seed_key, v_gain=.0005, v_noise=.0005, pq_gain=.003, pcc_gain=.002, pq_noise=.005):
        rng = np.random.default_rng(np.random.SeedSequence([20261125, 14, *seed_key]))
        self.v_noise, self.pq_noise = v_noise, pq_noise
        self.v_gain, self.v0_gain = v_gain*rng.normal(size=n), v_gain*rng.normal()
        self.p_gain, self.q_gain = pq_gain*rng.normal(size=n), pq_gain*rng.normal(size=n)
        self.pcc_gain = pcc_gain*rng.normal(size=2)
        self.rng = rng

    def read(self, p, q, v, v0, s_pcc):
        noise = lambda a, s: s*self.rng.normal(size=a.shape)
        return dict(P=p*(1+self.p_gain+noise(p, self.pq_noise)), Q=q*(1+self.q_gain+noise(q, self.pq_noise)),
                    V=v*(1+self.v_gain+noise(v, self.v_noise)), V0=v0*(1+self.v0_gain+noise(v0, self.v_noise)),
                    Ppcc=s_pcc.real*(1+self.pcc_gain[0]+noise(v0, self.pq_noise)),
                    Qpcc=s_pcc.imag*(1+self.pcc_gain[1]+noise(v0, self.pq_noise)))


def hourly(a):
    """Average of the four quarter-hours of every hour."""
    return a.reshape(len(a)//4, 4, *a.shape[1:]).mean(axis=1)


def read(true, meters, per_hour):
    """Meter readings at one value per hour (averages of the quarter-hours) or four."""
    f = hourly if per_hour == 1 else (lambda a: a)
    return meters.read(f(true["P"]), f(true["Q"]), f(true["V"]), f(true["V0"]), f(true["S_pcc"]))
