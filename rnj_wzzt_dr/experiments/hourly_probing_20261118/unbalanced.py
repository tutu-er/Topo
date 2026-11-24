"""Three-phase backward-forward sweep for the unbalanced Stage D evaluation.

Each household equivalent sits on one phase (seeded). Line self and mutual
impedances come from the positive-sequence library with Z0/Z1 (default 3),
Kron-reduced with the neutral earthed at the transformer:
Zs = (Z0 + 2 Z1)/3, Zm = (Z0 - Z1)/3. With equal phase shares the sweep
reduces exactly to the balanced sweep of ``physical.backend``.

A probe at a node is delivered on the phases of the enrolled devices that
supply it (one or two device units per node, each on a household's phase).
Each terminal's voltage is read on the phase of a seeded representative
household; its P and Q are the sums over phases (unchanged totals).
"""
from __future__ import annotations

import numpy as np

from rnj_wzzt_dr.physical import backend

A = np.exp(2j*np.pi/3)


class ThreePhase:
    def __init__(self, network, labels, households=None, *, seed=0, z0_over_z1=3., equal=False,
                 device_units=None):
        self.network, self.labels = network, tuple(labels)
        order, parent, incoming = backend._radial_order(network)
        self.order, self.parent = order, parent
        self.position = {bus: k for k, bus in enumerate(order)}
        zbase = network["base_kv"]**2/network["base_mva"]
        self.z1 = {c: complex(incoming[c]["r_ohm"], incoming[c]["x_ohm"])/zbase for c in order[1:]}
        self.k0 = float(z0_over_z1)
        n = len(self.labels)
        rng = np.random.default_rng(np.random.SeedSequence([33, int(seed)]))
        households = np.ones(n)*6 if households is None else np.asarray(households, float)
        counts = np.maximum(np.round(households).astype(int), 1)
        self.load_share = np.zeros((n, 3))
        self.reader = np.zeros(n, int)
        self.probe_share = np.zeros((n, 3))
        for i in range(n):
            phases = rng.integers(0, 3, counts[i])
            self.load_share[i] = np.bincount(phases, minlength=3)/counts[i]
            self.reader[i] = phases[rng.integers(0, counts[i])]
            units = 2 if device_units is None else int(device_units)
            supply = phases[rng.integers(0, counts[i], units)]
            self.probe_share[i] = np.bincount(supply, minlength=3)/units
        if equal:
            self.load_share[:] = 1/3; self.probe_share[:] = 1/3

    def _zabc(self, z1):
        z0 = self.k0*z1
        zs, zm = (z0+2*z1)/3, (z0-z1)/3
        return np.full((3, 3), zm)+np.eye(3)*(zs-zm)

    def phase_voltages(self, p, q, root, probe=None, *, tolerance=1e-10, max_iterations=200):
        """Complex phase voltages (T x buses x 3); p, q, probe in pu (3-phase base)."""
        p = np.asarray(p, float); q = np.asarray(q, float)
        T = p.shape[0]
        nb = len(self.order)
        s = np.zeros((T, nb, 3), complex)
        for k, label in enumerate(self.labels):
            col = self.position[label]
            s[:, col, :] = 3*(p[:, k, None]+1j*q[:, k, None])*self.load_share[k]
            if probe is not None:
                s[:, col, :] += 3*probe[:, k, None]*self.probe_share[k]
        source = np.asarray(root, float)[:, None]*np.array([1., A**2, A])[None, :]
        v = np.repeat(source[:, None, :], nb, axis=1)
        zmat = {c: self._zabc(self.z1[c]) for c in self.order[1:]}
        for it in range(max_iterations):
            current = np.conj(s/v)
            for c in reversed(self.order[1:]):
                current[:, self.position[self.parent[c]]] += current[:, self.position[c]]
            new = v.copy(); new[:, 0] = source
            for c in self.order[1:]:
                new[:, self.position[c]] = new[:, self.position[self.parent[c]]]-current[:, self.position[c]]@zmat[c].T
            err = float(np.max(np.abs(new-v)))
            v = new
            if err <= tolerance:
                return v
        raise RuntimeError("three-phase sweep did not converge")

    def terminal_voltages(self, p, q, root, probe=None):
        v = self.phase_voltages(p, q, root, probe)
        cols = [self.position[label] for label in self.labels]
        mags = np.abs(v[:, cols, :])
        return np.take_along_axis(mags, self.reader[None, :, None], axis=2)[:, :, 0]

    def all_phase_magnitudes(self, p, q, root, probe=None):
        v = self.phase_voltages(p, q, root, probe)
        cols = [self.position[label] for label in self.labels]
        return np.abs(v[:, cols, :])
