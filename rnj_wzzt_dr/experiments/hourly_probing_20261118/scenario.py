"""Physical days, meters and resources for the hourly probing development runs.

The bundled frozen banks of the protocol (data/) are not on this PC, so every
input is regenerated here from the case files with the load law of
``physical.backend.generate_day96`` extended to several days. Truth is used only
to simulate the physics and to evaluate; fitting sees metered arrays only.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from rnj_wzzt_dr.physical import backend

from . import unbalanced

SLOTS = 96
NOMINAL_V = 230.9


def network_info(case, network=None):
    network = network or backend.load_case(case)
    terminals = [row for row in network["buses"] if row["is_terminal"]]
    labels = tuple(row["bus_id"] for row in terminals)
    p_nominal = np.array([row["pd_kw"] for row in terminals])/network["base_kw"]
    q_nominal = np.array([row["qd_kvar"] for row in terminals])/network["base_kw"]
    return network, labels, p_nominal, q_nominal


def truth_family(network, labels):
    """Descendant-set supports of the physical network (singletons included)."""
    order, parent, _ = backend._radial_order(network)
    descendants = {node: set() for node in order}
    for index, label in enumerate(labels):
        descendants[label].add(index)
    supports = set()
    for child in reversed(order[1:]):
        if descendants[child]:
            supports.add(tuple(sorted(descendants[child])))
        descendants[parent[child]].update(descendants[child])
    return tuple(sorted(supports, key=lambda s: (len(s), s)))


def generate_load(case, seed, days, *, network=None):
    """True P/Q (pu, consumption positive) and physical root voltage for `days` days."""
    network, labels, p_nominal, q_nominal = network_info(case, network)
    n = len(labels)
    rng = np.random.default_rng(np.random.SeedSequence([20261120, int(seed), len(labels)]))
    shifts = rng.uniform(-np.pi, np.pi, n)
    phase = 2*np.pi*np.arange(SLOTS)/SLOTS
    daily = .65+.20*np.sin(phase-.5*np.pi)+.10*np.sin(2*phase-.9*np.pi)
    P, Q, root = [], [], []
    for _ in range(days):
        fluct = rng.normal(0., .04, (SLOTS, n))
        for t in range(1, SLOTS):
            fluct[t] += .8*fluct[t-1]
        scale = rng.normal(1., .05)
        pf = np.clip(scale*daily[:, None]+.12*np.sin(phase[:, None]+shifts)+fluct, .15, 1.4)
        qf = pf*np.clip(1.+.15*np.sin(phase[:, None]-shifts)+rng.normal(0., .04, (SLOTS, n)), .5, 1.5)
        P.append(pf*p_nominal); Q.append(qf*q_nominal)
        root.append(1.02+.003*(.7*np.sin(phase+.3)+.3*rng.normal(size=SLOTS)))
    return np.vstack(P), np.vstack(Q), np.concatenate(root)


@dataclass
class Meters:
    """Per-input meter model: random error, fixed gain bias and quantisation."""
    voltage_sigma: float
    pq_sigma: float
    bias: np.ndarray            # per terminal, relative
    root_bias: float
    resolution_v: float

    def read(self, rng, p, q, v, root):
        p_m = p*(1.+self.pq_sigma*rng.normal(size=p.shape))
        q_m = q*(1.+self.pq_sigma*rng.normal(size=q.shape))
        v_m = v*(1.+self.bias+self.voltage_sigma*rng.normal(size=v.shape))
        r_m = root*(1.+self.root_bias+self.voltage_sigma*rng.normal(size=root.shape))
        if self.resolution_v > 0:
            step = self.resolution_v/NOMINAL_V
            v_m, r_m = np.round(v_m/step)*step, np.round(r_m/step)*step
        return p_m, q_m, r_m[:, None]**2-v_m**2, v_m, r_m


def make_meters(settings, n, seed):
    rng = np.random.default_rng(np.random.SeedSequence([7, int(seed)]))
    bias = settings.meter_bias_sigma*rng.normal(size=n)
    root_bias = settings.meter_bias_sigma*rng.normal()
    return Meters(settings.voltage_sigma, settings.pq_sigma, bias, root_bias, settings.voltage_resolution_v)


class Physics:
    """True network evaluation, balanced (physical.backend) or three-phase."""

    def __init__(self, network, labels, settings, *, seed=0, households=None):
        self.network, self.labels, self.settings = network, labels, settings
        self.three_phase = None
        if settings.unbalanced:
            self.three_phase = unbalanced.ThreePhase(network, labels, households, seed=seed,
                                                     z0_over_z1=settings.z0_over_z1)

    def voltages(self, p, q, root, probe=None):
        if self.three_phase is None:
            pp = p if probe is None else p+probe
            v, _ = backend._ac_voltages(self.network, self.labels, pp, q, root)
            return v
        return self.three_phase.terminal_voltages(p, q, root, probe)


def resources(settings, p_mean_kw, base_kw):
    """Scenario R1 envelope per node: battery (+/-) and EV (night only), in kW."""
    households = np.maximum(p_mean_kw/settings.household_kw, 1.)
    enrolled = settings.enrolled_share
    battery_kw = households*settings.battery_share*enrolled*settings.battery_kw
    battery_kwh = households*settings.battery_share*enrolled*settings.battery_kwh
    ev_kw = households*settings.ev_share*enrolled*settings.ev_kw
    return dict(households=households, battery_kw=battery_kw, battery_kwh=battery_kwh, ev_kw=ev_kw,
                base_kw=base_kw)


def ev_available(settings, hour):
    start, stop = settings.ev_hours
    return hour >= start or hour < stop


def headroom(settings, res, hour, soc):
    """Upward (+, more consumption) and downward (-) headroom in kW for one hour."""
    lo, hi = settings.soc_bounds
    e = res["battery_kwh"]
    up_b = np.minimum(res["battery_kw"], np.maximum((hi-soc)*e/settings.battery_eta, 0.))
    dn_b = np.minimum(res["battery_kw"], np.maximum((soc-lo)*e*settings.battery_eta, 0.))
    ev = res["ev_kw"] if ev_available(settings, hour) else 0.*res["ev_kw"]
    return up_b+ev, dn_b+settings.ev_down_fraction*ev


def split_block(settings, res, hour, soc, u_kw):
    """Serve a node block from batteries first, EVs after; return battery part."""
    lo, hi = settings.soc_bounds
    e = res["battery_kwh"]
    cap_up = np.minimum(res["battery_kw"], np.maximum((hi-soc)*e/settings.battery_eta, 0.))
    cap_dn = np.minimum(res["battery_kw"], np.maximum((soc-lo)*e*settings.battery_eta, 0.))
    battery = np.where(u_kw >= 0, np.minimum(u_kw, cap_up), -np.minimum(-u_kw, cap_dn))
    return battery


def update_soc(settings, res, soc, battery_kw):
    e = np.maximum(res["battery_kwh"], 1e-9)
    gain = np.where(battery_kw >= 0, settings.battery_eta, 1./settings.battery_eta)
    return np.clip(soc+battery_kw*gain/e, 0., 1.)


def prices(hour):
    """Activation price (CNY/kWh) by period from resources/library.json: battery, EV."""
    periods = [(0, 6), (6, 9), (9, 17), (17, 22), (22, 24)]
    battery = [.30, .35, .30, .50, .35]
    ev = [.30, .60, .60, .45, .30]
    for k, (a, b) in enumerate(periods):
        if a <= hour < b:
            return battery[k], ev[k]
    return battery[-1], ev[-1]


def node_price(settings, res, hour):
    """Headroom-weighted activation price per node (CNY/kWh)."""
    pb, pe = prices(hour)
    ev = res["ev_kw"] if ev_available(settings, hour) else 0.*res["ev_kw"]
    total = res["battery_kw"]+ev
    return (pb*res["battery_kw"]+pe*ev)/np.maximum(total, 1e-9)
