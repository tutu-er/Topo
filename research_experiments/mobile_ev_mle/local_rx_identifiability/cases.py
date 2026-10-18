"""Frozen small physical subtrees and reproducible noisy P/Q/V records.

All fitting quantities are per unit.  ``truth`` is a separate evaluation-only
branch: pass ``observed``, ``meter``, ``domain`` and the declared candidates to
an estimator, never the whole returned object.  These are project synthetic
paper-style cases, not the original authors' numerical feeders or field data.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from .. import likelihood as lk
from ..reference_benchmark.networks import build_network


LOCAL_SELECTION = {
    "paper15": dict(root=3, hidden=6, terminals=(108, 109, 110)),
    "paper11": dict(root=3, hidden=5, terminals=(106, 107, 108)),
}
RATIO_BOUNDS = (.25, 1.5)
VOLTAGE_BOUNDS = (.90, 1.10)
ROOT_BOUNDS = (.97, 1.05)


def candidate_bank():
    """A common trunk/star and all three pair refinements, fixed in advance."""
    return tuple(lk.RadialCandidate.from_supports(name, 3, supports)
                 for name, supports in (("star", ()), ("pair01", ((0, 1),)),
                                        ("pair02", ((0, 2),)),
                                        ("pair12", ((1, 2),))))


def _canonical_name(network):
    name = {"soumalas11": "paper11", "samem11": "paper11"}.get(network, network)
    if name not in LOCAL_SELECTION:
        raise ValueError("network must be paper15 or paper11/soumalas11")
    return name


def public_domain(base_kw=200., base_kv=.4):
    """One line domain from *all* assets of both complete public case trees.

    The upper magnitude admits two series assets; the positive lower bound is
    the smallest complete-library asset.  No true local edge gets its own
    interval.  Component boxes are consequences of this magnitude/ratio domain;
    they alone do not enforce its magnitude lower bound.
    """
    networks = [build_network(name) for name in ("paper11", "paper15")]
    values = [complex(e["r_ohm"], e["x_ohm"])
              for n in networks for e in n["branches"]]
    zbase = float(base_kv) ** 2 / (float(base_kw) / 1000.)
    zmin_ohm = min(abs(z) for z in values)
    zmax_ohm = 2. * max(abs(z) for z in values)
    lo, hi = RATIO_BOUNDS
    zmin, zmax = zmin_ohm / zbase, zmax_ohm / zbase
    return dict(
        r_bounds=(zmin / np.sqrt(1. + hi**2), zmax / np.sqrt(1. + lo**2)),
        x_bounds=(zmin * lo / np.sqrt(1. + lo**2),
                  zmax * hi / np.sqrt(1. + hi**2)),
        z_min=zmin, z_max=zmax, ratio_bounds=RATIO_BOUNDS,
        ratio_min=lo, ratio_max=hi,
        v_bounds=VOLTAGE_BOUNDS, voltage_bounds=VOLTAGE_BOUNDS,
        v_min=VOLTAGE_BOUNDS[0], v_max=VOLTAGE_BOUNDS[1],
        root_bounds=ROOT_BOUNDS, root_min=ROOT_BOUNDS[0], root_max=ROOT_BOUNDS[1],
        library=dict(networks=("paper11", "paper15"), asset_count=len(values),
                     max_series_assets=2, z_min_ohm=zmin_ohm,
                     z_max_ohm=zmax_ohm,
                     source_sha256={n["name"]: n["provenance"]["source_sha256"]
                                    for n in networks},
                     ratio_rule="declared widening of synthetic material X/R to [0.25,1.5]",
                     lower_rule="minimum magnitude across both full unscaled asset libraries",
                     upper_rule="twice maximum magnitude across both full unscaled asset libraries"))


def _rng(seed, network, channel):
    code = 11 if network == "paper11" else 15
    return np.random.default_rng(np.random.SeedSequence([int(seed), code, int(channel)]))


def _ar(rng, count, width=1, rho=.8):
    """Stationary unit-variance AR(1), with no sample-variance rescaling."""
    z = rng.normal(size=(count, width))
    out = np.empty_like(z)
    out[0] = z[0]
    factor = np.sqrt(1. - rho**2)
    for t in range(1, count):
        out[t] = rho * out[t - 1] + factor * z[t]
    return out


def _trajectory(network, regime, seed, total, p0, q0):
    hours = np.arange(total, dtype=float) / 4.
    daily = .13 * np.cos(2. * np.pi * (hours - 19.) / 24.)
    morning = .06 * np.cos(4. * np.pi * (hours - 7.) / 24.)
    common = .84 + daily + morning
    common += .035 * _ar(_rng(seed, network, 1), total, rho=.9)[:, 0]
    independent = _ar(_rng(seed, network, 2), total, width=3, rho=.65)
    q_independent = _ar(_rng(seed, network, 3), total, width=3, rho=.7)
    q_common = _ar(_rng(seed, network, 4), total, rho=.85)[:, 0]
    phase = np.array([.15, 1.8, 3.6])
    individual_day = np.sin(2. * np.pi * hours[:, None] / 24. + phase)
    if regime == "ordinary":
        p_factor = common[:, None] + .07 * individual_day + .055 * independent
        q_factor = (.85 + .10 * np.cos(2. * np.pi * (hours - 18.) / 24.))[:, None]
        q_factor = q_factor + .06 * np.sin(2. * np.pi * hours[:, None] / 24. + phase + .8)
        q_factor += .045 * q_common[:, None] + .085 * q_independent
    else:
        # Weak independent variation is deliberately retained; Q is not k*P.
        p_factor = common[:, None] + .006 * independent
        q_factor = .98 * common[:, None] + .012 * q_common[:, None] + .006 * q_independent
    p_demand = p0[None, :] * p_factor
    q = q0[None, :] * q_factor
    # One predeclared terminal, unity power factor.  PV enters total P once.
    sun = np.maximum(0., np.sin(np.pi * ((hours % 24.) - 6.) / 12.))
    clouds = .90 + .04 * np.tanh(_ar(_rng(seed, network, 5), total, rho=.9)[:, 0])
    pv = np.zeros_like(p_demand)
    pv[:, 0] = .15 * p0[0] * sun * clouds
    p = p_demand - pv
    # Exogenous cut-boundary voltage, shared exactly by K/U paired cases.
    root = 1.01 + .008 * np.sin(2. * np.pi * (hours - 3.) / 24.)
    root += .003 * np.tanh(_ar(_rng(seed, network, 6), total, rho=.97)[:, 0])
    return p, q, root, pv, p_demand


def make_case(network, regime, seed, total=384, *, boundary="U"):
    """Make a 288-step history plus 96 held-out steps (15-minute sampling).

    Prefixes are reproducible when ``total`` changes.  U exposes no root
    trajectory.  K exposes the exact root as an *additional ideal calibrated
    sensor*; the original cut bus is unmetered.  No extra unmetered source is
    added in this first local identification experiment.
    """
    name = _canonical_name(network)
    regime = {"weak": "correlated", "strongly_correlated": "correlated"}.get(regime, regime)
    if regime not in ("ordinary", "correlated"):
        raise ValueError("regime must be ordinary or correlated")
    if boundary not in ("K", "U"):
        raise ValueError("boundary must be K or U")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(total, (bool, np.bool_)) or not isinstance(total, (int, np.integer)) or total < 1:
        raise ValueError("total must be a positive integer")
    native = build_network(name)
    selection = LOCAL_SELECTION[name]
    root, hidden, terminals = selection["root"], selection["hidden"], selection["terminals"]
    keep = {root, hidden, *terminals}
    branches = [deepcopy(e) for e in native["branches"]
                if e["from_bus"] in keep and e["to_bus"] in keep]
    buses = [deepcopy(b) for b in native["buses"] if b["bus_id"] in keep]
    by_bus = {b["bus_id"]: b for b in buses}
    if len(branches) != 4 or any(by_bus[b]["pd_kw"] or by_bus[b]["qd_kvar"] for b in (root, hidden)):
        raise RuntimeError("frozen subtree must contain four native edges and zero internal injections")
    p0_kw = np.array([by_bus[b]["pd_kw"] for b in terminals], float)
    q0_kvar = np.array([by_bus[b]["qd_kvar"] for b in terminals], float)
    base_kw, base_kv = native["base_kw"], native["base_kv"]
    zbase = base_kv**2 / (base_kw / 1000.)
    p0, q0 = p0_kw / base_kw, q0_kvar / base_kw
    candidates = candidate_bank()
    edges_by_child = {e["to_bus"]: e for e in branches}
    edge_order = [edges_by_child[hidden], *[edges_by_child[b] for b in terminals]]
    r = np.array([e["r_ohm"] / zbase for e in edge_order])
    x = np.array([e["x_ohm"] / zbase for e in edge_order])
    p, q, root_v, pv, p_demand = _trajectory(name, regime, int(seed), int(total), p0, q0)
    physical = lk.ac_predict(candidates[0], r, x, p, q, root_v=root_v)
    sigma_p = .01 * p0
    sigma_q = .01 * np.maximum(abs(q0), .2 * p0)
    sigma_v = np.full(3, .0005)
    observed = dict(
        p=p + _rng(seed, name, 11).normal(size=p.shape) * sigma_p,
        q=q + _rng(seed, name, 12).normal(size=q.shape) * sigma_q,
        v=physical["v"] + _rng(seed, name, 13).normal(size=p.shape) * sigma_v)
    if boundary == "K":
        observed["root_known"] = root_v.copy()
    domain = public_domain(base_kw, base_kv)
    # Declared operational envelope, identical across boundary/regime/seed.
    domain.update(p_bounds=(-p0.copy(), 2.5 * p0), q_bounds=(-q0.copy(), 2.5 * q0),
                  power_caps=2.5 * np.hypot(p0, q0))
    module = Path(__file__)
    nominal = dict(p=p0, q=q0, p_kw=p0_kw, q_kvar=q0_kvar,
                   base_kw=base_kw, base_kv=base_kv, zbase_ohm=zbase,
                   terminal_bus_ids=terminals)
    provenance = dict(
        schema="local_rx_case_v1", network=name, regime=regime, seed=int(seed),
        total=int(total), boundary=boundary, interval_minutes=15,
        training_prefixes=(96, 288), holdout_start=288, holdout_stop=384,
        source_kind="project_synthetic_paper_style_compatibility",
        original_paper_parameters=False, field_measurements=False,
        original_cut_bus_metered=False, cut_bus=root,
        known_boundary_is_additional_ideal_calibrated_sensor=boundary == "K",
        internal_loads_moved=0, retained_terminal_count=3,
        source_inputs_empty=True, pv_terminal_bus=terminals[0],
        pv_power_factor=1., pv_capacity_fraction=.15,
        pv_in_net_p_once=True, q_has_independent_variation=True,
        noise="independent Gaussian across channel, terminal and time; known fixed sigma; calibrated bias zero",
        temporal_profile_source="declared synthetic daily demand plus stationary AR(1); not public measured curves",
        root_process="exogenous slow daily voltage plus bounded AR(1), about one percent variation",
        subtree_selection="two physical three-terminal groups frozen before new meter noise",
        physical_parameter_source=deepcopy(native["provenance"]),
        generator_sha256=hashlib.sha256(module.read_bytes()).hexdigest())
    truth = dict(r=r, x=x, candidate_name="star", p=p, q=q,
                 root_v=root_v, v=physical["v"], pv_p=pv, demand_p=p_demand,
                 physical_tree=dict(branches=branches, buses=buses, root_bus=root,
                                    observation_bus_ids=terminals, source_bus=root,
                                    base_kv=base_kv, base_kw=base_kw),
                 ac_equation_residual=physical["ac_equation_residual"],
                 power_balance_residual=physical["power_balance_residual"])
    return dict(observed=observed,
                meter=dict(sigma_p=sigma_p, sigma_q=sigma_q, sigma_v=sigma_v),
                nominal=nominal, candidates=candidates,
                candidate_definitions=[dict(name=c.name, n=c.n, supports=c.supports)
                                       for c in candidates],
                domain=domain, provenance=provenance, truth=truth)


__all__ = ["LOCAL_SELECTION", "candidate_bank", "public_domain", "make_case"]
