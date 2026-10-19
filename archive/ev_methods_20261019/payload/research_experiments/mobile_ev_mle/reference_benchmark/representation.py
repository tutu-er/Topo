"""Exact measured-tree representation, including loads on internal buses.

An observed physical bus can be represented by a virtual leaf joined with
exactly zero impedance.  Its constant-power injection and measured voltage do
not change.  After this operation the existing terminal-tree AC and derivative
code applies.  The additional zero edges are bookkeeping, not new conductors.

If a series chain has no measured or source injection at its intermediate
nodes, every edge carries the same complex current.  Thus summing its R and X
gives precisely the original voltage difference and I^2(R+iX) loss.  A support
is the set of measured descendants: equal supports identify such chains.  The
source must be at the LCA of all measured buses, which is the end of the full-
support chain.  This prevents aggregating an edge across the source injection.

Together these two transformations preserve every measured complex voltage,
the root complex power, and the full nonlinear constant-power AC equations.
The topology target is the resulting measured-subtree hierarchy.  It does not
identify zero virtual edges or individual unobserved series sections.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..likelihood import RadialCandidate


def impedance_base_ohm(base_kv: float, base_kw: float) -> float:
    """Three-phase power base and line-to-line voltage base, both consistent."""
    if not np.isfinite([base_kv, base_kw]).all() or base_kv <= 0 or base_kw <= 0:
        raise ValueError("positive finite base_kv and three-phase base_kw required")
    return float(base_kv**2 / (base_kw / 1000.))


def _rows(branches):
    if hasattr(branches, "to_dict"):
        branches = branches.to_dict("records")
    rows = []
    for original_index, item in enumerate(branches):
        row = dict(item)
        if not bool(row.get("status", row.get("closed", row.get("is_true_closed", True)))):
            continue
        u, v = row["from_bus"], row["to_bus"]
        r, x = float(row["r_ohm"]), float(row["x_ohm"])
        if u == v or not np.isfinite([r, x]).all() or min(r, x) < 0:
            raise ValueError("closed branches must have distinct buses and finite nonnegative R/X")
        rows.append(dict(from_bus=u, to_bus=v, r_ohm=r, x_ohm=x,
                         original_index=original_index))
    if not rows:
        raise ValueError("at least one closed physical branch required")
    return rows


def _orient(branches, root_bus):
    rows = _rows(branches)
    adjacency = {}
    for i, row in enumerate(rows):
        u, v = row["from_bus"], row["to_bus"]
        adjacency.setdefault(u, []).append((v, i))
        adjacency.setdefault(v, []).append((u, i))
    if root_bus not in adjacency:
        raise ValueError("root bus absent from physical tree")
    parent = {root_bus: None}; edge_for = {}; order = [root_bus]
    for node in order:
        for neighbor, i in adjacency[node]:
            if neighbor == parent[node]:
                continue
            if neighbor in parent:
                raise ValueError("closed physical graph must be acyclic, without duplicate edges")
            parent[neighbor] = node; edge_for[neighbor] = i; order.append(neighbor)
    if len(order) != len(adjacency) or len(rows) != len(order)-1:
        raise ValueError("closed physical graph must be connected and radial")
    return rows, parent, edge_for, order


@dataclass(frozen=True)
class EncodedTree:
    candidate: RadialCandidate
    r: np.ndarray
    x: np.ndarray
    source_bus: Any
    zbase_ohm: float
    observation_bus_ids: tuple
    physical_edge_to_candidate_edge: tuple
    metadata: dict

    def record(self):
        return dict(candidate=self.candidate.metadata(), r=self.r.tolist(), x=self.x.tolist(),
                    source_bus=self.source_bus, zbase_ohm=self.zbase_ohm,
                    observation_bus_ids=list(self.observation_bus_ids),
                    physical_edge_to_candidate_edge=list(self.physical_edge_to_candidate_edge),
                    metadata=self.metadata)


def encode_physical_tree(branches, observation_bus_ids, root_bus, base_kv, base_kw,
                         source_bus=None, name="reference") -> EncodedTree:
    """Encode the simulator's physical tree without moving any injection.

    Inputs are a radial branch table in ohm and ordered nonroot measured buses.
    All nonroot injections must occur at these buses, apart from the one source.
    This function is for the simulator/physical audit; its returned true lines
    and true candidate must never enter the observed-only planner.

    Unobserved injection-free dangling branches may be removed, explicitly
    recorded as empty-support branches.  A source on such a branch is rejected.
    """
    physical_rows = branches.to_dict("records") if hasattr(branches, "to_dict") else tuple(branches)
    rows, parent, edge_for, order = _orient(physical_rows, root_bus)
    observed = tuple(observation_bus_ids)
    if len(observed) < 2 or len(set(observed)) != len(observed):
        raise ValueError("at least two distinct ordered measured buses required")
    if root_bus in observed or any(bus not in parent for bus in observed):
        raise ValueError("measured buses must be present nonroot buses")
    descendants = {bus: set() for bus in order}
    for j, bus in enumerate(observed):
        descendants[bus].add(j)
    for bus in reversed(order[1:]):
        descendants[parent[bus]].update(descendants[bus])
    universe = set(range(len(observed)))
    full_nodes = [bus for bus in order if descendants[bus] == universe]
    # Equal full supports form one root path; BFS order chooses its last node.
    interface = full_nodes[-1]
    if source_bus is not None and source_bus != interface:
        raise ValueError("source_bus must be the measured-bus LCA at the end of the full-support chain")
    support_for = {bus: tuple(sorted(descendants[bus])) for bus in order[1:]}
    supports = [s for s in support_for.values() if s]
    candidate = RadialCandidate.from_supports(name, len(observed), supports)
    indices = {tuple(s): i for i, s in enumerate(candidate.supports)}
    zbase = impedance_base_ohm(base_kv, base_kw)
    r = np.zeros(candidate.m); x = np.zeros(candidate.m)
    original_map = [None] * len(physical_rows)
    aggregation = [[] for _ in range(candidate.m)]; removed = []
    for bus in order[1:]:
        row = rows[edge_for[bus]]; source_index = row["original_index"]
        support = support_for[bus]
        if not support:
            removed.append(source_index)
            continue
        e = indices[support]; r[e] += row["r_ohm"]/zbase; x[e] += row["x_ohm"]/zbase
        original_map[source_index] = e; aggregation[e].append(source_index)
    virtual = [i for i, edges in enumerate(aggregation) if not edges]
    # An observed bus's support may include descendants; its virtual singleton
    # leaf consequently has zero R/X. Physical-leaf singleton lines stay real.
    return EncodedTree(candidate, r, x, interface, zbase, observed, tuple(original_map),
        dict(base_kv=float(base_kv), base_kw=float(base_kw),
             impedance_units="pu; supplied physical table in ohm", power_units="three-phase kW/kvar",
             root_bus=root_bus, source_bus=interface,
             source_rule="measured-bus LCA; physical full-support series chain end",
             source_components=["P", "Q"], constant_power=True,
             edge_physical_indices=aggregation, virtual_zero_edges=virtual,
             virtual_zero_supports=[list(candidate.supports[i]) for i in virtual],
             removed_zero_injection_empty_support_edges=removed,
             topology_scope="nontrivial hierarchy of measured subtrees; virtual/series edge identity excluded",
             exact_zero_edges_not_conductors=True, no_positive_service_edges_added=True))


def physical_tree_ac(branches, observation_bus_ids, root_bus, base_kv, base_kw,
                     p_kw, q_kvar, source_bus, source_p_kw=0., source_q_kvar=0.,
                     root_v=1., *, tol=2e-13, max_iter=1000):
    """Independent general-tree backward/forward oracle, including internal loads.

    It deliberately does not use RadialCandidate, supports, B, or zbus.  This
    audit implementation is not a planner and receives simulator-only inputs.
    """
    rows, parent, edge_for, order = _orient(branches, root_bus)
    observed = tuple(observation_bus_ids)
    if source_bus not in parent or root_bus in observed or len(set(observed)) != len(observed):
        raise ValueError("invalid measured/source bus map")
    if any(bus not in parent for bus in observed):
        raise ValueError("measured bus missing from physical tree")
    p = np.asarray(p_kw, float)
    if p.ndim == 1:
        p = p[None, :]
    q = np.asarray(q_kvar, float)
    if q.ndim == 1:
        q = q[None, :]
    if p.ndim != 2 or not len(p) or p.shape[1] != len(observed) or q.shape != p.shape:
        raise ValueError("P/Q must be matching time-by-measured-bus matrices")
    t = len(p); root = np.broadcast_to(np.asarray(root_v, float), (t,))
    sp = np.broadcast_to(np.asarray(source_p_kw, float), (t,))
    sq = np.broadcast_to(np.asarray(source_q_kvar, float), (t,))
    if not all(np.isfinite(a).all() for a in (p, q, root, sp, sq)) or np.min(root) <= 0:
        raise ValueError("finite powers and positive root voltage required")
    zbase = impedance_base_ohm(base_kv, base_kw)
    loads = {bus: np.zeros(t, complex) for bus in order}
    for j, bus in enumerate(observed):
        loads[bus] += (p[:, j]+1j*q[:, j])/base_kw
    loads[source_bus] += (sp+1j*sq)/base_kw
    voltage = {bus: root.astype(complex).copy() for bus in order}
    impedances = {bus: complex(rows[edge_for[bus]]["r_ohm"], rows[edge_for[bus]]["x_ohm"])/zbase
                  for bus in order[1:]}
    for iteration in range(1, max_iter+1):
        if min(np.min(abs(voltage[b])) for b in order) < .1:
            raise RuntimeError("independent AC oracle reached low voltage")
        currents = {bus: np.conj(loads[bus]/voltage[bus]) for bus in order}
        for bus in reversed(order[1:]):
            currents[parent[bus]] += currents[bus]
        new = {root_bus: root.astype(complex).copy()}
        for bus in order[1:]:
            new[bus] = new[parent[bus]]-impedances[bus]*currents[bus]
        error = max(np.max(abs(new[b]-voltage[b])) for b in order)
        voltage = new
        if not all(np.isfinite(v).all() for v in voltage.values()):
            raise RuntimeError("independent AC oracle produced nonfinite voltage")
        if error <= tol:
            break
    else:
        raise RuntimeError("independent AC oracle failed to converge")
    currents = {bus: np.conj(loads[bus]/voltage[bus]) for bus in order}
    for bus in reversed(order[1:]):
        currents[parent[bus]] += currents[bus]
    master = root*np.conj(currents[root_bus])
    losses = sum(impedances[b]*abs(currents[b])**2 for b in order[1:])
    total_load = sum(loads.values())
    balance = master-total_load-losses
    equation = max(np.max(abs(voltage[b]-voltage[parent[b]]+impedances[b]*currents[b]))
                   for b in order[1:])
    return dict(v=np.column_stack([abs(voltage[b]) for b in observed]),
                measured_voltage=np.column_stack([voltage[b] for b in observed]),
                master_p=master.real, master_q=master.imag,
                source_bus=source_bus, voltage_by_bus=voltage,
                ac_equation_residual=float(equation),
                power_balance_residual=float(np.max(abs(balance))),
                iterations=iteration)


def public_fit_options(*, base_kv, base_kw, r_ohm_upper, x_ohm_upper,
                        ratio_bounds, source_p_kw_bound, source_q_kvar_bound,
                        max_series_edges=1, voltage_bounds=(.8, 1.2),
                        multistart=2, max_nfev=500, tol=1e-8):
    """Common candidate domain from explicitly supplied *public* asset limits.

    There is no tree/true-parameter argument. max_series_edges is a declared
    worst-case aggregate-chain cap, not a true-path query. Lower endpoints 0
    include virtual meter/root edges and near-short alternatives. They provide
    no positive-length separation guarantee. The caller records the public
    provenance of these bounds separately.
    """
    zbase = impedance_base_ohm(base_kv, base_kw)
    values = (r_ohm_upper, x_ohm_upper, source_p_kw_bound, source_q_kvar_bound)
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("positive public line/source bounds required")
    if int(max_series_edges) != max_series_edges or max_series_edges < 1:
        raise ValueError("max_series_edges must be a positive integer")
    lo, hi = map(float, ratio_bounds)
    if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
        raise ValueError("public X/R bounds require 0 < lower < upper")
    return dict(r_bounds=(0., r_ohm_upper*max_series_edges/zbase),
                x_bounds=(0., x_ohm_upper*max_series_edges/zbase),
                ratio_bounds=(lo, hi),
                source_p_bounds=(-source_p_kw_bound/base_kw, source_p_kw_bound/base_kw),
                source_q_bounds=(-source_q_kvar_bound/base_kw, source_q_kvar_bound/base_kw),
                voltage_bounds=tuple(voltage_bounds), multistart=int(multistart),
                max_nfev=int(max_nfev), tol=float(tol))


__all__ = ["EncodedTree", "impedance_base_ohm", "encode_physical_tree", "physical_tree_ac",
           "public_fit_options"]
