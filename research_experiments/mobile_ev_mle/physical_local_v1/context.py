"""Root-connected context and exact reductions; no topology truth selection.

The full-context constructor can operate on any supplied candidate physical
tree. The state-dependent port function is strictly an oracle validation tool:
its exact port powers must not be supplied as observations to an estimator.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from ..reference_benchmark.representation import _orient, physical_tree_ac, impedance_base_ohm


def rooted_scope(network, focus_bus_ids):
    """Retain all candidate-tree physics; annotate paths, including side loads."""
    focus = tuple(focus_bus_ids)
    _, parent, edge_for, order = _orient(network["branches"], network["root_bus"])
    if not focus or len(set(focus)) != len(focus) or any(b not in parent for b in focus):
        raise ValueError("distinct present focus buses required")
    paths = []
    for b in focus:
        path = [b]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        paths.append(list(reversed(path)))
    keep = set().union(*map(set, paths))
    shared = set(paths[0]).intersection(*map(set, paths[1:]))
    junction = max(shared, key=lambda b: order.index(b))
    path_edges = [i for b, i in edge_for.items() if b in keep]
    offshoots = [(parent[b], b) for b in order[1:] if parent[b] in keep and b not in keep]
    return dict(root_bus=network["root_bus"], focus_bus_ids=list(focus),
        junction_bus=junction, paths=paths, path_bus_ids=[b for b in order if b in keep],
        path_edge_indices=path_edges, side_subtree_ports=offshoots,
        retained_physical_bus_count=len(network["buses"]),
        retained_measurement_bus_ids=list(network["observation_bus_ids"]),
        model="full candidate tree; local focus is an annotation, not a cut",
        all_original_edges_retained=True)


def compress_zero_injection_series(network, protected_buses=()):
    """Sum series Z only across unmeasured zero-injection non-source nodes."""
    out = deepcopy(network)
    protected = {network["root_bus"], network.get("source_bus", network["root_bus"]),
                 *network["observation_bus_ids"], *protected_buses}
    for e in out["branches"]:
        e["original_branch_ids"] = [e["branch_id"]]
    removed = []
    while True:
        adjacency = {b["bus_id"]: [] for b in out["buses"]}
        for i, e in enumerate(out["branches"]):
            adjacency[e["from_bus"]].append((e["to_bus"], i))
            adjacency[e["to_bus"]].append((e["from_bus"], i))
        targets = [b for b in out["buses"] if b["bus_id"] not in protected
                   and not b["pd_kw"] and not b["qd_kvar"]
                   and len(adjacency[b["bus_id"]]) == 2]
        if not targets:
            break
        b = targets[0]["bus_id"]
        (u, i), (v, j) = adjacency[b]
        a, c = out["branches"][i], out["branches"][j]
        merged = dict(branch_id=max(e["branch_id"] for e in out["branches"])+1,
            from_bus=u, to_bus=v, r_ohm=a["r_ohm"]+c["r_ohm"],
            x_ohm=a["x_ohm"]+c["x_ohm"], status=True, branch_type="exact_series",
            original_branch_ids=a["original_branch_ids"]+c["original_branch_ids"],
            length_km=(a["length_km"]+c["length_km"]
                       if a.get("length_km") is not None and c.get("length_km") is not None else None))
        out["branches"] = [e for k, e in enumerate(out["branches"]) if k not in (i, j)]+[merged]
        out["buses"] = [z for z in out["buses"] if z["bus_id"] != b]
        removed.append(b)
    out["metadata"]["series_contraction"] = dict(removed_bus_ids=removed,
        preserves="all measured complex voltages, root complex power and series losses",
        source_and_meter_buses_protected=True)
    return out


def port_reduction_oracle(network, focus_bus_ids, p_kw, q_kvar, root_v, state,
                          *, include_losses=True):
    """Validate a root-connected skeleton with exact or naive side ports.

    Receives a complete AC state and simulator P/Q: evaluation only. Exact
    powers preserve that state, not arbitrary future actions without updates.
    """
    scope = rooted_scope(network, focus_bus_ids)
    rows, parent, edge_for, order = _orient(network["branches"], network["root_bus"])
    keep = set(scope["path_bus_ids"])
    root = network["root_bus"]; ids = network["observation_bus_ids"]
    p = np.asarray(p_kw, float); q = np.asarray(q_kvar, float)
    if p.ndim != 2 or q.shape != p.shape or p.shape[1] != len(ids):
        raise ValueError("matching complete-tree P/Q arrays required")
    total = {b: np.zeros(len(p), complex) for b in order}
    for j, b in enumerate(ids):
        total[b] += p[:, j]+1j*q[:, j]
    subtree = {b: z.copy() for b, z in total.items()}
    for b in reversed(order[1:]):
        subtree[parent[b]] += subtree[b]
    port_records = []
    zbase = impedance_base_ohm(network["base_kv"], network["base_kw"])
    voltages = state["voltage_by_bus"]
    for a, b in scope["side_subtree_ports"]:
        e = rows[edge_for[b]]
        z = complex(e["r_ohm"], e["x_ohm"])/zbase
        if z == 0:
            raise ValueError("port oracle needs a positive physical connection edge")
        current = (voltages[a]-voltages[b])/z
        exact = voltages[a]*np.conj(current)*network["base_kw"]
        chosen = exact if include_losses else subtree[b]
        total[a] += chosen
        port_records.append(dict(attachment=a, subtree_root=b,
            max_omitted_loss_kva=float(np.max(abs(exact-subtree[b])))))
    reduced_ids = [b for b in order if b in keep and b != root]
    reduced_p = np.column_stack([total[b].real for b in reduced_ids])
    reduced_q = np.column_stack([total[b].imag for b in reduced_ids])
    reduced_edges = [e for e in network["branches"]
                     if e["from_bus"] in keep and e["to_bus"] in keep]
    result = physical_tree_ac(reduced_edges, reduced_ids, root, network["base_kv"],
        network["base_kw"], reduced_p, reduced_q, root,
        total[root].real, total[root].imag, root_v=root_v)
    return result, dict(evaluation_only=True, oracle_ports=include_losses,
        retained_bus_ids=scope["path_bus_ids"], ports=port_records,
        root_to_focus_edges_preserved=True,
        estimator_input_allowed=False, declared_scope="single operating trajectory port validation")


def local_refinements(candidate, focus_positions):
    """Predeclare pair refinements of a supplied candidate; retain all context.

    This routine has no measured future or truth parameter argument. It only
    proposes structure; it does not assert candidate coverage for unknown grids.
    """
    from ..likelihood import RadialCandidate
    focus = tuple(focus_positions)
    if len(focus) != 3 or len(set(focus)) != 3 or not all(0 <= i < candidate.n for i in focus):
        raise ValueError("three distinct observed positions required")
    bank = [candidate]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        support = tuple(sorted((focus[a], focus[b])))
        if support in candidate.supports:
            continue
        bank.append(RadialCandidate.from_supports(f"pair{a}{b}_full_context", candidate.n,
                                                 [*candidate.supports, support]))
    return tuple(bank)


__all__ = ["rooted_scope", "compress_zero_injection_series", "port_reduction_oracle",
           "local_refinements"]
