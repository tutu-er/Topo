"""Safe graph-only control bounds for the unchanged mobile service MILP.

The graph already specifies access/setup windows, travel time, starting and
terminal states. Dynamic programming adds the declared move and worker-cost
budgets. It does not solve a MILP, enumerate complete routes, or inspect AC
models. Battery, throughput and service constraints remain in the original
operation dictionary; this helper does not use them to claim feasibility.
"""
from __future__ import annotations

import math

import numpy as np


def graph_reachability(domain, *, cost_tolerance=1e-9):
    """Find connected arcs contained in a complete budget-feasible graph path.

    A state is (slot, site, connected). The extra DP coordinate counts travel
    arcs. For each state/count, retaining the cheapest prefix and suffix is
    sufficient because the only other graph resource is additive worker cost.
    The small explicit tolerance retains paths near the budget boundary; it
    can only weaken pruning. SOC/energy and destination-service feasibility
    are deliberately not asserted by this graph calculation.
    """
    if not np.isfinite(cost_tolerance) or cost_tolerance < 0:
        raise ValueError("cost_tolerance must be finite and nonnegative")
    cfg, ix = domain.config, domain.indices
    T, J = int(cfg.n_slots), int(cfg.n_sites)
    arcs = ix.arcs
    source = (0, int(cfg.initial_site), bool(cfg.initial_connected))
    nodes = {source}
    outgoing = {}
    max_cost = 0.
    for i, a in enumerate(arcs):
        # Zero-time setup is the sole within-slot transition. Sorting False
        # before True therefore gives a topological order even in that case.
        key = lambda n: (n[0], int(n[2]), n[1])
        if key(a.start) >= key(a.end):
            raise ValueError("Expected the mobile service acyclic graph")
        if not np.isfinite(a.driver_cost) or a.driver_cost < 0:
            raise ValueError("Graph costs must be finite and nonnegative")
        if a.kind == "travel" and a.arrival <= a.departure:
            raise ValueError("Travel must advance at least one slot")
        nodes.update((a.start, a.end))
        outgoing.setdefault(a.start, []).append(i)
        max_cost = max(max_cost, float(a.driver_cost))
    order = sorted(nodes, key=lambda n: (n[0], int(n[2]), n[1]))
    terminal = {n for n in nodes if n[0] == T
                and (cfg.terminal_site is None or n[1] == cfg.terminal_site)
                and (not cfg.terminal_connected or n[2])}
    moves = T if cfg.max_moves is None else min(T, int(cfg.max_moves))
    # At most T travel arcs, and at most one zero-time setup between successive
    # time-advancing arcs. This scale keeps ordinary rounding from deleting an
    # exactly-on-budget path without using an undisclosed physical tolerance.
    budget = math.inf if cfg.driver_budget is None else float(cfg.driver_budget)
    scale = max(1., 0. if math.isinf(budget) else budget, max_cost * (2 * T + 1))
    tolerance = float(cost_tolerance) * scale
    forward = {n: np.full(moves + 1, np.inf) for n in nodes}
    backward = {n: np.full(moves + 1, np.inf) for n in nodes}
    forward[source][0] = 0.
    for node in order:
        for i in outgoing.get(node, ()):
            a = arcs[i]
            step = int(a.kind == "travel")
            for m in range(moves + 1 - step):
                value = forward[node][m] + a.driver_cost
                if value < forward[a.end][m + step]:
                    forward[a.end][m + step] = value
    for node in terminal:
        backward[node][0] = 0.
    for node in reversed(order):
        for i in outgoing.get(node, ()):
            a = arcs[i]
            step = int(a.kind == "travel")
            for m in range(moves + 1 - step):
                value = backward[a.end][m] + a.driver_cost
                if value < backward[node][m + step]:
                    backward[node][m + step] = value

    reachable_arcs = np.zeros(len(arcs), bool)
    for i, a in enumerate(arcs):
        step = int(a.kind == "travel")
        for m in range(moves + 1 - step):
            p = forward[a.start][m]
            if not np.isfinite(p):
                continue
            remaining = moves - m - step
            s = np.min(backward[a.end][:remaining + 1], initial=np.inf)
            if np.isfinite(s) and math.fsum((p, a.driver_cost, s)) <= budget + tolerance:
                reachable_arcs[i] = True
                break
    arc_positions = {int(v): i for i, v in enumerate(ix.arc)}
    connected = np.zeros((T, J), bool)
    for t in range(T):
        for j in range(J):
            connected[t, j] = reachable_arcs[arc_positions[int(ix.connected_arc[t, j])]]
    minimum_cost = float(np.min(backward[source], initial=np.inf))
    graph_path_exists = bool(np.isfinite(minimum_cost) and minimum_cost <= budget + tolerance)
    return dict(connected=connected, arc=reachable_arcs,
                graph_path_exists=graph_path_exists,
                minimum_graph_worker_cost=None if not np.isfinite(minimum_cost) else minimum_cost,
                max_moves_layer=moves, cost_tolerance_absolute=tolerance,
                source=list(source), terminal_states=[list(n) for n in sorted(terminal)],
                scope="graph, initial/terminal state, move count and declared worker-cost budget",
                omitted_feasibility="SOC, terminal energy, throughput and arrival service remain original MILP constraints")


def tighten_operation_bounds(domain, *, cost_tolerance=1e-9):
    """Return the selector operation dictionary with only redundant bounds.

    The input domain/model is unchanged. All original rows, variable indices,
    control ordering and cross-site zero-product statements are preserved.
    Callers can use the returned dictionary wherever ``domain.as_dict()`` was
    used; ``domain.indices``, ``domain.check`` and ``domain.decode`` remain
    valid for its full operation vectors.

    Proof: every feasible integer flow is one source-to-terminal path in this
    DAG. If a connected arc belongs to no such budget-feasible path, its binary
    variable is zero. The original charging links then force c=d=u=0. Adding
    those bounds preserves the feasible set of the original MILP exactly.
    """
    reach = graph_reachability(domain, cost_tolerance=cost_tolerance)
    operation = domain.as_dict()
    # as_dict currently returns copies; copy the bounds explicitly as part of
    # this helper's no-mutation contract if that implementation later changes.
    operation["lower"] = np.asarray(operation["lower"], float).copy()
    operation["upper"] = np.asarray(operation["upper"], float).copy()
    nonzero_original_box = ((operation["control_lower"] != 0)
                            | (operation["control_upper"] != 0))
    mask = reach["connected"]
    tightened_variables = []
    for name in ("u", "charge", "discharge"):
        indices = getattr(domain.indices, name)[~mask].ravel()
        if np.any(operation["lower"][indices] > 0) or np.any(operation["upper"][indices] < 0):
            raise ValueError("An unreachable control excludes zero in its original box")
        operation["lower"][indices] = 0.
        operation["upper"][indices] = 0.
        tightened_variables.extend(int(i) for i in indices)
    controls = operation["control_indices"]
    operation["control_lower"] = operation["lower"][controls].copy()
    operation["control_upper"] = operation["upper"][controls].copy()
    reach["unreachable_control_count"] = int(np.count_nonzero(~mask))
    reach["newly_zero_control_count"] = int(np.count_nonzero(
        (~mask).ravel() & nonzero_original_box))
    reach["tightened_variable_indices"] = tightened_variables
    operation["graph_reachability"] = reach
    return operation
