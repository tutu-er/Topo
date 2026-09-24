"""Validation routines for terminal-load-only scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from rnj_wzzt.models.network import TerminalizedNetwork


def validate_terminal_load_only_scenario(
    net: TerminalizedNetwork,
    strict: bool = True,
    output_path: str | Path | None = None,
) -> dict:
    """Validate graph structure, observability labels, and load placement."""

    graph = net.to_networkx_graph()
    violations: list[str] = []
    is_connected = nx.is_connected(graph)
    is_acyclic = nx.is_tree(graph)
    if not is_connected:
        violations.append("graph is not connected")
    if not is_acyclic:
        violations.append("closed graph is not acyclic")

    roots = net.buses[net.buses["bus_type"].eq("root")]
    if len(roots) != 1 or int(roots.iloc[0]["bus_id"]) != net.root_bus:
        violations.append("root node is not unique or does not match net.root_bus")

    degree = dict(graph.degree())
    load_buses = net.buses[net.buses["has_load"]]
    for row in load_buses.itertuples():
        if int(row.bus_id) != net.root_bus and degree.get(int(row.bus_id), 0) != 1:
            violations.append(f"non-root load bus {int(row.bus_id)} is not a leaf")

    hidden = net.buses[net.buses["bus_type"].eq("hidden_internal")]
    if bool(hidden["has_load"].any()):
        violations.append("hidden_internal node has load")
    if bool(hidden["is_observed"].any()):
        violations.append("hidden_internal node is observed")

    terminals = net.buses[net.buses["bus_type"].eq("observed_terminal")]
    if bool((~terminals["has_load"]).any()):
        violations.append("observed_terminal node without load")
    root_rows = net.buses[net.buses["bus_id"].eq(net.root_bus)]
    if root_rows.empty or not bool(root_rows.iloc[0]["is_observed"]):
        violations.append("root is not observed")

    original_p = float(
        net.metadata.get("original_total_pd_kw", net.buses["pd_kw"].sum())
    )
    original_q = float(
        net.metadata.get("original_total_qd_kvar", net.buses["qd_kvar"].sum())
    )
    load_error_p = abs(float(net.buses["pd_kw"].sum()) - original_p)
    load_error_q = abs(float(net.buses["qd_kvar"].sum()) - original_q)
    if load_error_p > 1e-9 or load_error_q > 1e-9:
        violations.append("load is not conserved")

    hidden_degree_counts = (
        hidden["hidden_degree"].astype(int).value_counts().sort_index().to_dict()
    )
    degree_2_hidden = (
        hidden.loc[hidden["hidden_degree"].eq(2), "bus_id"].astype(int).tolist()
    )
    summary = {
        "is_connected": is_connected,
        "is_acyclic": is_acyclic,
        "root_bus": int(net.root_bus),
        "observed_terminal_count": int(len(terminals)),
        "hidden_internal_count": int(len(hidden)),
        "load_bus_count": int(len(load_buses)),
        "hidden_degree_distribution": {
            str(key): int(value) for key, value in hidden_degree_counts.items()
        },
        "degree_2_hidden_chains": degree_2_hidden,
        "degree_2_note": (
            "cannot be uniquely recovered from terminal-only measurements "
            "unless additional assumptions or meters are added"
        ),
        "load_conservation_error_pd_kw": load_error_p,
        "load_conservation_error_qd_kvar": load_error_q,
        "violations": violations,
        "strict": strict,
    }
    if strict and violations:
        raise ValueError(
            "terminal-load-only validation failed: " + "; ".join(violations)
        )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = ["validate_terminal_load_only_scenario"]
