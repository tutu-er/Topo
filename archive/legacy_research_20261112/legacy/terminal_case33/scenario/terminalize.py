"""Construct strict terminal-load-only variants of IEEE case33."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd

from terminal_case33.data.case33bw_raw import RawCase33
from terminal_case33.models.network import TerminalizedNetwork


def _raw_graph(raw: RawCase33) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(raw.buses["bus_id"].astype(int).tolist())
    closed = raw.branches[raw.branches["status"].astype(bool)]
    graph.add_edges_from((int(r.from_bus), int(r.to_bus)) for r in closed.itertuples())
    return graph


def _service_impedance(
    raw: RawCase33,
    attach_bus: int,
    mode: str,
    rng: np.random.Generator,
    service_length_m: float,
    service_length_random_range_m: tuple[float, float] | None,
    service_r_ohm_per_km: float | None,
    service_x_ohm_per_km: float | None,
) -> tuple[float, float, float]:
    """Generate service-edge impedance in Ohm."""

    length_m = service_length_m
    if service_length_random_range_m is not None:
        length_m = float(rng.uniform(*service_length_random_range_m))

    if mode in {"scaled_original", "random_perturbed"}:
        adj = raw.branches[
            raw.branches["status"].astype(bool)
            & ((raw.branches["from_bus"].eq(attach_bus)) | (raw.branches["to_bus"].eq(attach_bus)))
        ]
        med_r = float(adj["r_ohm"].median()) if not adj.empty else float(raw.branches["r_ohm"].median())
        med_x = float(adj["x_ohm"].median()) if not adj.empty else float(raw.branches["x_ohm"].median())
        r_ohm, x_ohm = 0.2 * med_r, 0.2 * med_x
    elif mode == "fixed_ohm":
        r_ohm = service_r_ohm_per_km if service_r_ohm_per_km is not None else 0.05
        x_ohm = service_x_ohm_per_km if service_x_ohm_per_km is not None else 0.02
    elif mode == "lv_cable_soumalas_style":
        r_km = service_r_ohm_per_km if service_r_ohm_per_km is not None else 1.380
        x_km = service_x_ohm_per_km if service_x_ohm_per_km is not None else 0.082
        r_ohm, x_ohm = r_km * length_m / 1000.0, x_km * length_m / 1000.0
    else:
        raise ValueError(f"unknown service_impedance_mode: {mode}")

    if mode == "random_perturbed":
        r_ohm *= float(rng.lognormal(mean=0.0, sigma=0.15))
        x_ohm *= float(rng.lognormal(mean=0.0, sigma=0.15))
    return float(r_ohm), float(x_ohm), float(length_m)


def terminalize_case33(
    raw: RawCase33,
    mode: str = "hybrid_leaf",
    service_impedance_mode: str = "scaled_original",
    service_length_m: float = 30.0,
    service_length_random_range_m: tuple[float, float] | None = None,
    service_r_ohm_per_km: float | None = None,
    service_x_ohm_per_km: float | None = None,
    keep_original_physical_leaves_observed: bool = True,
    aggregate_internal_loads_to_existing_leaf: bool = False,
    min_hidden_degree_target: int | None = 3,
    seed: int = 0,
) -> TerminalizedNetwork:
    """Create a terminal-load-only version of case33.

    Args mirror the project README. In the default ``hybrid_leaf`` mode, loads
    already located on physical leaves remain there, while every internal
    non-root load is moved to a newly attached terminal service node.
    """

    del aggregate_internal_loads_to_existing_leaf
    rng = np.random.default_rng(seed)
    graph = _raw_graph(raw)
    raw_leaves = {node for node, degree in graph.degree() if degree == 1 and node != raw.root_bus}
    load_by_bus = raw.buses.set_index("bus_id")[["pd_kw", "qd_kvar"]].to_dict("index")
    original_total_p = float(raw.buses["pd_kw"].sum())
    original_total_q = float(raw.buses["qd_kvar"].sum())

    buses = []
    for row in raw.buses.itertuples(index=False):
        bus_id = int(row.bus_id)
        buses.append(
            {
                "bus_id": bus_id,
                "original_bus_id": bus_id,
                "bus_type": "root" if bus_id == raw.root_bus else "hidden_internal",
                "pd_kw": float(row.pd_kw),
                "qd_kvar": float(row.qd_kvar),
                "is_observed": bus_id == raw.root_bus,
                "has_load": bool(row.pd_kw or row.qd_kvar),
                "is_terminal": False,
                "is_original_case33_bus": True,
                "hidden_degree": int(graph.degree(bus_id)),
                "note": "",
            }
        )

    branches = []
    for row in raw.branches.itertuples(index=False):
        branches.append(
            {
                "from_bus": int(row.from_bus),
                "to_bus": int(row.to_bus),
                "r_ohm": float(row.r_ohm),
                "x_ohm": float(row.x_ohm),
                "status": bool(row.status),
                "branch_type": "tie" if bool(row.is_tie) else "backbone",
                "original_branch_id": int(row.branch_id),
                "length_m": math.nan,
                "is_candidate": True,
                "is_true_closed": bool(row.status),
            }
        )

    bus_df = pd.DataFrame(buses).set_index("bus_id", drop=False)
    original_to_terminal: dict[int, int] = {}
    terminal_to_original: dict[int, int] = {}
    added_terminals: list[int] = []

    def add_terminal_for(original_bus: int) -> int:
        terminal = 1000 + int(original_bus)
        pd_kw = float(load_by_bus[original_bus]["pd_kw"])
        qd_kvar = float(load_by_bus[original_bus]["qd_kvar"])
        r_ohm, x_ohm, length_m = _service_impedance(
            raw,
            original_bus,
            service_impedance_mode,
            rng,
            service_length_m,
            service_length_random_range_m,
            service_r_ohm_per_km,
            service_x_ohm_per_km,
        )
        bus_df.loc[terminal] = {
            "bus_id": terminal,
            "original_bus_id": original_bus,
            "bus_type": "observed_terminal",
            "pd_kw": pd_kw,
            "qd_kvar": qd_kvar,
            "is_observed": True,
            "has_load": pd_kw != 0.0 or qd_kvar != 0.0,
            "is_terminal": True,
            "is_original_case33_bus": False,
            "hidden_degree": 1,
            "note": f"terminal load migrated from original bus {original_bus}",
        }
        bus_df.loc[original_bus, ["pd_kw", "qd_kvar", "has_load", "is_observed", "is_terminal"]] = [
            0.0,
            0.0,
            False,
            False,
            False,
        ]
        branches.append(
            {
                "from_bus": original_bus,
                "to_bus": terminal,
                "r_ohm": r_ohm,
                "x_ohm": x_ohm,
                "status": True,
                "branch_type": "service",
                "original_branch_id": math.nan,
                "length_m": length_m,
                "is_candidate": True,
                "is_true_closed": True,
            }
        )
        original_to_terminal[original_bus] = terminal
        terminal_to_original[terminal] = original_bus
        added_terminals.append(terminal)
        return terminal

    nonroot_loads = [
        int(r.bus_id)
        for r in raw.buses.itertuples(index=False)
        if int(r.bus_id) != raw.root_bus and (float(r.pd_kw) != 0.0 or float(r.qd_kvar) != 0.0)
    ]

    if mode == "hybrid_leaf":
        for bus_id in nonroot_loads:
            if bus_id in raw_leaves and keep_original_physical_leaves_observed:
                bus_df.loc[bus_id, ["bus_type", "is_observed", "is_terminal", "has_load", "note"]] = [
                    "observed_terminal",
                    True,
                    True,
                    True,
                    "original physical leaf kept observed",
                ]
                original_to_terminal[bus_id] = bus_id
                terminal_to_original[bus_id] = bus_id
            else:
                add_terminal_for(bus_id)
    elif mode == "all_loads_to_new_terminal_leaves":
        for bus_id in nonroot_loads:
            add_terminal_for(bus_id)
    elif mode == "aggregate_to_original_leaves":
        bus_df.loc[:, ["pd_kw", "qd_kvar", "has_load"]] = [0.0, 0.0, False]
        descendants = nx.single_source_shortest_path_length(graph, raw.root_bus)
        leaves = sorted(raw_leaves)
        for leaf in leaves:
            bus_df.loc[leaf, ["bus_type", "is_observed", "is_terminal", "note"]] = [
                "observed_terminal",
                True,
                True,
                "original physical leaf receives aggregated downstream load",
            ]
        oriented = nx.bfs_tree(graph, raw.root_bus)
        for bus_id in nonroot_loads:
            downstream_leaves = [
                leaf for leaf in leaves if nx.has_path(oriented, bus_id, leaf) and descendants[leaf] >= descendants[bus_id]
            ]
            if not downstream_leaves:
                downstream_leaves = [min(leaves, key=lambda leaf: nx.shortest_path_length(graph, bus_id, leaf))]
            share = 1.0 / len(downstream_leaves)
            for leaf in downstream_leaves:
                bus_df.loc[leaf, "pd_kw"] += float(load_by_bus[bus_id]["pd_kw"]) * share
                bus_df.loc[leaf, "qd_kvar"] += float(load_by_bus[bus_id]["qd_kvar"]) * share
                bus_df.loc[leaf, "has_load"] = True
            original_to_terminal[bus_id] = downstream_leaves[0]
        terminal_to_original = {int(v): int(k) for k, v in original_to_terminal.items() if v in leaves}
    else:
        raise ValueError(f"unknown terminalization mode: {mode}")

    bus_df.loc[raw.root_bus, ["bus_type", "pd_kw", "qd_kvar", "is_observed", "has_load", "is_terminal", "note"]] = [
        "root",
        0.0,
        0.0,
        True,
        False,
        False,
        "observable feeder root",
    ]
    for bus_id in bus_df.index:
        if bus_id != raw.root_bus and bus_df.loc[bus_id, "bus_type"] != "observed_terminal":
            bus_df.loc[bus_id, ["bus_type", "is_observed", "has_load", "is_terminal"]] = [
                "hidden_internal",
                False,
                False,
                False,
            ]

    branch_df = pd.DataFrame(branches)
    tmp = TerminalizedNetwork(
        buses=bus_df.reset_index(drop=True).sort_values("bus_id").reset_index(drop=True),
        branches=branch_df.reset_index(drop=True),
        root_bus=raw.root_bus,
        base_kv=raw.base_kv,
        base_mva=raw.base_mva,
        original_to_terminal=original_to_terminal,
        terminal_to_original=terminal_to_original,
        metadata={},
    )
    deg = dict(tmp.to_networkx_graph().degree())
    tmp.buses["hidden_degree"] = tmp.buses["bus_id"].map(deg).fillna(0).astype(int)
    hidden_degree_2 = tmp.buses[(tmp.buses["bus_type"].eq("hidden_internal")) & (tmp.buses["hidden_degree"].eq(2))]
    tmp.metadata = {
        "mode": mode,
        "service_impedance_mode": service_impedance_mode,
        "raw_leaves": sorted(raw_leaves),
        "added_terminals": added_terminals,
        "min_hidden_degree_target": min_hidden_degree_target,
        "degree_2_hidden_nodes": hidden_degree_2["bus_id"].astype(int).tolist(),
        "degree_2_identifiability_note": "cannot be uniquely recovered from terminal-only measurements unless additional assumptions or meters are added",
        "original_total_pd_kw": original_total_p,
        "original_total_qd_kvar": original_total_q,
        "terminal_total_pd_kw": float(tmp.buses["pd_kw"].sum()),
        "terminal_total_qd_kvar": float(tmp.buses["qd_kvar"].sum()),
        "lv_style_variant": service_impedance_mode == "lv_cable_soumalas_style",
    }
    return tmp

