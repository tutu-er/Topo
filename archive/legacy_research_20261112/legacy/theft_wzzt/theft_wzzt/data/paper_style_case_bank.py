"""Paper-style terminal-only LV case bank for robustness tests."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from theft_wzzt.data.paper_style_terminal_lv import build_paper_style_terminal_lv_case
from theft_wzzt.models.network import TerminalizedNetwork


def _z(r_km: float, x_km: float, length_m: float, scale: float) -> tuple[float, float]:
    return scale * r_km * length_m / 1000.0, scale * x_km * length_m / 1000.0


def _make_case(
    *,
    case_name: str,
    hidden: list[int],
    terminals: list[tuple[int, float, float]],
    edges: list[tuple[int, int, float, str]],
    root: int = 1,
    impedance_scale: float = 5.0,
    note: str,
) -> TerminalizedNetwork:
    """Build a terminal-only LV case from hidden nodes, terminals, and edge lengths."""

    base_kv = 0.4
    base_mva = 0.2
    degree = Counter()
    for u, v, _length, _kind in edges:
        degree[u] += 1
        degree[v] += 1
    rows = [(root, root, "root", 0.0, 0.0, True, False, False, True, degree[root], "observable LV transformer/root")]
    rows.extend(
        (bus, bus, "hidden_internal", 0.0, 0.0, False, False, False, True, degree[bus], "hidden zero-injection branching node")
        for bus in hidden
    )
    rows.extend(
        (bus, bus, "observed_terminal", pd_kw, qd_kvar, True, True, True, False, degree[bus], "observed terminal smart-meter load")
        for bus, pd_kw, qd_kvar in terminals
    )
    buses = pd.DataFrame(
        rows,
        columns=[
            "bus_id",
            "original_bus_id",
            "bus_type",
            "pd_kw",
            "qd_kvar",
            "is_observed",
            "has_load",
            "is_terminal",
            "is_original_case33_bus",
            "hidden_degree",
            "note",
        ],
    )
    overhead_r_km, overhead_x_km = 0.42, 0.34
    service_r_km, service_x_km = 0.55, 0.38
    branch_rows = []
    for idx, (u, v, length_m, branch_type) in enumerate(edges, start=1):
        if branch_type == "service":
            r, x = _z(service_r_km, service_x_km, length_m, impedance_scale)
        else:
            r, x = _z(overhead_r_km, overhead_x_km, length_m, impedance_scale)
        branch_rows.append(
            {
                "from_bus": u,
                "to_bus": v,
                "r_ohm": r,
                "x_ohm": x,
                "status": True,
                "branch_type": branch_type,
                "original_branch_id": idx,
                "length_m": length_m,
                "is_candidate": True,
                "is_true_closed": True,
            }
        )
    terminal_ids = [bus for bus, _pd, _qd in terminals]
    return TerminalizedNetwork(
        buses=buses,
        branches=pd.DataFrame(branch_rows),
        root_bus=root,
        base_kv=base_kv,
        base_mva=base_mva,
        original_to_terminal={bus: bus for bus in terminal_ids},
        terminal_to_original={bus: bus for bus in terminal_ids},
        metadata={
            "case_name": case_name,
            "style": note,
            "original_total_pd_kw": float(buses["pd_kw"].sum()),
            "original_total_qd_kvar": float(buses["qd_kvar"].sum()),
            "terminal_total_pd_kw": float(buses["pd_kw"].sum()),
            "terminal_total_qd_kvar": float(buses["qd_kvar"].sum()),
            "degree_2_hidden_nodes": [int(bus) for bus in hidden if degree[bus] == 2],
            "lv_style_variant": True,
            "impedance_scale": impedance_scale,
            "line_parameter_note": "balanced LV synthetic line library, overhead X/R=0.81, service X/R=0.69",
        },
    )


def build_soumalas_style_lv_11_case() -> TerminalizedNetwork:
    """Compact Soumalas-style LV hidden tree with 11 terminal meters."""

    hidden = [2, 3, 4, 5]
    terminals = [
        (101, 4.8, 1.6),
        (102, 6.1, 2.1),
        (103, 7.4, 2.5),
        (104, 5.6, 1.8),
        (105, 8.2, 2.9),
        (106, 6.7, 2.2),
        (107, 9.5, 3.3),
        (108, 5.1, 1.7),
        (109, 7.9, 2.6),
        (110, 6.3, 2.0),
        (111, 8.8, 3.0),
    ]
    edges = [
        (1, 2, 64.0, "backbone"),
        (1, 3, 82.0, "backbone"),
        (1, 4, 72.0, "backbone"),
        (3, 5, 56.0, "backbone"),
        (2, 101, 22.0, "service"),
        (2, 102, 31.0, "service"),
        (2, 103, 28.0, "service"),
        (3, 104, 24.0, "service"),
        (3, 105, 35.0, "service"),
        (5, 106, 20.0, "service"),
        (5, 107, 38.0, "service"),
        (5, 108, 27.0, "service"),
        (4, 109, 25.0, "service"),
        (4, 110, 34.0, "service"),
        (4, 111, 29.0, "service"),
    ]
    return _make_case(
        case_name="soumalas_style_lv_11",
        hidden=hidden,
        terminals=terminals,
        edges=edges,
        impedance_scale=5.2,
        note="Soumalas-style compact LV terminal-meter feeder with hidden branching nodes",
    )


def build_flynn_balanced_lv_16_case() -> TerminalizedNetwork:
    """Balanced binary-branch hidden tree inspired by recursive grouping tests."""

    hidden = [2, 3, 4, 5, 6, 7]
    terminals = [(201 + i, pd, q) for i, (pd, q) in enumerate([
        (5.2, 1.7), (7.3, 2.5), (6.4, 2.1), (9.1, 3.2),
        (4.9, 1.5), (8.4, 2.9), (6.8, 2.2), (10.2, 3.7),
        (5.7, 1.8), (7.6, 2.6), (6.2, 2.0), (9.8, 3.4),
        (5.4, 1.7), (8.9, 3.1), (6.6, 2.1), (10.8, 3.9),
    ])]
    edges = [
        (1, 2, 76.0, "backbone"),
        (1, 3, 79.0, "backbone"),
        (2, 4, 54.0, "backbone"),
        (2, 5, 61.0, "backbone"),
        (3, 6, 58.0, "backbone"),
        (3, 7, 52.0, "backbone"),
        # Uneven service lengths keep the hidden-tree distances identifiable.
        (4, 201, 15.0, "service"), (4, 202, 38.0, "service"), (4, 203, 56.0, "service"),
        (5, 204, 17.0, "service"), (5, 205, 39.0, "service"), (5, 206, 57.0, "service"),
        (6, 207, 16.0, "service"), (6, 208, 37.0, "service"), (6, 209, 55.0, "service"),
        (7, 210, 18.0, "service"), (7, 211, 40.0, "service"), (7, 212, 58.0, "service"),
        (2, 213, 18.0, "service"), (3, 214, 19.0, "service"),
        (5, 215, 28.0, "service"), (7, 216, 30.0, "service"),
    ]
    return _make_case(
        case_name="flynn_balanced_lv_16",
        hidden=hidden,
        terminals=terminals,
        edges=edges,
        impedance_scale=4.8,
        note="Flynn-style balanced hidden tree for recursive-grouping-like topology tests",
    )


def build_pengwah_mixed_lv_18_case() -> TerminalizedNetwork:
    """Unbalanced multi-lateral LV case inspired by mixed smart/interval meter tests."""

    hidden = [2, 3, 4, 5, 6, 7, 8]
    terminals = [(301 + i, pd, q) for i, (pd, q) in enumerate([
        (4.6, 1.4), (6.9, 2.4), (8.1, 2.8), (5.5, 1.8), (9.3, 3.4), (6.2, 2.0),
        (11.4, 4.2), (7.1, 2.5), (5.8, 1.9), (8.6, 3.0), (6.7, 2.3), (10.5, 3.9),
        (5.0, 1.6), (7.8, 2.7), (9.9, 3.5), (6.4, 2.1), (8.3, 2.9), (11.8, 4.4),
    ])]
    edges = [
        (1, 2, 68.0, "backbone"), (1, 3, 95.0, "backbone"), (1, 4, 74.0, "backbone"), (1, 5, 88.0, "backbone"),
        (2, 6, 46.0, "backbone"), (4, 7, 63.0, "backbone"), (5, 8, 57.0, "backbone"),
        (2, 301, 20.0, "service"), (2, 302, 35.0, "service"),
        (6, 303, 24.0, "service"), (6, 304, 31.0, "service"), (6, 305, 43.0, "service"),
        (3, 306, 22.0, "service"), (3, 307, 39.0, "service"), (3, 308, 28.0, "service"),
        (4, 309, 26.0, "service"), (7, 310, 20.0, "service"), (7, 311, 37.0, "service"), (7, 312, 45.0, "service"),
        (5, 313, 25.0, "service"), (5, 314, 34.0, "service"),
        (8, 315, 23.0, "service"), (8, 316, 29.0, "service"), (8, 317, 41.0, "service"), (8, 318, 36.0, "service"),
    ]
    return _make_case(
        case_name="pengwah_mixed_lv_18",
        hidden=hidden,
        terminals=terminals,
        edges=edges,
        impedance_scale=4.5,
        note="Pengwah-style mixed lateral LV feeder with unbalanced terminal clusters",
    )


CASE_BUILDERS = {
    "paper15": build_paper_style_terminal_lv_case,
    "soumalas11": build_soumalas_style_lv_11_case,
    "flynn16": build_flynn_balanced_lv_16_case,
    "pengwah18": build_pengwah_mixed_lv_18_case,
}

def build_case_bank_resource_assignment(net: TerminalizedNetwork) -> pd.DataFrame:
    """Assign heterogeneous demand/PV/wind/generator profiles to any case-bank feeder."""

    terminals = net.buses.loc[net.buses["bus_type"].eq("observed_terminal")].sort_values("bus_id")
    profiles = [
        ("base_load", "residential", 0.0, 0.0, 0.0),
        ("pv", "residential", 0.55, 0.0, 0.0),
        ("wind", "commercial", 0.0, 0.55, 0.0),
        ("pv", "commercial", 0.42, 0.22, 0.0),
        ("small_generator", "industrial", 0.18, 0.0, 0.48),
        ("wind", "residential", 0.15, 0.50, 0.0),
        ("small_generator", "industrial", 0.0, 0.20, 0.52),
        ("pv", "residential", 0.38, 0.0, 0.18),
    ]
    records = []
    for idx, row in enumerate(terminals.itertuples(index=False)):
        profile_class, customer_type, pv_ratio, wind_ratio, gen_ratio = profiles[idx % len(profiles)]
        pd_kw = float(row.pd_kw)
        pv_kw = pv_ratio * pd_kw
        wind_kw = wind_ratio * pd_kw
        gen_kw = gen_ratio * pd_kw
        der_capacity = pv_kw + wind_kw + gen_kw
        records.append(
            {
                "bus_id": int(row.bus_id),
                "pd_kw": pd_kw,
                "qd_kvar": float(row.qd_kvar),
                "profile_class": profile_class,
                "customer_type": customer_type,
                "der_capacity_kw": der_capacity,
                "pv_capacity_kw": pv_kw,
                "wind_capacity_kw": wind_kw,
                "small_generator_capacity_kw": gen_kw,
                "der_capacity_ratio_to_load": 0.0 if pd_kw == 0 else der_capacity / pd_kw,
                "profile_source_candidate": "NREL/NLR End-Use Load Profiles, NSRDB, WIND Toolkit, IEEE 1547-style DER reactive capability",
                "net_load_sign_convention": "net_P_load = demand_P - pv_P - wind_P - small_generator_P",
                "note": "generic mixed DER terminal profile for paper-style case bank",
            }
        )
    return pd.DataFrame(records)
