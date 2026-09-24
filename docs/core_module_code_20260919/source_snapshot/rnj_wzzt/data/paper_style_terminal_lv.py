"""Paper-style terminal-only LV feeder with hidden internal nodes.

This feeder is intentionally smaller and more balanced than terminalized
case33. It follows the topology assumptions common to Soumalas-style and
minimal-observability papers: the transformer/root is known, all smart meters
are at terminal consumers, and all internal branching nodes are hidden
zero-injection nodes. Hidden internal nodes have degree at least three.
"""

from __future__ import annotations

import pandas as pd

from rnj_wzzt.models.network import TerminalizedNetwork


def build_paper_style_terminal_lv_case() -> TerminalizedNetwork:
    """Build a 22-bus LV feeder with 15 terminal smart-meter loads.

    The structure has three main laterals from the root, each with a hidden
    upstream branch node and a hidden downstream branch node. This reduces the
    excessive common-path dominance of the long case33 backbone while preserving
    hidden-node topology-identification difficulty.
    """

    base_kv = 0.4
    base_mva = 0.2
    root = 1
    hidden = [2, 3, 4, 5, 6, 7]
    terminal_specs = [
        # bus, pd_kw, qd_kvar
        (101, 7.5, 2.6),
        (102, 5.8, 1.9),
        (103, 9.2, 3.1),
        (104, 6.4, 2.0),
        (105, 11.5, 4.0),
        (106, 4.6, 1.4),
        (107, 8.8, 2.8),
        (108, 13.5, 5.2),
        (109, 6.9, 2.1),
        (110, 10.4, 3.8),
        (111, 5.2, 1.6),
        (112, 12.6, 4.7),
        (113, 7.8, 2.5),
        (114, 14.2, 5.4),
        (115, 6.1, 1.8),
    ]

    rows = [
        (root, root, "root", 0.0, 0.0, True, False, False, True, 3, "observable LV transformer/root"),
    ]
    rows.extend(
        (bus, bus, "hidden_internal", 0.0, 0.0, False, False, False, True, 4, "hidden zero-injection branching node")
        for bus in hidden
    )
    rows.extend(
        (bus, bus, "observed_terminal", pd_kw, qd_kvar, True, True, True, False, 1, "observed terminal smart-meter load")
        for bus, pd_kw, qd_kvar in terminal_specs
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

    # Identification benchmark line library. The earlier Soumalas-style LV
    # service cable values had X/R near 0.06 on service drops, which made X
    # sensitivities numerically tiny. For this synthetic benchmark we use
    # distribution-line parameters with R and X in the same order of magnitude.
    overhead_r_km = 0.42
    overhead_x_km = 0.34
    service_r_km = 0.55
    service_x_km = 0.38

    impedance_scale = 5.0

    def z(r_km: float, x_km: float, length_m: float) -> tuple[float, float]:
        return impedance_scale * r_km * length_m / 1000.0, impedance_scale * x_km * length_m / 1000.0

    edge_specs = [
        (1, 2, *z(overhead_r_km, overhead_x_km, 92.0), "backbone", 92.0),
        (1, 3, *z(overhead_r_km, overhead_x_km, 74.0), "backbone", 74.0),
        (1, 4, *z(overhead_r_km, overhead_x_km, 88.0), "backbone", 88.0),
        (2, 5, *z(overhead_r_km, overhead_x_km, 58.0), "backbone", 58.0),
        (3, 6, *z(overhead_r_km, overhead_x_km, 67.0), "backbone", 67.0),
        (4, 7, *z(overhead_r_km, overhead_x_km, 52.0), "backbone", 52.0),
        (2, 101, *z(service_r_km, service_x_km, 26.0), "service", 26.0),
        (2, 102, *z(service_r_km, service_x_km, 34.0), "service", 34.0),
        (5, 103, *z(service_r_km, service_x_km, 21.0), "service", 21.0),
        (5, 104, *z(service_r_km, service_x_km, 38.0), "service", 38.0),
        (5, 105, *z(service_r_km, service_x_km, 29.0), "service", 29.0),
        (3, 106, *z(service_r_km, service_x_km, 24.0), "service", 24.0),
        (3, 107, *z(service_r_km, service_x_km, 36.0), "service", 36.0),
        (6, 108, *z(service_r_km, service_x_km, 31.0), "service", 31.0),
        (6, 109, *z(service_r_km, service_x_km, 22.0), "service", 22.0),
        (6, 110, *z(service_r_km, service_x_km, 40.0), "service", 40.0),
        (4, 111, *z(service_r_km, service_x_km, 27.0), "service", 27.0),
        (4, 112, *z(service_r_km, service_x_km, 35.0), "service", 35.0),
        (7, 113, *z(service_r_km, service_x_km, 23.0), "service", 23.0),
        (7, 114, *z(service_r_km, service_x_km, 37.0), "service", 37.0),
        (7, 115, *z(service_r_km, service_x_km, 30.0), "service", 30.0),
    ]
    branches = pd.DataFrame(
        [
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
            for idx, (u, v, r, x, branch_type, length_m) in enumerate(edge_specs, start=1)
        ]
    )
    metadata = {
        "case_name": "paper_style_terminal_lv_15",
        "style": "Soumalas/Park/Flynn-style minimal-observability synthetic LV feeder",
        "original_total_pd_kw": float(buses["pd_kw"].sum()),
        "original_total_qd_kvar": float(buses["qd_kvar"].sum()),
        "terminal_total_pd_kw": float(buses["pd_kw"].sum()),
        "terminal_total_qd_kvar": float(buses["qd_kvar"].sum()),
        "degree_2_hidden_nodes": [],
        "degree_2_identifiability_note": "no degree-2 hidden chains by construction",
        "lv_style_variant": True,
        "impedance_scale": impedance_scale,
        "voltage_design_note": "scaled LV feeder impedance targets terminal voltages around 0.94-1.00 pu with transformer secondary near 1.02 pu",
        "line_parameter_note": "synthetic R/X-balanced distribution-line library; overhead X/R=0.81, service X/R=0.69; impedances multiplied by scale factor 5.0",
        "rationale": "balanced hidden branching tree with terminal-only meters and heterogeneous service lengths",
    }
    terminals = [bus for bus, _, _ in terminal_specs]
    return TerminalizedNetwork(
        buses=buses,
        branches=branches,
        root_bus=root,
        base_kv=base_kv,
        base_mva=base_mva,
        original_to_terminal={bus: bus for bus in terminals},
        terminal_to_original={bus: bus for bus in terminals},
        metadata=metadata,
    )


def build_paper_style_resource_assignment(net: TerminalizedNetwork | None = None) -> pd.DataFrame:
    """Assign mixed load/PV/wind/small-generator profiles to terminal buses."""

    net = net or build_paper_style_terminal_lv_case()
    bus = net.buses.set_index("bus_id")
    specs = {
        101: ("base_load", "residential", 0.0, 0.0, 0.0),
        102: ("pv", "residential", 4.2, 0.0, 0.0),
        103: ("pv", "commercial", 4.0, 1.7, 0.0),
        104: ("wind", "residential", 0.0, 3.4, 0.0),
        105: ("small_generator", "industrial", 2.0, 0.0, 4.6),
        106: ("pv", "residential", 2.8, 1.2, 0.0),
        107: ("wind", "commercial", 0.0, 4.8, 1.4),
        108: ("small_generator", "industrial", 0.0, 2.8, 5.1),
        109: ("pv", "residential", 3.0, 0.0, 1.2),
        110: ("wind", "commercial", 2.6, 4.4, 0.0),
        111: ("base_load", "residential", 0.0, 0.0, 0.0),
        112: ("pv", "commercial", 5.5, 0.0, 2.2),
        113: ("wind", "residential", 1.6, 3.1, 0.0),
        114: ("small_generator", "industrial", 0.0, 0.0, 6.2),
        115: ("pv", "residential", 2.5, 2.0, 0.0),
    }
    records = []
    for terminal, (profile_class, customer_type, pv_kw, wind_kw, gen_kw) in specs.items():
        pd_kw = float(bus.loc[terminal, "pd_kw"])
        qd_kvar = float(bus.loc[terminal, "qd_kvar"])
        der_capacity = pv_kw + wind_kw + gen_kw
        records.append(
            {
                "bus_id": terminal,
                "pd_kw": pd_kw,
                "qd_kvar": qd_kvar,
                "profile_class": profile_class,
                "customer_type": customer_type,
                "der_capacity_kw": der_capacity,
                "pv_capacity_kw": pv_kw,
                "wind_capacity_kw": wind_kw,
                "small_generator_capacity_kw": gen_kw,
                "der_capacity_ratio_to_load": 0.0 if pd_kw == 0.0 else der_capacity / pd_kw,
                "profile_source_candidate": "NREL/NLR End-Use Load Profiles, NSRDB, WIND Toolkit, IEEE 1547-style DER reactive capability",
                "net_load_sign_convention": "net_P_load = demand_P - pv_P - wind_P - small_generator_P",
                "note": "mixed DER terminal profile for paper-style synthetic LV case",
            }
        )
    return pd.DataFrame(records)
