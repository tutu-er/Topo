"""Small terminal-load-only LV-style feeder.

The topology follows the minimal-observability style used in terminal-meter
papers: root voltage is known, internal branching nodes are hidden and
zero-injection, and every non-root load is attached to an observed terminal
leaf. It is intentionally smaller than case33 so hidden-node behavior is easy
to inspect visually and numerically.
"""

from __future__ import annotations

from ..models.network import TerminalizedNetwork

import pandas as pd


def build_small_terminal_lv_case() -> TerminalizedNetwork:
    """Build a compact 10-bus terminal-load-only synthetic LV feeder.

    The feeder has one observable root, three hidden internal branching nodes,
    and six observed terminal loads. Hidden internal nodes have degree at least
    three, so the example avoids degree-2 hidden chains that are not uniquely
    recoverable from terminal-only additive distances.
    """

    base_kv = 0.4
    base_mva = 0.1
    root = 1
    buses = pd.DataFrame(
        [
            # bus_id, original_bus_id, bus_type, pd_kw, qd_kvar, observed, has_load, terminal, original, degree, note
            (1, 1, "root", 0.0, 0.0, True, False, False, True, 2, "observable LV transformer/root"),
            (2, 2, "hidden_internal", 0.0, 0.0, False, False, False, True, 4, "hidden zero-injection branching node"),
            (3, 3, "hidden_internal", 0.0, 0.0, False, False, False, True, 3, "hidden zero-injection branching node"),
            (4, 4, "hidden_internal", 0.0, 0.0, False, False, False, True, 3, "hidden zero-injection branching node"),
            (101, 101, "observed_terminal", 2.4, 0.8, True, True, True, False, 1, "terminal smart-meter load"),
            (102, 102, "observed_terminal", 1.8, 0.6, True, True, True, False, 1, "terminal smart-meter load"),
            (103, 103, "observed_terminal", 3.2, 1.1, True, True, True, False, 1, "terminal smart-meter load"),
            (104, 104, "observed_terminal", 2.1, 0.7, True, True, True, False, 1, "terminal smart-meter load"),
            (105, 105, "observed_terminal", 4.0, 1.4, True, True, True, False, 1, "terminal smart-meter load"),
            (106, 106, "observed_terminal", 2.8, 0.9, True, True, True, False, 1, "terminal smart-meter load"),
        ],
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

    # Backbone uses Soumalas-style overhead-line scale; service drops use a
    # 30 m LV service cable scale. Values are in Ohm.
    overhead_r_km = 0.284
    overhead_x_km = 0.083
    service_r_km = 1.380
    service_x_km = 0.082

    def z(r_km: float, x_km: float, length_m: float) -> tuple[float, float]:
        return r_km * length_m / 1000.0, x_km * length_m / 1000.0

    edge_specs = [
        (1, 2, *z(overhead_r_km, overhead_x_km, 80.0), "backbone", 80.0),
        (1, 3, *z(overhead_r_km, overhead_x_km, 65.0), "backbone", 65.0),
        (2, 4, *z(overhead_r_km, overhead_x_km, 55.0), "backbone", 55.0),
        (2, 101, *z(service_r_km, service_x_km, 30.0), "service", 30.0),
        (2, 102, *z(service_r_km, service_x_km, 30.0), "service", 30.0),
        (3, 103, *z(service_r_km, service_x_km, 30.0), "service", 30.0),
        (3, 104, *z(service_r_km, service_x_km, 30.0), "service", 30.0),
        (4, 105, *z(service_r_km, service_x_km, 30.0), "service", 30.0),
        (4, 106, *z(service_r_km, service_x_km, 30.0), "service", 30.0),
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
        "case_name": "small_terminal_lv",
        "style": "Soumalas/Park minimal-observability synthetic LV feeder",
        "original_total_pd_kw": float(buses["pd_kw"].sum()),
        "original_total_qd_kvar": float(buses["qd_kvar"].sum()),
        "terminal_total_pd_kw": float(buses["pd_kw"].sum()),
        "terminal_total_qd_kvar": float(buses["qd_kvar"].sum()),
        "degree_2_hidden_nodes": [],
        "degree_2_identifiability_note": "no degree-2 hidden chains by construction",
        "lv_style_variant": True,
        "rationale": "compact terminal-only tree with hidden branching nodes of degree at least three",
    }
    return TerminalizedNetwork(
        buses=buses,
        branches=branches,
        root_bus=root,
        base_kv=base_kv,
        base_mva=base_mva,
        original_to_terminal={bus: bus for bus in [101, 102, 103, 104, 105, 106]},
        terminal_to_original={bus: bus for bus in [101, 102, 103, 104, 105, 106]},
        metadata=metadata,
    )
