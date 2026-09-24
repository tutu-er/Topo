"""Adapter for the open Norwegian industrial distribution-grid dataset.

The source files use a MATPOWER-like common per-unit representation and
hourly active-energy measurements. This adapter selects one radial, prunes
unmetered dead ends, and maps string bus names to the integer labels expected
by the topology-identification code.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from terminal_case33.models.network import TerminalizedNetwork


DATASET_DOI = "10.5281/zenodo.10361330"
DEFAULT_DATA_DIR = Path("data_external/norwegian_industrial_zenodo_7123537")


def _source_tables(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the semicolon-delimited source bus and branch tables."""

    directory = Path(data_dir)
    bus_path = directory / "bus.csv"
    branch_path = directory / "branch.csv"
    if not bus_path.exists() or not branch_path.exists():
        raise FileNotFoundError(
            f"missing {bus_path} or {branch_path}; run scripts/download_norwegian_industrial.py"
        )
    return pd.read_csv(bus_path, sep=";"), pd.read_csv(branch_path, sep=";")


def load_norwegian_industrial_radial(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    radial: int = 1,
    base_mva: float = 1.0,
    equivalent_base_kv: float = 11.0,
) -> TerminalizedNetwork:
    """Load one real radial as a terminal-load-only balanced equivalent.

    The published branch R/X values are retained as common per-unit values.
    They are converted to ohms on the supplied equivalent bases so the
    existing single-base AC solver converts them back to the same per-unit
    values. Ideal transformer and connection branches remain zero impedance.
    This is a balanced positive-sequence equivalent, not a detailed
    multi-voltage transformer model.
    """

    buses_raw, branches_raw = _source_tables(data_dir)
    root_label = f"r{int(radial)}v47.0b1"
    if root_label not in set(buses_raw["BUS_I"]):
        raise ValueError(f"radial {radial} is not present in the dataset")

    closed = branches_raw.loc[branches_raw["BR_STATUS"].eq(1)].copy()
    graph = nx.Graph()
    graph.add_nodes_from(buses_raw["BUS_I"].astype(str))
    graph.add_edges_from(closed[["F_BUS", "T_BUS"]].itertuples(index=False, name=None))
    component = set(nx.node_connected_component(graph, root_label))
    load_labels = {
        path.stem
        for path in Path(data_dir).glob(f"r{int(radial)}*.txt")
        if path.stem in component
    }
    if not load_labels:
        raise FileNotFoundError(f"no smart-meter files found for radial {radial} in {data_dir}")

    radial_graph = graph.subgraph(component).copy()
    changed = True
    while changed:
        changed = False
        removable = [
            node
            for node, degree in radial_graph.degree()
            if degree <= 1 and node != root_label and node not in load_labels
        ]
        if removable:
            radial_graph.remove_nodes_from(removable)
            changed = True

    retained = set(radial_graph.nodes())
    ordered_labels = list(
        nx.bfs_tree(radial_graph, root_label, sort_neighbors=sorted).nodes()
    )
    label_to_id = {label: index + 1 for index, label in enumerate(ordered_labels)}
    source_bus = buses_raw.set_index("BUS_I")
    z_base_ohm = equivalent_base_kv**2 / base_mva

    bus_rows = []
    for label in ordered_labels:
        bus_id = label_to_id[label]
        is_root = label == root_label
        is_terminal = label in load_labels
        if is_root:
            bus_type = "root"
        elif is_terminal:
            bus_type = "observed_terminal"
        else:
            bus_type = "hidden_internal"
        bus_rows.append(
            {
                "bus_id": bus_id,
                "original_bus_id": None,
                "bus_type": bus_type,
                "pd_kw": 0.0,
                "qd_kvar": 0.0,
                "is_observed": bool(is_root or is_terminal),
                "has_load": bool(is_terminal),
                "is_terminal": bool(is_terminal),
                "is_original_case33_bus": False,
                "hidden_degree": int(radial_graph.degree(label)) if not is_terminal else 0,
                "note": f"source_bus={label}; source_base_kv={float(source_bus.loc[label, 'BASE_KV']):g}",
            }
        )

    retained_edges = closed[
        closed["F_BUS"].isin(retained) & closed["T_BUS"].isin(retained)
    ].copy()
    branch_rows = []
    for branch_id, row in enumerate(retained_edges.itertuples(index=False), start=1):
        branch_rows.append(
            {
                "from_bus": label_to_id[str(row.F_BUS)],
                "to_bus": label_to_id[str(row.T_BUS)],
                "r_ohm": float(row.BR_R) * z_base_ohm,
                "x_ohm": float(row.BR_X) * z_base_ohm,
                "status": 1,
                "branch_type": "backbone" if (row.BR_R or row.BR_X) else "synthetic",
                "original_branch_id": branch_id,
                "length_m": None,
                "is_candidate": True,
                "is_true_closed": True,
            }
        )

    terminal_ids = {label_to_id[label] for label in load_labels}
    network = TerminalizedNetwork(
        buses=pd.DataFrame(bus_rows),
        branches=pd.DataFrame(branch_rows),
        root_bus=label_to_id[root_label],
        base_kv=float(equivalent_base_kv),
        base_mva=float(base_mva),
        original_to_terminal={node: node for node in terminal_ids},
        terminal_to_original={node: node for node in terminal_ids},
        metadata={
            "name": f"norwegian_industrial_radial_{radial}",
            "dataset_doi": DATASET_DOI,
            "dataset_version": "2.3",
            "source_record": "https://zenodo.org/records/10361330",
            "radial": int(radial),
            "source_root_bus": root_label,
            "source_bus_labels": {str(value): key for key, value in label_to_id.items()},
            "branch_parameter_interpretation": "published common per-unit R/X",
            "equivalent_model": "balanced common-per-unit model with ideal zero-impedance transformers",
            "pruned_unmetered_dead_ends": sorted(component - retained),
        },
    )
    if not nx.is_tree(network.to_networkx_graph()):
        raise ValueError("selected Norwegian radial is not a tree after pruning")
    if set(network.get_leaf_buses()) != terminal_ids:
        raise ValueError("published smart-meter buses are not exactly the retained leaves")
    return network


def canonicalize_norwegian_terminal_radial(
    net: TerminalizedNetwork,
    zero_impedance_tolerance_pu: float = 1e-10,
    service_r_pu: float = 5e-4,
    service_x_pu: float = 4e-4,
) -> TerminalizedNetwork:
    """Return the identifiable terminal-tree form of a Norwegian radial.

    Zero-impedance edges between unobserved buses are contracted, hidden
    degree-2 chains are replaced by their series impedance, and a zero-ohm
    edge incident to a terminal meter is replaced by a short positive service
    impedance. Terminal identities are retained so the published meter files
    still map to the same observed leaves.
    """

    if zero_impedance_tolerance_pu < 0.0:
        raise ValueError("zero_impedance_tolerance_pu must be nonnegative")
    if service_r_pu <= 0.0 or service_x_pu <= 0.0:
        raise ValueError("service impedances must be positive")
    graph = net.to_networkx_graph()
    if not nx.is_tree(graph):
        raise ValueError("canonicalization requires a radial tree")
    terminals = set(net.load_buses())
    root = int(net.root_bus)
    z_base_ohm = net.base_kv**2 / net.base_mva
    zero_tolerance_ohm = zero_impedance_tolerance_pu * z_base_ohm
    service_r_ohm = service_r_pu * z_base_ohm
    service_x_ohm = service_x_pu * z_base_ohm

    contracted_nodes: list[dict] = []
    while True:
        depths = nx.single_source_shortest_path_length(graph, root)
        zero_edge = next(
            (
                (int(u), int(v))
                for u, v, data in graph.edges(data=True)
                if u not in terminals
                and v not in terminals
                and abs(float(data["r_ohm"])) + abs(float(data["x_ohm"]))
                <= zero_tolerance_ohm
            ),
            None,
        )
        if zero_edge is None:
            break
        u, v = zero_edge
        keep, remove = (u, v) if depths[u] <= depths[v] else (v, u)
        for neighbor, data in list(graph[remove].items()):
            neighbor_i = int(neighbor)
            if neighbor_i == keep:
                continue
            graph.add_edge(keep, neighbor_i, **dict(data))
        graph.remove_node(remove)
        contracted_nodes.append({"removed": remove, "retained": keep})

    replaced_terminal_edges: list[dict] = []
    for u, v, data in graph.edges(data=True):
        impedance = abs(float(data["r_ohm"])) + abs(float(data["x_ohm"]))
        if impedance > zero_tolerance_ohm:
            continue
        if u not in terminals and v not in terminals:
            raise RuntimeError("nonterminal zero edge remained after contraction")
        data["r_ohm"] = service_r_ohm
        data["x_ohm"] = service_x_ohm
        data["branch_type"] = "canonical_service"
        replaced_terminal_edges.append(
            {"edge": [int(u), int(v)], "r_pu": service_r_pu, "x_pu": service_x_pu}
        )

    suppressed_nodes: list[dict] = []
    while True:
        removable = next(
            (
                int(node)
                for node in graph.nodes
                if node != root and node not in terminals and graph.degree(node) == 2
            ),
            None,
        )
        if removable is None:
            break
        left, right = [int(node) for node in graph.neighbors(removable)]
        left_data = dict(graph.edges[left, removable])
        right_data = dict(graph.edges[removable, right])
        graph.remove_node(removable)
        graph.add_edge(
            left,
            right,
            r_ohm=float(left_data["r_ohm"]) + float(right_data["r_ohm"]),
            x_ohm=float(left_data["x_ohm"]) + float(right_data["x_ohm"]),
            branch_type="canonical_series",
        )
        suppressed_nodes.append(
            {"removed": removable, "between": [left, right]}
        )

    source_buses = net.buses.set_index("bus_id")
    bus_rows = []
    for node in nx.bfs_tree(graph, root).nodes():
        node_i = int(node)
        source = source_buses.loc[node_i]
        is_root = node_i == root
        is_terminal = node_i in terminals
        bus_rows.append(
            {
                "bus_id": node_i,
                "original_bus_id": source["original_bus_id"],
                "bus_type": "root" if is_root else "observed_terminal" if is_terminal else "hidden_internal",
                "pd_kw": float(source["pd_kw"]),
                "qd_kvar": float(source["qd_kvar"]),
                "is_observed": bool(is_root or is_terminal),
                "has_load": bool(is_terminal),
                "is_terminal": bool(is_terminal),
                "is_original_case33_bus": False,
                "hidden_degree": int(graph.degree(node_i)) if not is_terminal else 0,
                "note": f"{source['note']}; canonical identifiable representation",
            }
        )

    branch_rows = []
    for branch_id, (parent, child) in enumerate(nx.bfs_edges(graph, root), start=1):
        data = graph.edges[parent, child]
        branch_rows.append(
            {
                "from_bus": int(parent),
                "to_bus": int(child),
                "r_ohm": float(data["r_ohm"]),
                "x_ohm": float(data["x_ohm"]),
                "status": 1,
                "branch_type": str(data.get("branch_type", "canonical")),
                "original_branch_id": branch_id,
                "length_m": None,
                "is_candidate": True,
                "is_true_closed": True,
            }
        )

    metadata = dict(net.metadata)
    metadata.update(
        {
            "name": f"{net.metadata.get('name', 'norwegian_radial')}_canonical",
            "canonical_identifiable_form": True,
            "canonicalization": "contract hidden zero-Z edges, suppress degree-2 hidden chains, retain terminal meters with short positive service Z",
            "zero_impedance_tolerance_pu": float(zero_impedance_tolerance_pu),
            "canonical_service_r_pu": float(service_r_pu),
            "canonical_service_x_pu": float(service_x_pu),
            "contracted_zero_impedance_nodes": contracted_nodes,
            "suppressed_degree_2_hidden_nodes": suppressed_nodes,
            "replaced_zero_terminal_edges": replaced_terminal_edges,
        }
    )
    canonical = TerminalizedNetwork(
        buses=pd.DataFrame(bus_rows),
        branches=pd.DataFrame(branch_rows),
        root_bus=root,
        base_kv=net.base_kv,
        base_mva=net.base_mva,
        original_to_terminal={terminal: terminal for terminal in terminals},
        terminal_to_original={terminal: terminal for terminal in terminals},
        metadata=metadata,
    )
    canonical_graph = canonical.to_networkx_graph()
    if not nx.is_tree(canonical_graph):
        raise RuntimeError("canonical Norwegian radial is not a tree")
    if set(canonical.get_leaf_buses()) != terminals:
        raise RuntimeError("canonicalization changed terminal-meter leaves")
    if any(canonical_graph.degree(node) == 2 for node in canonical.hidden_buses()):
        raise RuntimeError("canonicalization left a degree-2 hidden node")
    if any(
        abs(float(row.r_ohm)) + abs(float(row.x_ohm)) <= zero_tolerance_ohm
        for row in canonical.branches.itertuples()
    ):
        raise RuntimeError("canonicalization left a zero-impedance edge")
    return canonical

def load_norwegian_active_power(
    net: TerminalizedNetwork,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    start: str | pd.Timestamp | None = None,
    periods: int | None = None,
) -> pd.DataFrame:
    """Load aligned hourly smart-meter active energy as average power in pu.

    Each source record is hourly Load_kWh; therefore its numeric value is also
    the average kW over that hour. Only the common timestamp intersection is
    retained, and columns are mapped to integer terminal bus IDs.
    """

    label_by_id = {
        int(bus_id): str(label)
        for bus_id, label in net.metadata["source_bus_labels"].items()
    }
    series = []
    for terminal in net.load_buses():
        label = label_by_id[int(terminal)]
        path = Path(data_dir) / f"{label}.txt"
        frame = pd.read_csv(path, sep=";")
        timestamp = pd.to_datetime(frame["Timestamp"], dayfirst=True)
        values = pd.Series(frame["Load_kWh"].to_numpy(dtype=float), index=timestamp, name=int(terminal))
        series.append(values[~values.index.duplicated(keep="first")])
    aligned = pd.concat(series, axis=1, join="inner").sort_index()
    if start is not None:
        aligned = aligned.loc[aligned.index >= pd.Timestamp(start)]
    if periods is not None:
        aligned = aligned.iloc[: int(periods)]
    if aligned.empty:
        raise ValueError("no common smart-meter timestamps remain after slicing")
    return aligned / (1000.0 * net.base_mva)
