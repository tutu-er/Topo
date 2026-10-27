"""Read-only SimBench topology/data audit; no power flow or topology fitting.

Read the frozen source CSVs, keep their original selected rows, and contract
only switches whose recorded cond is 1. No impedance or load is fabricated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path


SUBNET = "LV1.101"
TABLES = ["Node", "Line", "LineType", "Load", "RES", "Switch", "Transformer", "TransformerType", "ExternalNet"]
WHEEL_SHA256 = "8b48bdab2445286a96649eb18bb09476d000a0e212fd57fdba83399df54af4de"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def value(row, key):
    return row[key].strip()


def adjacency(nodes, edges):
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def reachable(adj, start, excluded=None):
    found, queue = {start}, deque([start])
    while queue:
        a = queue.popleft()
        for b in adj[a]:
            if excluded is not None and frozenset((a, b)) == excluded:
                continue
            if b not in found:
                found.add(b)
                queue.append(b)
    return found


def node_order(name):
    return int(name.split(" Bus ")[1])


def audit(source_dir, wheel_dir):
    tables = {name: load_csv(source_dir / (name + ".csv")) for name in TABLES}
    selected = {
        name: [row for row in tables[name] if value(row, "subnet") == SUBNET]
        for name in ["Line", "Load", "RES", "Switch", "Transformer"]
    }
    selected["ExternalNet"] = [row for row in tables["ExternalNet"] if value(row, "subnet").split("_")[0] == SUBNET]
    node_ids = set()
    for table, columns in {"Line": ["nodeA", "nodeB"], "Switch": ["nodeA", "nodeB"], "Transformer": ["nodeHV", "nodeLV"], "Load": ["node"], "RES": ["node"], "ExternalNet": ["node"]}.items():
        for row in selected[table]:
            node_ids.update(value(row, col) for col in columns)
    selected["Node"] = [row for row in tables["Node"] if value(row, "id") in node_ids]
    selected["LineType"] = [row for row in tables["LineType"] if value(row, "id") in {value(x, "type") for x in selected["Line"]}]
    selected["TransformerType"] = [row for row in tables["TransformerType"] if value(row, "id") in {value(x, "type") for x in selected["Transformer"]}]
    nodes = {value(row, "id"): row for row in selected["Node"]}
    assert len(nodes) == len(selected["Node"]) == len(node_ids), "Missing/duplicate selected node IDs"
    parent = {n: n for n in nodes}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    switch_states = Counter(value(row, "cond") for row in selected["Switch"])
    assert set(switch_states) <= {"0", "1"}, "Unexpected switch-state encoding"
    closed_switches = [row for row in selected["Switch"] if value(row, "cond") == "1"]
    for row in closed_switches:
        parent[find(value(row, "nodeB"))] = find(value(row, "nodeA"))
    groups = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    representatives = {}
    for component in groups.values():
        busbars = [n for n in component if value(nodes[n], "type") == "busbar"]
        assert len(busbars) == 1, "Selected switch component lacks a unique physical busbar"
        representatives.update({n: busbars[0] for n in component})

    raw_edges = [(value(row, "nodeA"), value(row, "nodeB")) for row in selected["Line"] + closed_switches]
    raw_edges += [(value(row, "nodeHV"), value(row, "nodeLV")) for row in selected["Transformer"]]
    raw_adj = adjacency(node_ids, raw_edges)
    raw_connected = len(reachable(raw_adj, next(iter(node_ids)))) == len(node_ids)
    assert raw_connected and len(raw_edges) == len(node_ids) - 1, "Original operating graph is not a tree"
    assert len(selected["Transformer"]) == 1
    trafo = selected["Transformer"][0]
    root = representatives[value(trafo, "nodeLV")]
    line_types = {value(row, "id"): row for row in selected["LineType"]}
    edges = []
    for row in selected["Line"]:
        typ = line_types[value(row, "type")]
        length = float(value(row, "length"))
        edges.append({"source_id": value(row, "id"), "a": representatives[value(row, "nodeA")], "b": representatives[value(row, "nodeB")], "length_km": length, "line_type": value(row, "type"), "r_ohm": length * float(value(typ, "r")), "x_ohm": length * float(value(typ, "x"))})
    lv_nodes = {n for n in set(representatives.values()) if float(value(nodes[n], "vmR")) == 0.4}
    adj = adjacency(lv_nodes, [(row["a"], row["b"]) for row in edges])
    assert all(row["a"] != row["b"] for row in edges), "Line collapsed to a self-loop"
    assert len({frozenset((row["a"], row["b"])) for row in edges}) == len(edges), "Parallel lines need multigraph treatment"
    assert len(reachable(adj, root)) == len(lv_nodes) and len(edges) == len(lv_nodes) - 1
    load_nodes = {representatives[value(row, "node")] for row in selected["Load"]}
    leaves = sorted([n for n in lv_nodes if n != root and len(adj[n]) == 1], key=node_order)
    internal_loads = sorted(load_nodes - set(leaves), key=node_order)
    observed = sorted(load_nodes, key=node_order)
    assert root not in load_nodes
    # Direct downstream sets, including an observed internal node itself.
    block_sets = []
    for edge in edges:
        side = reachable(adj, edge["b"], frozenset((edge["a"], edge["b"])))
        if root in side:
            side = lv_nodes - side
        support = side & load_nodes
        edge["observed_downstream_support"] = sorted(support, key=node_order)
        block_sets.append(support)
    laminar = all(not (a & b) or a <= b or b <= a for a in block_sets for b in block_sets)
    assert laminar and all(block_sets)
    paths = []
    for leaf in leaves:
        queue = deque([(root, [root])])
        seen = {root}
        while queue:
            a, path = queue.popleft()
            if a == leaf:
                paths.append(path)
                break
            for b in sorted(adj[a], key=node_order):
                if b not in seen:
                    seen.add(b)
                    queue.append((b, path + [b]))

    wheel_files = sorted(wheel_dir.glob("simbench-1.6.3-*.whl")) if wheel_dir.exists() else []
    wheel = {"package": "simbench", "version": "1.6.3", "expected_sha256": WHEEL_SHA256, "local_file_present": bool(wheel_files)}
    if wheel_files:
        assert len(wheel_files) == 1
        wheel.update({"filename": wheel_files[0].name, "actual_sha256": sha256(wheel_files[0])})
        assert wheel["actual_sha256"] == WHEEL_SHA256, "Frozen wheel hash mismatch"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "python_version": platform.python_version(),
        "network_code": "1-LV-rural1--0-sw", "subnet": SUBNET,
        "source": {"wheel": wheel, "official_repository": "https://github.com/e2nIEE/simbench", "dataset_description": "https://simbench.de/en/download/datasets/", "csvs": {name + ".csv": {"sha256": sha256(source_dir / (name + ".csv")), "size_bytes": (source_dir / (name + ".csv")).stat().st_size, "full_row_count": len(tables[name])} for name in TABLES}},
        "original_selected_rows": selected,
        "original_operating_graph": {"nodes": len(node_ids), "closed_edges_including_switches_transformer": len(raw_edges), "connected": raw_connected, "tree": True, "switch_count": len(selected["Switch"]), "switch_state_counts": dict(switch_states), "edge_list": raw_edges},
        "closed_switch_contraction": {"rule": "Contract only recorded cond=1 ideal switches; no line impedance changed", "node_to_busbar": representatives},
        "lv_graph": {"root": root, "nodes": sorted(lv_nodes, key=node_order), "node_count": len(lv_nodes), "line_count": len(edges), "connected": True, "tree": True, "root_degree": len(adj[root]), "leaves": leaves, "load_nodes": observed, "internal_load_nodes": internal_loads, "root_to_leaf_paths": paths, "lines": edges},
        "load_summary": {"row_count": len(selected["Load"]), "unique_load_bus_count": len(load_nodes), "p_kw": 1000 * sum(float(value(row, "pLoad")) for row in selected["Load"]), "q_kvar": 1000 * sum(float(value(row, "qLoad")) for row in selected["Load"]), "profile_counts": dict(Counter(value(row, "profile") for row in selected["Load"])), "note": "Original profile labels include L1/L2 and H0; substituting iFlex households is an explicit new load scenario."},
        "res_summary": {"row_count": len(selected["RES"]), "p_kw": 1000 * sum(float(value(row, "pRES")) for row in selected["RES"]), "type_counts": dict(Counter(value(row, "type") for row in selected["RES"])), "removed_or_disabled": False},
        "line_summary": {"length_km_total": sum(e["length_km"] for e in edges), "length_km_min": min(e["length_km"] for e in edges), "length_km_max": max(e["length_km"] for e in edges), "line_type_count": len(line_types), "types": [{"id": value(t, "id"), "r_ohm_per_km": float(value(t, "r")), "x_ohm_per_km": float(value(t, "x")), "i_max_A": float(value(t, "iMax")), "b_original_csv_value": float(value(t, "b"))} for t in line_types.values()]},
        "wzzt_compatibility": {"mathematical_laminar_representation_for_all_metered_load_nodes": laminar, "physical_nonzero_support_count": len(block_sets), "nontrivial_support_count": sum(1 < len(s) < len(observed) for s in block_sets), "physical_singleton_support_count": sum(len(s) == 1 for s in block_sets), "terminal_only_scenario_valid": len(internal_loads) == 0, "production_adapter_status": "pending", "notes": ["Core shared-path block representation remains valid for observed internal load nodes; downstream support includes each such node itself.", "A measured ancestor i of j can satisfy R_ii=R_ij; zero singleton weight is allowed and no positive service line should be invented.", "Production terminal-load-only validator rejects non-leaf loads; RNJ zero labeled edges, ordered-QP positive margin, decoding and metrics require separate compatibility audit.", "Using all 13 load points as measured points does not reproduce the original hidden-internal terminal-only benchmark assumption.", "AC voltages, reactive-demand trajectories and iFlex household mapping have not been prepared by this audit."]},
        "scope": {"power_flow_run": False, "wzzt_run": False, "source_files_modified": False, "synthetic_service_lines_added": False, "loads_relocated": False, "pv_removed": False, "verdict": "topology_data_confirmed_production_adapter_pending"},
    }


def main():
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--source-dir", type=Path, default=here / "simbench_source/simbench/networks/1-complete_data-mixed-all-0-sw")
    parser.add_argument("--wheel-dir", type=Path, default=here / "packages")
    parser.add_argument("--output", type=Path, default=here / "outputs/topology_audit.json")
    args = parser.parse_args()
    result = audit(args.source_dir, args.wheel_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "verdict": result["scope"]["verdict"], "lv_nodes": result["lv_graph"]["node_count"], "lines": result["lv_graph"]["line_count"], "leaves": result["lv_graph"]["leaves"], "internal_load_nodes": result["lv_graph"]["internal_load_nodes"], "switch_states": result["original_operating_graph"]["switch_state_counts"]}, indent=2))


if __name__ == "__main__":
    main()
