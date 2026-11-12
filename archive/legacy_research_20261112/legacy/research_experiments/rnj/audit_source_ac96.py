"""Independently audit saved AC96 source data without importing its generator.

Reads only saved network, raw AC state, measurements and source amplitudes.
Generator diagnostics and its profile/position helper functions are never used.
No topology estimator, power-flow solver, or core module is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
VARIANTS = ("no_source", "constant", "intermittent", "smooth")
OBSERVED_FILES = {
    "P_terminal.csv", "Q_terminal.csv", "V_terminal.csv", "root_voltage.csv",
    "drop_target.csv", "P0_measured.csv", "Q0_measured.csv", "metadata.json",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path: Path, *, integer_columns=False) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(f"nonunique index or columns: {path}")
    if integer_columns:
        frame.columns = frame.columns.astype(int)
    return frame


def vector(path: Path) -> np.ndarray:
    frame = read_table(path)
    if frame.shape != (96, 1):
        raise ValueError(f"expected a 96-point vector: {path}, got {frame.shape}")
    return frame.iloc[:, 0].to_numpy(dtype=float)


def max_abs(value) -> float:
    array = np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.0


def forbidden_keys(value, prefix=""):
    """Find saved truth/location keys in the estimator-visible metadata."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            location = f"{prefix}.{key}" if prefix else str(key)
            lower = str(key).lower()
            if any(token in lower for token in
                   ("truth", "source_bus", "source_location", "physical_load", "_true", "clade")):
                found.append(location)
            found.extend(forbidden_keys(item, location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(forbidden_keys(item, f"{prefix}[{index}]"))
    return found


class Audit:
    def __init__(self):
        self.checks = []

    def require(self, name, condition, *, details=None):
        self.checks.append({"check": name, "passed": bool(condition), "details": details})

    def close(self, name, left, right=0.0, *, tolerance=1e-9):
        difference = np.asarray(left) - np.asarray(right)
        finite = bool(np.isfinite(difference).all())
        residual = max_abs(difference) if finite else None
        self.checks.append({
            "check": name, "passed": finite and residual <= tolerance,
            "max_abs_residual": residual, "tolerance": tolerance,
        })

    def result(self):
        failed = [item for item in self.checks if not item["passed"]]
        return {"passed": not failed, "check_count": len(self.checks),
                "failed_count": len(failed), "checks": self.checks}


def read_network(path: Path):
    buses = pd.read_csv(path / "buses.csv")
    branches = pd.read_csv(path / "branches.csv")
    metadata = read_json(path / "metadata.json")
    root = int(metadata["root_bus"])
    graph = nx.Graph()
    for row in buses.to_dict("records"):
        graph.add_node(int(row["bus_id"]), **row)
    for row in branches.to_dict("records"):
        closed = row.get("is_true_closed", True)
        if isinstance(closed, str):
            closed = closed.lower() == "true"
        if closed:
            graph.add_edge(int(row["from_bus"]), int(row["to_bus"]), **row)
    if not nx.is_tree(graph) or root not in graph:
        raise ValueError("saved network must be a rooted connected tree")
    directed = nx.bfs_tree(graph, root)
    depth = nx.single_source_shortest_path_length(graph, root)
    candidates = [node for node in graph
                  if graph.nodes[node]["bus_type"] == "hidden_internal"
                  and depth[node] >= 2 and directed.out_degree(node) >= 2]
    selection_scope = "depth_at_least_2_hidden_branching"
    if not candidates:
        candidates = [node for node in directed.successors(root)
                      if graph.nodes[node]["bus_type"] == "hidden_internal"]
        selection_scope = "root_child_hidden_fallback"
    if not candidates:
        raise ValueError("no eligible source location")
    downstream_pd = {node: sum(float(graph.nodes[child]["pd_kw"])
                              for child in nx.descendants(directed, node)
                              if graph.nodes[child]["bus_type"] == "observed_terminal")
                     for node in candidates}
    source_bus = min(candidates, key=lambda node: (-downstream_pd[node], node))
    return buses, branches, metadata, graph, directed, source_bus, selection_scope


def expected_profile(variant, mean_kw):
    position = np.arange(96)
    if variant == "no_source":
        return np.zeros(96), np.zeros(96)
    if variant == "constant":
        p = np.full(96, mean_kw)
        return p, .4 * p
    if variant == "intermittent":
        on = ((position >= 32) & (position < 48)) | ((position >= 64) & (position < 80))
        p = np.where(on, 3.0 * mean_kw, 0.0)
        return p, .4 * p
    theta = 2.0 * np.pi * position / 96.0
    p = mean_kw * (1.0 - .6 * np.cos(theta))
    return p, p * (.4 + .15 * np.sin(theta))


def audit_variant(path, network, variant, audit):
    buses, _, meta, graph, directed, expected_bus, _ = network
    tag = f"{path.parent.name}/{variant}"
    obs_path, truth_path, ac_path = path / "observed", path / "truth", path / "truth" / "ac"
    files = {item.name for item in obs_path.iterdir() if item.is_file()}
    audit.require(f"{tag}: observed file allowlist", files == OBSERVED_FILES,
                  details={"extra": sorted(files - OBSERVED_FILES), "missing": sorted(OBSERVED_FILES - files)})
    leaked = forbidden_keys(read_json(obs_path / "metadata.json"))
    source_metadata = read_json(path / "source" / "metadata.json")
    leaked += forbidden_keys(source_metadata, "source")
    audit.require(f"{tag}: estimator metadata excludes truth and source location", not leaked, details=leaked)

    observed = {key: read_table(obs_path / f"{key}.csv", integer_columns=True)
                for key in ("P_terminal", "Q_terminal", "V_terminal", "drop_target")}
    terminal_labels = buses.loc[buses["bus_type"] == "observed_terminal", "bus_id"].astype(int).tolist()
    audit.require(f"{tag}: 96 original terminal observations",
                  all(frame.shape == (96, len(terminal_labels)) and list(frame.columns) == terminal_labels
                      for frame in observed.values()))
    indices = [read_table(item).index for item in (*obs_path.glob("*.csv"), *truth_path.glob("*.csv"), *ac_path.glob("*.csv"))]
    audit.require(f"{tag}: all physical and observed time indices agree",
                  bool(indices) and list(indices[0]) == list(range(96))
                  and all(index.equals(indices[0]) for index in indices))
    for label, frame in observed.items():
        audit.require(f"{tag}: {label} finite", np.isfinite(frame.to_numpy(dtype=float)).all())

    truth = {key: read_table(truth_path / f"{key}.csv", integer_columns=True)
             for key in ("P_physical", "Q_physical", "P_terminal_base", "Q_terminal_base", "V_terminal_true")}
    root_true = vector(truth_path / "root_voltage_true.csv")
    root_measured = vector(obs_path / "root_voltage.csv")
    p_source, q_source = vector(truth_path / "P_unmetered.csv"), vector(truth_path / "Q_unmetered.csv")
    truth_meta = read_json(truth_path / "metadata.json")
    source_bus = truth_meta["source_bus_id"]
    expected_source_bus = None if variant == "no_source" else expected_bus
    audit.require(f"{tag}: source position independently selected from network",
                  source_bus == expected_source_bus, details={"saved": source_bus, "expected": expected_source_bus})
    base_kw = 1000.0 * float(meta["base_mva"])
    mean_kw = .05 * float(buses.loc[buses["bus_type"] == "observed_terminal", "pd_kw"].sum())
    expected_p, expected_q = expected_profile(variant, mean_kw)
    audit.close(f"{tag}: source active profile kW", p_source * base_kw, expected_p, tolerance=1e-10)
    audit.close(f"{tag}: source reactive profile kvar", q_source * base_kw, expected_q, tolerance=1e-10)
    expected_p_energy = 0.0 if variant == "no_source" else 24.0 * mean_kw
    expected_q_energy = .4 * expected_p_energy
    audit.close(f"{tag}: daily source active energy kWh", p_source.sum() * base_kw * .25,
                expected_p_energy, tolerance=1e-10)
    audit.close(f"{tag}: daily source reactive energy kvarh", q_source.sum() * base_kw * .25,
                expected_q_energy, tolerance=1e-10)
    rank = int(np.linalg.matrix_rank(np.column_stack((p_source, q_source))))
    audit.require(f"{tag}: source channel rank", rank == (0 if variant == "no_source" else 2 if variant == "smooth" else 1),
                  details={"rank": rank})
    amplitudes = read_table(path / "source" / "amplitudes.csv")
    audit.require(f"{tag}: source amplitudes only P/Q", set(amplitudes.columns) == {"P_pu", "Q_pu", "P_kw", "Q_kvar"})
    audit.close(f"{tag}: supplied source P matches physical truth", amplitudes["P_pu"], p_source, tolerance=1e-13)
    audit.close(f"{tag}: supplied source Q matches physical truth", amplitudes["Q_pu"], q_source, tolerance=1e-13)
    audit.close(f"{tag}: supplied source active units", amplitudes["P_kw"], amplitudes["P_pu"] * base_kw, tolerance=1e-10)
    audit.close(f"{tag}: supplied source reactive units", amplitudes["Q_kvar"], amplitudes["Q_pu"] * base_kw, tolerance=1e-10)
    for label, source in (("P", p_source), ("Q", q_source)):
        frame = truth[f"{label}_physical"]
        base_frame = truth[f"{label}_terminal_base"].reindex(columns=frame.columns, fill_value=0.0)
        expected = base_frame.to_numpy().copy()
        if expected_source_bus is not None:
            expected[:, frame.columns.get_loc(expected_source_bus)] += source
        audit.close(f"{tag}: {label} physical injections equal terminal load plus source",
                    frame, expected, tolerance=1e-13)
    audit.close(f"{tag}: observed squared voltage target",
                observed["drop_target"], root_measured[:, None]**2 - observed["V_terminal"].to_numpy()**2,
                tolerance=1e-12)

    ac_meta = read_json(ac_path / "metadata.json")
    edges = ac_meta["branch_edges"]
    edge_set = {(int(item["from_bus"]), int(item["to_bus"])) for item in edges}
    audit.require(f"{tag}: AC orientation matches independently rooted network", edge_set == set(directed.edges()))
    labels = [item["label"] for item in edges]
    nodes = [int(value) for value in read_table(ac_path / "V_bus_mag.csv").columns]
    vm = read_table(ac_path / "V_bus_mag.csv", integer_columns=True).loc[:, nodes].to_numpy()
    va = read_table(ac_path / "V_bus_angle_deg.csv", integer_columns=True).loc[:, nodes].to_numpy()
    voltage = vm * np.exp(1j * np.deg2rad(va))
    node_pos = {node: index for index, node in enumerate(nodes)}
    v_from = voltage[:, [node_pos[int(item["from_bus"])] for item in edges]]
    v_to = voltage[:, [node_pos[int(item["to_bus"])] for item in edges]]
    def branch_array(key):
        return read_table(ac_path / f"{key}.csv").loc[:, labels].to_numpy(dtype=float)
    s_from = branch_array("branch_p_from_pu") + 1j * branch_array("branch_q_from_pu")
    s_to = branch_array("branch_p_to_pu") + 1j * branch_array("branch_q_to_pu")
    stored_loss = branch_array("branch_loss_p_pu") + 1j * branch_array("branch_loss_q_pu")
    current = np.conj(s_from / v_from)
    current_mag = branch_array("I_branch_mag")
    z_base = float(meta["base_kv"])**2 / float(meta["base_mva"])
    impedance = np.array([complex(float(graph.edges[int(item["from_bus"]), int(item["to_bus"])]["r_ohm"]),
                                  float(graph.edges[int(item["from_bus"]), int(item["to_bus"])]["x_ohm"])) / z_base
                          for item in edges])
    audit.close(f"{tag}: current magnitude from sending power", current_mag, np.abs(current))
    audit.close(f"{tag}: receiving power from voltage/current", s_to, v_to * np.conj(current))
    audit.close(f"{tag}: branch loss equals sending minus receiving", stored_loss, s_from - s_to)
    audit.close(f"{tag}: branch loss equals impedance times current squared", stored_loss, impedance * current_mag**2)
    audit.close(f"{tag}: complex branch voltage equation", v_to, v_from - impedance * current)
    audit.close(f"{tag}: squared branch voltage equation", np.abs(v_to)**2,
                np.abs(v_from)**2 - 2.0 * np.real(np.conj(impedance) * s_from) + np.abs(impedance)**2 * current_mag**2)
    nodal = np.zeros(voltage.shape, dtype=complex)
    for index, item in enumerate(edges):
        nodal[:, node_pos[int(item["to_bus"])]] += s_to[:, index]
        nodal[:, node_pos[int(item["from_bus"])]] -= s_from[:, index]
    load = truth["P_physical"].reindex(columns=nodes, fill_value=0.0).to_numpy() + 1j * truth["Q_physical"].reindex(columns=nodes, fill_value=0.0).to_numpy()
    root = int(meta["root_bus"])
    nonroot = [node_pos[node] for node in nodes if node != root]
    audit.close(f"{tag}: every nonroot AC node balances P/Q", nodal[:, nonroot], load[:, nonroot], tolerance=1e-7)
    head = [index for index, item in enumerate(edges) if int(item["from_bus"]) == root]
    master = s_from[:, head].sum(axis=1)
    audit.close(f"{tag}: feeder head power balances physical load plus branch loss",
                master, load.sum(axis=1) + stored_loss.sum(axis=1), tolerance=1e-7)
    audit.close(f"{tag}: saved true master P", vector(truth_path / "P0_true.csv"), master.real)
    audit.close(f"{tag}: saved true master Q", vector(truth_path / "Q0_true.csv"), master.imag)
    audit.close(f"{tag}: saved total active loss", vector(truth_path / "P_loss_true.csv"), stored_loss.real.sum(axis=1))
    audit.close(f"{tag}: saved total reactive loss", vector(truth_path / "Q_loss_true.csv"), stored_loss.imag.sum(axis=1))
    audit.close(f"{tag}: root voltage agrees with AC state", voltage[:, node_pos[root]], root_true)
    audit.close(f"{tag}: terminal voltage agrees with AC state", truth["V_terminal_true"], vm[:, [node_pos[node] for node in terminal_labels]])
    audit.require(f"{tag}: AC convergence flags", read_table(ac_path / "converged.csv").iloc[:, 0].astype(bool).all())
    return {"observed": observed, "truth": truth, "root_true": root_true, "root_measured": root_measured,
            "p0_true": master.real, "q0_true": master.imag,
            "p0_measured": vector(obs_path / "P0_measured.csv"),
            "q0_measured": vector(obs_path / "Q0_measured.csv")}


def noise_standard_scores(observed, truth):
    # Relative standard scores compare common RNG draws even when voltage or
    # master-power magnitudes change physically across source profiles.
    return (np.asarray(observed) - np.asarray(truth)) / np.maximum(np.abs(truth), 1e-12)


def audit_recorded_hashes(directory, audit):
    """Rehash the listed files; recorded completion flags are not evidence."""
    manifest = read_json(directory / "manifest.json")
    file_hashes = manifest["file_sha256"]
    source_hashes = read_json(directory / "code_environment.json")["source_sha256"]
    repository = Path(__file__).resolve().parents[2]
    counts = {"data_files_checked": 0, "source_files_checked": 0}
    for scope, root, entries in (("data", directory, file_hashes), ("source", repository, source_hashes)):
        for relative, expected in entries.items():
            path = (root / relative).resolve()
            allowed = path.is_relative_to(root.resolve())
            # Source inputs are only the explicitly recorded Python files;
            # no git diff, unrelated working-tree files, or credentials read.
            allowed = allowed and (scope != "source" or path.suffix == ".py")
            present = allowed and path.is_file()
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if present else None
            audit.require(f"{scope} sha256: {relative}", present and actual == expected,
                          details={"expected": expected, "actual": actual, "path_allowed": allowed})
            counts[f"{scope}_files_checked"] += 1
    required = {str(path.relative_to(directory).as_posix())
                for case in CASES for path in (directory / case).rglob("*") if path.is_file()}
    required.update(("run_config.json", "code_environment.json", "README.md", "summary.csv"))
    audit.require("data hash inventory covers all saved case files", required == set(file_hashes),
                  details={"missing_hash": sorted(required - set(file_hashes)),
                           "unexpected_hash": sorted(set(file_hashes) - required)})
    audit.require("source hash inventory is nonempty", bool(source_hashes))
    return counts


def audit_dataset(directory: Path):
    audit = Audit()
    hash_counts = audit_recorded_hashes(directory, audit)
    summary = {}
    for case in CASES:
        network = read_network(directory / case / "network")
        summary[case] = {"source_bus_id": network[5], "selection_scope": network[6],
                         "nominal_total_pd_kw": float(network[0]["pd_kw"].sum())}
        variants = {}
        for variant in VARIANTS:
            variants[variant] = audit_variant(directory / case / variant, network, variant, audit)
        baseline = variants["no_source"]
        for variant in VARIANTS[1:]:
            item = variants[variant]
            prefix = f"{case}/{variant}: paired"
            for label in ("P", "Q"):
                audit.close(f"{prefix} terminal {label} physical profile", item["truth"][f"{label}_terminal_base"],
                            baseline["truth"][f"{label}_terminal_base"], tolerance=1e-13)
                audit.close(f"{prefix} terminal {label} observations", item["observed"][f"{label}_terminal"],
                            baseline["observed"][f"{label}_terminal"], tolerance=1e-13)
            audit.close(f"{prefix} physical root", item["root_true"], baseline["root_true"], tolerance=1e-13)
            audit.close(f"{prefix} observed root", item["root_measured"], baseline["root_measured"], tolerance=1e-13)
            audit.close(f"{prefix} normalized voltage noise",
                        noise_standard_scores(item["observed"]["V_terminal"], item["truth"]["V_terminal_true"]),
                        noise_standard_scores(baseline["observed"]["V_terminal"], baseline["truth"]["V_terminal_true"]), tolerance=1e-12)
            for label in ("p", "q"):
                audit.close(f"{prefix} normalized master {label.upper()} noise",
                            noise_standard_scores(item[f"{label}0_measured"], item[f"{label}0_true"]),
                            noise_standard_scores(baseline[f"{label}0_measured"], baseline[f"{label}0_true"]), tolerance=1e-10)
    return {"audit": "independent saved-state AC/source audit; no generator, core, solver or diagnostics imported",
            "dataset": str(directory.resolve()), "conditions_expected": 16, "case_summary": summary,
            "hash_checks": hash_counts, **audit.result()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to DIRECTORY/independent_audit.json")
    args = parser.parse_args()
    output = args.output or args.directory / "independent_audit.json"
    result = audit_dataset(args.directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "check_count", "failed_count", "conditions_expected")}, ensure_ascii=False))
    if not result["passed"]:
        for item in result["checks"]:
            if not item["passed"]:
                print(json.dumps(item, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
