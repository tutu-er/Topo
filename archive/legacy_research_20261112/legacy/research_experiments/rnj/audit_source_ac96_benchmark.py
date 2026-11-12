"""Recompute the AC96 benchmark evidence from saved estimator outputs.

This auditor imports no topology estimator, scorer, Source implementation, or
benchmark runner.  Network/truth files are opened only here, after estimation.
The physical R/X matrices are built from intersections of independently rooted
physical paths, using the squared-voltage factor 2 and the saved impedance base.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
VARIANTS = ("no_source", "constant", "intermittent", "smooth")
METHODS = ("NJ", "RNJ", "WZZT", "RNJ+WZZT")
METRICS = ("clade_f1", "precision", "recall", "r_relative_error", "x_relative_error")
ARRAY_NAMES = {"R", "X", "P", "Q", "Y", "P_source", "Q_source"}
HASH_INPUT_NAMES = {
    "observed/P_terminal.csv", "observed/Q_terminal.csv", "observed/V_terminal.csv",
    "observed/root_voltage.csv", "observed/drop_target.csv", "observed/P0_measured.csv",
    "observed/Q0_measured.csv", "observed/metadata.json", "source/amplitudes.csv",
    "source/metadata.json",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_table(path):
    return pd.read_csv(path, index_col=0, float_precision="round_trip")


def _true(value):
    return str(value).strip().lower() in {"true", "1"}


def physical_truth(network_directory: Path, terminal_labels):
    """Construct squared-voltage R/X by physical root-path intersections.

    Terminal labels are bus IDs.  No saved sensitivity matrix, reported clade,
    core helper or benchmark scoring function participates in this calculation.
    """
    network_directory = Path(network_directory)
    meta = read_json(network_directory / "metadata.json")
    buses = pd.read_csv(network_directory / "buses.csv")
    branches = pd.read_csv(network_directory / "branches.csv")
    root = int(meta["root_bus"])
    nodes = {int(value) for value in buses["bus_id"]}
    labels = [int(value) for value in terminal_labels]
    if len(set(labels)) != len(labels) or not set(labels) <= nodes:
        raise ValueError("terminal labels must be distinct saved bus IDs")
    adjacency = {node: [] for node in nodes}
    edges, impedances = [], []
    zbase = float(meta["base_kv"]) ** 2 / float(meta["base_mva"])
    if not np.isfinite(zbase) or zbase <= 0:
        raise ValueError("saved impedance base must be finite and positive")
    for row in branches.to_dict("records"):
        if not _true(row.get("is_true_closed", True)):
            continue
        a, b = int(row["from_bus"]), int(row["to_bus"])
        if a not in nodes or b not in nodes or a == b:
            raise ValueError("invalid physical edge")
        index = len(edges)
        edges.append((a, b))
        impedances.append((float(row["r_ohm"]) / zbase, float(row["x_ohm"]) / zbase))
        adjacency[a].append((b, index))
        adjacency[b].append((a, index))
    if len(edges) != len(nodes) - 1 or root not in nodes:
        raise ValueError("saved physical feeder is not a tree")
    paths = {root: frozenset()}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for other, edge in adjacency[node]:
            if other in paths:
                continue
            paths[other] = paths[node] | {edge}
            queue.append(other)
    if len(paths) != len(nodes):
        raise ValueError("saved physical feeder is disconnected")
    incidence = np.array([[int(edge in paths[node]) for edge in range(len(edges))]
                          for node in labels], dtype=float)
    rx = np.asarray(impedances, dtype=float)
    r = (incidence * (2.0 * rx[:, 0])) @ incidence.T
    x = (incidence * (2.0 * rx[:, 1])) @ incidence.T
    clades = {frozenset(labels[i] for i in np.flatnonzero(incidence[:, edge]))
              for edge in range(len(edges))}
    clades = {item for item in clades if 2 <= len(item) < len(labels)}
    return {"R": r, "X": x, "clades": clades, "terminal_labels": labels,
            "edge_count": len(edges), "zbase_ohm": zbase}


def clade_metrics(predicted, truth, terminal_labels):
    universe = set(terminal_labels)
    def normalized(values):
        result = set()
        for support in values:
            value = frozenset(support)
            if not value <= universe:
                raise ValueError("predicted clade has a nonterminal label")
            if 2 <= len(value) < len(universe):
                result.add(value)
        return result
    predicted_set, true_set = normalized(predicted), normalized(truth)
    matched = len(predicted_set & true_set)
    precision = matched / len(predicted_set) if predicted_set else float(not true_set)
    recall = matched / len(true_set) if true_set else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"clade_f1": f1, "precision": precision, "recall": recall,
            "predicted_clades": len(predicted_set), "true_clades": len(true_set),
            "matched_clades": matched}


def relative_error(estimated, true):
    denominator = float(np.linalg.norm(true))
    return float(np.linalg.norm(np.asarray(estimated) - true) / denominator) if denominator else None


class Audit:
    def __init__(self):
        self.checks = []

    def require(self, name, condition, **details):
        self.checks.append({"check": name, "passed": bool(condition), **details})

    def close(self, name, actual, expected, *, atol=1e-8, rtol=1e-7):
        try:
            actual = np.asarray(actual, dtype=float)
            expected = np.asarray(expected, dtype=float)
            same_shape = actual.shape == expected.shape
            finite = np.isfinite(actual).all() and np.isfinite(expected).all()
            residual = float(np.max(np.abs(actual - expected))) if same_shape and finite and actual.size else 0.0
            passed = same_shape and finite and np.allclose(actual, expected, atol=atol, rtol=rtol)
        except (TypeError, ValueError):
            passed, residual = False, None
        self.require(name, passed, max_abs_residual=residual, atol=atol, rtol=rtol)

    def result(self):
        failed = [item for item in self.checks if not item["passed"]]
        return {"passed": not failed, "check_count": len(self.checks),
                "failed_count": len(failed), "checks": self.checks}


def optimality_certificate(event):
    """Check the numerical certificate without interpreting status 0 as gap 0."""
    if event.get("status") != 0 or event.get("success") is not True:
        return False
    binary_count = event.get("binary_variable_count")
    if binary_count == 0:
        return True
    if not isinstance(binary_count, int) or binary_count < 0:
        return False
    primal, dual = event.get("objective"), event.get("dual_bound")
    if primal is not None and dual is not None:
        if np.isfinite(primal) and np.isfinite(dual) and abs(primal - dual) <= 1e-10:
            return True
    gap = event.get("mip_gap")
    return gap is not None and np.isfinite(gap) and 0 <= gap <= 1e-10


def check_atom_model(model, arrays, audit, *, prefix="model", active_tolerance=1e-8):
    """Rebuild actual atom matrices, Source coefficients and full residuals."""
    n = len(model["terminal_labels"])
    supports = model.get("support_indices")
    is_atom_model = supports is not None
    if is_atom_model:
        supports = [tuple(int(i) for i in support) for support in supports]
        support_sets = [set(support) for support in supports]
        audit.require(f"{prefix}: nonempty unique support positions", all(
            support and len(support) == len(set(support)) and all(0 <= i < n for i in support)
            for support in supports) and len({frozenset(support) for support in supports}) == len(supports))
        audit.require(f"{prefix}: laminar supports", all(
            a <= b or b <= a or a.isdisjoint(b)
            for index, a in enumerate(support_sets) for b in support_sets[index + 1:]))
        r = np.asarray(model.get("r_values"), dtype=float)
        x = np.asarray(model.get("x_values"), dtype=float)
        k = len(supports)
        audit.require(f"{prefix}: atom weight dimensions", r.shape == (k,) and x.shape == (k,))
        if r.shape != (k,) or x.shape != (k,):
            return None
        z = np.zeros((k, n), dtype=float)
        for index, support in enumerate(supports):
            if any(i < 0 or i >= n for i in support):
                return None
            z[index, list(support)] = 1.0
        for name, values in (("r", r), ("x", x)):
            bound = model.get(f"{name}_upper_bound")
            valid_bound = bound is not None and np.isfinite(bound) and bound > 0
            audit.require(f"{prefix}: {name} bounded atom weights", valid_bound and
                          np.isfinite(values).all() and np.all(values >= -1e-7) and
                          np.all(values <= float(bound) + 1e-7))
        audit.close(f"{prefix}: R reconstructed from atoms", arrays["R"], (z.T * r) @ z)
        audit.close(f"{prefix}: X reconstructed from atoms", arrays["X"], (z.T * x) @ z)
        labels = model["terminal_labels"]
        active_clades = [frozenset(labels[i] for i in support)
                         for support, rr, xx in zip(supports, r, x)
                         if (rr > active_tolerance or xx > active_tolerance) and 2 <= len(support) < n]
        actual_clades = {frozenset(value) for value in model.get("clades", []) if 2 <= len(value) < n}
        audit.require(f"{prefix}: reported clades match active atoms", set(active_clades) == actual_clades)
        source_active = bool(np.any(arrays["P_source"] != 0) or np.any(arrays["Q_source"] != 0))
        h_value = model.get("source_h")
        if source_active:
            h = np.asarray(h_value, dtype=float)
            audit.require(f"{prefix}: full binary source h", h.shape == (k,) and
                          np.isfinite(h).all() and np.all(np.abs(h - np.rint(h)) <= 1e-6) and
                          np.all(h >= -1e-6) and np.all(h <= 1 + 1e-6))
            if h.shape != (k,) or not np.isfinite(h).all():
                return None
            audit.require(f"{prefix}: non-root source path nonempty", h.sum() >= 1 - 1e-6)
            constraints_ok = True
            for a_index, a in enumerate(support_sets):
                for b_index, b in enumerate(support_sets[a_index + 1:], start=a_index + 1):
                    if a < b:
                        constraints_ok &= h[a_index] <= h[b_index] + 1e-6
                    elif b < a:
                        constraints_ok &= h[b_index] <= h[a_index] + 1e-6
                    elif a.isdisjoint(b):
                        constraints_ok &= h[a_index] + h[b_index] <= 1 + 1e-6
            audit.require(f"{prefix}: complete ancestor and disjoint source path constraints", constraints_ok)
            audit.close(f"{prefix}: source g_r from atom h", model.get("source_g_r"), (r * h) @ z)
            audit.close(f"{prefix}: source g_x from atom h", model.get("source_g_x"), (x * h) @ z)
        else:
            audit.close(f"{prefix}: no-source g_r", model.get("source_g_r"), np.zeros(n))
            audit.close(f"{prefix}: no-source g_x", model.get("source_g_x"), np.zeros(n))
        events = model.get("solver_events", [])
        audit.require(f"{prefix}: solver evidence retained", isinstance(events, list) and bool(events))
        for event_index, event in enumerate(events):
            audit.require(f"{prefix}: solve {event_index} acceptance explicitly recorded",
                          isinstance(event.get("accepted"), bool))
            if event.get("accepted") is True:
                audit.require(f"{prefix}: accepted solve {event_index} certified", optimality_certificate(event))
    else:
        audit.require(f"{prefix}: baseline matrices have regression origin",
                      model.get("matrix_origin") == "source_adjusted_ordered_regression")
    g_r = np.asarray(model.get("source_g_r"), dtype=float)
    g_x = np.asarray(model.get("source_g_x"), dtype=float)
    audit.require(f"{prefix}: Source nuisance coefficient dimensions", g_r.shape == (n,) and g_x.shape == (n,)
                  and np.isfinite(g_r).all() and np.isfinite(g_x).all())
    if g_r.shape != (n,) or g_x.shape != (n,):
        return None
    prediction = arrays["P"] @ arrays["R"].T + arrays["Q"] @ arrays["X"].T
    if not is_atom_model:
        source_design = np.column_stack((arrays["P_source"].reshape(-1), arrays["Q_source"].reshape(-1)))
        expected_g = np.linalg.lstsq(source_design, arrays["Y"] - prediction, rcond=None)[0]
        audit.close(f"{prefix}: Source nuisance g independently refit after projection", np.vstack((g_r, g_x)), expected_g,
                    atol=1e-7, rtol=1e-6)
    prediction += arrays["P_source"].reshape(-1, 1) * g_r + arrays["Q_source"].reshape(-1, 1) * g_x
    train_mae = float(np.mean(np.abs(arrays["Y"] - prediction)))
    audit.close(f"{prefix}: full training mean absolute objective", model.get("train_mae"), train_mae,
                atol=1e-9, rtol=1e-7)
    if is_atom_model:
        accepted_events = [event for event in model.get("solver_events", []) if event.get("accepted") is True]
        audit.require(f"{prefix}: actual model has an accepted solve", bool(accepted_events))
        if accepted_events:
            audit.close(f"{prefix}: last accepted solver objective independently reproduced",
                        accepted_events[-1].get("objective"), train_mae, atol=1e-7, rtol=1e-7)
    return train_mae


def _audit_input_hashes(job_directory, condition_directory, copy_directory, audit, prefix):
    hash_path = job_directory / "inputs_hash.json"
    audit.require(f"{prefix}: input hash manifest exists", hash_path.is_file())
    if not hash_path.is_file():
        return
    content = read_json(hash_path)
    entries = content.get("file_sha256", content)
    audit.require(f"{prefix}: exact observed/source input inventory", set(entries) == HASH_INPUT_NAMES,
                  missing=sorted(HASH_INPUT_NAMES - set(entries)), extra=sorted(set(entries) - HASH_INPUT_NAMES))
    for relative, expected in entries.items():
        normalized = str(relative).replace("\\", "/")
        path = (condition_directory / normalized).resolve()
        allowed = normalized in HASH_INPUT_NAMES and path.is_relative_to(condition_directory.resolve())
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if allowed and path.is_file() else None
        audit.require(f"{prefix}: estimator input sha256 {relative}", allowed and digest == expected)
        copied_path = (copy_directory / normalized).resolve()
        copy_allowed = normalized in HASH_INPUT_NAMES and copied_path.is_relative_to(copy_directory.resolve())
        copied_digest = hashlib.sha256(copied_path.read_bytes()).hexdigest() if copy_allowed and copied_path.is_file() else None
        audit.require(f"{prefix}: copied estimator input sha256 {relative}",
                      copy_allowed and copied_digest == expected and copied_digest == digest)
    copied_inventory = {path.relative_to(copy_directory).as_posix()
                        for path in copy_directory.rglob("*") if path.is_file()}
    audit.require(f"{prefix}: copied input directory excludes unlisted files", copied_inventory == HASH_INPUT_NAMES,
                  missing=sorted(HASH_INPUT_NAMES - copied_inventory), extra=sorted(copied_inventory - HASH_INPUT_NAMES))


def audit_benchmark(directory: Path, dataset_directory: Path, *, active_tolerance=1e-8):
    directory, dataset_directory = Path(directory), Path(dataset_directory)
    audit, rows = Audit(), []
    expected = {f"{case}__{variant}__{method}" for case in CASES for variant in VARIANTS for method in METHODS}
    jobs_directory = directory / "jobs"
    actual = {path.name for path in jobs_directory.iterdir() if path.is_dir()} if jobs_directory.is_dir() else set()
    audit.require("all 64 attempted jobs retained", actual == expected,
                  missing=sorted(expected - actual), extra=sorted(actual - expected))
    for case in CASES:
        for variant in VARIANTS:
            for method in METHODS:
                key = f"{case}__{variant}__{method}"
                job_directory = jobs_directory / key
                result_path = job_directory / "result.json"
                row = {"case": case, "variant": variant, "method": method,
                       "status": "missing", **{name: None for name in METRICS}}
                rows.append(row)
                audit.require(f"{key}: result checkpoint exists", result_path.is_file())
                if not result_path.is_file():
                    continue
                result = read_json(result_path)
                row.update({name: result.get(name) for name in ("status", "stop_reason", "elapsed_seconds", "search_complete")})
                audit.require(f"{key}: job identity matches checkpoint", all(result.get(name) == value
                              for name, value in (("case", case), ("variant", variant), ("method", method))))
                elapsed = result.get("elapsed_seconds")
                audit.require(f"{key}: elapsed time finite and retained", elapsed is not None and
                              np.isfinite(elapsed) and elapsed >= 0)
                condition_directory = dataset_directory / case / variant
                copy_directory = directory / "inputs" / f"{case}__{variant}"
                _audit_input_hashes(job_directory, condition_directory, copy_directory, audit, key)
                forbidden_paths = [path.name for path in job_directory.rglob("*")
                                   if path.name.lower() in {"truth", "network", "buses.csv", "branches.csv"}]
                audit.require(f"{key}: worker artifact directory excludes physical truth", not forbidden_paths,
                              leaked_paths=forbidden_paths)
                model_path, npz_path = job_directory / "model.json", job_directory / "model.npz"
                artifacts_exist = model_path.is_file() and npz_path.is_file()
                declared_model = result.get("has_model")
                audit.require(f"{key}: result explicitly declares model availability", isinstance(declared_model, bool))
                metrics = result.get("metrics") or {}
                if not artifacts_exist:
                    audit.require(f"{key}: absent model matches availability declaration", declared_model is False)
                    audit.require(f"{key}: missing output metrics remain null", all(metrics.get(name) is None for name in METRICS))
                    audit.require(f"{key}: missing model has explicit failed/timeout status",
                                  result.get("status") in {"failed", "incomplete", "method_timeout", "timeout", "error", "worker_failed"})
                    continue
                try:
                    model = read_json(model_path)
                    has_model = model.get("has_model")
                    audit.require(f"{key}: saved model availability matches result", isinstance(has_model, bool)
                                  and has_model == declared_model)
                    with np.load(npz_path, allow_pickle=False) as saved:
                        audit.require(f"{key}: exact saved estimator array inventory", set(saved.files) == ARRAY_NAMES,
                                      array_keys=saved.files)
                        arrays = {name: saved[name].copy() for name in ARRAY_NAMES}
                    n = len(model["terminal_labels"])
                    audit.require(f"{key}: all saved arrays finite", all(np.isfinite(value).all() for value in arrays.values()))
                    audit.require(f"{key}: 96-point estimator array dimensions", all(arrays[name].shape == (96, n)
                                  for name in ("P", "Q", "Y")) and all(arrays[name].shape in {(96,), (96, 1)}
                                  for name in ("P_source", "Q_source")))
                    for name, filename in (("P", "P_terminal.csv"), ("Q", "Q_terminal.csv"), ("Y", "drop_target.csv")):
                        table = _read_table(condition_directory / "observed" / filename)
                        audit.require(f"{key}: {name} terminal order matches input", [int(value) for value in table.columns] == model["terminal_labels"])
                        audit.close(f"{key}: {name} exactly matches observed input", arrays[name], table.to_numpy(), atol=1e-13, rtol=0)
                    source = _read_table(condition_directory / "source" / "amplitudes.csv")
                    for name, column in (("P_source", "P_pu"), ("Q_source", "Q_pu")):
                        audit.close(f"{key}: {name} exactly matches supplied amplitudes", arrays[name].reshape(-1),
                                    source[column].to_numpy(), atol=1e-13, rtol=0)
                    if has_model is not True:
                        audit.require(f"{key}: unavailable model uses empty R/X placeholders",
                                      arrays["R"].shape == (0, 0) and arrays["X"].shape == (0, 0))
                        audit.require(f"{key}: unavailable model metrics remain null", all(metrics.get(name) is None for name in METRICS))
                        audit.require(f"{key}: unavailable model has explicit failure or incomplete status",
                                      result.get("status") in {"failed", "incomplete", "method_timeout", "timeout", "error", "worker_failed"}
                                      and model.get("status") in {"failed", "incomplete", "method_timeout", "timeout", "error", "worker_failed"})
                        continue
                    audit.require(f"{key}: available R/X matrix dimensions", arrays["R"].shape == (n, n)
                                  and arrays["X"].shape == (n, n))
                    row["train_mae_recomputed"] = check_atom_model(model, arrays, audit, prefix=key,
                                                                    active_tolerance=active_tolerance)
                    truth = physical_truth(dataset_directory / case / "network", model["terminal_labels"])
                    computed = clade_metrics(model.get("clades", []), truth["clades"], model["terminal_labels"])
                    computed["r_relative_error"] = relative_error(arrays["R"], truth["R"])
                    computed["x_relative_error"] = relative_error(arrays["X"], truth["X"])
                    row.update(computed)
                    for name in METRICS:
                        audit.close(f"{key}: reported {name} independently reproduced", metrics.get(name), computed[name],
                                    atol=1e-10, rtol=1e-8)
                except Exception as error:
                    audit.require(f"{key}: saved output fully auditable", False,
                                  exception=f"{type(error).__name__}: {error}")
    return {"audit": "independent physical-path and saved-model benchmark audit; no core/scorer imports",
            "directory": str(directory.resolve()), "dataset_directory": str(dataset_directory.resolve()),
            "jobs_expected": 64, "jobs_present": len(actual), "active_atom_tolerance": active_tolerance,
            "metrics_recomputed": rows, **audit.result()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dataset-directory", type=Path, default=Path("outputs/source_ac96_20261018"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--active-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    result = audit_benchmark(args.directory, args.dataset_directory, active_tolerance=args.active_tolerance)
    output = args.output or args.directory / "independent_benchmark_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(result["metrics_recomputed"]).to_csv(output.with_name("metrics_recomputed.csv"), index=False,
                                                    float_format="%.17g")
    print(json.dumps({name: result[name] for name in ("passed", "check_count", "failed_count", "jobs_expected", "jobs_present")}))
    if not result["passed"]:
        for check in result["checks"]:
            if not check["passed"]:
                print(json.dumps(check, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
