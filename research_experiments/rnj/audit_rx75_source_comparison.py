"""Independent, final-artifact-only audit of the thirty RX75 Source jobs.

No driver, floor adapter, runtime estimator or solver is imported. Running
models are never scored as final outcomes. Partial audits retain all thirty
planned jobs and explicitly separate unselected/pending/final evidence.
Physical truth is used only for posthoc scores and scale-assumption diagnostics;
the complete true clade set is never contracted to improve a score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from research_experiments.rnj.audit_source_ac96_benchmark import (
    Audit, clade_metrics, optimality_certificate, physical_truth, relative_error,
)
from research_experiments.rnj import audit_source_ac288 as evidence
from research_experiments.rnj.audit_rnj_initialization import audit_initialization

ROOT = Path(__file__).resolve().parents[2]
CASE = "soumalas11"
METHODS = ("NJ", "RNJ", "WZZT", "RNJ+WZZT", "RNJ+WZZT_tree_init")
VARIANTS = ("no_source", "rank2_source")
FINAL = {"complete", "incomplete", "failed"}
TOL = 1e-10
ARRAYS = {"R", "X", "P", "Q", "Y", "P_source", "Q_source", "row_scenario", "row_position",
          "calibration_R", "calibration_X", "calibration_source_g"}
EXPECTED_CORE = {"max_atoms": 21, "r_upper_bound": 2., "x_upper_bound": 2.,
    "bound_safety_factor": 2., "bound_expansion_factor": 2., "max_bound_expansions": 4,
    "bound_active_fraction": .995, "improvement_abs_tol": 1e-10, "improvement_rel_tol": 1e-8,
    "zero_tolerance": 1e-9, "validation_blocks": 4, "mip_rel_gap": 0., "presolve": True, "disp": False}
EXPECTED_SCREEN = {"bootstrap_replicates": 100, "block_length": 4, "confidence_threshold": .75,
    "maximum_candidate_count": 2, "tolerance_factor": .16, "seed": 20260903,
    "maximum_subclade_frequency": .1, "fine_tolerance_ratio": .5}


def expected_jobs():
    return {f"{CASE}__replicate_{r}__{v}__{m}":
            {"case": CASE, "replicate": r, "variant": v, "method": m}
            for r in range(3) for v in VARIANTS for m in METHODS}


def independent_calibration(r, x):
    matrices = [np.asarray(value, dtype=float) for value in (r, x)]
    if (any(a.ndim != 2 or a.shape[0] != a.shape[1] or not a.size or not np.isfinite(a).all()
            for a in matrices) or matrices[0].shape != matrices[1].shape):
        raise ValueError("finite matching square calibration matrices required")
    matrices = [(a+a.T)*.5 for a in matrices]
    means = []
    for matrix in matrices:
        d = np.diag(matrix)[:, None]+np.diag(matrix)[None, :]-2*matrix
        positive = d[d > 0.]
        means.append(max(float(positive.mean()) if positive.size else 1., 1e-12))
    a, b = .75/means[0], .25/means[1]
    median = float(np.median(np.diag(a*matrices[0]+b*matrices[1])))
    return {"mu_r": means[0], "mu_x": means[1], "a": a, "b": b,
            "median_root_depth": median, "factor": .16, "tau": .16*max(median, 1e-12)}


def resolution_atoms(supports, r, x, calibration):
    r, x = np.asarray(r, dtype=float), np.asarray(x, dtype=float)
    if r.shape != (len(supports),) or x.shape != r.shape or not np.isfinite([r, x]).all():
        raise ValueError("one finite coefficient pair per support required")
    weights = calibration["a"]*r+calibration["b"]*x
    keep = [len(s) == 1 or value+TOL >= calibration["tau"] for s, value in zip(supports, weights)]
    return weights, keep


def independent_tree_resolution(tree, labels, tau):
    """Contract via a rooted union-find, then recompute descendant clades."""
    root, labels = int(tree["root"]), set(map(int, labels))
    adjacency, weights = {}, {}
    for left, right, weight in tree["edges"]:
        left, right, weight = int(left), int(right), float(weight)
        edge = frozenset((left, right))
        if left == right or edge in weights or not np.isfinite(weight) or weight < 0:
            raise ValueError("invalid baseline tree edge")
        weights[edge] = weight
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    parent, order = {root: None}, [root]
    for node in order:
        for other in adjacency.get(node, []):
            if other == parent[node]:
                continue
            if other in parent:
                raise ValueError("baseline tree has a cycle")
            parent[other] = node
            order.append(other)
    if set(order) != set(adjacency) or not labels <= set(order):
        raise ValueError("baseline tree disconnected or missing terminals")
    union = {node: node for node in order}
    def find(node):
        while union[node] != node:
            union[node] = union[union[node]]
            node = union[node]
        return node
    contracted = []
    for node in order[1:]:
        if node not in labels and weights[frozenset((node, parent[node]))]+TOL < tau:
            union[find(node)] = find(parent[node])
            contracted.append((parent[node], node, weights[frozenset((node, parent[node]))]))
    compact = [(find(a), find(b), w) for edge, w in weights.items()
               for a, b in [tuple(edge)] if find(a) != find(b)]
    def clades(edges):
        adj = {root: []}
        for a, b, _ in edges:
            adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
        par, stack = {root: None}, [root]
        for a in stack:
            for b in adj[a]:
                if b != par[a]:
                    if b in par: raise ValueError("contracted tree has a cycle")
                    par[b] = a; stack.append(b)
        descendants = {a: ({a} if a in labels else set()) for a in stack}
        for a in reversed(stack[1:]): descendants[par[a]].update(descendants[a])
        return {frozenset(s) for a, s in descendants.items() if a != root and 1 < len(s) < len(labels)}
    return {"family": clades([(a, b, w) for edge, w in weights.items() for a, b in [tuple(edge)]]),
            "resolution": clades(compact), "edges": compact, "contracted_edges": contracted}


def true_edge_scales(network, labels, calibration):
    """Diagnose physical non-singleton edges, including the root stem.

    The online floor applies to a full-terminal support too, although the
    topology F1 excludes that trivial clade. Report both domains explicitly.
    """
    network = Path(network)
    meta = evidence.read_json(network/"metadata.json")
    with (network/"branches.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    adjacency, oriented = {}, []
    for row in rows:
        if str(row.get("is_true_closed", True)).lower() not in {"true", "1"}: continue
        a, b = int(row["from_bus"]), int(row["to_bus"])
        adjacency.setdefault(a, []).append((b, row)); adjacency.setdefault(b, []).append((a, row))
    root, labels = int(meta["root_bus"]), set(map(int, labels))
    if root not in adjacency or not labels <= set(adjacency):
        raise ValueError("physical network missing root or requested terminals")
    parent, order = {root: None}, [root]
    for a in order:
        for b, row in adjacency[a]:
            if b == parent[a]: continue
            if b in parent: raise ValueError("physical network is not a tree")
            parent[b] = a; order.append(b); oriented.append((a, b, row))
    if set(order) != set(adjacency):
        raise ValueError("physical network disconnected")
    descendants = {a: ({a} if a in labels else set()) for a in order}
    for a in reversed(order[1:]): descendants[parent[a]].update(descendants[a])
    zbase = float(meta["base_kv"])**2/float(meta["base_mva"])
    if not np.isfinite(zbase) or zbase <= 0:
        raise ValueError("positive finite impedance base required")
    values = []
    for a, b, row in oriented:
        if not 1 < len(descendants[b]): continue
        r, x = 2*float(row["r_ohm"])/zbase, 2*float(row["x_ohm"])/zbase
        strength = calibration["a"]*r+calibration["b"]*x
        values.append({"from_bus": a, "to_bus": b, "clade": sorted(descendants[b]),
                       "is_root_stem": len(descendants[b]) == len(labels),
                       "r_coefficient": r, "x_coefficient": x, "strength": strength,
                       "below_tau": strength+TOL < calibration["tau"]})
    count = sum(v["below_tau"] for v in values)
    proper = [v for v in values if not v["is_root_stem"]]
    stems = [v for v in values if v["is_root_stem"]]
    return {"edges": values, "nontrivial_internal_edge_count": len(values),
            "proper_internal_edge_count": len(proper), "root_stem_edge_count": len(stems),
            "proper_internal_min_true_strength": min((v["strength"] for v in proper), default=None),
            "proper_internal_below_tau_count": sum(v["below_tau"] for v in proper),
            "root_stem_below_tau_count": sum(v["below_tau"] for v in stems),
            "min_true_strength": min((v["strength"] for v in values), default=None),
            "below_tau_count": count, "tau": calibration["tau"], "absolute_tolerance": TOL,
            "assumption_satisfied": count == 0, "complete_truth_used_for_scoring": True}


def _sparse_rows(saved):
    rows = [{} for _ in saved["constraint_lower"]]
    for i, j, value in zip(saved["constraint_row_index"], saved["constraint_col_index"], saved["constraint_data"]):
        row = rows[int(i)]; row[int(j)] = row.get(int(j), 0.)+float(value)
    return rows


def audit_floor_domain(event, saved, n, partition, calibration, audit, prefix):
    """Inspect actual original COO rows, not a regenerated adapter model."""
    slices = event["variable_slices"]
    r = slices.get("old_r", slices.get("r")); x = slices.get("old_x", slices.get("x"))
    if r is None or x is None: raise ValueError("raw atom-weight slices missing")
    k = r[1]-r[0]
    audit.require(f"{prefix}: raw old weights include frozen singleton prefix", k >= n and x[1]-x[0] == k)
    rows = _sparse_rows(saved)
    lower, upper = saved["constraint_lower"], saved["constraint_upper"]
    def has(entries, lo=-np.inf, hi=np.inf):
        return any(set(row) == set(entries) and all(np.isclose(row[j], value, atol=1e-12, rtol=1e-10)
                   for j, value in entries.items()) and np.isclose(lower[i], lo, atol=1e-12, rtol=1e-10)
                   and np.isclose(upper[i], hi, atol=1e-12, rtol=1e-10)
                   for i, row in enumerate(rows))
    a, b, tau = (calibration[name] for name in ("a", "b", "tau"))
    audit.require(f"{prefix}: positive frozen geometry and feasible coefficient capacity",
                  a > 0 and b > 0 and tau > 0 and np.isfinite([a, b, tau]).all()
                  and a*float(saved["upper_bounds"][r[0]])+b*float(saved["upper_bounds"][x[0]])+1e-12 >= tau)
    active = slices.get("edge_active")
    audit.require(f"{prefix}: one edge_active per old non-singleton only",
                  (active is None and k == n) or (active is not None and active[1]-active[0] == k-n))
    if active is not None:
        for offset, atom in enumerate(range(n, k)):
            ri, xi, ei = r[0]+atom, x[0]+atom, active[0]+offset
            ur, ux = saved["upper_bounds"][ri], saved["upper_bounds"][xi]
            audit.require(f"{prefix}: edge[{atom}] binary activation domain",
                          saved["integrality"][ei] == 1 and saved["lower_bounds"][ei] == 0 and saved["upper_bounds"][ei] == 1)
            audit.require(f"{prefix}: edge[{atom}] zero-or-floor original rows",
                          has({ri: 1., ei: -ur}, hi=0.) and has({xi: 1., ei: -ux}, hi=0.)
                          and has({ri: a, xi: b, ei: -tau}, lo=0.))
            containing = [row for row in rows if ei in row]
            audit.require(f"{prefix}: edge[{atom}] activation does not alter Source h constraints",
                          len(containing) == 3 and all(set(row) <= {ri, xi, ei} for row in containing))
    if "new_r" in slices:
        ri, xi = slices["new_r"][0], slices["new_x"][0]
        audit.require(f"{prefix}: new atom compares positive resolved extensions directly",
                      has({ri: a, xi: b}, lo=tau))
        z = slices["z"]
        audit.require(f"{prefix}: group membership dimension matches original-terminal partition", z[1]-z[0] == len(partition))
        cardinality = {z[0]+i: float(len(group)) for i, group in enumerate(partition)}
        audit.require(f"{prefix}: original-terminal cardinality row is >=2", has(cardinality, lo=2.))
        if "primal" in saved:
            value = saved["primal"]
            count = sum(len(group) for i, group in enumerate(partition) if value[z[0]+i] > .5)
            audit.require(f"{prefix}: incumbent new support has at least two original terminals", count >= 2)
            audit.require(f"{prefix}: incumbent new coefficient reaches the fixed resolution",
                          a*value[ri]+b*value[xi]+1e-6 >= tau)


def _edges(values):
    return sorted((min(int(a), int(b)), max(int(a), int(b)), float(w)) for a, b, w in values)


def _sets(values): return {frozenset(map(int, v)) for v in values}


def _metrics(value, truth, labels):
    result = {}
    for kind in ("family", "active", "resolution"):
        score = clade_metrics(value[kind], truth["clades"], labels)
        result.update({f"{kind}_{name}": score[name] for name in ("clade_f1", "precision", "recall")})
    result.update(clade_f1=result["resolution_clade_f1"], r_relative_error=relative_error(value["R"], truth["R"]),
                  x_relative_error=relative_error(value["X"], truth["X"]))
    return result


def posthoc_path_metrics(path, reconstructed, labels, calibration, truth):
    """Score an already selected certified path; this function never selects.

    In particular, a better truth score at path0 leaves the original last
    point selected. This separates a fixed-tree joint refit from support search.
    """
    if len(path) != len(reconstructed):
        raise ValueError("one reconstruction required per certified path point")
    rows = []
    for index, (point, value) in enumerate(zip(path, reconstructed)):
        _, keep = resolution_atoms(point["support_indices"], point["r_values"], point["x_values"], calibration)
        resolved = {frozenset(labels[j] for j in support)
            for support, retained in zip(point["support_indices"], keep) if retained and 1 < len(support) < len(labels)}
        rows.append({"path_index": index, "iteration": point["iteration"], "support_count": len(point["support_indices"]),
            "train_mae": value["loss"], "selected_last_point": index == len(path)-1,
            "stage": "fixed_initial_family_same_L1_Source_objective" if index == 0 else "greedy_extension_after_fixed_family",
            "truth_used_for_selection": False, **_metrics({**value, "resolution": resolved}, truth, labels)})
    return rows


def initialization_diagnostic(supplement, initial_supports, n, *, tree_init):
    """Distinguish selected atomic groups from ordinary RNJ tree supports."""
    excluded = "no unreported non-singleton initial supports are frozen"
    applicable = [c for c in supplement["checks"] if not (tree_init and excluded in c["check"])]
    return {"check_count": len(applicable), "passed": all(c["passed"] for c in applicable), "checks": applicable,
        "excluded_checks": [c for c in supplement["checks"] if c not in applicable],
        "exclusion_reason": "tree_init explicitly adds same-screen full clades as ordinary supports" if tree_init else None,
        "selected_count": supplement.get("selected_count"),
        "selected_true_complete_group_count": supplement.get("selected_true_complete_group_count"),
        "selected_false_complete_group_count": supplement.get("selected_false_complete_group_count"),
        "empty_strict_terminal_groups": supplement.get("empty_initialization"),
        "empty_strict_terminal_groups_scope": "only strict bootstrap-selected atomic groups; ordinary initial tree supports may remain",
        "initial_support_count": len(initial_supports),
        "ordinary_nontrivial_initial_support_count": sum(1 < len(s) < n for s in initial_supports),
        "ordinary_full_root_stem_initial_support_count": sum(len(s) == n for s in initial_supports)}


def audit_model(folder, model, arrays, job, network, audit):
    key, labels = job["job_id"], list(map(int, model["terminal_labels"]))
    initialization_report = None
    path_values = []
    audit.require(f"{key}: exact saved-array inventory", set(arrays) == ARRAYS)
    calibration = independent_calibration(arrays["calibration_R"], arrays["calibration_X"])
    stored = evidence.read_json(folder/"rx75_calibration.json")
    for name, value in calibration.items():
        audit.close(f"{key}: independent calibration {name}", stored[name], value, atol=1e-12)
        audit.close(f"{key}: model keeps fixed calibration {name}", model["rx75_calibration"][name], value, atol=1e-12)
    audit.require(f"{key}: calibration is explicitly pre-tree observed FWL, without truth",
                  stored.get("matrix_origin") == "pre_tree_ordered_FWL_regression" and stored.get("truth_used") is False)
    audit.require(f"{key}: fixed classification tolerance is explicit", model.get("resolution_absolute_tolerance") == TOL)
    audit.require(f"{key}: all 288 rows fit without validation", model.get("training_rows") == 288
                  and model.get("validation_supplied") is False and model.get("validation_mae") is None and model.get("validation_se") is None)
    source = np.column_stack((arrays["P_source"], arrays["Q_source"]))
    expected_g = np.linalg.lstsq(source, arrays["Y"]-arrays["P"]@arrays["calibration_R"].T
                                 -arrays["Q"]@arrays["calibration_X"].T, rcond=None)[0]
    audit.close(f"{key}: calibration uses unrestricted FWL response, no position oracle",
                arrays["calibration_source_g"], expected_g, atol=1e-8)
    rank = int(np.linalg.matrix_rank(source))
    audit.require(f"{key}: paired known source has predeclared rank", rank == (2 if job["variant"] == "rank2_source" else 0))
    if "WZZT" not in job["method"]:
        value = evidence.audit_baseline(model, arrays, labels, audit, key)
        tree = independent_tree_resolution(model["tree"], labels, calibration["tau"])
        audit.require(f"{key}: raw family clades are independently traversed tree", _sets(model["family_clades"]) == tree["family"])
        audit.close(f"{key}: raw baseline R remains the pre-tree calibration", arrays["R"], arrays["calibration_R"])
        audit.close(f"{key}: raw baseline X remains the pre-tree calibration", arrays["X"], arrays["calibration_X"])
        audit.require(f"{key}: independent hidden-child contraction matches reported edges", _edges(model["resolution_tree"]["edges"]) == _edges(tree["edges"]))
        audit.require(f"{key}: independent contracted-edge inventory", _edges(model["resolution_tree"]["contracted_edges"]) == _edges(tree["contracted_edges"]))
        value["resolution"] = tree["resolution"]
        audit.require(f"{key}: baseline path is absent", not model.get("path"))
    else:
        path = model["path"]
        audit.require(f"{key}: actual fit parameters preserve frozen policy with dimension-specific atom cap",
                      model.get("core_parameters") == {**EXPECTED_CORE, "max_atoms": 2*len(labels)-1})
        chosen = evidence.audit_path(path, arrays, labels, model["initial_supports"], model["terminal_groups"],
                                     model["r_upper_bound"], model["x_upper_bound"], audit, key)
        value = chosen["selected"]
        path_values = chosen["path_reconstructed"]
        audit.require(f"{key}: original last certified point stays selected", model["selected_path_index"] == len(path)-1
                      and model["last_path_index"] == len(path)-1)
        for i, point in enumerate(path):
            audit.require(f"{key}: path[{i}] retains original singleton prefix", point["support_indices"][:len(labels)] == [[j] for j in range(len(labels))])
            strengths, keep = resolution_atoms(point["support_indices"], point["r_values"], point["x_values"], calibration)
            r, x = np.asarray(point["r_values"]), np.asarray(point["x_values"])
            valid = [len(s) == 1 or (abs(rv) <= 1e-8 and abs(xv) <= 1e-8) or w+TOL >= calibration["tau"]
                     for s, rv, xv, w in zip(point["support_indices"], r, x, strengths)]
            audit.require(f"{key}: path[{i}] every old non-singleton is zero or resolved", all(valid))
        strengths, keep = resolution_atoms(model["support_indices"], model["r_values"], model["x_values"], calibration)
        retained = [i for i, flag in enumerate(keep) if flag]
        audit.close(f"{key}: resolution strengths independently recomputed", model["resolution_atoms"]["strengths"], strengths)
        audit.require(f"{key}: resolution keeps leaves and >=tau with fixed absolute tolerance",
                      model["resolution_atoms"]["keep"] == keep and model["resolution_atoms"]["retained_indices"] == retained
                      and model["resolution_atoms"]["contracted_indices"] == [i for i, flag in enumerate(keep) if not flag]
                      and model["resolution_atoms"]["absolute_tolerance"] == TOL)
        for name in ("a", "b", "tau"):
            audit.close(f"{key}: resolution records fixed {name}", model["resolution_atoms"][name], calibration[name], atol=1e-12)
        value["resolution"] = {frozenset(labels[j] for j in model["support_indices"][i]) for i in retained
                               if 1 < len(model["support_indices"][i]) < len(labels)}
        for field, expected in (("r_values", path[-1]["r_values"]), ("x_values", path[-1]["x_values"]),
                                ("source_h", path[-1].get("source_h"))):
            audit.require(f"{key}: final {field} is the selected path value", model.get(field) == expected)
        for field, expected in (("source_g_r", path[-1].get("source_response_r")), ("source_g_x", path[-1].get("source_response_x"))):
            audit.require(f"{key}: final {field} remains selected Source response", model.get(field) == expected)
        evidence.audit_solver_events(folder, model, job, audit, key)
        evidence.audit_progress(folder, model, arrays, labels, audit, key)
        evidence.audit_stopping(model, audit, key)
        for event in model.get("solver_events", []):
            if event.get("has_incumbent"):
                with np.load(folder/event["primal_file"], allow_pickle=False) as raw:
                    audit_floor_domain(event, raw, len(labels), model["terminal_groups"], calibration, audit, f"{key}: floor solve[{event['sequence']}]")
        for snapshot in sorted((folder/"progress").glob("*.json")):
            saved = evidence.read_json(snapshot)
            for name, expected in calibration.items():
                audit.close(f"{key}: {snapshot.name} calibration fixed across restarts {name}", saved["rx75_calibration"][name], expected, atol=1e-12)
        pointer = evidence.read_json(folder/"last_certified_model.json")
        audit.require(f"{key}: atomic certified pointer binds exactly one JSON/NPZ pair",
                      len(pointer["files"]) == 2 and {Path(p).name for p in pointer["files"]} == {"model.json", "model.npz"})
        for relative, checksum in pointer["files"].items():
            target = (folder/relative).resolve()
            audit.require(f"{key}: atomic certified pointer hash {relative}", target.is_relative_to(folder.resolve())
                          and target.is_file() and evidence.file_hash(target) == checksum)
            if target.is_relative_to(folder.resolve()) and target.is_file():
                if target.name == "model.json":
                    saved_model = evidence.read_json(target)
                    audit.require(f"{key}: final path matches atomic certified model", saved_model["path"] == path)
                elif target.name == "model.npz":
                    with np.load(target, allow_pickle=False) as pair:
                        audit.require(f"{key}: final arrays match atomic certified model pair", set(pair.files) == set(arrays))
                        for name in set(pair.files) & set(arrays):
                            audit.close(f"{key}: atomic certified {name} pair agrees", arrays[name], pair[name], atol=0., rtol=0.)
        if job["method"].startswith("RNJ+WZZT"):
            screen_parameters = evidence.read_json(folder/"rnj_screen.json")["parameters"]
            audit.require(f"{key}: actual bootstrap parameters preserve paired seed and screening policy",
                          screen_parameters == {**EXPECTED_SCREEN, "seed": EXPECTED_SCREEN["seed"]+job["replicate"]})
            supplement = audit_initialization(folder, network)
            initialization_report = initialization_diagnostic(supplement, model["initial_supports"], len(labels),
                                                              tree_init=job["method"].endswith("_tree_init"))
            audit.require(f"{key}: independent RNJ group/checkpoint audit", initialization_report["passed"])
            if job["method"].endswith("_tree_init"):
                screen, initialization = evidence.read_json(folder/"rnj_screen.json"), evidence.read_json(folder/"tree_initialization.json")
                positions = {label: i for i, label in enumerate(labels)}
                expected = {frozenset((i,)) for i in range(len(labels))} | {
                    frozenset(positions[int(label)] for label in clade) for clade in screen["full_clades"]+screen["selected_support_labels"]}
                audit.require(f"{key}: declared tree-init ordinary supports exactly from same screen",
                              {frozenset(s) for s in model["initial_supports"]} == expected
                              and initialization.get("truth_used") is False
                              and initialization["requested_terminal_groups"] == screen["requested_terminal_groups"])
    audit.close(f"{key}: saved R rebuilt independently", arrays["R"], value["R"], atol=1e-10)
    audit.close(f"{key}: saved X rebuilt independently", arrays["X"], value["X"], atol=1e-10)
    for kind in ("family", "active", "resolution"):
        audit.require(f"{key}: reported {kind} clades match independent reconstruction", _sets(model[kind+"_clades"]) == value[kind])
    audit.close(f"{key}: saved recomputed full training loss", model["train_mae_recomputed"], value["loss"], atol=1e-9)
    truth = physical_truth(network, labels)
    posthoc = posthoc_path_metrics(model.get("path", []), path_values, labels, calibration, truth)
    if initialization_report is not None and posthoc:
        initialization_report.update(initial_family_clade_f1=posthoc[0]["family_clade_f1"],
                                     initial_active_clade_f1=posthoc[0]["active_clade_f1"],
                                     initial_resolution_clade_f1=posthoc[0]["resolution_clade_f1"],
                                     initial_score_scope="full initial family; active/resolution after same-objective fixed-family refit; complete physical truth")
    return {"metrics_recomputed": _metrics(value, truth, labels),
            "truth_resolution_assumption": true_edge_scales(network, labels, calibration),
            "calibration_recomputed": calibration, "baseline_matrix_origin": model.get("matrix_origin"),
            "rnj_initialization_audit": initialization_report, "path_metrics_posthoc": posthoc,
            "path_diagnostic_scope": "same final certified path; path0 is fixed-initial-family joint refit, later points include greedy support search; truth never selects a point"}


def _hash_check(root, relative, checksum, audit, prefix):
    path = (root/relative).resolve()
    audit.require(prefix, path.is_relative_to(root.resolve()) and path.is_file() and evidence.file_hash(path) == checksum)


def audit_comparison(directory, dataset_directory=None, *, only_jobs=None):
    directory = Path(directory).resolve()
    audit, rows, calibrations = Audit(), [], {}
    protocol = evidence.read_json(directory/"protocol.json")
    catalog = expected_jobs()
    supplied = {j["job_id"]: j for j in protocol["jobs"]}
    audit.require("all thirty predeclared conditions retained", set(supplied) == set(catalog) and len(protocol["jobs"]) == 30)
    for key, expected in catalog.items():
        job = supplied.get(key, {})
        audit.require(f"{key}: case/replicate/source/method identity", all(job.get(k) == v for k, v in expected.items()))
        wzzt = "WZZT" in expected["method"]
        audit.require(f"{key}: predeclared solve/whole-method budgets", job.get("per_solve_seconds") == (6000. if wzzt else None)
                      and job.get("method_wall_seconds") == (30000. if wzzt else 300.))
        audit.require(f"{key}: paired bootstrap seed", job.get("screen_parameters", {}).get("seed") == 20260903+expected["replicate"])
    for key, expected in {"expected_jobs": 30, "main_jobs": 24, "initializer_ablation_jobs": 6,
                          "training_samples": 288, "validation_samples": 0, "validation_supplied": False,
                          "independent_test_samples": 0, "resolution_absolute_tolerance": TOL,
                          "solver_threads": 1, "blas_threads": 1, "truth_used_for_selection": False}.items():
        audit.require(f"protocol {key}", protocol.get(key) == expected)
    audit.require("Soumalas atom cap is 2*11-1", protocol["core_parameters"].get("max_atoms") == 21)
    audit.require("frozen core numerical and bound policy", protocol["core_parameters"] == EXPECTED_CORE)
    audit.require("frozen complete RNJ screening policy", protocol.get("screen_parameters") == EXPECTED_SCREEN)
    if only_jobs is not None and not set(only_jobs) <= set(catalog): raise ValueError("unknown --only-job")
    selected = set(catalog) if only_jobs is None else set(only_jobs)
    signature = evidence.file_hash(directory/"protocol.json")
    manifest_path = directory/"dataset_manifest.json"
    audit.require("copied dataset manifest fingerprint", manifest_path.is_file() and evidence.file_hash(manifest_path) == protocol["dataset_manifest_sha256"])
    manifest = evidence.read_json(manifest_path)
    for relative, checksum in protocol.get("source_sha256", {}).items():
        _hash_check(directory/"source_snapshot", relative, checksum, audit, f"frozen research source {relative}")
    network = directory/"evaluation_network"
    prefix = CASE+"/network/"
    network_files = {key[len(prefix):]: value for key, value in manifest["file_sha256"].items() if key.startswith(prefix)}
    audit.require("frozen evaluation network complete inventory", bool(network_files)
                  and {p.relative_to(network).as_posix() for p in network.rglob("*") if p.is_file()} == set(network_files))
    for relative, checksum in network_files.items(): _hash_check(network, relative, checksum, audit, f"physical evaluation network {relative}")
    if dataset_directory is not None:
        dataset = Path(dataset_directory).resolve()
        audit.require("external dataset manifest unchanged", evidence.file_hash(dataset/"manifest.json") == protocol["dataset_manifest_sha256"])
        for relative, checksum in manifest["file_sha256"].items(): _hash_check(dataset, relative, checksum, audit, f"original dataset {relative}")
    runtime = Path(protocol["runtime_directory"])
    audit.require("frozen runtime manifest fingerprint", evidence.file_hash(runtime/"runtime_manifest.json") == protocol["runtime_manifest_sha256"])
    runtime_manifest = evidence.read_json(runtime/"runtime_manifest.json")
    for field, folder in (("runtime_sha256", runtime/"rnj_wzzt"), ("source_sha256", runtime/"source_snapshot/rnj_wzzt")):
        audit.require(f"{field} retains source inventory", bool(runtime_manifest[field]) and "__init__.py" in runtime_manifest[field])
        for relative, checksum in runtime_manifest[field].items(): _hash_check(folder, relative, checksum, audit, f"{field} {relative}")
    for key, identity in catalog.items():
        folder = directory/"jobs"/key
        row = {"job_id": key, **identity, "audited": False, "status": "not_run", "has_model": False,
               "metrics_recomputed": None, "truth_resolution_assumption": None}
        rows.append(row)
        if not (folder/"result.json").is_file():
            row["status"] = "running" if (folder/"execution.json").exists() else "not_run"
            continue  # Never open a running model/checkpoint for final grading.
        try:
            result = evidence.read_json(folder/"result.json")
            row.update(status=result.get("status"), has_model=result.get("has_model"), stop_reason=result.get("stop_reason"),
                       compute_seconds=result.get("compute_seconds", result.get("elapsed_seconds")))
            if key not in selected or result.get("status") not in FINAL: continue
            row["audited"] = True
            audit.require(f"{key}: final status explicit", result.get("status") in FINAL)
            audit.require(f"{key}: result protocol hash", result.get("protocol_sha256") == signature)
            audit.require(f"{key}: search completeness is operational status", result.get("search_complete") is (result.get("status") == "complete"))
            audit.require(f"{key}: failure/time remains finite", row["compute_seconds"] is not None
                          and np.isfinite(row["compute_seconds"]) and row["compute_seconds"] >= 0)
            condition = Path(CASE)/f"replicate_{identity['replicate']}"/identity["variant"]
            copied = directory/"inputs"/condition
            expected_inputs = {f"scenario_{i}/{name}" for i in range(3) for name in evidence.INPUT_NAMES}
            inventory = evidence.read_json(folder/"inputs_hash.json")["file_sha256"]
            payload = evidence.read_json(folder/"payload.json")
            audit.require(f"{key}: payload uses only frozen observations/source", Path(payload["input_dir"]).resolve() == copied
                          and payload["input_sha256"] == inventory and payload["protocol_sha256"] == signature)
            audit.require(f"{key}: exact observed/source whitelist", set(inventory) == expected_inputs
                          and {p.relative_to(copied).as_posix() for p in copied.rglob("*") if p.is_file()} == expected_inputs)
            for relative, checksum in inventory.items():
                audit.require(f"{key}: copied input matches generation manifest {relative}", manifest["file_sha256"].get((condition/relative).as_posix()) == checksum)
                _hash_check(copied, relative, checksum, audit, f"{key}: worker input {relative}")
            if not result.get("has_model"):
                audit.require(f"{key}: failed/unavailable model retained without scores", result.get("status") != "complete"
                              and all(value is None for value in result.get("metrics", {}).values()))
                continue
            model = evidence.read_json(folder/"model.json")
            audit.require(f"{key}: final model/result availability agrees", model.get("has_model") is True)
            with np.load(folder/"model.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            labels = list(map(int, model["terminal_labels"]))
            audit.require(f"{key}: production terminal dimension matches protocol", len(labels) == protocol.get("terminal_count", 11))
            evidence.audit_training_arrays(arrays, copied, labels, audit, key)
            use = evidence.read_json(folder/"data_use.json")
            audit.require(f"{key}: actual data-use record has no validation", use.get("training_samples") == 288 and use.get("validation_samples") == 0)
            imports = evidence.read_json(folder/"runtime_import.json")
            evidence.audit_runtime_import(imports, runtime, audit, key,
                                         completed=result.get("status") == "complete")
            origins = imports.get("experiment_module_origins")
            # An outer-killed worker may have only its initial import record.
            # Completed jobs must provide all actual experiment helper origins.
            audit.require(f"{key}: completed experiment helper origins retained",
                          result.get("status") != "complete" or isinstance(origins, dict) and bool(origins))
            for name, origin in (origins or {}).items():
                target, snapshot = Path(origin).resolve(), (directory/"source_snapshot").resolve()
                relative = target.relative_to(snapshot).as_posix() if target.is_relative_to(snapshot) else None
                audit.require(f"{key}: experiment module {name} uses frozen bytes",
                              relative in protocol["source_sha256"] and target.is_file()
                              and evidence.file_hash(target) == protocol["source_sha256"].get(relative))
            details = audit_model(folder, model, arrays, supplied[key], network, audit)
            row.update(details)
            for name, value in details["metrics_recomputed"].items():
                audit.close(f"{key}: physical-truth metric {name}", result.get("metrics", {}).get(name), value, atol=1e-10)
            audit.require(f"{key}: main clade score uses resolution while full truth stays intact", result.get("primary_topology_metric") == "resolution_clade_f1")
            calibrations[key] = (arrays["calibration_R"], arrays["calibration_X"], details["calibration_recomputed"])
        except Exception as error:
            audit.require(f"{key}: saved evidence is independently readable", False, error=f"{type(error).__name__}: {error}")
            row["audit_error"] = f"{type(error).__name__}: {error}"
    case_checks = []
    for key, calibration in calibrations.items():
        identity = catalog[key]
        baseline = f"{CASE}__replicate_{identity['replicate']}__{identity['variant']}__NJ"
        if baseline not in calibrations:
            # A partial audit may select only one WZZT job. Read a FINAL NJ
            # calibration pair, never its running checkpoint, for this case check.
            folder = directory/"jobs"/baseline
            if (folder/"result.json").is_file() and evidence.read_json(folder/"result.json").get("status") in FINAL and (folder/"model.npz").is_file():
                with np.load(folder/"model.npz", allow_pickle=False) as archive:
                    rr, xx = archive["calibration_R"], archive["calibration_X"]
                    reference = (rr, xx, independent_calibration(rr, xx))
            else:
                case_checks.append({"job_id": key, "NJ_reference": baseline, "checked": False, "reason": "NJ final calibration unavailable"})
                continue
        else: reference = calibrations[baseline]
        same = np.allclose(calibration[0], reference[0], atol=1e-10, rtol=1e-10) and np.allclose(calibration[1], reference[1], atol=1e-10, rtol=1e-10)
        audit.require(f"{key}: fixed calibration is the paired NJ baseline matrix", same)
        case_checks.append({"job_id": key, "NJ_reference": baseline, "checked": True, "passed": bool(same)})
    counts = dict(Counter(row["status"] for row in rows))
    result = audit.result()
    return {**result, "expected_jobs": 30, "main_jobs": 24, "initializer_ablation_jobs": 6,
            "selected_jobs": len(selected), "audited_final_jobs": sum(row["audited"] for row in rows),
            "available_final_jobs": sum(row["status"] in FINAL for row in rows),
            "run_complete": all(row["status"] in FINAL for row in rows), "status_counts": counts,
            "scope": "final saved evidence only; pending and unselected jobs retain the thirty-job denominator",
            "rows": rows, "cross_method_calibration_checks": case_checks,
            "protocol_sha256": signature, "truth_used_for_selection": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"outputs/rx75_source_comparison_20261018")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--only-job", nargs="+")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    report = audit_comparison(args.run_dir, args.dataset_dir, only_jobs=args.only_job)
    target = args.output_prefix
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    rows = [{**{k: v for k, v in row.items() if k not in {"metrics_recomputed", "truth_resolution_assumption", "calibration_recomputed", "rnj_initialization_audit", "path_metrics_posthoc"}},
             **(row.get("metrics_recomputed") or {}),
             **{f"truth_scale_{k}": v for k, v in (row.get("truth_resolution_assumption") or {}).items()
                if k in {"min_true_strength", "below_tau_count", "assumption_satisfied"}}} for row in report["rows"]]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with target.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    path_rows = [{"job_id": row["job_id"], "method": row["method"], "variant": row["variant"], "replicate": row["replicate"],
                  "final_job_status": row["status"], **point}
                 for row in report["rows"] for point in row.get("path_metrics_posthoc", [])]
    path_fields = list(dict.fromkeys(k for row in path_rows for k in row)) or ["job_id", "path_index", "train_mae"]
    with target.with_name(target.name+"_paths").with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=path_fields); writer.writeheader(); writer.writerows(path_rows)
    print(json.dumps({key: report[key] for key in ("passed", "check_count", "failed_count", "expected_jobs", "audited_final_jobs", "run_complete")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
