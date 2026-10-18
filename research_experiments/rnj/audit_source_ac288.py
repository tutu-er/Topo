"""Independent saved-evidence audit for the 3 x 96 AC/Source experiment.

Only saved arrays, JSON and physical network tables are read.  This module does
not import the experiment driver, the runtime package, a topology estimator or
a solver.  In particular, Source attachment truth never enters a model check.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_experiments.rnj.audit_source_ac96_benchmark import (
    Audit, clade_metrics, optimality_certificate, physical_truth, relative_error,
)
from research_experiments.rnj.audit_source_ac96_full_history import tree_clades

CASE = "pengwah18"
VARIANTS = ("no_source", "constant", "intermittent", "smooth")
METHODS = ("NJ", "RNJ", "WZZT", "RNJ+WZZT")
ARRAY_NAMES = {"R", "X", "P", "Q", "Y", "P_source", "Q_source", "row_scenario", "row_position"}
ACTIVE_TOLERANCE = 1e-8
SELECTION_PARAMETERS = {
    "bootstrap_replicates": 100, "block_length": 4,
    "confidence_threshold": .75, "maximum_candidate_count": 2,
    "tolerance_factor": .16, "maximum_subclade_frequency": .1,
    "fine_tolerance_ratio": .5,
}
STOP_REASONS = {
    "max_atoms", "no_feasible_extension", "extension_not_proven_optimal",
    "no_significant_one_atom_gain_certified_by_dual_bound",
    "extension_without_incumbent", "no_significant_one_atom_gain",
    "bound_expansion_limit", "no_significant_one_atom_gain_after_pruning",
    "repeated_family_after_zero_pruning", "laminar_family_limit",
}
CORE_PARAMETERS = {
    "max_atoms": 35, "r_upper_bound": 2., "x_upper_bound": 2.,
    "bound_safety_factor": 2., "bound_expansion_factor": 2., "max_bound_expansions": 4,
    "bound_active_fraction": .995, "improvement_abs_tol": 1e-10, "improvement_rel_tol": 1e-8,
    "zero_tolerance": 1e-9, "validation_blocks": 4, "mip_rel_gap": 0., "presolve": True, "disp": False,
}
INPUT_NAMES = {"observed/P_terminal.csv", "observed/Q_terminal.csv", "observed/V_terminal.csv",
               "observed/root_voltage.csv", "observed/drop_target.csv", "observed/P0_measured.csv",
               "observed/Q0_measured.csv", "observed/metadata.json", "source/amplitudes.csv", "source/metadata.json"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_partition(groups, n):
    """Normalize a supplied COMPLETE partition; missing terminals are errors."""
    normalized = tuple(sorted(tuple(sorted(map(int, group))) for group in groups))
    flat = [i for group in normalized for i in group]
    if (any(not group or len(group) != len(set(group)) for group in normalized)
            or sorted(flat) != list(range(n))):
        raise ValueError("terminal groups must partition every original terminal exactly once")
    return normalized


def expected_partition(selected_labels, labels):
    """Build indivisible selected blocks plus every uncovered singleton."""
    positions = {int(label): i for i, label in enumerate(labels)}
    groups = []
    used = set()
    for clade in selected_labels:
        group = tuple(sorted(positions[int(label)] for label in clade))
        if len(group) < 3 or len(set(group)) != len(group) or used.intersection(group):
            raise ValueError("selected atomic boundary blocks must be disjoint and have size >= 3")
        used.update(group)
        groups.append(group)
    groups.extend((i,) for i in range(len(labels)) if i not in used)
    return canonical_partition(groups, len(labels))


def atom_reconstruction(point, labels):
    """Reconstruct WZZT matrices and both structural scoring definitions."""
    n = len(labels)
    supports = tuple(tuple(map(int, support)) for support in point["support_indices"])
    sets = tuple(frozenset(support) for support in supports)
    if (any(not support or len(support) != len(set(support))
            or any(i < 0 or i >= n for i in support) for support in supports)
            or len(set(sets)) != len(sets)):
        raise ValueError("supports must be distinct nonempty sets of original terminal positions")
    if any(not (a <= b or b <= a or a.isdisjoint(b))
           for k, a in enumerate(sets) for b in sets[k + 1:]):
        raise ValueError("saved atom family is not laminar")
    r, x = (np.asarray(point[f"{name}_values"], dtype=float) for name in ("r", "x"))
    if r.shape != (len(supports),) or x.shape != r.shape or not np.isfinite([r, x]).all():
        raise ValueError("one finite R/X weight is required for each atom")
    z = np.zeros((len(supports), n))
    for k, support in enumerate(supports):
        z[k, list(support)] = 1.
    family = {frozenset(labels[i] for i in s) for s in supports if 1 < len(s) < n}
    active = {frozenset(labels[i] for i in s) for s, a, b in zip(supports, r, x, strict=True)
              if 1 < len(s) < n and max(a, b) > ACTIVE_TOLERANCE}
    return {"R": (z.T * r) @ z, "X": (z.T * x) @ z, "z": z,
            "supports": supports, "sets": sets, "r": r, "x": x,
            "family": family, "active": active}


def audit_source_path(point, reconstructed, source_active, audit, prefix):
    """Check the full single-Source h and its original-terminal responses."""
    n = reconstructed["z"].shape[1]
    fields = ("source_response_r", "source_response_x")
    if not source_active:
        audit.require(f"{prefix}: ordinary model has no Source h", point.get("source_h") is None)
        for field in fields:
            value = point.get(field)
            audit.require(f"{prefix}: ordinary {field} absent or zero",
                          value is None or np.array_equal(np.asarray(value), np.zeros(n)))
        return np.zeros(n), np.zeros(n)
    h = np.asarray(point.get("source_h"), dtype=float)
    k = len(reconstructed["supports"])
    valid = (h.shape == (k,) and np.isfinite(h).all()
             and np.all(np.abs(h - np.rint(h)) <= 1e-6)
             and np.all((h >= -1e-6) & (h <= 1. + 1e-6)))
    audit.require(f"{prefix}: Source h covers every original atom and is binary", valid)
    if not valid:
        raise ValueError("cannot reconstruct an invalid full Source path")
    audit.require(f"{prefix}: Source non-root path is nonempty", h.sum() >= 1. - 1e-6)
    path_ok = True
    for i, a in enumerate(reconstructed["sets"]):
        for j, b in enumerate(reconstructed["sets"][i + 1:], start=i + 1):
            if a < b:
                path_ok &= h[i] <= h[j] + 1e-6
            elif b < a:
                path_ok &= h[j] <= h[i] + 1e-6
            else:
                path_ok &= h[i] + h[j] <= 1. + 1e-6
    audit.require(f"{prefix}: Source path is ancestor complete and excludes disjoint branches", path_ok)
    responses = []
    for name, field in zip(("r", "x"), fields, strict=True):
        response = (reconstructed[name] * h) @ reconstructed["z"]
        audit.close(f"{prefix}: {field} independently rebuilt from h and all atom weights",
                    point.get(field), response, atol=1e-9)
        responses.append(response)
    return tuple(responses)


def prediction_and_loss(arrays, r, x, response_r, response_x):
    prediction = (arrays["P"] @ r.T + arrays["Q"] @ x.T
                  + np.asarray(arrays["P_source"]).reshape(-1, 1) * response_r
                  + np.asarray(arrays["Q_source"]).reshape(-1, 1) * response_x)
    return prediction, float(np.abs(arrays["Y"] - prediction).mean())


def audit_point(point, arrays, labels, initial_supports, terminal_groups,
                r_bound, x_bound, audit, prefix):
    """Audit one accepted model without re-solving or using topology truth."""
    value = atom_reconstruction(point, labels)
    initial = {frozenset(s) for s in initial_supports}
    groups = canonical_partition(terminal_groups, len(labels))
    audit.require(f"{prefix}: every fixed initial support retained", initial <= set(value["sets"]))
    # Frozen leaf atoms remain original-dimensional and may split a group.
    # Only later atoms obey the union-of-whole-groups domain restriction.
    future = [s for s in value["sets"] if s not in initial]
    atomic = all(not (s & set(group)) or set(group) <= s for s in future for group in groups)
    audit.require(f"{prefix}: every later support is a union of complete terminal groups", atomic)
    if "support_labels" in point:
        audit.require(f"{prefix}: atom positions and labels agree", point["support_labels"]
                      == [[labels[i] for i in s] for s in value["supports"]])
    for name, bound in (("r", r_bound), ("x", x_bound)):
        weights = value[name]
        audit.require(f"{prefix}: {name} weights satisfy actual coefficient bound",
                      np.isfinite(bound) and bound > 0 and np.all(weights >= -1e-8)
                      and np.all(weights <= bound + 1e-8))
        if f"{name}_matrix" in point:
            audit.close(f"{prefix}: {name.upper()} matrix is atom reconstruction",
                        point[f"{name}_matrix"], value[name.upper()], atol=1e-10)
    active = bool(np.any(arrays["P_source"] != 0.) or np.any(arrays["Q_source"] != 0.))
    gr, gx = audit_source_path(point, value, active, audit, prefix)
    _, loss = prediction_and_loss(arrays, value["R"], value["X"], gr, gx)
    audit.close(f"{prefix}: complete 288-row Source training objective", point["train_mae"], loss,
                atol=1e-9, rtol=1e-7)
    audit.require(f"{prefix}: validation MAE is absent", point.get("validation_mae") is None)
    audit.require(f"{prefix}: validation SE is absent", point.get("validation_se") is None)
    diagnostic = point.get("solver", point.get("diagnostics"))
    audit.require(f"{prefix}: accepted point has strict LP/MIP certificate",
                  isinstance(diagnostic, dict) and optimality_certificate(diagnostic))
    if isinstance(diagnostic, dict):
        # After zero pruning a certified fixed-family refit can differ from the
        # extension objective by at most the configured acceptance tolerance.
        audit.close(f"{prefix}: accepted point objective agrees with certified solve",
                    diagnostic.get("objective"), loss, atol=2e-9, rtol=1e-7)
    return value | {"source_response_r": gr, "source_response_x": gx, "loss": loss}


def audit_path(path, arrays, labels, initial_supports, terminal_groups,
               r_bound, x_bound, audit, prefix):
    if not path:
        raise ValueError("a saved model requires a nonempty accepted path")
    initial = {frozenset(s) for s in initial_supports}
    previous = None
    reconstructed = []
    for i, point in enumerate(path):
        tag = f"{prefix}: path[{i}]"
        value = audit_point(point, arrays, labels, initial_supports, terminal_groups,
                            r_bound, x_bound, audit, tag)
        if i == 0:
            audit.require(f"{tag}: first point is exactly fixed initial family", set(value["sets"]) == initial)
        if previous is not None:
            gain = previous - value["loss"]
            tolerance = 1e-10 + 1e-8 * max(abs(previous), np.finfo(float).eps)
            audit.require(f"{tag}: accepted gain exceeds historical stopping tolerance", gain > tolerance - 1e-10)
            audit.close(f"{tag}: accepted gain independently reproduced", point["accepted_gain"], gain,
                        atol=2e-9, rtol=1e-7)
        previous = value["loss"]
        reconstructed.append(value)
    return {"selected_path_index": len(path) - 1, "selected": reconstructed[-1],
            "path_reconstructed": reconstructed}


def audit_baseline(model, arrays, labels, audit, prefix):
    """FWL nuisance response check after the saved ordered R/X projection."""
    audit.require(f"{prefix}: baseline matrix origin is the Source-adjusted ordered regression",
                  model.get("matrix_origin") == "source_adjusted_ordered_regression_before_tree_inference")
    r, x = arrays["R"], arrays["X"]
    residual = arrays["Y"] - arrays["P"] @ r.T - arrays["Q"] @ x.T
    source = np.column_stack((arrays["P_source"], arrays["Q_source"]))
    expected = np.linalg.lstsq(source, residual, rcond=None)[0]
    gr, gx = (np.asarray(model[name], dtype=float) for name in ("source_g_r", "source_g_x"))
    audit.close(f"{prefix}: FWL Source nuisance independently refit on all 288 rows",
                np.vstack((gr, gx)), expected, atol=1e-8, rtol=1e-6)
    audit.require(f"{prefix}: baseline Source h is not invented", model.get("source_h") is None)
    prediction, loss = prediction_and_loss(arrays, r, x, gr, gx)
    audit.close(f"{prefix}: baseline full training loss", model.get("train_mae", model.get("train_mae_recomputed")),
                loss, atol=1e-9)
    family = tree_clades(model["tree"], labels)
    return {"R": r, "X": x, "family": family, "active": family,
            "loss": loss, "prediction": prediction}


def recompute_metrics(value, truth, labels):
    metrics = {}
    for kind in ("family", "active"):
        score = clade_metrics(value[kind], truth["clades"], labels)
        metrics.update({f"{kind}_{name}": score[name] for name in ("clade_f1", "precision", "recall")})
    metrics.update(clade_f1=metrics["family_clade_f1"],
                   r_relative_error=relative_error(value["R"], truth["R"]),
                   x_relative_error=relative_error(value["X"], truth["X"]))
    return metrics


def audit_training_arrays(arrays, variant_directory, labels, audit, prefix):
    """Read the three allowed input blocks directly, without using the loader."""
    expected = {name: [] for name in ("P", "Q", "Y", "P_source", "Q_source")}
    scenario_descriptions = []
    for index in range(3):
        original = Path(variant_directory) / f"scenario_{index}"
        time_labels = None
        for name, filename in (("P", "P_terminal"), ("Q", "Q_terminal"), ("Y", "drop_target")):
            frame = pd.read_csv(original / "observed" / f"{filename}.csv", index_col="t",
                                float_precision="round_trip")
            audit.require(f"{prefix}: scenario {index} {name} retains 96 original-terminal rows",
                          frame.shape == (96, len(labels)) and frame.index.is_unique
                          and list(map(int, frame.columns)) == labels)
            if time_labels is None:
                time_labels = frame.index
            audit.require(f"{prefix}: scenario {index} {name} original row labels aligned",
                          frame.index.equals(time_labels))
            expected[name].append(frame.to_numpy())
        source = pd.read_csv(original / "source/amplitudes.csv", index_col="t", float_precision="round_trip")
        audit.require(f"{prefix}: scenario {index} Source row labels match observations",
                      source.shape[0] == 96 and source.index.equals(time_labels))
        expected["P_source"].append(source["P_pu"].to_numpy())
        expected["Q_source"].append(source["Q_pu"].to_numpy())
        scenario_descriptions.append({"scenario_index": index, "time_labels": time_labels.tolist()})
    for name, blocks in expected.items():
        audit.close(f"{prefix}: {name} uses all 3x96 rows in original scenario/source order",
                    arrays[name], np.concatenate(blocks), atol=0, rtol=0)
    audit.close(f"{prefix}: scenario identity retained for every training row",
                arrays["row_scenario"], np.repeat(np.arange(3), 96), atol=0, rtol=0)
    audit.close(f"{prefix}: local position retained for every training row",
                arrays["row_position"], np.tile(np.arange(96), 3), atol=0, rtol=0)
    for name in ("P", "Q", "Y"):
        audit.require(f"{prefix}: {name} is finite 288 by original-terminal array",
                      arrays[name].shape == (288, len(labels)) and np.isfinite(arrays[name]).all())
    for name in ("P_source", "Q_source"):
        audit.require(f"{prefix}: {name} is a finite 288-row fixed amplitude",
                      arrays[name].shape == (288,) and np.isfinite(arrays[name]).all())
    return scenario_descriptions


def _frequency_records(values):
    if not isinstance(values, list):
        raise ValueError("frequency evidence must use explicit clade/frequency records")
    pairs = [(frozenset(map(int, row["clade"])), float(row["frequency"])) for row in values]
    if len({clade for clade, _ in pairs}) != len(pairs):
        raise ValueError("frequency evidence repeats a clade")
    return dict(pairs)


def audit_screen(screen, model, labels, audit, prefix):
    """Independently apply the final peer-block and internal-subclade veto."""
    diagnostics = screen["diagnostics"]
    audit.require(f"{prefix}: screening uses all 288 training rows", screen.get("training_sample_count") == 288)
    audit.require(f"{prefix}: screening uses no validation rows", screen.get("validation_used") is False)
    parameters = screen["parameters"]
    for key, expected in SELECTION_PARAMETERS.items():
        audit.require(f"{prefix}: screening parameter {key} matches final atomic policy", parameters.get(key) == expected)
    audit.require(f"{prefix}: screening preserves historical bootstrap seed", parameters.get("seed") == 20260903)
    peers = {frozenset(map(int, c)) for c in diagnostics["full_peers"]}
    full = {frozenset(map(int, c)) for c in screen["full_clades"]}
    audit.require(f"{prefix}: peer blocks belong to full RNJ tree", peers <= full)
    confidence = _frequency_records(screen["confidence_records"])
    audit.require(f"{prefix}: peer frequencies count all 100 bootstrap replicates", all(
        np.isfinite(v) and 0 <= v <= 1 and abs(100 * v - round(100 * v)) <= 1e-8
        for v in confidence.values()))
    sub_frequency = _frequency_records(diagnostics["subclade_frequency_records"])
    valid = diagnostics["valid_bootstrap_replicates"]
    invalid = diagnostics["unidentifiable_bootstrap_replicates"]
    audit.require(f"{prefix}: identifiable and nonidentifiable replicate denominator retained",
                  isinstance(valid, int) and isinstance(invalid, int) and 0 <= valid <= 100
                  and 0 <= invalid <= 100 and valid + invalid == 100)
    audit.require(f"{prefix}: refinement frequencies use only identifiable replicates", all(
        np.isfinite(v) and 0 <= v <= 1 and
        (abs(valid * v - round(valid * v)) <= 1e-8 if valid else v == 1.)
        for v in sub_frequency.values()))
    audit.require(f"{prefix}: every peer has a saved refinement frequency", set(sub_frequency) == peers)
    vetoed = {frozenset(map(int, c)) for c in diagnostics["full_fine_subclades"]}
    audit.require(f"{prefix}: full finer-tree veto only names peer blocks", vetoed <= peers)
    identifiable = diagnostics["source_adjusted_identifiable"]
    audit.require(f"{prefix}: source identifiability is explicit", isinstance(identifiable, bool))
    eligible = {c for c in peers if len(c) >= 3 and confidence.get(c, 0.) >= .75
                and sub_frequency.get(c, 1.) <= .1 and c not in vetoed and identifiable and valid > 0}
    audit.require(f"{prefix}: eligible blocks independently follow peer/frequency/refinement veto",
                  {frozenset(c) for c in diagnostics["eligible"]} == eligible)
    chosen, occupied = [], set()
    for clade in sorted(eligible, key=lambda c: (-confidence[c], -len(c), tuple(sorted(c)))):
        if len(chosen) >= 2:
            break
        if occupied.isdisjoint(clade):
            chosen.append(clade)
            occupied.update(clade)
    expected_labels = [sorted(c) for c in chosen]
    audit.require(f"{prefix}: selected blocks follow final stable-disjoint ordering",
                  screen["selected_support_labels"] == expected_labels)
    position = {label: i for i, label in enumerate(labels)}
    expected_indices = [[position[label] for label in c] for c in expected_labels]
    audit.require(f"{prefix}: selected labels map to exact original terminal positions",
                  screen["selected_support_indices"] == expected_indices)
    requested = model["requested_terminal_groups"]
    audit.require(f"{prefix}: requested indivisible groups are exactly selected size>=3 blocks",
                  requested == expected_indices)
    partition = expected_partition(expected_labels, labels)
    audit.require(f"{prefix}: actual partition keeps selected blocks and all remaining leaves",
                  canonical_partition(model["terminal_groups"], len(labels)) == partition)
    initial = {frozenset((i,)) for i in range(len(labels))} | {frozenset(s) for s in expected_indices}
    audit.require(f"{prefix}: fixed atom family consists of original singletons and selected blocks",
                  {frozenset(s) for s in model["initial_supports"]} == initial)
    return expected_labels


def audit_raw_primal(event, saved, audit, prefix):
    """Recompute the original objective/linear/bound/integer feasibility."""
    primal = np.asarray(saved["primal"], dtype=float)
    count = event["variable_count"]
    audit.require(f"{prefix}: raw primal is finite with original variable count",
                  primal.shape == (count,) and np.isfinite(primal).all())
    if primal.shape != (count,) or not np.isfinite(primal).all():
        return
    for name in ("objective_coefficients", "lower_bounds", "upper_bounds", "integrality"):
        audit.require(f"{prefix}: original model {name} retained", name in saved and np.shape(saved[name]) == (count,))
    if not all(name in saved for name in ("objective_coefficients", "lower_bounds", "upper_bounds", "integrality")):
        return
    objective = float(np.asarray(saved["objective_coefficients"]) @ primal)
    audit.close(f"{prefix}: raw primal objective independently recomputed", event["objective"], objective,
                atol=1e-9, rtol=1e-7)
    if "primal_objective_recomputed" in event:
        audit.close(f"{prefix}: observer raw objective matches independent arithmetic", event["primal_objective_recomputed"],
                    objective, atol=1e-12)
    lower, upper = np.asarray(saved["lower_bounds"]), np.asarray(saved["upper_bounds"])
    bound_violation = float(max(np.max(lower - primal, initial=0.), np.max(primal - upper, initial=0.), 0.))
    integer = np.flatnonzero(np.asarray(saved["integrality"]) != 0)
    integer_violation = float(np.max(np.abs(primal[integer] - np.rint(primal[integer])), initial=0.))
    audit.require(f"{prefix}: raw primal satisfies variable bounds", bound_violation <= 1e-6,
                  max_violation=bound_violation)
    audit.require(f"{prefix}: raw primal satisfies integrality", integer_violation <= 1e-6,
                  max_violation=integer_violation)
    for field, value in (("max_bound_violation", bound_violation), ("max_integrality_violation", integer_violation)):
        if field in event:
            audit.close(f"{prefix}: observer {field} matches independent arithmetic", event[field], value, atol=1e-12)
    if "integer_variable_count" in event:
        audit.require(f"{prefix}: integer variable count agrees with original model", event["integer_variable_count"] == len(integer))
    constraint_names = ("constraint_row_index", "constraint_col_index", "constraint_data", "constraint_lower", "constraint_upper")
    have_constraints = all(name in saved for name in constraint_names)
    audit.require(f"{prefix}: original sparse linear constraints retained", have_constraints)
    if have_constraints:
        row = np.asarray(saved["constraint_row_index"], dtype=int)
        col = np.asarray(saved["constraint_col_index"], dtype=int)
        coefficients = np.asarray(saved["constraint_data"], dtype=float)
        lower = np.asarray(saved["constraint_lower"], dtype=float)
        upper = np.asarray(saved["constraint_upper"], dtype=float)
        activity = np.bincount(row, weights=coefficients * primal[col], minlength=len(lower))
        violation = float(max(np.max(lower - activity, initial=0.), np.max(activity - upper, initial=0.), 0.))
        audit.require(f"{prefix}: raw primal satisfies original linear constraints", violation <= 1e-6,
                      max_violation=violation)
        if "max_constraint_violation" in event:
            audit.close(f"{prefix}: observer linear-constraint violation independently reproduced",
                        event["max_constraint_violation"], violation, atol=1e-12)
    dual = event.get("dual_bound")
    if dual is not None and np.isfinite(dual):
        audit.require(f"{prefix}: minimization dual bound does not exceed primal objective",
                      float(dual) <= objective + 1e-8)


def audit_stopping(model, audit, prefix):
    """Check local stopping semantics; never label this global topology optimality."""
    reason = model["stop_reason"]
    path = model.get("path", [])
    attempts = model.get("attempted_extensions", [])
    if reason not in STOP_REASONS:
        audit.require(f"{prefix}: external interruption is explicitly incomplete/failed",
                      model.get("status") in {"incomplete", "failed", "running"})
        return
    if reason == "max_atoms" and path:
        audit.require(f"{prefix}: max_atoms denotes iteration cap even after zero pruning",
                      path[-1]["iteration"] == model["core_parameters"]["max_atoms"])
    if reason == "laminar_family_limit" and path:
        audit.require(f"{prefix}: laminar family count reached its structural cap",
                      len(path[-1]["support_indices"]) >= 2 * len(model["terminal_labels"]) - 1)
    if reason in {"extension_not_proven_optimal", "extension_without_incumbent",
                  "no_significant_one_atom_gain_certified_by_dual_bound", "no_feasible_extension"}:
        audit.require(f"{prefix}: extension stop preserves its attempted solve", bool(attempts))
        if not attempts:
            return
        event = attempts[-1]["diagnostics"]
        if reason == "extension_not_proven_optimal":
            audit.require(f"{prefix}: unproved extension was not accepted as optimal", not optimality_certificate(event))
        elif reason == "no_feasible_extension":
            audit.require(f"{prefix}: infeasibility stop retains solver infeasible status", event.get("status") == 2)
        elif reason == "no_significant_one_atom_gain_certified_by_dual_bound" and path:
            current = float(path[-1]["train_mae"])
            tolerance = 1e-10 + 1e-8 * max(abs(current), np.finfo(float).eps)
            lower = event.get("dual_bound")
            audit.require(f"{prefix}: current one-atom no-gain statement is backed by finite lower bound",
                          lower is not None and np.isfinite(lower) and current - lower <= tolerance + 1e-10)


def audit_protocol(protocol, audit):
    for name, expected in {"schema_version": 3, "case": CASE, "variants": list(VARIANTS),
                           "methods": list(METHODS), "expected_jobs": 16, "total_samples": 288,
                           "training_samples": 288, "validation_samples": 0, "validation_supplied": False,
                           "independent_test_samples": 0, "scenario_count": 3, "samples_per_scenario": 96,
                           "selection": "last_certified_path_point_without_validation", "observed_root_label": 1,
                           "rx75_weight": .75, "tree_tolerance_factor": .16,
                           "minimum_atomic_group_size": 3, "rnj_atomic_block_rule": "complete_direct_terminal_children",
                           "solver_threads": 1, "blas_threads": 1, "truth_used_for_selection": False}.items():
        audit.require(f"protocol {name} is predeclared", protocol.get(name) == expected)
    audit.require("all historical core controls are explicit and unchanged", protocol.get("core_parameters") == CORE_PARAMETERS)
    audit.require("all atomic bootstrap controls are explicit and unchanged",
                  protocol.get("screen_parameters") == {**SELECTION_PARAMETERS, "seed": 20260903})
    expected = {f"{CASE}__{v}__{m}" for v in VARIANTS for m in METHODS}
    jobs = {job["job_id"]: job for job in protocol.get("jobs", [])}
    audit.require("protocol retains all sixteen method/Source conditions", set(jobs) == expected and len(protocol.get("jobs", [])) == 16)
    for key, job in jobs.items():
        is_milp = job["method"] in {"WZZT", "RNJ+WZZT"}
        audit.require(f"{key}: declared single-solve budget", job.get("per_solve_seconds") == (6000. if is_milp else None))
        audit.require(f"{key}: declared whole-method budget", job.get("method_wall_seconds") == (30000. if is_milp else 300.))
    return jobs


class _StripObserver(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == "observe":
            return None
        if node.name == "fit_laminar_l1_sensitivity":
            pairs = [(arg, default) for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
                     if arg.arg != "progress_callback"]
            node.args.kwonlyargs = [arg for arg, _ in pairs]
            node.args.kw_defaults = [default for _, default in pairs]
        return self.generic_visit(node)

    def visit_If(self, node):
        if any(isinstance(item, ast.Name) and item.id == "progress_callback" for item in ast.walk(node.test)):
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "observe":
            return None
        return self.generic_visit(node)


def audit_fingerprints(directory, dataset, protocol, audit):
    manifest_path = dataset / "manifest.json"
    audit.require("dataset manifest is the frozen generation manifest",
                  manifest_path.is_file() and file_hash(manifest_path) == protocol.get("dataset_manifest_sha256"))
    if manifest_path.is_file():
        for relative, expected in read_json(manifest_path).get("file_sha256", {}).items():
            path = (dataset / relative).resolve()
            audit.require(f"data fingerprint {relative}", path.is_relative_to(dataset.resolve())
                          and path.is_file() and file_hash(path) == expected)
    for relative, expected in protocol.get("source_sha256", {}).items():
        frozen = directory / "source_snapshot" / relative
        audit.require(f"frozen caller fingerprint {relative}", frozen.is_file() and file_hash(frozen) == expected)
    runtime = Path(protocol["runtime_directory"])
    manifest = runtime / "runtime_manifest.json"
    audit.require("runtime manifest fingerprint matches declared package",
                  manifest.is_file() and file_hash(manifest) == protocol["runtime_manifest_sha256"])
    if not manifest.is_file():
        return
    metadata = read_json(manifest)
    for field, root in (("runtime_sha256", runtime / "rnj_wzzt"),
                        ("source_sha256", runtime / "source_snapshot/rnj_wzzt")):
        hashes = metadata[field]
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}}
        audit.require(f"{field} exact package inventory", actual == set(hashes))
        for relative, expected in hashes.items():
            path = root / relative
            audit.require(f"{field} {relative}", path.is_file() and file_hash(path) == expected)
    changed = [name for name in metadata["runtime_sha256"]
               if metadata["runtime_sha256"][name] != metadata["source_sha256"][name]]
    audit.require("runtime changed only the declared passive observer module", changed == ["estimation/laminar_l1_milp.py"])
    original = ast.parse((runtime / "source_snapshot/rnj_wzzt/estimation/laminar_l1_milp.py").read_text(encoding="utf-8"))
    patched = ast.parse((runtime / "rnj_wzzt/estimation/laminar_l1_milp.py").read_text(encoding="utf-8"))
    restored = _StripObserver().visit(deepcopy(patched))
    audit.require("stripping passive observer restores whole original module AST",
                  ast.dump(original, include_attributes=False) == ast.dump(restored, include_attributes=False))


def audit_input_hashes(directory, dataset, variant, job_dir, audit, prefix):
    expected = {f"scenario_{i}/{name}" for i in range(3) for name in INPUT_NAMES}
    inventory = read_json(job_dir / "inputs_hash.json")["file_sha256"]
    audit.require(f"{prefix}: estimator receives only the thirty allowed observed/Source files", set(inventory) == expected)
    copied = directory / "inputs" / variant
    actual = {p.relative_to(copied).as_posix() for p in copied.rglob("*") if p.is_file()}
    audit.require(f"{prefix}: worker input directory has no network or Source-position files", actual == expected)
    for relative, checksum in inventory.items():
        source, duplicate = dataset / CASE / variant / relative, copied / relative
        audit.require(f"{prefix}: original and copied input hash {relative}", source.is_file() and duplicate.is_file()
                      and file_hash(source) == checksum and file_hash(duplicate) == checksum)


def audit_solver_events(job_dir, model, job, audit, prefix):
    events = model.get("solver_events", [])
    audit.require(f"{prefix}: initializer and every attempted solve retained", len(events)
                  >= 1 + len(model.get("attempted_extensions", [])) + model.get("bound_expansions", 0))
    for i, event in enumerate(events, start=1):
        tag = f"{prefix}: solver[{i}]"
        audit.require(f"{tag}: sequence retained", event.get("sequence") == i)
        path = job_dir / "solver" / f"{i:06d}.json"
        audit.require(f"{tag}: individual raw record matches combined evidence", path.is_file() and read_json(path) == event)
        audit.require(f"{tag}: requested time equals declared solve budget", event.get("requested_time_limit_seconds") == job["per_solve_seconds"])
        effective, remaining = event.get("time_limit_seconds"), event.get("remaining_wall_before_solve")
        audit.require(f"{tag}: effective time respects solve and remaining-method budgets",
                      effective is not None and remaining is not None and 0 < effective <= job["per_solve_seconds"]
                      and effective <= remaining)
        audit.require(f"{tag}: reported certificate follows status AND strict gap/bound rule",
                      event.get("optimality_certified") is optimality_certificate(event))
        if event.get("has_incumbent"):
            target = (job_dir / event["primal_file"]).resolve()
            safe = target.is_relative_to(job_dir.resolve()) and target.is_file()
            audit.require(f"{tag}: raw incumbent and original model retained locally", safe)
            if safe:
                with np.load(target, allow_pickle=False) as saved:
                    audit_raw_primal(event, saved, audit, tag)
        if optimality_certificate(event):
            audit.require(f"{tag}: certified solve has retained incumbent", event.get("has_incumbent") is True)


def audit_progress(job_dir, model, arrays, labels, audit, prefix):
    snapshots = [read_json(path) for path in sorted((job_dir / "progress").glob("*.json"))]
    audit.require(f"{prefix}: full passive checkpoints retained", bool(snapshots))
    pending = None
    restarted = 0
    for i, snapshot in enumerate(snapshots):
        tag = f"{prefix}: checkpoint[{i}]"
        chosen = audit_path(snapshot["path"], arrays, labels, snapshot["initial_supports"], snapshot["terminal_groups"],
                            snapshot["r_upper_bound"], snapshot["x_upper_bound"], audit, tag)
        audit.require(f"{tag}: no-validation checkpoint selects current last point",
                      snapshot["selected_path_index"] == chosen["selected_path_index"])
        audit.require(f"{tag}: original-terminal partition retained",
                      canonical_partition(snapshot["terminal_groups"], len(labels))
                      == canonical_partition(model["terminal_groups"], len(labels)))
        if i == 0:
            audit.require(f"{tag}: original bound-two initialization retained", snapshot["event"] == "initialized"
                          and snapshot["bound_expansions"] == 0 and snapshot["r_upper_bound"] == 2.
                          and snapshot["x_upper_bound"] == 2. and len(snapshot["path"]) == 1)
        if snapshot["event"] == "bounds_expansion_requested":
            pending = snapshot
        elif snapshot["event"] == "bounds_restarted":
            restarted += 1
            audit.require(f"{tag}: restart has retained pre-expansion path", pending is not None)
            if pending is not None:
                extension = pending["attempted_extensions"][-1]
                audit.require(f"{tag}: bound-triggering extension strictly certified", optimality_certificate(extension["diagnostics"]))
                for name in ("r", "x"):
                    old = float(pending[f"{name}_upper_bound"])
                    touch = np.max(extension[f"{name}_values"], initial=0.) >= .995 * old
                    audit.close(f"{tag}: only touched {name} coefficient bound doubles", snapshot[f"{name}_upper_bound"],
                                old * (2. if touch else 1.), atol=1e-12)
                audit.require(f"{tag}: expansion count increments by one",
                              snapshot["bound_expansions"] == pending["bound_expansions"] + 1)
            audit.require(f"{tag}: restart resets to initial certified family", len(snapshot["path"]) == 1)
            pending = None
    audit.require(f"{prefix}: all completed bound restarts preserved", restarted == model.get("bound_expansions", 0))
    if snapshots:
        last = snapshots[-1]
        audit.require(f"{prefix}: final actual model retains last checkpoint path", model["path"] == last["path"])
        for name in ("r_upper_bound", "x_upper_bound", "bound_expansions"):
            audit.close(f"{prefix}: final {name} equals checkpoint", model[name], last[name], atol=1e-12)
    return {"checkpoints": len(snapshots), "completed_bound_restarts": restarted,
            "pending_expansion_at_stop": pending is not None}


def audit_benchmark(directory, dataset_directory=None, *, only_jobs=None):
    directory = Path(directory).resolve()
    dataset = Path(dataset_directory or ROOT / "outputs/source_ac288_20261018").resolve()
    audit, rows, common = Audit(), [], {}
    protocol = read_json(directory / "protocol.json")
    jobs = audit_protocol(protocol, audit)
    audit_fingerprints(directory, dataset, protocol, audit)
    signature = file_hash(directory / "protocol.json")
    if only_jobs is not None and not set(only_jobs) <= set(jobs):
        raise ValueError("only_jobs contains a condition outside the frozen sixteen jobs")
    expected = [key for key in jobs if only_jobs is None or key in only_jobs]
    for key in expected:
        job = jobs[key]
        folder = directory / "jobs" / key
        result_path = folder / "result.json"
        row = {"job_id": key, "case": CASE, "variant": job["variant"], "method": job["method"],
               "status": "missing", "has_model": False, "metrics_recomputed": None}
        rows.append(row)
        audit.require(f"{key}: outcome or explicit failure retained", result_path.is_file())
        if not result_path.is_file():
            continue
        try:
            result = read_json(result_path)
            row.update(status=result.get("status"), has_model=result.get("has_model"),
                       elapsed_seconds=result.get("elapsed_seconds"), stop_reason=result.get("stop_reason"))
            audit.require(f"{key}: status is complete/incomplete/failed", result.get("status") in {"complete", "incomplete", "failed"})
            audit.require(f"{key}: search completeness agrees with recorded status",
                          result.get("search_complete") is (result.get("status") == "complete"))
            audit.require(f"{key}: outcome is bound to exact frozen protocol", result.get("protocol_sha256") == signature)
            audit.require(f"{key}: model availability is explicit", isinstance(result.get("has_model"), bool))
            elapsed = result.get("elapsed_seconds")
            audit.require(f"{key}: finite nonnegative time retained even after failure",
                          elapsed is not None and np.isfinite(elapsed) and elapsed >= 0)
            if "compute_seconds" in result:
                audit.close(f"{key}: reported compute time equals complete method elapsed time", result["compute_seconds"], elapsed,
                            atol=1e-9, rtol=0)
            audit_input_hashes(directory, dataset, job["variant"], folder, audit, key)
            has_files = (folder / "model.json").is_file() and (folder / "model.npz").is_file()
            audit.require(f"{key}: available model has complete saved artifacts", not result.get("has_model") or has_files)
            if not result.get("has_model"):
                audit.require(f"{key}: unavailable model keeps metrics empty/null", all(v is None for v in result.get("metrics", {}).values()))
                audit.require(f"{key}: unavailable model cannot be complete", result.get("status") != "complete")
            if not has_files:
                continue
            model = read_json(folder / "model.json")
            audit.require(f"{key}: result and model availability agree", model.get("has_model") is result.get("has_model"))
            with np.load(folder / "model.npz", allow_pickle=False) as saved:
                audit.require(f"{key}: exact original-array inventory retained", set(saved.files) == ARRAY_NAMES)
                arrays = {name: saved[name].copy() for name in saved.files}
            if set(arrays) != ARRAY_NAMES:
                continue
            labels = list(map(int, model["terminal_labels"]))
            audit_training_arrays(arrays, dataset / CASE / job["variant"], labels, audit, key)
            common[key] = {name: value for name, value in arrays.items() if name not in {"R", "X"}}
            use = read_json(folder / "data_use.json")
            for name, value in {"total_samples": 288, "training_samples": 288, "validation_samples": 0,
                                "validation_supplied": False, "terminal_labels": labels,
                                "selection": "last_certified_path_point_without_validation"}.items():
                audit.require(f"{key}: declared data-use {name} agrees with all-row fit", use.get(name) == value)
            audit.require(f"{key}: three original scenario row ranges retained",
                          [block["row_range"] for block in use["scenarios"]] == [[0, 96], [96, 192], [192, 288]])
            imported = read_json(folder / "runtime_import.json")
            frozen_core = Path(protocol["runtime_directory"]) / "rnj_wzzt/estimation/laminar_l1_milp.py"
            audit.require(f"{key}: worker imported the exact frozen package", Path(imported["core_file"]).resolve() == frozen_core.resolve())
            if not result.get("has_model"):
                audit.require(f"{key}: unavailable R/X matrices are empty", arrays["R"].shape == (0, 0) and arrays["X"].shape == (0, 0))
                continue
            for name, expected_value in {"schema_version": 3, "training_rows": 288, "validation_supplied": False,
                                         "validation_mae": None, "validation_se": None}.items():
                audit.require(f"{key}: model {name} matches no-validation protocol", model.get(name) == expected_value)
            for name in ("R", "X"):
                matrix = arrays[name]
                audit.require(f"{key}: {name} finite symmetric original-terminal matrix", matrix.shape == (len(labels), len(labels))
                              and np.isfinite(matrix).all() and np.allclose(matrix, matrix.T, atol=1e-8, rtol=0))
                audit.require(f"{key}: {name} obeys nonnegative ordered sensitivity bounds",
                              np.min(matrix, initial=0.) >= -1e-8
                              and np.all(matrix <= np.minimum(np.diag(matrix)[:, None], np.diag(matrix)[None, :]) + 1e-8))
            if job["method"] in {"NJ", "RNJ"}:
                value = audit_baseline(model, arrays, labels, audit, key)
                diagnostic = model["regression_diagnostics"]
                design = np.column_stack((arrays["P"], arrays["Q"]))
                source = np.column_stack((arrays["P_source"], arrays["Q_source"]))
                u, singular, _ = np.linalg.svd(source, full_matrices=False)
                rank = int(np.count_nonzero(singular > max(source.shape) * np.finfo(float).eps * singular[0]))
                basis = u[:, :rank]
                residualized = design - basis @ (basis.T @ design)
                for name, expected_value in {"source_rank": rank, "source_location_used": False,
                                             "source_path_constraints": False, "intercept_added": False,
                                             "raw_design_rank": int(np.linalg.matrix_rank(design)),
                                             "projected_design_rank": int(np.linalg.matrix_rank(residualized)),
                                             "collapse_or_tolerance_factor": .16}.items():
                    audit.require(f"{key}: front-end {name} independently checked", diagnostic.get(name) == expected_value)
            else:
                audit.require(f"{key}: all core controls equal frozen protocol", model.get("core_parameters") == CORE_PARAMETERS)
                audit.require(f"{key}: no-validation choice explicitly preserved",
                              model.get("selection") == "last_certified_path_point_without_validation")
                if job["method"] == "RNJ+WZZT":
                    screen = read_json(folder / "rnj_screen.json")
                    audit.require(f"{key}: separate screening evidence matches saved model", model.get("bootstrap") == screen)
                    row["selected_blocks_recomputed"] = audit_screen(screen, model, labels, audit, key)
                else:
                    audit.require(f"{key}: pure search uses original singletons only",
                                  {frozenset(s) for s in model["initial_supports"]} == {frozenset((i,)) for i in range(len(labels))})
                    audit.require(f"{key}: pure search has no requested atomic restriction", model["requested_terminal_groups"] == [])
                    audit.require(f"{key}: pure actual partition is all original singletons",
                                  canonical_partition(model["terminal_groups"], len(labels)) == tuple((i,) for i in range(len(labels))))
                audit_solver_events(folder, model, job, audit, key)
                path = audit_path(model["path"], arrays, labels, model["initial_supports"], model["terminal_groups"],
                                  model["r_upper_bound"], model["x_upper_bound"], audit, key)
                chosen = path["selected_path_index"]
                value = path["selected"]
                audit.require(f"{key}: selected and last path indexes equal no-validation last point",
                              model["selected_path_index"] == model["last_path_index"] == chosen)
                selected = model["path"][chosen]
                for field in ("support_indices", "support_labels", "r_values", "x_values", "source_h"):
                    audit.require(f"{key}: actual {field} equals last certified point", model[field] == selected[field])
                for suffix in ("r", "x"):
                    actual, expected_value = model.get(f"source_g_{suffix}"), selected.get(f"source_response_{suffix}")
                    audit.require(f"{key}: actual Source response {suffix} equals last certified point", actual == expected_value)
                row["progress_audit"] = audit_progress(folder, model, arrays, labels, audit, key)
                audit_stopping(model, audit, key)
                audit.require(f"{key}: successful bound expansions respect historical cap", 0 <= model["bound_expansions"] <= 4)
            audit.close(f"{key}: saved R reproduces actual selected model", arrays["R"], value["R"], atol=1e-10)
            audit.close(f"{key}: saved X reproduces actual selected model", arrays["X"], value["X"], atol=1e-10)
            audit.close(f"{key}: final saved training objective independently recomputed",
                        model.get("train_mae", model.get("train_mae_recomputed")), value["loss"], atol=1e-9)
            for field, expected_value in (("clades", value["family"]), ("family_clades", value["family"]), ("active_clades", value["active"])):
                audit.require(f"{key}: independently reproduced {field}", {frozenset(c) for c in model[field]} == expected_value)
            # Truth enters only here, after all estimator-input/model checks.
            truth = physical_truth(dataset / CASE / "network", labels)
            metrics = recompute_metrics(value, truth, labels)
            for name, expected_value in metrics.items():
                audit.close(f"{key}: independently recomputed metric {name}", result["metrics"].get(name), expected_value, atol=1e-10)
            row["metrics_recomputed"] = metrics
        except Exception as exc:
            audit.require(f"{key}: complete saved evidence is independently auditable", False,
                          exception=f"{type(exc).__name__}: {exc}")
    for variant in VARIANTS:
        values = {key: arrays for key, arrays in common.items() if jobs[key]["variant"] == variant}
        if values:
            reference = next(iter(values.values()))
            for key, arrays in values.items():
                for name in reference:
                    audit.close(f"{key}: all methods receive identical {variant} {name}", arrays[name], reference[name], atol=0, rtol=0)
        nj_path, rnj_path = (directory / "jobs" / f"{CASE}__{variant}__{method}" / "model.npz" for method in ("NJ", "RNJ"))
        if nj_path.is_file() and rnj_path.is_file():
            with np.load(nj_path, allow_pickle=False) as nj, np.load(rnj_path, allow_pickle=False) as rnj:
                if nj["R"].size and rnj["R"].size:
                    for name in ("R", "X"):
                        audit.close(f"{variant}: NJ/RNJ share the same all-row front-end {name}", nj[name], rnj[name], atol=1e-12)
    counts = {method: {status: sum(row["method"] == method and row["status"] == status for row in rows)
                       for status in ("complete", "incomplete", "failed", "missing")} for method in METHODS}
    return {"audit": "independent AC288 no-validation/atomic-Source/physical-path audit",
            "directory": str(directory), "dataset_directory": str(dataset), "jobs_expected": len(expected),
            "jobs_present": sum(row["status"] != "missing" for row in rows), "status_counts_by_method": counts,
            "metrics_recomputed": rows, **audit.result()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dataset-directory", type=Path, default=ROOT / "outputs/source_ac288_20261018")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only-job", nargs="+")
    args = parser.parse_args()
    result = audit_benchmark(args.directory, args.dataset_directory, only_jobs=args.only_job)
    output = args.output or args.directory / "independent_ac288_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = [{**{key: value for key, value in row.items() if key not in {"metrics_recomputed", "progress_audit", "selected_blocks_recomputed"}},
             **(row["metrics_recomputed"] or {})} for row in result["metrics_recomputed"]]
    pd.DataFrame(rows).to_csv(output.with_name("metrics_recomputed.csv"), index=False, float_format="%.17g")
    print(json.dumps({name: result[name] for name in ("passed", "check_count", "failed_count", "jobs_expected", "jobs_present")}))
    for check in result["checks"]:
        if not check["passed"]:
            print(json.dumps(check, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
