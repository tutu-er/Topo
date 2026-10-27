"""Observed-data adapter to the production free-support WZZT implementation.

This module never receives a physical network or truth labels during fitting.
It uses the production ``milp_only`` initialization (terminal singletons),
the production joint weight refits and the production validation selection.
All P/Q are per unit; terminal and contemporaneous root V are per unit.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter
from typing import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd

from rnj_wzzt.estimation import laminar_l1_milp as core
from rnj_wzzt.estimation.preprocessing import squared_voltage_drop_from_observed_root


DEFAULT_CONFIG = {
    "solver": "highs",
    "time_limit_seconds": 30.0,
    "coefficient_bound": 2.0,
    "max_atoms": None,
    "max_bound_expansions": 4,
    "validation_blocks": 4,
    "zero_tolerance": 1e-9,
    "improvement_abs_tol": 1e-10,
    "improvement_rel_tol": 1e-8,
    "mip_rel_gap": 0.0,
    "presolve": True,
    "disp": False,
}


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path | None, data: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(_plain(data), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _finite_frame(frame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty pandas DataFrame")
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(f"{name} must have unique time indices and terminal columns")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains nonfinite measurements")
    return frame.astype(float).copy()


def make_observed_scenario(
    p_pu: pd.DataFrame,
    q_pu: pd.DataFrame,
    v_pu: pd.DataFrame,
    v_root_pu: pd.Series,
    name: str,
) -> dict:
    """Validate exact meter alignment and construct the production raw target.

The function deliberately rejects silent intersections, sorting and dropped
timestamps. No centering, fitted offsets, measurement repair or truth enters.
Negative net P/Q are permitted; physical feasibility belongs to AC validation.
"""
    p = _finite_frame(p_pu, "p_pu")
    q = _finite_frame(q_pu, "q_pu")
    v = _finite_frame(v_pu, "v_pu")
    for key, frame in (("q_pu", q), ("v_pu", v)):
        if not frame.columns.equals(p.columns) or not frame.index.equals(p.index):
            raise ValueError(f"{key} must have exactly the P terminal columns and time order")
    if not isinstance(v_root_pu, pd.Series) or not v_root_pu.index.equals(p.index):
        raise ValueError("v_root_pu must be a Series with exactly the P time index")
    root = v_root_pu.astype(float).copy()
    if not np.isfinite(root.to_numpy()).all() or (root <= 0).any() or (v <= 0).any().any():
        raise ValueError("terminal and root voltage magnitudes must be finite and positive")
    target = squared_voltage_drop_from_observed_root(v, root)
    assert target.index.equals(p.index) and target.columns.equals(p.columns)
    return {
        "name": str(name),
        "P_terminal": p,
        "Q_terminal": q,
        "V_terminal": v,
        "V_root_observed": root,
        "drop_target": target,
        "units": {"P_terminal": "pu", "Q_terminal": "pu", "V_terminal": "pu", "V_root_observed": "pu", "drop_target": "pu2"},
    }


def _normalize_observed(scenarios: Sequence[dict], split: str) -> list[dict]:
    if not scenarios:
        raise ValueError(f"{split} must contain at least one independent scenario")
    result = []
    labels = None
    names = set()
    for number, s in enumerate(scenarios):
        reconstructed = make_observed_scenario(
            s["P_terminal"], s["Q_terminal"], s["V_terminal"],
            s["V_root_observed"], s.get("name", f"{split}_{number}"),
        )
        if s.get("units") != reconstructed["units"]:
            raise ValueError(f"{split} units must be the explicit per-unit observation contract")
        if "drop_target" not in s or not isinstance(s["drop_target"], pd.DataFrame):
            raise ValueError(f"{split} requires the observed-root drop_target DataFrame")
        old = s["drop_target"]
        new = reconstructed["drop_target"]
        if not old.index.equals(new.index) or not old.columns.equals(new.columns) or not np.array_equal(old.to_numpy(), new.to_numpy()):
            raise ValueError("drop_target differs from measured root squared minus terminal squared")
        if labels is not None and not reconstructed["P_terminal"].columns.equals(labels):
            raise ValueError("all scenarios must use exactly the same terminal order")
        labels = reconstructed["P_terminal"].columns
        if reconstructed["name"] in names:
            raise ValueError("scenario names must be unique within a split")
        names.add(reconstructed["name"])
        result.append(reconstructed)
    return result


def _configuration(config: dict | None, terminal_count: int) -> dict:
    supplied = {} if config is None else dict(config)
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unsupported WZZT configuration keys: {sorted(unknown)}")
    effective = {**DEFAULT_CONFIG, **supplied}
    if effective["solver"] not in ("highs", "gurobi"):
        raise ValueError("solver must be highs or gurobi")
    if effective["mip_rel_gap"] != 0:
        raise ValueError("this adapter retains the production exact-step mip_rel_gap=0")
    if effective["max_atoms"] is None:
        effective["max_atoms"] = 2 * terminal_count - 1
    if not isinstance(effective["max_atoms"], int) or not terminal_count <= effective["max_atoms"] <= 2 * terminal_count - 1:
        raise ValueError("max_atoms must include the terminal singleton initialization")
    return effective


def solver_availability() -> dict:
    """Package inventory only; an optional Gurobi license is not probed here."""
    return {"highs_via_scipy": True, "gurobipy_installed": importlib.util.find_spec("gurobipy") is not None,
            "gurobi_license_checked": False}


def _diagnostic(diag) -> dict:
    d = _plain(diag)
    d["numerical_optimum_certified"] = core.solver_diagnostics_prove_optimality(diag)
    return d


def _path(point, previous=None) -> dict:
    d = _plain(point)
    d["solver"] = _diagnostic(point.solver)
    previous_family = set() if previous is None else set(previous.support_indices)
    d["removed_support_indices_since_previous_path_point"] = _plain(sorted(previous_family - set(point.support_indices)))
    return d


def fit_observed_wzzt(
    training: Sequence[dict],
    validation: Sequence[dict],
    config: dict | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Call the actual production WZZT once after a complete observation batch.

The returned status reports algorithm execution, not true topology recovery.
A time-limited unproven extension is retained as incomplete with the last
certified fit. Fixed-support LP failure is recorded as failed without a model.
"""
    train = _normalize_observed(training, "training")
    valid = _normalize_observed(validation, "validation")
    labels = train[0]["P_terminal"].columns
    if not valid[0]["P_terminal"].columns.equals(labels):
        raise ValueError("training and validation terminal order must agree")
    # Split independence is a caller/data-provider obligation. Distinct scenario
    # names help detect accidental object reuse but do not certify independence.
    if {s["name"] for s in train} & {s["name"] for s in valid}:
        raise ValueError("training and validation scenario names must be disjoint")
    effective = _configuration(config, len(labels))
    initial_supports = [(i,) for i in range(len(labels))]
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    core_path = Path(core.__file__).resolve()
    started = perf_counter()
    events = []
    last_snapshot = {}
    receipt = {
        "entrypoint": "rnj_wzzt.estimation.laminar_l1_milp.fit_laminar_l1_sensitivity",
        "core_file": str(core_path), "core_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
        "initializer": "milp_only_terminal_singletons", "rnj_stage_called": False,
        "production_default_pipeline_called": False, "production_wzzt_core_called": True,
        "support_search_mode": "all_nonempty_distinct_laminar_compatible_extensions",
        "initial_support_indices": initial_supports, "terminal_labels": labels.tolist(),
        "training_scenarios": [s["name"] for s in train], "validation_scenarios": [s["name"] for s in valid],
        "training_time_rows": sum(len(s["P_terminal"]) for s in train),
        "validation_time_rows": sum(len(s["P_terminal"]) for s in valid),
        "observed_target": "V_root_observed^2 - V_terminal^2; raw; no intercept",
        "config": effective, "global_tree_optimality_claim": False,
        "status": "running", "has_model": False,
    }
    _write_json(checkpoint, receipt)

    def progress(state):
        nonlocal last_snapshot
        events.append({"event": state["event"], "elapsed_seconds": perf_counter()-started,
                       "path_length": len(state["path"]), "attempt_count": len(state["attempted_extensions"]),
                       "bound_expansions": state["bound_expansions"], "stop_reason": state["stop_reason"]})
        snapshot = {**receipt, "events": events, "status": "running",
                    "last_certified_path": [_path(p, state["path"][i-1] if i else None) for i,p in enumerate(state["path"])],
                    "attempted_extensions": [{**_plain(a), "diagnostics": _diagnostic(a.diagnostics)} for a in state["attempted_extensions"]]}
        last_snapshot = snapshot
        _write_json(checkpoint, snapshot)

    try:
        fitted = core.fit_laminar_l1_sensitivity(
            train, validation_scenarios=valid, initial_supports=initial_supports,
            max_atoms=effective["max_atoms"], r_upper_bound=effective["coefficient_bound"],
            x_upper_bound=effective["coefficient_bound"], max_bound_expansions=effective["max_bound_expansions"],
            validation_blocks=effective["validation_blocks"], zero_tolerance=effective["zero_tolerance"],
            improvement_abs_tol=effective["improvement_abs_tol"], improvement_rel_tol=effective["improvement_rel_tol"],
            time_limit=effective["time_limit_seconds"], mip_rel_gap=effective["mip_rel_gap"],
            presolve=effective["presolve"], disp=effective["disp"], solver=effective["solver"],
            progress_callback=progress,
        )
    except Exception as exc:
        failed = {**receipt, **last_snapshot, "status": "failed", "has_model": False,
                  "exception_type": type(exc).__name__, "exception": str(exc),
                  "elapsed_seconds": perf_counter()-started, "events": events}
        _write_json(checkpoint, failed)
        return _plain(failed)

    incomplete = fitted.stop_reason in {"extension_not_proven_optimal", "extension_without_incumbent", "bound_expansion_limit"}
    if fitted.stop_reason == "max_atoms" and effective["max_atoms"] < 2*len(labels)-1:
        incomplete = True
    chosen = fitted.path[fitted.selected_path_index]
    # Independently recompute selected losses from observed matrices.
    train_mae, _, _ = core.evaluate_l1_matrices(train, fitted.r_matrix, fitted.x_matrix)
    valid_mae, valid_se, valid_blocks = core.evaluate_l1_matrices(valid, fitted.r_matrix, fitted.x_matrix,
                                                                  blocks_per_scenario=effective["validation_blocks"])
    if not np.isclose(train_mae, fitted.train_mae, atol=1e-10, rtol=1e-7):
        raise RuntimeError("selected training loss does not reproduce")
    if not np.isclose(valid_mae, fitted.validation_mae, atol=1e-10, rtol=1e-7):
        raise RuntimeError("selected validation loss does not reproduce")
    result = {**receipt, "status": "incomplete" if incomplete else "complete", "has_model": True,
              "elapsed_seconds": perf_counter()-started, "events": events,
              "summary": fitted.summary(), "selected": _path(chosen),
              "path": [_path(p, fitted.path[i-1] if i else None) for i,p in enumerate(fitted.path)],
              "attempted_extensions": [{**_plain(a), "diagnostics": _diagnostic(a.diagnostics)} for a in fitted.attempted_extensions],
              "selected_loss_recomputed": {"train_mae_pu2": train_mae, "validation_mae_pu2": valid_mae,
                                           "validation_se_pu2": valid_se, "validation_block_means_pu2": valid_blocks.tolist()},
              "validation_rule": "production one-standard-error rule on stored path; no truth input",
              "topology_recovery_evaluated": False}
    result = _plain(result)
    _write_json(checkpoint, result)
    return result


def summarize_fit(fit_result: dict) -> dict:
    """Compact execution/loss record for a strategy table; no truth inference."""
    summary = fit_result.get("summary", {})
    attempts = fit_result.get("attempted_extensions", [])
    return {
        "status": fit_result["status"], "has_model": fit_result["has_model"],
        "stop_reason": summary.get("stop_reason"),
        "selected_path_index": summary.get("selected_path_index"),
        "path_length": summary.get("path_length", len(fit_result.get("last_certified_path", []))),
        "selected_train_mae_pu2": summary.get("train_mae"),
        "selected_validation_mae_pu2": summary.get("validation_mae"),
        "selected_support_count": len(summary["support_labels"]) if "support_labels" in summary else None,
        "attempt_count": len(attempts),
        "unproved_attempt_count": sum(not a["diagnostics"]["numerical_optimum_certified"] for a in attempts),
        "last_attempt_status": attempts[-1]["diagnostics"]["status"] if attempts else None,
        "last_attempt_mip_gap": attempts[-1]["diagnostics"]["mip_gap"] if attempts else None,
        "bound_expansions": summary.get("bound_expansions"),
        "elapsed_seconds": fit_result.get("elapsed_seconds"),
        "exception_type": fit_result.get("exception_type"),
    }


def evaluate_effective_supports(
    fit_result: dict,
    truth_supports: Iterable[Iterable[Hashable]],
    *,
    effective_tolerance: float | None = None,
) -> dict:
    """Separate simulation-truth evaluation; never called during planning/fit.

Truth supports must already refer to effective physical lines (R or X > 0).
The numerical threshold is reported; low loss is not a truth probability.
"""
    if not fit_result.get("has_model"):
        return {"status": "unavailable_no_model", "full_effective_family_recovered": None}
    tolerance = fit_result["config"]["zero_tolerance"] if effective_tolerance is None else float(effective_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("effective_tolerance must be finite and nonnegative")
    selected = fit_result["selected"]
    predicted = {frozenset(s) for s,r,x in zip(selected["support_labels"], selected["r_values"], selected["x_values"], strict=True)
                 if r > tolerance or x > tolerance}
    truth = {frozenset(s) for s in truth_supports}
    labels = frozenset(fit_result["terminal_labels"])
    if any(not s or not s <= labels for s in truth):
        raise ValueError("truth support outside measured terminal labels")
    def serialized(family):
        return [sorted(s, key=str) for s in sorted(family, key=lambda s:(len(s), tuple(sorted(map(str,s)))))]
    def score(p, t):
        matched=p&t
        precision=len(matched)/len(p) if p else float(not t)
        recall=len(matched)/len(t) if t else float(not p)
        return {"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.,
                "matched_count":len(matched),"predicted_count":len(p),"true_count":len(t),
                "missing_supports":serialized(t-p),"extra_supports":serialized(p-t)}
    p_non={s for s in predicted if 1<len(s)<len(labels)}
    t_non={s for s in truth if 1<len(s)<len(labels)}
    return {"status":"evaluated", "fit_status":fit_result["status"], "effective_tolerance":tolerance,
            "full_effective_family_recovered":predicted==truth,"nontrivial_family_recovered":p_non==t_non,
            "all_effective_supports":score(predicted,truth),"nontrivial_supports":score(p_non,t_non),
            "predicted_effective_supports":serialized(predicted),"truth_effective_supports":serialized(truth),
            "probability_claim":False}
