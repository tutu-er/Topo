"""Standalone observed-data bridge for the frozen finite-pool best strategy.

No simulation truth, old model bank, parent result or unrelated variant is input.
All three returned models share the newly built training-only pool and fits.
"""
from __future__ import annotations
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback
import numpy as np
import pandas as pd
import scipy
from research_experiments.wzzt_dr_selection.end_to_end import wzzt_adapter as observed
from .engine import build_bank, PoolLP, search, canonical
from .observations import normalize, input_hash
from .selection import contract, measure, select, package


def source_hashes():
    from rnj_wzzt.estimation import multiscenario, constrained_least_squares
    from rnj_wzzt.graph import rooted_hierarchy, rooted_neighbor_joining
    from research_experiments.rnj import rooted_study_baselines, rx75_edge_floor, sensitivity_geometry
    files = [Path(__file__).parent/name for name in
             ("__init__.py", "adapter.py", "observations.py", "engine.py", "selection.py", "provenance.json")]
    files += [Path(module.__file__) for module in (observed, multiscenario, constrained_least_squares,
              rooted_hierarchy, rooted_neighbor_joining, rooted_study_baselines, rx75_edge_floor, sensitivity_geometry)]
    # Hash the full local core source closure, including geometry and target
    # helpers reached indirectly by the public functions above.
    files.extend(Path(multiscenario.__file__).parents[1].rglob("*.py"))
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def fit_observed(training, validation, output_dir, *, root_label=1, search_seconds=25., resume=False,lp_method=1):
    """Build a pool on observed training data and return three explicit models.

    Scenarios follow end_to_end.wzzt_adapter.make_observed_scenario. Daily
    centering uses timestamps, or explicit day_ids for non-datetime inputs.
    Validation selects starts/models but never creates the candidate pool.
    Resume only reuses a completed exact-input/source match; unfinished search
    checkpoints are retained and never silently restarted with a new budget.
    """
    if isinstance(lp_method,(bool,np.bool_)) or not isinstance(lp_method,(int,np.integer)) or lp_method not in (0,1,2):
        raise ValueError("lp_method must be an integer in {0, 1, 2}")
    lp_method=int(lp_method)
    train, days, counts = normalize(training, "training")
    valid, valid_days, valid_counts = normalize(validation, "validation")
    labels = train[0].columns.tolist()
    if not valid[0].columns.equals(train[0].columns):
        raise ValueError("training and validation terminal order must agree exactly")
    if not isinstance(root_label, (int, np.integer)) or any(not isinstance(v, (int, np.integer)) for v in labels):
        raise ValueError("current NJ/RNJ geometry requires integer physical labels")
    if root_label in labels:
        raise ValueError("root cannot be a terminal")
    if isinstance(train[0].index, pd.DatetimeIndex) and isinstance(valid[0].index, pd.DatetimeIndex):
        if len(train[0].index.intersection(valid[0].index)):
            raise ValueError("training and validation timestamps must be disjoint")
    if not np.isfinite(search_seconds) or search_seconds <= 0:
        raise ValueError("search_seconds must be positive")
    inputs = {"training":input_hash(train, days), "validation":input_hash(valid, valid_days)}
    sources = source_hashes()
    identity = {"input_sha256":inputs, "source_sha256":sources,
                "root_label":int(root_label), "search_seconds":float(search_seconds),"lp_method":lp_method}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    out = Path(output_dir)
    result_path, checkpoint = out/"result.json", out/"checkpoint.json"
    if result_path.exists():
        old = json.loads(result_path.read_text(encoding="utf-8"))
        if resume and old.get("status") == "complete" and old.get("fingerprint") == fingerprint:
            return old
        raise FileExistsError("existing result; only an unchanged completed run may resume")
    if checkpoint.exists():
        raise FileExistsError("unfinished search checkpoint; do not restart or extend its budget implicitly")
    out.mkdir(parents=True, exist_ok=True)
    lock = out/"fit.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid":os.getpid(), "fingerprint":fingerprint}, stream)
    started = perf_counter()
    lp = None
    result = dict(status="running", stage="building_pool", fingerprint=fingerprint, **identity,
        terminal_labels=labels, training_rows=len(train[0]), validation_rows=len(valid[0]),
        training_day_counts=counts, validation_day_counts=valid_counts,
        initialization="complete_raw_RNJ_families", truth_input_allowed=False,
        candidate_pool_policy="rebuild from observed training each batch; raw and physical-day demean",
        final_strategy="multistart_original_rule", models={}, searches={},
        settings=dict(coefficient_bound=2., lp_time_limit_seconds=15., lp_threads=1,
            lp_method=lp_method, feasibility_tolerance=1e-9, optimality_tolerance=1e-9,
            dense_qp_max_iterations=5000, per_path_seconds=float(search_seconds),
            max_extra_starts=2, max_steps=3*len(labels), relative_gain_threshold=.001,
            final_validation_fraction=.03, resolution_factor=.16,
            final_refit=False, AC_gate=False),
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                         numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
        timing_scope="independent complete call; bank/start preparation included in total; each path budget may overrun by current LP",
        historical_equivalence="same declared algorithm rules; standalone cache and wall time can change truncated paths",
        checkpoint_policy="completed stages and accepted points retained; interrupted solver state not resumable")
    def save(stage):
        result.update(stage=stage, elapsed_seconds=perf_counter()-started)
        if lp is not None:
            result["LP_solves"] = lp.solves
            result["lp_cached_family_count"] = len(lp.cache)
        observed._write_json(checkpoint, result)
    try:
        save("building_pool")
        bank, pool, fronts, calibrations = build_bank(train, day_ids=days, root_label=root_label)
        result.update(pool_support_indices=pool,
                      pool_support_labels=[[labels[i] for i in s] for s in pool],
                      fronts=fronts, calibrations=calibrations,
                      generated_tree_count=len(bank), pool_size=len(pool))
        lp = PoolLP(train, pool, "raw", bound=2.,lp_method=lp_method)
        result["settings"]["actual_lp_method"]=int(lp.model.Params.Method)
        result["environment"]["gurobi"] = list(lp.gp.gurobi.version())
        fitted_bank = []
        for tree in bank:
            if tree["preprocess"] != "raw" or tree["method"] != "RNJ":
                continue
            point = measure(dict(lp.fit(tree["family"]), origin="tree_bank",
                                 config=tree["config"], native_family=tree["family"]), train, valid)
            fitted_bank.append(point)
        if len(fitted_bank) != 15:
            raise RuntimeError("expected fifteen raw RNJ bank sources")
        result["bank"] = fitted_bank
        cal = calibrations["raw"]
        result["models"]["rnj_bank"] = package(select(fitted_bank, cal), labels, train, valid)
        save("bank_complete")
        primary = next(p for p in fitted_bank if p["config"] == "raw_RX75_0.16")
        def run_path(name, initial):
            def progress(trajectory):
                result["searches"][name] = trajectory
                save(name)
            path = search(lp, initial, seconds=search_seconds, repair=True,
                          max_steps=3*len(labels), progress_callback=progress)
            result["searches"][name] = path
            return path
        main_path = run_path("primary", primary["native_family"])
        main_points = [measure(dict(p, origin="tree_initialized_correction", path_index=i), train, valid)
                       for i, p in enumerate(main_path["path"])]
        candidates = fitted_bank + main_points
        result["models"]["baseline25"] = package(select(candidates, cal), labels, train, valid)
        save("primary_complete")
        options = []
        for index, point in enumerate(fitted_bank):
            cut = contract(point, cal)
            fitted = measure(dict(lp.fit(cut["supports"]), source_index=index,
                                  source_config=point["config"]), train, valid)
            options.append(fitted)
        ranked = sorted(enumerate(options), key=lambda t:(t[1]["validation_mae"], len(t[1]["supports"]), t[0]))
        seen = {canonical(primary["native_family"])}
        extra = []
        for index, point in ranked:
            family = canonical(point["supports"])
            if family in seen:
                continue
            seen.add(family)
            extra.append((index, point))
            if len(extra) == 2:
                break
        result["start_preparation"] = options
        result["extra_start_sources"] = [i for i, _ in extra]
        save("extra_starts_prepared")
        for serial, (index, point) in enumerate(extra):
            name = f"multistart_{serial}"
            trajectory = run_path(name, point["supports"])
            trajectory.update(source_index=index, source_config=fitted_bank[index]["config"],
                              source_family=point["supports"])
            candidates.extend(measure(dict(p, origin=name, path_index=i,
                                            source_config=fitted_bank[index]["config"]), train, valid)
                              for i, p in enumerate(trajectory["path"]))
            save(name+"_complete")
        result["models"]["multistart_original_rule"] = package(select(candidates, cal), labels, train, valid)
        result["selection_candidates"] = candidates
        result["search_stop_statuses"] = {k:v["reason"] for k, v in result["searches"].items()}
        result["all_local_scans_complete"] = all(v == "pool_local_optimum" for v in result["search_stop_statuses"].values())
        result.update(status="complete", stage="finished", has_model=True,
                      elapsed_seconds=perf_counter()-started, LP_solves=lp.solves,
                      lp_cached_family_count=len(lp.cache), global_tree_optimality_claim=False)
        if any(s.get("status") != 2 or not s.get("accepted") for s in lp.solves):
            raise RuntimeError("nonoptimal fixed-family LP in completed result")
        observed._write_json(result_path, result)
        observed._write_json(checkpoint, result)
        return observed._plain(result)
    except Exception as exc:
        result.update(status="failed", has_model=bool(result["models"]),
                      exception_type=type(exc).__name__, exception=str(exc),
                      traceback=traceback.format_exc(), elapsed_seconds=perf_counter()-started)
        if lp is not None:
            result["LP_solves"] = lp.solves
        observed._write_json(result_path, result)
        raise
    finally:
        if lp is not None:
            lp.close()
        lock.unlink(missing_ok=True)
