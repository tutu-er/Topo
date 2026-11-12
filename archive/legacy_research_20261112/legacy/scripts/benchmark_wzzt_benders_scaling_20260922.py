"""Serial, matched full-domain one-extension Benders/MILP benchmark.

Run with Python -B. Each method runs in a fresh process after a tiny LP/MILP
warm-up. Core/prototype imports are read-only; wrappers record diagnostics and
complete Benders events without changing their algorithms. No candidate pool.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import traceback

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "rnj_wzzt_core" / "rnj_wzzt"
METHODS = ("original_milp", "benders")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    files = list(CORE.rglob("*.py")) + [ROOT / "experiments/wzzt_benders_prototype.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def build_case(spec):
    """Construct observations independently; truth never enters search arguments."""
    if spec["case_family"] == "archived_ac":
        path = Path(spec["archived_job"])
        if path.is_dir():
            path = path / "inputs.npz"
        scenarios = []
        loaded_keys = []
        with np.load(path, allow_pickle=False) as archive:
            indices = sorted(int(k.split("_")[1]) for k in archive.files
                             if k.startswith("train_") and k.endswith("_P_terminal"))
            for index in indices:
                item = {"name": f"train_{index}"}
                for key in ("P_terminal", "Q_terminal", "drop_target"):
                    archive_key = f"train_{index}_{key}"
                    item[key] = np.array(archive[archive_key], dtype=float, copy=True)
                    loaded_keys.append(archive_key)
                scenarios.append(item)
        if not scenarios:
            raise ValueError("No train observation arrays found")
        n = scenarios[0]["P_terminal"].shape[1]
        meta = {"source": str(path.resolve()), "source_sha256": sha(path),
                "loaded_keys": loaded_keys, "truth_used": False,
                "note": "Archived AC observations; only a singleton-family extension, not full-tree recovery."}
        bounds = (2.0, 2.0)
    else:
        n, count = spec["n"], spec["samples_per_scenario"]
        rng = np.random.default_rng(spec["case_seed"])
        truth = tuple(range(n // 2))
        r, x = np.diag(rng.uniform(0.05, 0.10, n)), np.diag(rng.uniform(0.03, 0.07, n))
        r[np.ix_(truth, truth)] += 0.20
        x[np.ix_(truth, truth)] += 0.12
        scenarios = []
        for index in range(2):
            p = rng.normal(0.9 + 0.15 * index, 0.35, (count, n))
            q = 0.15 * (p - p.mean(axis=0)) + rng.normal(0.25, 0.30, (count, n))
            intercept = rng.uniform(-0.12, 0.12, n)
            target = p @ r.T + q @ x.T + intercept + rng.normal(0.0, 0.003, (count, n))
            scenarios.append({"name": f"scenario_{index}", "P_terminal": p,
                              "Q_terminal": q, "drop_target": target})
        bounds = (0.8, 0.6)
        meta = {"generation_only_true_new_support": truth,
                "noise_standard_deviation": 0.003,
                "new_r": 0.20, "new_x": 0.12,
                "truth_used_for_search": False}
    metadata = {"n": n, "scenario_count": len(scenarios),
                "samples_per_scenario": [len(s["P_terminal"]) for s in scenarios],
                "observation_count": sum(s["P_terminal"].size for s in scenarios),
                "old_support_count": n, "r_upper_bound": bounds[0],
                "x_upper_bound": bounds[1], **meta}
    return scenarios, tuple((i,) for i in range(n)), bounds, metadata


def independent_mae(scenarios, supports, support, r, x, intercepts):
    if support is None:
        return None
    n = scenarios[0]["P_terminal"].shape[1]
    family = (*supports, tuple(support))
    if len(r) != len(family) or len(x) != len(family):
        raise ValueError("Incomplete refitted coefficients")
    rm, xm = np.zeros((n, n)), np.zeros((n, n))
    for atom, rv, xv in zip(family, r, x):
        rm[np.ix_(atom, atom)] += rv
        xm[np.ix_(atom, atom)] += xv
    residuals = []
    for index, scenario in enumerate(scenarios):
        predicted = scenario["P_terminal"] @ rm.T + scenario["Q_terminal"] @ xm.T + intercepts[index]
        residuals.append(np.abs(scenario["drop_target"] - predicted).ravel())
    return float(np.mean(np.concatenate(residuals)))


def warmup():
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    linprog([1.0], A_ub=[[-1.0]], b_ub=[-0.5], bounds=[(0, 1)], method="highs")
    milp([1.0], integrality=[1], bounds=Bounds([0], [1]),
         constraints=LinearConstraint([[1.0]], [0.5], [np.inf]))


def worker(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    out = Path(config["result_path"])
    event_path = out.with_suffix(".events.jsonl")
    before = manifest()
    record = {"specification": config, "status": "started", "events_file": str(event_path)}
    events = []
    event_file = event_path.open("w", encoding="utf-8", buffering=1)
    started = None

    def emit(event):
        event = clean(event)
        events.append(event)
        event_file.write(json.dumps(event, allow_nan=False) + "\n")
        event_file.flush()

    try:
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "rnj_wzzt_core"))
        from experiments import wzzt_benders_prototype as prototype
        from rnj_wzzt.estimation import laminar_l1_milp as original
        if config["method"] == "original_milp" and "intercepts" not in original.ExtensionSolution.__dataclass_fields__:
            raise RuntimeError("Historical Benders comparison requires the intercept core in artifacts/observed_root_model_20260928/before; current core uses a different model.")
        scenarios, supports, bounds, meta = build_case(config)
        record["input"] = meta
        warmup()
        budget = config["time_limit"]
        common = dict(r_upper_bound=bounds[0], x_upper_bound=bounds[1], time_limit=budget)
        if config["method"] == "original_milp":
            original_run = original._run_milp
            captured = {}

            def observed_run(variables, rows, **kwargs):
                # Make both time budgets include model construction, while keeping the
                # core's model, solver options, and returned results unchanged.
                kwargs["time_limit"] = max(0.001, budget - (perf_counter() - started))
                call_started = perf_counter()
                raw, diagnostics = original_run(variables, rows, **kwargs)
                captured.update(raw=raw, slices=variables.slices, diagnostics=diagnostics)
                if raw.x is not None and np.isfinite(raw.x).all():
                    matrix = rows.build(variables.size)
                    activity = matrix.A @ raw.x
                    integrality = np.asarray(variables.integrality, dtype=bool)
                    captured["feasibility_error"] = max(
                        0.0, float(np.max(matrix.lb - activity)), float(np.max(activity - matrix.ub)),
                        float(np.max(np.asarray(variables.lower) - raw.x)),
                        float(np.max(raw.x - np.asarray(variables.upper))),
                        float(np.max(np.abs(raw.x[integrality] - np.rint(raw.x[integrality])))))
                emit({"event": "original_solver", "elapsed_seconds": perf_counter() - started,
                      "call_seconds": perf_counter() - call_started,
                      "diagnostics": asdict(diagnostics)})
                return raw, diagnostics

            original._run_milp = observed_run
            started = perf_counter()
            try:
                result = original.solve_best_laminar_extension_l1(
                    scenarios, supports, candidate_supports=None, mip_rel_gap=0.0, **common)
            finally:
                wall = perf_counter() - started
                original._run_milp = original_run
            diagnostics = asdict(result.diagnostics)
            certified = original.solver_diagnostics_prove_optimality(result.diagnostics)
            raw = captured["raw"]
            raw_fit = None
            if raw.x is not None and np.isfinite(raw.x).all():
                blocks = captured["slices"]
                values = np.asarray(raw.x)
                zs = values[blocks["z"]]
                support = tuple(int(i) for i in np.flatnonzero(np.rint(zs)))
                rv = np.r_[values[blocks["old_r"]], values[blocks["new_r"]]]
                xv = np.r_[values[blocks["old_x"]], values[blocks["new_x"]]]
                intercepts = values[blocks["intercepts"]].reshape(len(scenarios), meta["n"])
                admissible = (np.max(np.abs(zs - np.rint(zs))) <= 1e-6
                              and original.is_admissible_extension(support, supports, meta["n"]))
                recomputed = independent_mae(scenarios, supports, support, rv, xv, intercepts) if admissible else None
                raw_fit = {"support": support, "r_values": rv, "x_values": xv,
                           "intercepts": intercepts, "independent_mae": recomputed,
                           "admissible_integer_support": admissible,
                           "feasibility_error": captured.get("feasibility_error"),
                           "objective_reproduction_error": None if recomputed is None else abs(recomputed - raw.fun),
                           "raw_x": values,
                           "variable_slices": {k: [v.start, v.stop] for k, v in blocks.items()}}
            lb = diagnostics["dual_bound"]
            ub = None if raw_fit is None else raw_fit["independent_mae"]
            gap = None if lb is None or ub is None else max(0.0, ub - lb)
            record.update(status="completed", wall_time_seconds=wall,
                          solver_time_seconds=diagnostics["runtime_seconds"],
                          objective=result.objective, returned_support=result.support,
                          incumbent=raw_fit, lower_bound=lb, upper_bound=ub,
                          absolute_gap=gap, relative_gap=None if gap is None else gap / max(abs(ub), 1e-15),
                          proven_optimal=certified, diagnostics=diagnostics,
                          stop_reason=diagnostics["message"],
                          node_count=diagnostics["node_count"], lp_count=None, cut_count=0,
                          result=asdict(result))
        else:
            master_old = prototype.BendersExtensionProblem.solve_master
            sub_old = prototype.BendersExtensionProblem.solve_subproblem
            bounds_seen = {"lb": 0.0, "ub": float("inf")}

            def observed_master(problem, cuts, time_limit=None):
                tick = perf_counter()
                result = master_old(problem, cuts, time_limit=time_limit)
                seconds = perf_counter() - tick
                lb = getattr(result, "mip_dual_bound", None)
                if lb is not None and np.isfinite(lb):
                    bounds_seen["lb"] = max(bounds_seen["lb"], float(lb))
                support = None if result.x is None else np.flatnonzero(np.rint(result.x[:problem.n])).tolist()
                emit({"event": "master", "cut_count_before": len(cuts), "call_seconds": seconds,
                      "elapsed_seconds": perf_counter() - started, "status": int(result.status),
                      "message": str(result.message), "objective": getattr(result, "fun", None),
                      "solver_dual_bound": lb, "node_count": getattr(result, "mip_node_count", None),
                      "solver_mip_gap": getattr(result, "mip_gap", None), "support": support,
                      "lower_bound": bounds_seen["lb"], "upper_bound": bounds_seen["ub"],
                      "absolute_gap": max(0.0, bounds_seen["ub"] - bounds_seen["lb"]),
                      "variable_count": len(problem.master_c),
                      "constraint_count": problem.master_a.shape[0] + len(cuts)})
                return result

            def observed_subproblem(problem, y, time_limit=None):
                tick = perf_counter()
                support = tuple(i for i in range(problem.n) if y[problem.pair_lookup[(i, i)]] > 0.5)
                try:
                    result = sub_old(problem, y, time_limit=time_limit)
                except Exception as exc:
                    emit({"event": "subproblem_failure", "call_seconds": perf_counter() - tick,
                          "elapsed_seconds": perf_counter() - started, "support": support,
                          "error": str(exc), "status": getattr(exc, "status", None)})
                    raise
                seconds = perf_counter() - tick
                recomputed = independent_mae(scenarios, supports, support,
                                             np.clip(result.r_values, 0, bounds[0]),
                                             np.clip(result.x_values, 0, bounds[1]), result.intercepts)
                bounds_seen["ub"] = min(bounds_seen["ub"], recomputed)
                emit({"event": "subproblem", "call_seconds": seconds,
                      "elapsed_seconds": perf_counter() - started, "support": support,
                      "objective": result.objective, "independent_mae": recomputed,
                      "dual_objective": result.dual_objective, "primal_dual_gap": result.primal_dual_gap,
                      "feasibility_error": result.primal_feasibility_error,
                      "lower_bound": bounds_seen["lb"], "upper_bound": bounds_seen["ub"],
                      "absolute_gap": max(0.0, bounds_seen["ub"] - bounds_seen["lb"]),
                      "variable_count": problem.variable_count, "constraint_count": problem.sub_a.shape[0]})
                return result

            prototype.BendersExtensionProblem.solve_master = observed_master
            prototype.BendersExtensionProblem.solve_subproblem = observed_subproblem
            started = perf_counter()
            try:
                result = prototype.solve_best_laminar_extension_benders(
                    scenarios, supports, max_iterations=config["max_iterations"],
                    absolute_tolerance=1e-10, relative_tolerance=0.0, **common)
            finally:
                wall = perf_counter() - started
                prototype.BendersExtensionProblem.solve_master = master_old
                prototype.BendersExtensionProblem.solve_subproblem = sub_old
            recomputed = independent_mae(scenarios, supports, result.support,
                                         result.r_values, result.x_values, result.intercepts)
            result_summary = asdict(result)
            result_summary.pop("cuts")  # Full gradients are unnecessary for timing analysis.
            record.update(status="completed", wall_time_seconds=wall,
                          solver_time_seconds=sum(e["call_seconds"] for e in events),
                          master_time_seconds=sum(e["call_seconds"] for e in events if e["event"] == "master"),
                          subproblem_time_seconds=sum(e["call_seconds"] for e in events if e["event"].startswith("subproblem")),
                          objective=result.objective, returned_support=result.support,
                          incumbent={"support": result.support, "r_values": result.r_values,
                                     "x_values": result.x_values, "intercepts": result.intercepts,
                                     "independent_mae": recomputed,
                                     "objective_reproduction_error": None if recomputed is None else abs(recomputed - result.objective)},
                          lower_bound=result.lower_bound, upper_bound=result.upper_bound,
                          absolute_gap=result.absolute_gap,
                          relative_gap=None if result.absolute_gap is None else result.absolute_gap / max(abs(result.upper_bound), 1e-15),
                          proven_optimal=result.proven_optimal, stop_reason=result.stop_reason,
                          node_count=sum(e.get("node_count") or 0 for e in events),
                          master_count=sum(e["event"] == "master" for e in events),
                          lp_count=sum(e["event"].startswith("subproblem") for e in events),
                          cut_count=len(result.cuts), result=result_summary)
        if record.get("incumbent") is not None:
            truth = meta.get("generation_only_true_new_support")
            record["incumbent"]["matches_generation_support"] = (
                None if truth is None else tuple(record["incumbent"]["support"] or ()) == tuple(truth))
        emit({"event": "final", "elapsed_seconds": wall,
              "lower_bound": record["lower_bound"], "upper_bound": record["upper_bound"],
              "absolute_gap": record["absolute_gap"], "proven_optimal": record["proven_optimal"],
              "stop_reason": record["stop_reason"]})
    except Exception as exc:
        record.update(status="error", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc(), proven_optimal=False,
                      wall_time_seconds=None if started is None else perf_counter() - started)
    finally:
        event_file.close()
        after = manifest()
        changes = [k for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)]
        record["source_integrity"] = {"unchanged": not changes, "changed_paths": changes,
                                      "before": before, "after": after}
        record["events"] = events
        record["execution"] = {"python": sys.version, "executable": sys.executable,
                               "platform": platform.platform(), "numpy": np.__version__,
                               "scipy": scipy.__version__, "bytecode_disabled": sys.dont_write_bytecode,
                               "warmup_excluded_from_timing": True,
                               "event_logging_included_in_wall_time": True}
        write_json(out, record)
    return 0 if record["status"] == "completed" else 1


def comparisons(records):
    groups = {}
    for row in records:
        groups.setdefault(row["specification"]["case_id"], {})[row["specification"]["method"]] = row
    result = []
    for case, methods in groups.items():
        if not all(name in methods for name in METHODS):
            continue
        a, b = methods["original_milp"], methods["benders"]
        both = bool(a.get("proven_optimal") and b.get("proven_optimal"))
        av, bv = a.get("upper_bound"), b.get("upper_bound")
        difference = None if av is None or bv is None else abs(av - bv)
        result.append({"case_id": case, "both_certified": both,
                       "incumbent_objective_absolute_difference": difference,
                       "equal_optimum_verified_at_1e_minus_7": None if not both else difference <= 1e-7,
                       "benders_over_original_wall_ratio": (
                           b["wall_time_seconds"] / a["wall_time_seconds"]
                           if a.get("wall_time_seconds") and b.get("wall_time_seconds") else None),
                       "speed_ratio_is_time_to_certification": both})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="*", default=[4, 8, 12, 18])
    parser.add_argument("--samples-per-scenario", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--archived-job", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/wzzt_benders_scaling_20260922")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    sys.path.insert(0, str(ROOT / "rnj_wzzt_core"))
    from rnj_wzzt.estimation.laminar_l1_milp import ExtensionSolution
    if "intercepts" not in ExtensionSolution.__dataclass_fields__:
        parser.error("Benders retains free intercepts but current core does not. Replay the matched historical comparison with artifacts/observed_root_model_20260928/before.")
    if any(n < 4 for n in args.sizes) or args.repeats < 1 or args.samples_per_scenario < 4 or args.time_limit <= 0:
        parser.error("sizes >= 4, repeats >= 1, samples >= 4, time limit > 0 required")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before = manifest()
    cases = []
    for n in args.sizes:
        for repeat in range(args.repeats):
            cases.append({"case_id": f"synthetic_n{n}_t{args.samples_per_scenario}_r{repeat}",
                          "case_family": "synthetic_single_clade", "n": n,
                          "samples_per_scenario": args.samples_per_scenario,
                          "case_seed": args.seed + 1000 * n + repeat, "repeat": repeat})
    for archive in args.archived_job:
        resolved = archive.resolve()
        label = resolved.name if resolved.is_dir() else resolved.parent.name
        for repeat in range(args.repeats):
            cases.append({"case_id": f"archived_{label}_r{repeat}", "case_family": "archived_ac",
                          "archived_job": str(resolved), "repeat": repeat})
    records = []
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Full-domain ONE extension from all singleton supports; no full-tree claim.",
              "configuration": vars(args), "candidate_pool": None,
              "time_budget": "Same wall budget including model construction; worker import/warm-up excluded.",
              "certification_tolerance": {"original": "core strict certificate (1e-10)",
                                          "benders": "absolute 1e-10 + relative 0"},
              "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(__file__), **before},
              "cases": records, "comparisons": []}
    write_json(args.output_dir / "benchmark_report.json", report)
    for index, case in enumerate(cases):
        order = METHODS if index % 2 == 0 else tuple(reversed(METHODS))
        for method in order:
            stem = f"{case['case_id']}__{method}"
            result_path = args.output_dir / f"{stem}.json"
            if result_path.exists():
                raise FileExistsError(f"Refusing to overwrite an existing job: {result_path}")
            spec = {**case, "method": method, "method_order": list(order),
                    "time_limit": args.time_limit, "max_iterations": args.max_iterations,
                    "result_path": str(result_path)}
            config_path = args.output_dir / f"{stem}.config.json"
            write_json(config_path, spec)
            print(f"START {stem}; budget={args.time_limit:g}s", flush=True)
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(config_path)]
            try:
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                           timeout=args.time_limit + 45.0, env=os.environ.copy())
                if not result_path.exists():
                    write_json(result_path, {"specification": spec, "status": "worker_failed",
                                            "proven_optimal": False, "returncode": completed.returncode,
                                            "stdout": completed.stdout, "stderr": completed.stderr})
            except subprocess.TimeoutExpired as exc:
                write_json(result_path, {"specification": spec, "status": "watchdog_timeout",
                                        "proven_optimal": False, "watchdog_seconds": args.time_limit + 45,
                                        "note": "Partial events remain in the adjacent events.jsonl file."})
            row = json.loads(result_path.read_text(encoding="utf-8"))
            records.append(row)
            report["comparisons"] = comparisons(records)
            after = manifest()
            report["source_integrity"] = {"unchanged": before == after,
                                          "changed_paths": [k for k in before if before[k] != after.get(k)]}
            write_json(args.output_dir / "benchmark_report.json", report)
            print(json.dumps(clean({"case": case["case_id"], "method": method,
                                    "status": row["status"], "seconds": row.get("wall_time_seconds"),
                                    "certified": row.get("proven_optimal"), "LB": row.get("lower_bound"),
                                    "UB": row.get("upper_bound"), "cuts": row.get("cut_count"),
                                    "stop": row.get("stop_reason")}), allow_nan=False), flush=True)
    print(f"Report: {args.output_dir / 'benchmark_report.json'}", flush=True)
    return 0 if all(row["status"] == "completed" for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
