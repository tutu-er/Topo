"""Small, reproducible full-domain Benders / original-MILP comparison.

Run with ``python -B scripts/run_wzzt_benders_demo.py``.  Only the requested
output directory is written; the original estimator is imported read-only.
These synthetic cases check one extension, not the global forward tree search.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import traceback

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "rnj_wzzt_core" / "rnj_wzzt"
EXPECTED_LAMINAR_SHA256 = (
    "b5155baf7ee86471e4c721122c04a2bd00431cbd195c9054183640c97ec5d208"
)
CASE_NAMES = (
    "n3_empty_noiseless",
    "n3_singletons_noisy",
    "n4_two_branches_noisy",
    "n4_nested_noisy",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def core_manifest() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): file_sha256(path)
        for path in sorted(CORE.rglob("*.py"))
    }


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def generate_case(name: str, seed: int) -> dict:
    """Generate observations independently of either solver implementation."""
    definitions = {
        "n3_empty_noiseless": (3, (), (0, 1), 0.0),
        "n3_singletons_noisy": (3, ((0,), (1,), (2,)), (0, 1), 0.004),
        "n4_two_branches_noisy": (4, ((0, 1), (2, 3)), (0, 1, 2, 3), 0.004),
        "n4_nested_noisy": (4, ((0, 1, 2, 3), (0,), (2, 3)), (0, 1), 0.004),
    }
    n, supports, new_support, noise_scale = definitions[name]
    rng = np.random.default_rng(seed)
    true_supports = (*supports, new_support)
    r_values = np.linspace(0.065, 0.225, len(true_supports))
    x_values = np.linspace(0.045, 0.135, len(true_supports))
    r_matrix = np.zeros((n, n))
    x_matrix = np.zeros((n, n))
    for support, r_value, x_value in zip(true_supports, r_values, x_values):
        r_matrix[np.ix_(support, support)] += r_value
        x_matrix[np.ix_(support, support)] += x_value
    scenarios = []
    true_intercepts = []
    for scenario_index, count in enumerate((12, 15)):
        p = rng.normal(0.9 + 0.15 * scenario_index, 0.35, size=(count, n))
        q = rng.normal(0.25 - 0.04 * scenario_index, 0.3, size=(count, n))
        intercept = rng.uniform(-0.12, 0.12, n)
        noise = rng.normal(0.0, noise_scale, size=(count, n))
        target = p @ r_matrix.T + q @ x_matrix.T + intercept + noise
        scenarios.append(
            {"name": f"scenario_{scenario_index}", "P_terminal": p,
             "Q_terminal": q, "drop_target": target}
        )
        true_intercepts.append(intercept)
    return {
        "name": name, "seed": seed, "n": n, "supports": supports,
        "r_upper_bound": 0.8, "x_upper_bound": 0.6,
        "noise_standard_deviation": noise_scale, "scenarios": scenarios,
        "generation_only_truth": {
            "supports": true_supports, "r_values": r_values,
            "x_values": x_values, "intercepts": true_intercepts,
        },
    }


def direct_mae(case: dict, result) -> float | None:
    """Reconstruct predictions without calling core or prototype helpers."""
    if result.support is None or result.objective is None:
        return None
    n = case["n"]
    family = (*case["supports"], tuple(result.support))
    r_matrix = np.zeros((n, n))
    x_matrix = np.zeros((n, n))
    if len(result.r_values) != len(family) or len(result.x_values) != len(family):
        raise ValueError("Solver did not return refitted weights for every old/new support")
    for support, r_value, x_value in zip(family, result.r_values, result.x_values):
        r_matrix[np.ix_(support, support)] += r_value
        x_matrix[np.ix_(support, support)] += x_value
    residuals = []
    for index, scenario in enumerate(case["scenarios"]):
        prediction = (scenario["P_terminal"] @ r_matrix.T
                      + scenario["Q_terminal"] @ x_matrix.T
                      + result.intercepts[index])
        residuals.append(np.abs(scenario["drop_target"] - prediction).reshape(-1))
    return float(np.mean(np.concatenate(residuals)))


def run_case(case: dict, args, solve_benders, solve_original, prove_original) -> dict:
    record = {"input": jsonable(case), "methods": {}}
    results = {}
    for name, solver in (("original_full_domain_milp", solve_original),
                         ("full_domain_benders", solve_benders)):
        started = perf_counter()
        try:
            kwargs = {
                "r_upper_bound": case["r_upper_bound"],
                "x_upper_bound": case["x_upper_bound"],
                "time_limit": args.time_limit,
            }
            if name == "original_full_domain_milp":
                kwargs.update(candidate_supports=None, mip_rel_gap=0.0)
            else:
                kwargs.update(max_iterations=args.max_iterations,
                              absolute_tolerance=1e-8, relative_tolerance=1e-8)
            result = solver(case["scenarios"], case["supports"], **kwargs)
            elapsed = perf_counter() - started
            independently_recomputed_mae = direct_mae(case, result)
            certified = (prove_original(result.diagnostics)
                         if name == "original_full_domain_milp"
                         else bool(result.proven_optimal))
            record["methods"][name] = {
                "wall_time_seconds": elapsed, "proven_optimal": certified,
                "independently_recomputed_mae": independently_recomputed_mae,
                "objective_reproduction_error": (
                    None if independently_recomputed_mae is None
                    else abs(independently_recomputed_mae - result.objective)),
                "result": jsonable(result),
            }
            results[name] = result
        except Exception as exc:
            record["methods"][name] = {
                "wall_time_seconds": perf_counter() - started,
                "proven_optimal": False, "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(),
            }
    if len(results) == 2 and all(item.objective is not None for item in results.values()):
        original = results["original_full_domain_milp"]
        benders = results["full_domain_benders"]
        difference = abs(float(benders.objective) - float(original.objective))
        record["comparison"] = {
            "objective_absolute_difference": difference,
            "support_equal": tuple(original.support or ()) == tuple(benders.support or ()),
            "both_certified": all(item["proven_optimal"] for item in record["methods"].values()),
            "objectives_match_at_1e_minus_7": difference <= 1e-7,
            "note": "Equal objectives may have different optimal supports or coefficients.",
        }
    else:
        record["comparison"] = {"both_certified": False,
                                "objectives_match_at_1e_minus_7": False}
    record["passed"] = bool(
        record["comparison"]["both_certified"]
        and record["comparison"]["objectives_match_at_1e_minus_7"]
        and all(item.get("objective_reproduction_error") is not None
                and item["objective_reproduction_error"] <= 1e-7
                for item in record["methods"].values())
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "outputs" / "wzzt_benders_prototype_20260922" / "demo_report.json"))
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--cases", nargs="+", choices=CASE_NAMES, default=list(CASE_NAMES))
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--max-iterations", type=int, default=200)
    args = parser.parse_args()
    before = core_manifest()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "rnj_wzzt_core"))
    from experiments.wzzt_benders_prototype import solve_best_laminar_extension_benders
    from rnj_wzzt.estimation.laminar_l1_milp import (
        solve_best_laminar_extension_l1, solver_diagnostics_prove_optimality,
    )
    records = []
    for name in args.cases:
        case = generate_case(name, args.seed + CASE_NAMES.index(name))
        record = run_case(case, args, solve_best_laminar_extension_benders,
                          solve_best_laminar_extension_l1, solver_diagnostics_prove_optimality)
        records.append(record)
        print(json.dumps({"case": name, "passed": record["passed"],
                          "comparison": record["comparison"]}, ensure_ascii=False), flush=True)
    after = core_manifest()
    manifest_diff = sorted(key for key in set(before) | set(after)
                           if before.get(key) != after.get(key))
    original_key = "rnj_wzzt_core/rnj_wzzt/estimation/laminar_l1_milp.py"
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Small synthetic full-domain one-extension comparison; no full-tree or speedup claim.",
        "candidate_pool": None, "complexity_penalty": 0.0,
        "execution": {"python": sys.version, "executable": sys.executable,
                      "platform": platform.platform(), "numpy": np.__version__,
                      "scipy": scipy.__version__, "bytecode_writes_disabled": sys.dont_write_bytecode},
        "configuration": vars(args) | {"output": str(args.output)},
        "source_sha256": {
            "scripts/run_wzzt_benders_demo.py": file_sha256(Path(__file__)),
            "experiments/wzzt_benders_prototype.py": file_sha256(
                ROOT / "experiments" / "wzzt_benders_prototype.py"),
        },
        "core_integrity": {
            "unchanged": not manifest_diff, "changed_paths": manifest_diff,
            "file_count": len(before), "before": before, "after": after,
            "laminar_matches_preimplementation_sha256": before.get(original_key) == EXPECTED_LAMINAR_SHA256,
            "expected_laminar_sha256": EXPECTED_LAMINAR_SHA256,
        },
        "cases": records,
        "all_passed": bool(records) and all(record["passed"] for record in records) and not manifest_diff,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
