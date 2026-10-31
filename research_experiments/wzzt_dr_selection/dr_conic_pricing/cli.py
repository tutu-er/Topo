"""JSON -> SOCP/MISOCP -> independent response qualification, no topology claims."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import traceback

from .examples import synthetic_problem, resource_day_problem
from .model import PriceProblem
from .sampling import validate_frozen_plan
from .solver import solve_price


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":"))+"\n").encode("utf-8")


def write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix+".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    folder = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.glob("*.py"))}


def environment():
    versions = {}
    for name in ("numpy", "scipy", "gurobipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.system(), packages=versions)


def run(args):
    raw = Path(args.problem).read_bytes()
    problem = PriceProblem.from_dict(json.loads(raw.decode("utf-8-sig")))
    out = Path(args.output)
    # A fresh directory prevents accidental replacement/restart of ended runs.
    out.mkdir(parents=True, exist_ok=False)
    allowed = None if args.allowed is None else [int(v) for v in args.allowed.split(",") if v]
    protocol = dict(status="frozen", created_utc=datetime.now(timezone.utc).isoformat(),
                    input_sha256=hashlib.sha256(raw).hexdigest(),
                    canonical_problem_sha256=hashlib.sha256(canonical_bytes(problem.to_dict())).hexdigest(),
                    source_sha256=source_hashes(), environment=environment(),
                    mode=args.mode, allowed_contracts=allowed, time_limit=args.time_limit,
                    relative_gap=args.relative_gap, threads=args.threads, solver_seed=args.solver_seed,
                    validation_seed=args.validation_seed, validation_samples=args.samples,
                    probability_scope="at-least-confidence excitation; not complete topology recovery")
    write_json(out/"protocol.json", protocol)
    write_json(out/"problem.json", problem.to_dict())
    try:
        result = solve_price(problem, mode=args.mode, allowed_contracts=allowed,
                             time_limit=args.time_limit, relative_gap=args.relative_gap,
                             threads=args.threads, seed=args.solver_seed)
        write_json(out/"optimization.json", result)
        if result["feasible_incumbent"] and args.samples:
            validation = validate_frozen_plan(
                problem, result["amplitudes"], threshold=result["achieved_benefit_lower_bound"],
                samples=args.samples, seed=args.validation_seed,
                numerical_tolerance=result.get("numerical_check_tolerance", 1e-7))
            write_json(out/"validation.json", validation)
        write_json(out/"ended.json", dict(status="complete", optimization_status=result["status"],
                   source_sha256=source_hashes(), topology_identification_executed=False))
        print(json.dumps(dict(output=str(out), status=result["status"],
                              achieved_benefit_lower_bound=result["achieved_benefit_lower_bound"]),
                         ensure_ascii=False))
        return 0 if result["feasible_incumbent"] else 2
    except Exception as error:
        write_json(out/"ended.json", dict(status="failed", error_type=type(error).__name__,
                   message=str(error), traceback=traceback.format_exc(),
                   source_sha256=source_hashes(), topology_identification_executed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    example = commands.add_parser("example", help="write explicit synthetic JSON (no empirical calibration)")
    example.add_argument("--output", required=True)
    example.add_argument("--resource-day", action="store_true", help="96-slot DR+EV input using existing resource declarations")
    solve = commands.add_parser("solve", help="solve a frozen JSON and qualify independent responses")
    solve.add_argument("--problem", required=True)
    solve.add_argument("--output", required=True, help="must not already exist")
    solve.add_argument("--mode", required=True, choices=("socp", "misocp"))
    solve.add_argument("--allowed", help="comma-separated contract indices for a fixed-type SOCP; empty string=none")
    solve.add_argument("--time-limit", type=float, default=60.)
    solve.add_argument("--relative-gap", type=float, default=1e-6)
    solve.add_argument("--threads", type=int, default=1)
    solve.add_argument("--solver-seed", type=int, default=0)
    solve.add_argument("--samples", type=int, default=20000)
    solve.add_argument("--validation-seed", type=int, default=41031)
    args = parser.parse_args(argv)
    if args.command == "example":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            problem = resource_day_problem() if args.resource_day else synthetic_problem()
            json.dump(problem.to_dict(), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        return 0
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
