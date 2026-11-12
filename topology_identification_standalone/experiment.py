"""Small command-line experiments using only this directory's code and data.

The algorithms receive training and validation observations. Physical truth
and test observations are opened only after every method for an input has
finished fitting. Failures retain their place in the planned denominator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import csv
import hashlib
from importlib import metadata
from itertools import permutations
import json
from pathlib import Path
import re
import sys
from time import perf_counter
import traceback

import numpy as np

if __package__:
    from . import algorithm, data
else:
    import algorithm
    import data


ROOT = Path(__file__).resolve().parent
FACTORIAL = ("000", "100", "010", "001", "110", "101", "011", "111")
ALL_METHODS = ("rnj", "nj", *FACTORIAL)
ALIASES = {"pure_wzzt": "000", "rnj_wzzt": "100", "rte": "111"}
MAIN_METHODS = ("rnj", "nj", "000", "100", "111")
TERMINAL_STATUSES = {"complete", "failed", "budget_exceeded"}


def json_value(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, default=json_value, ensure_ascii=False,
                                    indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    text = json.dumps(value, default=json_value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def environment():
    packages = {}
    for name in ("numpy", "scipy", "gurobipy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return dict(python=sys.version, packages=packages)


def methods_from_names(names):
    methods = []
    for name in names:
        for item in name.split(","):
            item = ALIASES.get(item.lower(), item.lower())
            if item not in ALL_METHODS:
                raise ValueError(f"Unknown method {item!r}; use {', '.join(ALL_METHODS)}")
            if item not in methods:
                methods.append(item)
    if not methods:
        raise ValueError("Choose at least one method")
    return methods


def response_matrices(point, n):
    if is_dataclass(point):
        point = asdict(point)
    z = np.asarray([[i in C for i in range(n)] for C in point["supports"]], float)
    return (z.T*np.asarray(point["r"]))@z, (z.T*np.asarray(point["x"]))@z


def score_observations(observed, point):
    R, X = response_matrices(point, observed.n)
    residual = observed.y-observed.p@R.T-observed.q@X.T
    return dict(mae=float(np.mean(abs(residual))), mse=float(np.mean(residual**2)))


def topology_scores(point, truth, n):
    """Clade scores exclude the singleton and all-terminal response bases."""
    predicted = {tuple(sorted(C)) for C in point["supports"] if 1 < len(C) < n}
    correct = {tuple(sorted(C)) for C in truth if 1 < len(C) < n}
    tp, fp, fn = len(predicted & correct), len(predicted-correct), len(correct-predicted)
    return dict(F1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 1., exact=predicted == correct,
                TP=tp, FP=fp, FN=fn, predicted_clades=len(predicted), true_clades=len(correct))


def selected_point(result):
    selected = result["selected"]
    if is_dataclass(selected):
        selected = asdict(selected)
    return selected["point"] if "point" in selected else selected


def linear_toy(seed=7, points=96):
    """A six-terminal linear-response test, not a physical paper-network run."""
    n = 6
    family = [*((i,) for i in range(n)), tuple(range(n)), (0, 1), (2, 3), (0, 1, 2, 3)]
    r = np.asarray([.02]*n+[.015, .055, .065, .09])
    point = dict(supports=family, r=r.tolist(), x=(.6*r).tolist())
    R, X = response_matrices(point, n)
    splits = []
    for split, count in enumerate((points, 3*points, points)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, split]))
        p, q = rng.uniform(.03, .12, (count, n)), rng.uniform(.01, .06, (count, n))
        y = p@R.T+q@X.T+rng.normal(0., 2e-5, (count, n))
        splits.append(data.Observation(p, q, y, tuple(range(101, 107))))
    return (*splits, family, dict(case="linear_toy6", seed=seed, points=points,
                generator="linear_response", additive_y_sigma=2e-5,
                physical_AC_case=False, historical_scores_reproduced=False))


def local_file(path):
    """Bundled catalog paths must stay inside the independently copied folder."""
    resolved = (ROOT/path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("A bundled data path escapes the standalone folder")
    return resolved


def select_jobs(catalog, requested):
    requested = [part for item in requested for part in item.split(",")]
    if requested == ["all"]:
        return list(catalog)
    if requested == ["pilot"]:
        selected = [job for job, meta in catalog.items()
                    if meta["seed_index"] == 0 and meta["voltage_sigma"] == .0005]
        if len(selected) != 4:
            raise ValueError("The bundled pilot must contain one fixed input per network")
        return selected
    unknown = set(requested)-set(catalog)
    if unknown:
        raise ValueError("Unknown bundled job(s): "+", ".join(sorted(unknown)))
    return list(dict.fromkeys(requested))


def experiment_config(args):
    short_example = args.command == "demo" or (args.command == "ablation" and args.dataset == "toy")
    seconds = float(args.time_limit if args.time_limit is not None else (30. if short_example else 14400.))
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("--time-limit must be positive and finite")
    return algorithm.SearchConfig(alpha=args.alpha, block_limit=args.block_limit, total_seconds=seconds,
        ambiguity_kappa=getattr(args, "ambiguity_kappa", .003),
        per_mip_seconds=min(3600., seconds), qp_seconds=min(300., max(1., seconds/4)),
        output_reserve_seconds=min(600., seconds/10))


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def ablation_effects(rows):
    """Paired module effects and all six addition orders from the same eight cells.

    The addition orders are ways to read the factorial table, not six new
    execution pipelines. A negative contribution remains a negative result.
    """
    effects, orders, shapley = [], [], []
    jobs = list(dict.fromkeys(row["job"] for row in rows))
    for job in jobs:
        bank = {row["method"]: row for row in rows if row["job"] == job}
        if not all(method in bank for method in FACTORIAL):
            continue
        paired = {}
        for bit, module in enumerate("RTE"):
            contributions = []
            for off in FACTORIAL:
                if off[bit] != "0":
                    continue
                on = off[:bit]+"1"+off[bit+1:]
                old, new = bank[off], bank[on]
                valid = old["status"] == new["status"] == "complete"
                effect = dict(job=job, module=module, off=off, on=on,
                              status="paired" if valid else "incomplete_pair",
                              delta_F1=new["F1"]-old["F1"] if valid else None,
                              delta_exact=int(new["exact"])-int(old["exact"]) if valid else None,
                              delta_seconds=new["total_seconds"]-old["total_seconds"] if valid else None)
                effects.append(effect); paired[(off, on)] = effect
                if valid:
                    # Three players: endpoint contexts have weight 1/3;
                    # each one-other-module context has weight 1/6.
                    weight = 1/6 if off.count("1") == 1 else 1/3
                    contributions.append((weight, effect))
            shapley.append(dict(job=job, module=module, planned_pairs=4, available_pairs=len(contributions),
                F1=sum(w*e["delta_F1"] for w, e in contributions) if len(contributions) == 4 else None,
                exact=sum(w*e["delta_exact"] for w, e in contributions) if len(contributions) == 4 else None,
                seconds=sum(w*e["delta_seconds"] for w, e in contributions) if len(contributions) == 4 else None))
        for order in permutations("RTE"):
            off = "000"
            for step, module in enumerate(order, 1):
                bit = "RTE".index(module)
                on = off[:bit]+"1"+off[bit+1:]
                orders.append(dict(order="".join(order), step=step, **paired[(off, on)]))
                off = on
    return effects, orders, shapley


def ablation_interactions(rows):
    """Second- and third-order differences, preserving incomplete cell counts."""
    interactions = []
    for job in dict.fromkeys(row["job"] for row in rows):
        bank = {row["method"]: row for row in rows if row["job"] == job}
        if not all(method in bank for method in FACTORIAL):
            continue
        contrasts = []
        for pair in ((0, 1), (0, 2), (1, 2)):
            remaining = next(i for i in range(3) if i not in pair)
            for context in (0, 1):
                cells = []
                for left, right, coefficient in ((0, 0, 1), (1, 0, -1), (0, 1, -1), (1, 1, 1)):
                    bits = [0, 0, 0]
                    bits[remaining], bits[pair[0]], bits[pair[1]] = context, left, right
                    cells.append(("".join(map(str, bits)), coefficient))
                contrasts.append(("".join("RTE"[i] for i in pair), "RTE"[remaining], context, cells))
        third = [(method, (-1)**(3-method.count("1"))) for method in FACTORIAL]
        contrasts.append(("RTE", None, None, third))
        for name, condition, value, cells in contrasts:
            available = sum(bank[method]["status"] == "complete" for method, _ in cells)
            paired = available == len(cells)
            differences = {}
            for source, target in (("F1", "delta_F1"), ("exact", "delta_exact"), ("total_seconds", "delta_seconds")):
                differences[target] = sum(coefficient*bank[method][source] for method, coefficient in cells) if paired else None
            interactions.append(dict(job=job, interaction=name, condition_module=condition, condition_value=value,
                cells=",".join(method for method, _ in cells), planned_cells=len(cells), available_cells=available,
                status="paired" if paired else "incomplete_cells", **differences))
    return interactions


def summarize(output, plan):
    """One row per planned method/input, including pending and failed rows."""
    rows = []
    for job in plan["jobs"]:
        for method in plan["methods"]:
            file = output/"runs"/safe_name(job)/(method+".json")
            record = read_json(file) if file.exists() else dict(status="pending")
            row = dict(job=job, method=method, status=record["status"], F1=None, exact=None,
                       TP=None, FP=None, FN=None, fresh_seconds=record.get("fresh_seconds"),
                       common_seconds=record.get("common_seconds"), total_seconds=record.get("total_seconds"))
            row.update(record.get("metrics", {}))
            rows.append(row)
    write_csv(output/"results.csv", rows)
    effects, orders, shapley = ablation_effects(rows)
    write_csv(output/"ablation_effects.csv", effects)
    write_csv(output/"module_addition_orders.csv", orders)
    write_csv(output/"shapley.csv", shapley)
    write_csv(output/"interactions.csv", ablation_interactions(rows))
    methods = []
    for method in plan["methods"]:
        group = [row for row in rows if row["method"] == method]
        valid = [row for row in group if row["status"] == "complete"]
        exact = sum(bool(row["exact"]) for row in valid)
        methods.append(dict(method=method, planned=len(group), completed=len(valid),
            failed=sum(row["status"] == "failed" for row in group),
            budget_exceeded=sum(row["status"] == "budget_exceeded" for row in group),
            pending=sum(row["status"] not in TERMINAL_STATUSES for row in group),
            F1_completed_mean=float(np.mean([row["F1"] for row in valid])) if valid else None,
            exact=exact, exact_rate_planned=exact/len(group),
            total_seconds_completed=sum(row["total_seconds"] for row in valid)))
    summary = dict(planned=len(rows), completed=sum(row["status"] == "complete" for row in rows),
                   failed=sum(row["status"] == "failed" for row in rows),
                   budget_exceeded=sum(row["status"] == "budget_exceeded" for row in rows),
                   pending=sum(row["status"] not in TERMINAL_STATUSES for row in rows), methods=methods)
    write_json(output/"summary.json", summary)
    return summary


def run_experiment(args):
    config = experiment_config(args)
    names = args.methods or (ALL_METHODS if args.command == "ablation" else MAIN_METHODS)
    methods = methods_from_names(names)
    bundled = args.command == "benchmark" or (args.command == "ablation" and args.dataset == "bundled")
    if bundled and not data.verify_bundled_data()["passed"]:
        raise ValueError("The bundled observations or case files differ from their manifest")
    catalog = data.load_catalog() if bundled else {}
    jobs = select_jobs(catalog, args.jobs) if bundled else [f"{args.case}_seed{args.seed}_N{args.points}"]
    output = Path(args.output).resolve() if args.output else ROOT/"results"/args.command
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name: sha256(p) for p in sorted(ROOT.glob("*.py"))}
    inputs = {job: {s: sha256(local_file(catalog[job][s])) for s in ("train", "validation", "test")}
              for job in jobs} if bundled else {}
    plan = dict(methods=methods, jobs=jobs, config=asdict(config), source_sha256=source_hashes,
                bundled_manifest_sha256=sha256(ROOT/"data/manifest.json") if bundled else None,
                environment=environment(),
                input_sha256=inputs, bundled_observations=bundled, seed=args.seed, case=args.case,
                points=args.points, shared_reference="one training-only reference per input, charged to each method",
                information_boundary="fit receives train/validation only; test/truth used for post-fit metrics",
                result_scope="readable standalone reconstruction; not a reproduction of historical reported scores")
    signature = digest(plan)
    if (output/"plan.json").exists() and digest(read_json(output/"plan.json")) != signature:
        raise ValueError("Output already contains a different frozen plan; choose another --output")
    write_json(output/"plan.json", plan)
    for job in jobs:
        files = [output/"runs"/safe_name(job)/(method+".json") for method in methods]
        for file in files:
            if file.exists() and read_json(file).get("plan_sha256") != signature:
                raise ValueError("A saved method result belongs to another plan")
        if all(file.exists() and read_json(file)["status"] in TERMINAL_STATUSES for file in files):
            continue
        metadata = catalog.get(job, {})
        synthetic = None
        if bundled:
            train, validation = (data.load_observations(local_file(metadata[s])) for s in ("train", "validation"))
        else:
            synthetic = linear_toy(args.seed, args.points) if args.case == "toy6" else data.generate_synthetic(
                args.case, seed=args.seed, points=args.points)
            train, validation = synthetic[:2]
        common_started = perf_counter()
        try:
            common = algorithm.prepare_common(train, config)
            common_seconds = perf_counter()-common_started
            preparation_error = None
        except Exception:
            common, common_seconds = None, perf_counter()-common_started
            preparation_error = traceback.format_exc()
        for method in methods:
            file = output/"runs"/safe_name(job)/(method+".json")
            if file.exists():
                continue
            started = perf_counter()
            try:
                if preparation_error:
                    raise RuntimeError("Shared training preprocessing failed:\n"+preparation_error)
                result = algorithm.fit(train, validation, method=method, config=config, common=common)
                if {p.name: sha256(p) for p in sorted(ROOT.glob("*.py"))} != source_hashes:
                    raise RuntimeError("The standalone source changed while this experiment was running")
                point = selected_point(result)
                score_observations(validation, point)
                record = dict(job=job, method=method, status="fitted", result=result,
                              plan_sha256=signature, fresh_seconds=perf_counter()-started,
                              common_seconds=common_seconds)
                record["total_seconds"] = record["fresh_seconds"]+common_seconds
                exceeded = bool(result.get("total_cap_exceeded")) or record["total_seconds"] > config.total_seconds+.001
                record["budget_exceeded"] = exceeded
                if exceeded:
                    record["status"] = "budget_exceeded"
                write_json(file, record)
            except Exception:
                record = dict(job=job, method=method, status="failed", error=traceback.format_exc(),
                    plan_sha256=signature, fresh_seconds=perf_counter()-started, common_seconds=common_seconds)
                record["total_seconds"] = record["fresh_seconds"]+common_seconds
                write_json(file, record)
            print(job, method, record["status"], flush=True)
        # No truth or independent test observations enter prepare_common/fit.
        if bundled:
            test = data.load_observations(local_file(metadata["test"]))
            truth = data.case_truth(metadata["case"], train.labels)
        else:
            test, truth = synthetic[2:4]
        for method in methods:
            file = output/"runs"/safe_name(job)/(method+".json")
            record = read_json(file)
            if record["status"] not in ("fitted", "complete"):
                continue
            point = selected_point(record["result"])
            metrics = topology_scores(point, truth, train.n)
            for split, observed in (("train", train), ("validation", validation), ("test", test)):
                metrics.update({split+"_"+key: value for key, value in score_observations(observed, point).items()})
            record.update(status="complete", metrics=metrics, truth_test_used_only_for_postfit_scoring=True)
            write_json(file, record)
        summarize(output, plan)
    summary = summarize(output, plan)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "benchmark", "ablation"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--methods", nargs="+", help="rnj nj 000..111; aliases: pure_wzzt rnj_wzzt rte")
    parser.add_argument("--jobs", nargs="+", default=["pilot"], help="bundled job IDs, pilot (4) or all (48)")
    parser.add_argument("--dataset", choices=("toy", "bundled"), default="toy", help="ablation input source")
    parser.add_argument("--case", default="toy6", help="demo/toy ablation: toy6 or a bundled physical case")
    parser.add_argument("--points", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--time-limit", type=float, help="seconds per method/input; default demo=30, bundled=14400")
    parser.add_argument("--alpha", type=float, default=.003)
    parser.add_argument("--block-limit", type=int, default=5)
    parser.add_argument("--ambiguity-kappa", type=float, default=.003,
                        help="relative selection-MSE ambiguity threshold; larger retains more pairs for DR")
    args = parser.parse_args(argv)
    if args.points < 8:
        parser.error("--points must be at least eight")
    try:
        result = run_experiment(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 1 if result["failed"] or result["pending"] or result["budget_exceeded"] else 0
