"""Paired one-move searches on the archived first anchored tree of 21 development feeders."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import pickle
import platform
from statistics import mean
from time import perf_counter

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

from ptid.cases import library
from ptid.cases.paths import OUTPUTS, STUDY
from ptid.method.model import Fitter, Model
from ptid.method.search import PRICING_TOL, best_addition
from ptid.method.sets import canonical, children, compatible
from ptid.runs.four_offline import digest, independent_H, metrics, save

SOURCE = STUDY / "experiments/four_feeders/random_time_limits.json"
PUBLISHED = STUDY / "experiments/line_exchange/results.json"
PILOT = ("dev_large_003_middle", "dev_huge_003_middle", "dev_huge_004_end", "dev_huge_005_none")
ARMS = ("initial", "add", "greedy_exchange", "exact_exchange", "add_or_greedy", "add_or_exact")
BUDGETS = (60, 240, 600)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_sha(source):
    return sha256(Path(source).read_text(encoding="utf-8").encode()).hexdigest()


def load_input(name, outputs):
    panel = read(SOURCE)
    original = next(row for row in panel["rows"] if row["case"] == name)
    library.LIBRARY = outputs / "case_library_20261125"
    record = library.open_case(name)
    observed = library.readings(record)
    data = {key: observed[key] for key in ("P", "Q", "Y", "V0sq")}
    hashes = digest(data)
    if hashes != original["observations"]:
        raise ValueError(f"Changed measurement history: {name}")
    settings = panel["new_settings"]
    model = Model(**data, lam=settings["lam"], p_s=settings["p_s"])
    stage = original["new"]["steps"][1]
    tree = canonical(stage["tree"])
    H = Fitter(model, tree).crit(tree) * model.N / 2
    if abs(H - stage["H"]) > 1e-6 or model.N != original["new"]["N"]:
        raise ValueError(f"Changed initial score: {name}")
    base = outputs / "topology_study_random_dev"
    previous_sha = panel["environment"]["calculation_sha256"]
    old = base / "arms" / f"{name}__proposed.json"
    directory = None
    if old.is_file():
        control = read(old)["run_contract"]
        if control["calculation_sha256"] != previous_sha:
            raise ValueError(f"Different original solver contract: {name}")
        sig = fingerprint(dict(st=settings, inputs=hashes, original=control))
        directory = base / "budget_30s" / sig
    if directory is None or not directory.is_dir():
        # The published anchors are sufficient to reproduce this fixed-tree experiment
        # when the historical solver's private subproblem cache is unavailable.
        prior = next(r for r in read(PUBLISHED)["rows"] if r["name"] == name)
        snapshot = prior["source"]
        if (snapshot["observations"] != hashes or snapshot["record_sha256"] != original["record_sha256"]
                or snapshot["original_calculation_sha256"] != previous_sha
                or snapshot["pool_size"] != stage["pool"] or canonical(prior["arms"]["initial"]["tree"]) != tree
                or not set(canonical(prior["anchors"]))<=set(tree)):
            raise ValueError(f"Published anchors or measurements inconsistent: {name}")
        return record, data, model, tree, canonical(prior["anchors"]), dict(
            observations=hashes, record_sha256=original["record_sha256"],domain_sha256=snapshot["domain_sha256"],
            pool_size=stage["pool"],original_H=H,environment=previous_sha)
    matches = {}
    for path in directory.glob("*.pkl"):
        payload = pickle.loads(path.read_bytes())
        if not (isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[1], dict)):
            continue
        info = payload[1]
        domain = info.get("domain")
        if domain is None or not isinstance(domain["require"], list):
            continue
        if info["pool"] != stage["pool"] or canonical(payload[0]) != tree:
            continue
        if abs(info["crit"] * model.N / 2 - H) > 1e-6:
            continue
        key = fingerprint(domain)
        if stage.get("domain_sha256", key) == key:
            matches[key] = domain
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely match original anchored domain for {name}: {list(matches)}")
    domain_sha, domain = matches.popitem()
    if any(len(group) != 1 for group in domain["require"]):
        raise ValueError(f"Unexpected load-alternative constraint: {name}")
    anchors = canonical(group[0] for group in domain["require"])
    if not set(anchors) <= set(tree):
        raise ValueError(f"Infeasible archived anchors: {name}")
    return record, data, model, tree, anchors, dict(observations=hashes, record_sha256=original["record_sha256"],
        domain_sha256=domain_sha, pool_size=len(domain["lines"]), original_H=H,
        environment=previous_sha)


def baseline(model, tree, path):
    """Same one-step add domain as the original method, with fresh enlarged budgets if needed."""
    if path.is_file():
        return read(path)
    scale = model.N / 2
    start_H = Fitter(model, tree).crit(tree) * scale
    trials = []
    best_H, best_line = start_H, None
    for seconds in BUDGETS:
        started = perf_counter()
        line, info = best_addition(model, tree, max_children=9, node_seconds=seconds)
        H = start_H - info["gain"] * scale
        if line is not None and H < best_H:
            best_H, best_line = H, line
        wide = info["wide_nodes"]
        # Narrow nodes are exhaustive. Previous runs used the same model and tree.
        covers = all(n["bound"] is not None and n["bound"] * scale >= best_H - max(
            PRICING_TOL * scale, 1e-4 * abs(best_H)) for n in wide)
        trials.append(dict(limit=seconds, seconds=perf_counter()-started, added=line, H=H, wide_nodes=wide,
                           narrow_candidates=info["evaluated"], within_tolerance=covers))
        if covers:
            break
    result = dict(tree=canonical([*tree, best_line] if best_line is not None else tree),
                  H=best_H, added=best_line, trials=trials,
                  seconds=sum(t["seconds"] for t in trials), certified=trials[-1]["within_tolerance"])
    save(path, result)
    return result


def worker(name, outputs, step):
    record, data, model, tree, anchors, origin = load_input(name, outputs)
    cache_root = outputs / "topology_study_line_exchange"
    key = fingerprint(dict(source=origin, K=9, budgets=BUDGETS))
    directory = cache_root / "cases" / name / key
    directory.mkdir(exist_ok=True, parents=True)
    added = baseline(model, tree, directory / "baseline.json")
    if step == "baseline":
        print(name, "baseline", round(added["H"], 4), flush=True)
        return
    from ptid.runs import line_exchange
    algorithm_digest = source_sha(line_exchange.__file__)
    path = directory / algorithm_digest
    path.mkdir(exist_ok=True)
    results = {}
    for method in ("greedy", "miqp"):
        cache = path / f"{method}.json"
        records = read(cache) if cache.is_file() else []
        grouped = {tuple(r["removed"]): r for r in records}
        def update(position):
            grouped[tuple(position["removed"])] = position
            save(cache, sorted(grouped.values(), key=lambda r: (len(r["removed"]), r["removed"])))
        result = line_exchange.exchange(model, tree, anchors, method, node_seconds=BUDGETS[0],
                                        cached=grouped, on_result=update)
        if method == "miqp":
            result = relax_positions(line_exchange, model, tree, result, path)
        results[method] = result
    receipt = score(name, record, data, model, tree, anchors, origin, added, results, algorithm_digest)
    save(cache_root / "receipts" / f"{name}.json", receipt)
    print(name, {k: round(v["score"]["f1"], 4) for k,v in receipt["arms"].items()}, flush=True)


def relax_positions(kernel, model, tree, result, directory):
    """Continue any unclosed wide position that could beat the current best exchange."""
    relax = directory / "extended.json"
    history = read(relax) if relax.is_file() else {}
    rows = list(result["positions"])
    retries = []
    for limit in BUDGETS[1:]:
        for i, old in enumerate(rows):
            key = tuple(old["removed"])
            label = ",".join(map(str, key))
            trials = history.get(label, {})
            saved = trials.get(str(limit))
            if old["certified"] and saved is None:
                continue
            best_H = min([result["initial_H"], *(r["H"] for r in rows)])
            tolerance = max(PRICING_TOL * model.N / 2, 1e-4 * abs(best_H))
            if saved is None and old.get("bound_H") is not None and old["bound_H"] >= best_H - tolerance:
                continue
            if saved is None:
                saved = kernel.evaluate_position(model, tree, key, "miqp", node_seconds=limit)
                history.setdefault(label, {})[str(limit)] = saved
                save(relax, history)
            attempts = [result["positions"][i], *(r for budget,r in sorted(
                history[label].items(),key=lambda item:int(item[0])) if int(budget)<=limit)]
            strongest = dict(min(attempts, key=lambda r:r["H"]))
            bounds = [r["bound_H"] for r in attempts if r.get("bound_H") is not None]
            strongest["bound_H"] = max(bounds, default=None)
            strongest["certified"] = (strongest["bound_H"] is not None and
                strongest["H"] - strongest["bound_H"] <= max(PRICING_TOL * model.N / 2, 1e-4 * abs(strongest["H"])))
            strongest["attempts"] = [dict(limit=r["node_seconds"], H=r["H"], bound_H=r.get("bound_H"),
                status=r["status"], seconds=r["seconds"]) for r in attempts]
            retries.append(dict(removed=old["removed"], added=saved["added"], H=saved["H"],
                                limit=limit, seconds=saved["seconds"]))
            rows[i] = strongest
    # Keep the short-budget incumbents and timing as a separate record; resumed runs never
    # supply longer-budget rows to an original-budget solve. Use only exact cached domains.
    winner = min(rows, key=lambda r:r["H"], default=None)
    best_tree, H, old_group, new_group = tree, result["initial_H"], None, None
    if winner is not None and H-winner["H"]>PRICING_TOL*model.N/2:
        old_group,new_group=tuple(winner["removed"]),tuple(winner["added"])
        best_tree=canonical([*(x for x in tree if x!=old_group),new_group])
        H=winner["H"]
    bounds=[r.get("bound_H") for r in rows]
    lower=min([result["initial_H"],*bounds]) if all(b is not None for b in bounds) else None
    return dict(result, tree=best_tree, H=H, gain_H=result["initial_H"]-H, removed=old_group, added=new_group,
        short_budget_positions=result["positions"] if retries else None, positions=rows, extensions=retries,
        bound_H=lower,certified=lower is not None and H-lower<=max(PRICING_TOL*model.N/2,1e-4*abs(H)),
        position_seconds=result["position_seconds"]+sum(r["seconds"] for r in retries))


def score(name, record, data, model, tree, anchors, origin, added, results, kernel_sha):
    scale = model.N / 2
    def minimum(first, second):
        return second if first["H"]-second["H"] > PRICING_TOL*scale else first
    initial = dict(tree=tree, H=origin["original_H"])
    arms = dict(initial=initial, add=added, greedy_exchange=results["greedy"], exact_exchange=results["miqp"],
                add_or_greedy=minimum(added, results["greedy"]), add_or_exact=minimum(added, results["miqp"]))
    matched_seconds = min(r["position_seconds"] for r in results.values())
    for label, scan in results.items():
        deadline = matched_seconds + 1e-9
        events = [*(scan.get("short_budget_positions") or scan["positions"]), *scan.get("extensions", [])]
        incumbent = initial
        elapsed = 0.
        count = 0
        for event in events:
            elapsed += event["seconds"]
            if elapsed > deadline:
                break
            count += 1
            if incumbent["H"]-event["H"] > PRICING_TOL*scale:
                removed = tuple(event["removed"])
                incumbent = dict(tree=canonical([*(g for g in tree if g!=removed),event["added"]]),
                                 H=event["H"])
        arms["matched_" + label] = dict(incumbent, completed_positions_or_retries=count)
    truth = library.truth(record)
    parameters = {k:getattr(model,k) for k in ("gamma","sigma2","alpha","N")}
    audited = {}
    checks = {}
    for label, arm in arms.items():
        final = canonical(arm["tree"])
        if not set(anchors)<=set(final) or not all(compatible(a,b) for a,b in combinations(final,2)):
            raise ValueError(f"An infeasible tree was returned: {name} {label}")
        if final not in checks:
            checks[final] = independent_H(data,dict(model=parameters, gate={"load":False}, tree=final))
        H = checks[final]
        if abs(H-arm["H"]) > 1e-5:
            raise ValueError(f"Independent score mismatch: {name}/{label}/{arm['H']}/{H}")
        new, lost = set(final)-set(tree), set(tree)-set(final)
        if len(lost)>1 or len(new)>1 or H>initial["H"]+1e-6:
            raise ValueError(f"Invalid one-step selection in {name}/{label}")
        audited[label] = dict(tree=final,H=arm["H"], independent_H=H, score=metrics(final,truth),
            removed=canonical(lost),added=canonical(new),true_added=canonical(new&truth),
            false_added=canonical(new-truth),true_removed=canonical(lost&truth),false_removed=canonical(lost-truth),
            action="exchange" if lost else "add" if new else "keep")
        if label.startswith("matched_"):
            audited[label]["completed_positions_or_retries"]=arm["completed_positions_or_retries"]
    scans = {m:{k:v for k,v in z.items() if k!="tree"} for m,z in results.items()}
    return dict(name=name, meters=record["n"], cohort="diagnostic_four" if name in PILOT else "remaining_seventeen",
                source=origin, kernel_sha256=kernel_sha, anchors=anchors, arms=audited, baseline=added, exchange=scans,
                matched_position_seconds=matched_seconds)


def collect(names, outputs, dest):
    base = outputs / "topology_study_line_exchange" / "receipts"
    rows = [read(base / f"{name}.json") for name in names]
    from ptid.runs import line_exchange
    expected = source_sha(line_exchange.__file__)
    if any(r["kernel_sha256"]!=expected or r["name"]!=n for n,r in zip(names,rows)):
        raise ValueError("A receipt has a different configuration or input name")
    summary = {}
    for cohort, sample in (("all",rows), ("diagnostic_four",[r for r in rows if r["name"] in PILOT]),
                           ("remaining_seventeen",[r for r in rows if r["name"] not in PILOT])):
        if not sample:
            continue
        summary[cohort] = {}
        for arm in (*ARMS,"matched_greedy","matched_miqp"):
            vals=[r["arms"][arm] for r in sample]
            summary[cohort][arm]=dict(n=len(vals),mean_F1=mean(a["score"]["f1"] for a in vals),
                true_added=sum(len(a["true_added"]) for a in vals),false_added=sum(len(a["false_added"]) for a in vals),
                true_removed=sum(len(a["true_removed"]) for a in vals),false_removed=sum(len(a["false_removed"]) for a in vals))
    import gurobipy as gp
    import numpy as np
    import scipy
    environment = dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                       gurobi=list(gp.gurobi.version()))
    previous = read(SOURCE)["environment"]
    if any(previous[k] != v for k,v in environment.items()):
        raise ValueError(f"Numerical environment differs from the archived 30-second experiment: {environment}")
    for row in rows:
        code_sha = row["source"].pop("environment")
        if code_sha != previous["calculation_sha256"]:
            raise ValueError(f"Previous solver source mismatch on {row['name']}")
        row["source"]["original_calculation_sha256"] = code_sha
    report=dict(protocol=dict(origin="First anchored Tree MIQP output in the published 30-second random comparison",
            observations="Same eight-day metered P, Q, Y and root squared voltage; identical hard anchors",
            selection="At most one change from the common original tree; also compare addition-or-exchange on common base",
            cohorts="The four diagnostic cases were chosen from retrospective structural checks. All 21 including the other seventeen are development feeders.",
            timing="Single-thread BLAS and Gurobi. Matched arms take completed position returns in their original order up to the shorter cumulative position-compute time, excluding shared initialization; no interim result is credited for an unfinished position.",
            stopping="Per-position enlargement applies when its valid bound still allows a better exchange. A missed best-group certificate is relative to a restricted add or exchange domain.",
            node_budgets_s=BUDGETS, narrow_node_max=9, seed=7, pilot=PILOT,
            evaluation="Truth read for reporting after all arms have selected; an independent explicit BVLS refit verifies every returned tree",
            environment=environment, solver_threads=1, blas_threads=1,
            previous_report_sha256=sha256(SOURCE.read_bytes()).hexdigest(), kernel_sha256=expected),
            summary=summary)
    # One JSON line per feeder keeps the full receipts small and the diff reviewable.
    dump=lambda v: json.dumps(v,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    dest.parent.mkdir(exist_ok=True,parents=True)
    tmp=dest.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(dump(report)[:-1]+',"rows":[\n'+",\n".join(dump(r) for r in rows)+"\n]}\n",encoding="utf-8")
    tmp.replace(dest)


def verify(path, outputs, names):
    """Check saved paired results against the original measurements without rerunning the search."""
    rows = {r["name"]:r for r in read(path)["rows"]}
    worst = 0.
    for name in names:
        row = rows[name]
        record, data, model, tree, anchors, origin = load_input(name, outputs)
        truth = library.truth(record)
        if (row["source"]["observations"] != origin["observations"] or canonical(row["anchors"]) != anchors
                or canonical(row["arms"]["initial"]["tree"])!=tree):
            raise ValueError(f"Archived initial conditions differ: {name}")
        removable = set(tree)-set(anchors)
        for method in ("greedy","miqp"):
            positions=row["exchange"][method]["positions"]
            if {tuple(p["removed"]) for p in positions}!=removable or len(positions)!=len(removable):
                raise ValueError(f"A contraction is missing or repeated: {name}/{method}")
            for p in positions:
                remaining=set(tree)-{tuple(p["removed"])}
                kids=children(remaining,model.n)[tuple(p["parent"])]
                selected=set(p["added"])
                parts=[g for g in kids if set(g)<=selected]
                if (canonical(p["children"])!=kids or p["width"]!=len(kids)
                        or not 2<=len(parts)<len(kids) or set(i for g in parts for i in g)!=selected
                        or (p["bound_H"] is not None and p["bound_H"]>p["H"]+1e-5)):
                    raise ValueError(f"Invalid fixed-position receipt: {name}/{method}")
        checks={}
        parameters={k:getattr(model,k) for k in ("gamma","sigma2","alpha","N")}
        for label,arm in row["arms"].items():
            final=canonical(arm["tree"])
            if final not in checks:
                checks[final]=independent_H(data,dict(model=parameters,gate={"load":False},tree=final))
            delta=abs(checks[final]-arm["H"])
            worst=max(worst,delta)
            if (delta>1e-5 or not set(anchors)<=set(final) or metrics(final,truth)!=arm["score"]
                    or not all(compatible(a,b) for a,b in combinations(final,2))):
                raise ValueError(f"Final tree, objective or accuracy mismatch: {name}/{label}")
    print(f"{len(names)} feeders verified, maximum independent H difference {worst:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs",type=Path,default=OUTPUTS)
    parser.add_argument("--cases",nargs="+",help="Verified feeder names, or pilot / other / all")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--stage",choices=("run","baseline","collect","verify"),default="run")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    names = [r["case"] for r in read(SOURCE)["rows"]]
    choice=args.cases or ["all"]
    if choice==["pilot"]:
        names=list(PILOT)
    elif choice==["other"]:
        names=[n for n in names if n not in PILOT]
    elif choice!=["all"]:
        if not set(choice)<=set(names) or len(set(choice))!=len(choice):
            parser.error("Choose unique verified names or pilot, other, all")
        names=choice
    if args.workers<1:
        parser.error("--workers must be positive")
    outputs=args.outputs.resolve()
    if args.stage=="verify":
        verify(args.out or PUBLISHED,outputs,names)
        return 0
    target=args.out or outputs/"topology_study_line_exchange"/"comparison.json"
    if args.stage!="collect":
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            jobs=[ex.submit(worker,n,outputs,args.stage) for n in names]
            for job in as_completed(jobs):
                job.result()
    if args.stage!="baseline":
        collect(names,outputs,target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
