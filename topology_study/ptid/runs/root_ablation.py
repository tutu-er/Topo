"""Trace recovery of near-root lines on the original passive benchmark."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from time import perf_counter
from unittest.mock import patch


PUBLISHED = Path(__file__).resolve().parents[2] / "experiments" / "root_ablation" / "results.json"


def physical_edges(rec):
    """Map a retained meter grouping to each physical edge that supports it."""
    from ptid.cases import physics
    order, parent, incoming = physics.radial_order(rec["network"])
    depth = {order[0]: 0}
    below = {node: set() for node in order}
    for i, bus in enumerate(rec["labels"]):
        below[bus].add(i)
    for bus in order[1:]:
        depth[bus] = depth[parent[bus]] + 1
    for bus in reversed(order[1:]):
        below[parent[bus]].update(below[bus])
    degree = Counter(parent[b] for b in order[1:])
    truth = {}
    for child in order[1:]:
        g = tuple(sorted(below[child]))
        r = incoming[child]["r_ohm"]
        if 1 < len(g) < len(rec["labels"]) and r >= .003:
            truth.setdefault(g, []).append(dict(parent_bus=parent[child], child_bus=child,
                                                physical_depth=depth[child], parent_children=degree[parent[child]],
                                                resistance_ohm=float(r)))
    return truth


def survey(root, arms):
    """Use only archived observations and estimates; reveal truth when scoring."""
    from ptid.cases import library
    from ptid.runs.four_feeders import CASES
    from ptid.runs.four_records import read_published
    from ptid.runs.random_dev_compare import selected
    from ptid.runs.four_offline import digest
    from rnj_wzzt_dr.physical.backend import load_case

    rows, candidates = [], []
    library.LIBRARY = root

    def collect(name, record, truth, nj, rnj, proposed, interim, status, series):
        geometry = physical_edges(record)
        if set(geometry) != set(truth):
            raise ValueError(f"Physical edges differ from the reference tree in {name}")
        n = record["n"]
        counter = {}
        for metric in ("first_two_physical_edges", "first_two_levels", "majority"):
            counter[metric] = Counter()
        for g in sorted(truth, key=lambda x: (len(x),x)):
            level = 1 + sum(set(g)<set(h) for h in truth)
            scopes = [s for s, active in (("first_two_physical_edges", any(
                edge["physical_depth"] <= 2 for edge in geometry[g])), ("first_two_levels", level<=2),
                ("majority", 2*len(g)>n)) if active]
            for scope in scopes:
                c = counter[scope]
                c["truth"] += 1
                c["miss_nj"] += g not in nj
                c["miss_rnj"] += g not in rnj
                if g not in nj or g not in rnj:
                    c["miss_either"] += 1
                    if proposed is not None:
                        c["either_miss_with_final"] += 1
                        c["either_miss_recovered_final"] += g in proposed
                if g not in nj and g not in rnj:
                    c["miss_both"] += 1
                    if proposed is not None:
                        c["both_miss_with_final"] += 1
                        c["recovered_final"] += g in proposed
                    elif interim is not None:
                        c["both_miss_with_checkpoint_only"] += 1
                        c["present_at_last_checkpoint_only"] += g in interim
            if scopes and g not in nj and g not in rnj and (
                (proposed is not None and g in proposed) or (interim is not None and g in interim)):
                candidates.append(dict(case=name, grouping=list(g), level=level, meters=n,
                                       scopes=scopes, physical_edges=geometry[g], original_status=status))
        rows.append(dict(case=name, series=series, status=status, meters=n, counts=counter))

    def groups(x):
        return set(map(tuple, x))

    for name in CASES:
        def archived(method):
            return read_published(f"offline_{name}_{method}.json")
        proposed = archived("blind")
        nj, rnj = archived("nj"), archived("rnj")
        if not nj["input_sha256"] == rnj["input_sha256"] == proposed["input_sha256"]:
            raise ValueError(f"Input mismatch in {name}")
        network = load_case(name)
        labels = sorted(b["bus_id"] for b in network["buses"] if b["is_terminal"])
        collect(name, dict(network=network, labels=labels, n=len(labels)), groups(proposed["truth"]),
                groups(nj["tree"]), groups(rnj["tree"]), groups(proposed["tree"]), None,"ok","named")
    for name in selected():
        if not (root/"dev"/(name+".pkl")).exists():
            rows.append(dict(case=name, series="random", status="unverified_input"))
            continue
        record = library.open_case(name)
        history = library.readings(record)
        sample = digest({k:history[k] for k in ("P","Q","Y","V0sq")})
        def receipt(method):
            r = json.loads((arms/(name+"__"+method+".json")).read_text(encoding="utf-8"))
            if r["case"] != name or r["method"] != method:
                raise ValueError(f"Wrong arm receipt for {name}/{method}")
            if r["status"] == "ok" and r["history_sha256"] != sample:
                raise ValueError(f"Input mismatch in {name}/{method}")
            return r
        nj, rnj, proposed = map(receipt, ("nj","rnj","proposed"))
        checkpoints = proposed.get("completed_restricted_solves") or []
        collect(name, record, library.truth(record), groups(nj["tree"]), groups(rnj["tree"]),
                groups(proposed["tree"]) if proposed["status"] == "ok" else None,
                groups(checkpoints[-1]["tree"]) if checkpoints else None, proposed["status"], "random")
    return rows, candidates

def as_tree(lines):
    return {tuple(g) for g in lines}


def locate(groups, lines, n):
    """For a missing line, report its smallest enclosing node in the estimated tree."""
    from ptid.method.sets import children
    family = children(lines, n)
    result = []
    for group in groups:
        key = set(group)
        node = min((k for k in family if key <= set(k)), key=len)
        kids = family[node]
        parts = [k for k in kids if set(k) <= key]
        result.append(dict(present=tuple(group) in family, enclosing_meters=len(node), children=len(kids),
                           reachable_by_one_add=key != set(node) and set().union(*map(set, parts)) == key))
    return result


def trace(name, groups, root, arms):
    """Observe production solver boundaries without changing candidates, starts or limits."""
    from ptid.cases import library
    from ptid.runs.four_offline import digest, settings
    from ptid.runs.random_dev_compare import cached_resistance, contract

    module = import_module("ptid.method.identify")
    search = import_module("ptid.method.search")
    library.LIBRARY = root
    rec = library.open_case(name)
    n = rec["n"]
    saved = json.loads((arms/f"{name}__proposed.json").read_text(encoding="utf-8"))
    if saved["run_contract"] != contract(name, "proposed", 120):
        raise ValueError(f"Changed source or settings for {name}")
    h = library.readings(rec)
    data = {k: h[k] for k in ("P","Q","Y","V0sq")}
    st = settings()
    records = []
    start = perf_counter()

    def stage(kind, tree, **details):
        row = dict(stage=kind, targets=locate(groups, tree, n), **details)
        records.append(row)

    original_solve = module.solve
    def solve(model, pool, starts, seconds, st, require=()):
        tree, result = original_solve(model, pool, starts, seconds, st, require)
        index = len([r for r in records if r["stage"] == "restricted_output"])
        scale = model.N / 2
        records.append(dict(stage="domain", solve=index, groups=[tuple(g) in as_tree(pool) for g in groups],
                            number_of_candidates=len(result["domain"]["lines"])))
        stage("warm", result["warm_tree"], solve=index, H=result["warm_crit"]*scale)
        if result["solver_tree"] is not None:
            stage("raw_tree_miqp", result["solver_tree"], solve=index, H=result["solver_crit"]*scale)
        stage("restricted_output", tree, solve=index, H=result["crit"]*scale, status=result["status"],
              lower_H=None if result["bound"] is None else result["bound"]*scale, gap_closed=result["gap_closed"])
        return tree, result

    original_build = module.builder_trees
    def build(*args):
        from ptid.method.builders import NAMES
        starts = original_build(*args)
        records.append(dict(stage="nine_initial_builders", builders={
            method:[tuple(g) in as_tree(t) for g in groups] for method,t in zip(NAMES,starts)}))
        return starts

    original_node = search.node_miqp
    def node(model, tree, kids, seconds, *args, **kwargs):
        out = original_node(model, tree, kids, seconds, *args, **kwargs)
        selected = out[0]
        records.append(dict(stage="wide_node_miqp", width=len(kids),
                            groups=[selected is not None and tuple(g)==tuple(selected) for g in groups]))
        return out

    original_price = module.price
    def price(model, tree, universe, require, seconds, st):
        ans, info = original_price(model, tree, universe, require, seconds, st)
        stage("after_node_search", ans, added_as_target=[tuple(g) in as_tree(info["added"]) for g in groups])
        return ans, info

    with ExitStack() as stack:
        stack.enter_context(cached_resistance(arms, name))
        for owner, attribute, replacement in ((module,"solve",solve), (module,"builder_trees",build),
                                               (search,"node_miqp",node), (module,"price",price)):
            stack.enter_context(patch.object(owner,attribute,replacement))
        answer = module.identify(**data, st=st)
    stage("replay_final", answer["tree"], H=answer["model"]["crit"]*answer["model"]["N"]/2)
    baseline = json.loads((arms/f"{name}__nj.json").read_text(encoding="utf-8"))
    if digest(data) != baseline["history_sha256"]:
        raise ValueError(f"Historical input mismatch: {name}")
    return dict(case=name, targets=groups, settings=asdict(st), input_sha256=digest(data),
                input_record_sha256=saved["run_contract"]["record_sha256"], seconds=perf_counter()-start,
                original_status=saved["status"], original_final_matches=(
                    as_tree(answer["tree"]) == as_tree(saved["tree"]) if saved["status"] == "ok" else None),
                numpy=saved["run_contract"]["numpy"], records=records)


def focus_groups(candidates):
    """Trace every joint miss recovered in a final tree, plus physical-depth ≤2 checkpoint hits."""
    selected = {}
    for c in candidates:
        if c["original_status"] == "ok" or "first_two_physical_edges" in c["scopes"]:
            selected.setdefault(c["case"], []).append(c["grouping"])
    return {name: sorted(groups,key=len,reverse=True) for name,groups in selected.items()}


def build_report(rows, candidates, traces, arms):
    expected = focus_groups(candidates)
    if {r["case"]:r["targets"] for r in traces} != expected:
        raise ValueError("Traces do not match the complete declared selection")
    for item in traces:
        name = item["case"]
        receipt = json.loads((arms/f"{name}__proposed.json").read_text(encoding="utf-8"))
        nj = json.loads((arms/f"{name}__nj.json").read_text(encoding="utf-8"))
        if item["input_sha256"] != nj["history_sha256"] or item["settings"] != receipt["run_contract"]["settings"]:
            raise ValueError(f"Trace inputs or solver settings differ on {name}")
        if receipt["status"] == "ok" and item["original_final_matches"] is not True:
            raise ValueError(f"Repeated final tree differs on {name}")
        steps = [r for r in item["records"] if r["stage"] == "restricted_output"]
        checkpoints = receipt.get("completed_restricted_solves") or []
        matches = []
        for i, checkpoint in enumerate(checkpoints):
            targets = [tuple(g) in set(map(tuple,checkpoint["tree"])) for g in item["targets"]]
            same = targets == [t["present"] for t in steps[i]["targets"]]
            # H = T*n*crit/2 with 192 hourly rows; no extra PCC channel in this benchmark.
            n = next(r["meters"] for r in rows if r["case"]==name)
            diff = steps[i]["H"] - checkpoint["crit"]*192*n/2
            matches.append(dict(step=i, target_membership_matches=same, objective_difference_H=diff))
        item["comparison_to_original_checkpoints"] = matches
        if any(not m["target_membership_matches"] or abs(m["objective_difference_H"])>1e-4 for m in matches):
            raise ValueError(f"Replay differs at archived checkpoints on {name}")
    totals = {}
    for series in ("named","random"):
        totals[series] = {}
        for scope in ("first_two_physical_edges","first_two_levels","majority"):
            total = Counter()
            for row in rows:
                if row["series"] == series and "counts" in row:
                    total.update(row["counts"][scope])
            totals[series][scope] = total
    return dict(
        definition=dict(physical="Incoming edge distance from the transformer in edges, including trunk and short edges.",
                        level="One plus number of strict retained ancestor groups; an all-meter trunk and singleton service leads are excluded.",
                        majority="More than half of the feeder's meters are downstream, as in PROTOCOL_ID section 6.",
                        target="Downstream meter indices start at zero. A scored group has 2..n-1 meters and a physical supporting line of at least 3 milliohms. A group with more than one such edge is counted once."),
        selection="Inspect both independent NJ/RNJ implementations on the four named and all 21 verified random inputs. Among joint misses, trace all final recoveries in any of the three scopes and all original timeout checkpoint recoveries in the first two physical edges. Cases and lines were selected retrospectively for diagnosis.",
        stages="Nine production builders, initial pool, locally converged warm tree, raw Tree MIQP incumbent after coefficient refit, selected tree after optional local cleanup, anchored search, Node additions, final tree. The external registered baselines have their own distance tolerances. The solver sees only the common eight-day P/Q/Y/V0sq history. Line membership is scored outside the solver.",
        budgets="Original random runs had 120 seconds wall time. The three supplemental traces retain all identification settings (4 seconds per MIQP and 1 second per first local pass) and run through completion; timings are elapsed wall time with parallel runs on the same host.",
        cases=rows, totals=totals, jointly_missed_groups_recovered_in_final_or_checkpoint=candidates,
        traces=traces)


def print_summary(report):
    print("series scope | true / either misses / joint misses | final recoveries / joint misses having a final tree")
    for series, scopes in report["totals"].items():
        for scope,c in scopes.items():
            print(series,scope, "|",c.get("truth",0),c.get("miss_either",0),c.get("miss_both",0),"|",
                  c.get("recovered_final",0),c.get("both_miss_with_final",0))
    for t in report["traces"]:
        warm = next(r for r in t["records"] if r["stage"]=="warm")
        final = t["records"][-1]
        print(t["case"], "meters per target",[len(g) for g in t["targets"]],
              "present in first warm tree",[r["present"] for r in warm["targets"]],
              "present in replay final tree",[r["present"] for r in final["targets"]],
              "original",t["original_status"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild",action="store_true",help="Read local SHA-verified feeders and original job receipts; rerun the selected solver traces")
    ap.add_argument("--outputs",type=Path,help="Parent of case_library_20261125 and topology_study_random_dev")
    ap.add_argument("--out",type=Path,help="Optional destination for the JSON report")
    args = ap.parse_args()
    if not args.rebuild:
        report = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    else:
        if not args.outputs:
            ap.error("--rebuild requires --outputs pointing to the original results cache")
        root = args.outputs / "case_library_20261125"
        arms = args.outputs / "topology_study_random_dev" / "arms"
        rows, candidates = survey(root, arms)
        traces = [trace(n,groups,root,arms) for n,groups in focus_groups(candidates).items()]
        report = build_report(rows,candidates,traces,arms)
    if args.out:
        if args.out.resolve() == PUBLISHED.resolve():
            ap.error("Choose a new output path for the working copy of the results")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=1,allow_nan=False)+"\n",encoding="utf-8")
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
