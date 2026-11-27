"""Trace baseline root-side misses on one additional load variant per verified small/medium feeder."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from importlib import import_module
import json
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

from ptid.runs.root_ablation import PUBLISHED, as_tree, locate, physical_edges

CLOCK = (
    "The 120 s include the shared first solve and each arm subprocess (including its own startup "
    "and builder replay); exclude initial disk reads, NJ/RNJ and nine-tree screening. "
    "This supplementary set does not enter the original 120 comparison jobs."
)


class WarmOnly(Exception):
    """Stop the screening solve immediately before its first integer optimisation."""


def hits(groups, tree):
    lines = as_tree(tree)
    return [g in lines for g in groups]


def score_warm_additions(model, warm, targets):
    """Post-hoc bounded refit of a single missing, compatible line against the observed warm tree."""
    from ptid.method.model import Fitter
    from ptid.method.sets import compatible

    tree = as_tree(warm)
    checks = []
    for g in targets:
        if g in tree or any(not compatible(g, k) for k in tree):
            continue
        enlarged = tree | {g}
        fit = Fitter(model, enlarged)
        old, new = fit.crit(tree), fit.crit(enlarged)
        checks.append(
            dict(
                group=g,
                baseline_crit=old,
                added_crit=new,
                delta_H=(new - old) * model.N / 2,
                penalty_H=model.alpha * model.N / 2,
                residual_gain_H=(fit.rss(tree)[0] - fit.rss(enlarged)[0]) / 2,
            )
        )
    return checks


def solve_record(info, groups):
    from ptid.method.sets import compatible

    required = info["domain"]["require"]
    return dict(
        warm=hits(groups, info["warm_tree"]),
        warm_crit=info["warm_crit"],
        raw=None if info["solver_tree"] is None else hits(groups, info["solver_tree"]),
        raw_crit=info["solver_crit"],
        output=hits(groups, info["tree"]),
        crit=info["crit"],
        lower=info["bound"],
        gap_closed=info["gap_closed"],
        status=info["status"],
        candidates=info["pool"],
        includes=hits(groups, info["domain"]["lines"]),
        incompatible_anchor_sets=[
            sum(all(not compatible(g, a) for a in block) for block in required) for g in groups
        ],
    )


def frozen_solve(module, model, pool, warm, require, seconds, st):
    """Run the production Tree MIQP from an exact recorded local optimum and keep its usual post-solve cleanup."""
    from ptid.method.model import Fitter

    warm = frozenset(as_tree(warm))
    before = Fitter(model, pool).crit(warm)
    engine, local = module.tree_miqp, module.local_search
    dispatched = False

    def replay_start(*args, **kwargs):
        if not dispatched:
            return warm, before
        return local(*args, **kwargs)

    def dispatch(*args, **kwargs):
        nonlocal dispatched
        dispatched = True
        return engine(*args, **kwargs)

    with patch.object(module, "tree_miqp", dispatch), patch.object(module, "local_search", replay_start):
        _, info = module.solve(model, pool, [warm], seconds, st, require)
    return info


def condition_on_line(module, model, pool, warm, missing, regular, st):
    """Use truth only for this diagnostic. Both bounds refer to the same initial pool."""
    from ptid.method.sets import compatible

    g = tuple(missing)
    warm = as_tree(warm)
    if g not in as_tree(pool) or not all(compatible(g, t) for t in warm):
        raise ValueError("Expected a feasible target addition in the initial domain")
    seconds = 60
    _, info = module.solve(model, pool, [warm | {g}], seconds, st, require=[(g,)])
    checked = solve_record(info, [g])
    if checked["output"] != [True]:
        raise AssertionError("Forced line missing from returned tree")
    scale = model.N / 2
    margin = (info["bound"] - regular["crit"]) * scale if info["bound"] is not None else None
    return dict(
        group=g,
        seconds_limit=seconds,
        solver_seconds=info["miqp_seconds"],
        regular_upper_H=regular["crit"] * scale,
        regular_lower_H=None if regular["lower"] is None else regular["lower"] * scale,
        forced_upper_H=info["crit"] * scale,
        forced_lower_H=None if info["bound"] is None else info["bound"] * scale,
        forced_lower_minus_regular_upper_H=margin,
        conditional_solver=checked,
    )


def tree_changes(info, rec, external, final_tree):
    """Score every changed initial-domain line after the decisions, including away from the root."""
    warm, raw, returned = (as_tree(info[key] or []) for key in ("warm_tree", "solver_tree", "tree"))
    final = as_tree(final_tree)
    physical = physical_edges(rec)
    observed = [warm, returned]
    if info["solver_tree"] is not None:
        observed.append(raw)
    changes = sorted(set.union(*observed) - set.intersection(*observed), key=lambda g: (len(g), g))
    return [
        dict(
            group=g,
            edges=physical.get(g, []),
            nj=g in external["nj"],
            rnj=g in external["rnj"],
            warm=g in warm,
            solver=g in raw if info["solver_tree"] is not None else None,
            returned=g in returned,
            final=g in final,
        )
        for g in changes
    ]


def budget_replay(root, published, out):
    from ptid.cases import library
    from ptid.method import builders
    from ptid.method.model import Model
    from ptid.method.sets import canonical
    from ptid.runs.four_offline import digest, metrics, save, settings
    from ptid.runs.random_dev_compare import cached_resistance
    from ptid.runs.passive_baselines import run_baseline

    library.LIBRARY = root
    module = import_module("ptid.method.identify")
    report = dict(
        protocol="Same two selected feeders, blind inputs and recorded local warm tree. One CPU thread, "
        "no whole-process deadline. The four-second anchored domain is reconstructed by a new four-second "
        "replay. Pool, anchor requirements and warm tree are recorded for every restricted solve.",
        cases=[],
    )

    def describe(info, groups):
        entry = solve_record(info, groups)
        entry.update(
            warm_tree=info["warm_tree"],
            pool=info["domain"]["lines"],
            require=info["domain"]["require"],
            returned=info["tree"],
            solver_tree=info["solver_tree"],
            solver_seconds=info["miqp_seconds"],
        )
        return entry

    def same_input(a, b):
        return (
            as_tree(a["domain"]["lines"]) == as_tree(b["domain"]["lines"])
            and a["domain"]["require"] == b["domain"]["require"]
            and as_tree(a["warm_tree"]) == as_tree(b["warm_tree"])
        )

    for source in published["cases"]:
        if source["status"] != "paired":
            continue
        name = source["case"]
        rec = library.open_case(name)
        h = library.readings(rec)
        data = {k: h[k] for k in ("P", "Q", "Y", "V0sq")}
        if digest(data) != source["history_sha256"]:
            raise ValueError("History differs from the original short run")
        targets = list(map(tuple, source["jointly_missed"]))
        st = settings()
        model = Model(**data, u=None, lam=st.lam, p_s=st.p_s)
        with cached_resistance(root.parent / "topology_study_random_dev" / "arms", name):
            initial = builders.builder_trees(**data)
            external = {m: as_tree(run_baseline(m, **data)["tree"]) for m in ("nj", "rnj")}
        pool = canonical(g for t in initial for g in t)
        warm = source["screen_warm"]["tree"]
        row = dict(case=name, targets=targets, history_sha256=digest(data), caps={})
        report["cases"].append(row)
        previous = None
        for seconds in (4, 30):
            config = replace(
                st, short_seconds=seconds, long_seconds=seconds, node_seconds=seconds, flip_seconds=seconds
            )
            began = perf_counter()
            first = frozen_solve(module, model, pool, warm, (), seconds, config)
            item = dict(initial=describe(first, targets), settings=asdict(config))
            row["caps"][str(seconds)] = item
            save(out, report)
            print(name, seconds, "first", first["status"], first["crit"], first["bound"], flush=True)
            result = module.refine(model, first["tree"], pool, config, initial_solve=first)
            anchored = next(r for r in result["rounds"] if "solve" in r)
            info = anchored["solve"]
            item.update(
                anchored=describe(info, targets),
                added=anchored["price"]["added"],
                node_scans=[
                    dict(narrow=s["enumerated"], wide=s["wide_nodes"])
                    for s in anchored["price"].get("scans", [])
                ],
                final=hits(targets, result["tree"]),
                final_tree=result["tree"],
                final_score=metrics(result["tree"], library.truth(rec)),
                initial_changes=tree_changes(first, rec, external, result["tree"]),
                seconds=perf_counter() - began,
                stop=result["stop"],
                flip_checks=[r["label_info"] for r in result["rounds"]],
            )
            save(out, report)
            print(name, seconds, "final", item["final"], "seconds", round(item["seconds"], 1), flush=True)
            if seconds == 4:
                previous = info
                old = source["arms"]["C"]
                item["matches_archived_target_membership"] = (
                    source["common_first"]["output"] == item["initial"]["output"]
                    and old["rounds"][0]["anchored"]["output"] == item["anchored"]["output"]
                    and old["final"] == item["final"]
                )
                historical = old["rounds"][0]["anchored"]
                item["archived_anchored_candidate_count"] = historical["candidates"]
                item["archived_anchored_crit"] = historical["crit"]
            else:
                identical = same_input(previous, info)
                item["same_anchored_input_as_4s"] = identical
                if identical:
                    item["on_original_anchored_domain"] = "same as anchored"
                else:
                    d = previous["domain"]
                    check = frozen_solve(
                        module, model, d["lines"], previous["warm_tree"], d["require"], seconds, config
                    )
                    item["on_original_anchored_domain"] = describe(check, targets)
                save(out, report)
        # Extend just the original first domain, whose four-second result supplied both published arms.
        regular = row["caps"]["30"]["initial"]
        for seconds in (120, 240):
            if regular["gap_closed"]:
                break
            check = frozen_solve(module, model, pool, warm, (), seconds, st)
            regular = describe(check, targets)
            row[f"initial_{seconds}s"] = regular
            regular["changes_from_warm"] = tree_changes(check, rec, external, row["caps"]["30"]["final_tree"])
            save(out, report)
            print(name, seconds, "first", check["status"], check["crit"], check["bound"], flush=True)
        row["conditional_line_checks"] = [
            condition_on_line(module, model, pool, warm, item["group"], regular, st)
            for item in source["warm_single_additions"]
        ]
        save(out, report)
    return report


def summary(rows):
    observed = [r for r in rows if "near_root" in r]
    paired = [r for r in rows if r["status"] == "paired"]
    groups = [g for r in observed for g in r["near_root"]]
    return dict(
        verified=len(observed),
        missing_inputs=len(rows) - len(observed),
        near_root_groups=len(groups),
        missed_nj=sum(not g["nj"] for g in groups),
        missed_rnj=sum(not g["rnj"] for g in groups),
        missed_both=sum(not g["nj"] and not g["rnj"] for g in groups),
        jointly_missed_still_absent_at_warm=sum(
            not p for r in observed if "screen_warm" in r for p in r["screen_warm"]["present"]
        ),
        paired_feeders=len(paired),
        first_tree_first_recoveries=sum(
            not warm and raw and kept
            for r in paired
            for warm, raw, kept in zip(
                r["screen_warm"]["present"], r["common_first"]["raw"] or [], r["common_first"]["output"]
            )
        ),
        arms={
            m: {s: sum(r["arms"][m]["status"] == s for r in paired) for s in ("ok", "wall_limit", "failed")}
            for m in ("B", "C")
        },
    )


def arm(job, output, mode):
    """Both arms start from the identical hashed input and completed first solve."""
    from ptid.method.identify import identify
    from ptid.runs.four_offline import digest, metrics, save, settings

    with job.open("rb") as source:
        data, first, groups, truth, fingerprint = pickle.load(source)
    if digest(data) != fingerprint:
        raise ValueError("Child observations differ from the screening history")
    st = replace(settings(), wide_node_miqp=mode == "C")
    search = import_module("ptid.method.search")
    original_node = search.node_miqp
    attempts = []

    def observe_node(model, tree, kids, seconds, **kwargs):
        answer = original_node(model, tree, kids, seconds, **kwargs)
        line, _, status, bound = answer
        attempts.append(dict(width=len(kids), selected=line, status=status, lower=bound))
        return answer

    start = perf_counter()
    with patch.object(search, "node_miqp", observe_node):
        result = identify(**data, st=st, paired_first=first)
    rounds = []
    for row in result["rounds"]:
        if "solve" not in row:
            continue
        price = row["price"]
        rounds.append(
            dict(
                anchored=solve_record(row["solve"], groups),
                added=price["added"],
                scans=[
                    dict(enumerated=s["enumerated"], by_miqp=s["by_miqp"], wide_nodes=s["wide_nodes"])
                    for s in price.get("scans", [])
                ],
                after_addition=None if price["re_solve"] is None else solve_record(price["re_solve"], groups),
            )
        )
    save(
        output,
        dict(
            status="ok",
            seconds=perf_counter() - start,
            rounds=rounds,
            wide_solutions=attempts,
            final=hits(groups, result["tree"]),
            final_tree=result["tree"],
            crit=result["model"]["crit"],
            score=metrics(result["tree"], truth),
        ),
    )


def compare(root):
    from ptid.cases import library
    from ptid.method import builders
    from ptid.method.model import Model
    from ptid.method.sets import canonical
    from ptid.runs.four_offline import digest, settings
    from ptid.runs.passive_baselines import run_baseline
    from ptid.runs.random_dev_compare import cached_resistance, selected

    module = import_module("ptid.method.identify")
    library.LIBRARY = root
    st = settings()
    rows, selected_count = [], 0
    declared = {r["name"]: r["sha256"] for r in library.manifest()}
    for old in sorted(selected(), key=lambda x: ("_small_" not in x, x)):
        if library.kind(old) not in ("small", "medium"):
            continue
        family, original = old.rsplit("_", 1)
        location = library.LOCATIONS[(library.LOCATIONS.index(original) + 1) % len(library.LOCATIONS)]
        name = family + "_" + location
        row = dict(case=name, previous_variant=old)
        rows.append(row)
        if not (root / "dev" / (name + ".pkl")).is_file():
            row["status"] = "input_unavailable"
            continue
        rec = library.open_case(name)
        h = library.readings(rec)
        data = {k: h[k] for k in ("P", "Q", "Y", "V0sq")}
        with cached_resistance(root.parent / "topology_study_random_dev" / "arms", name):
            baselines = {m: run_baseline(m, **data) for m in ("nj", "rnj")}
            initial = builders.builder_trees(**data)
        if any(r["status"] != "ok" for r in baselines.values()):
            raise RuntimeError(f"Baseline failed for {name}")
        geometry = physical_edges(rec)
        truth = library.truth(rec)
        if set(geometry) != truth:
            raise ValueError(f"Physical and identifiable reference differ in {name}")
        groups = []
        for g in sorted(truth, key=lambda v: (-len(v), v)):
            level = 1 + sum(set(g) < set(other) for other in truth)
            if level <= 2 or any(e["physical_depth"] <= 2 for e in geometry[g]):
                groups.append((g, level))
        nj, rnj = (as_tree(baselines[m]["tree"]) for m in ("nj", "rnj"))
        joint = [g for g, _ in groups if g not in nj and g not in rnj]
        row.update(
            status="baseline_screen",
            meters=rec["n"],
            record_sha256=declared[name],
            history_sha256=digest(data),
            near_root=[
                dict(
                    group=g,
                    level=level,
                    edges=geometry[g],
                    nj=g in nj,
                    rnj=g in rnj,
                    initial_builders=[m for m, t in zip(builders.NAMES, initial) if g in t],
                )
                for g, level in groups
            ],
            jointly_missed=joint,
        )
        if not joint:
            continue
        model = Model(data["P"], data["Q"], data["Y"], data["V0sq"], None, st.lam, p_s=st.p_s)
        pool = canonical(g for tree in initial for g in tree)
        starts = list(dict.fromkeys(frozenset(t) for t in initial))
        real_miqp = module.tree_miqp

        def at_warm(*args, **kwargs):
            warm = kwargs["warm"]
            row["screen_warm"] = dict(
                present=hits(joint, warm),
                crit=kwargs["known"],
                positions=locate(joint, warm, rec["n"]),
                tree=sorted(warm),
            )
            if all(row["screen_warm"]["present"]) or selected_count >= 2:
                raise WarmOnly
            return real_miqp(*args, **kwargs)

        started = perf_counter()
        limit = st.short_seconds if rec["n"] <= st.large_n else st.long_seconds
        try:
            with patch.object(module, "tree_miqp", at_warm):
                _, first = module.solve(model, pool, starts, limit, st)
        except WarmOnly:
            row["status"] = "all_recovered_in_warm" if all(row["screen_warm"]["present"]) else "trace_cap"
            continue
        selected_count += 1
        first_seconds = perf_counter() - started
        row.update(
            status="paired", first_seconds=first_seconds, common_first=solve_record(first, joint), arms={}
        )
        row["warm_single_additions"] = score_warm_additions(model, row["screen_warm"]["tree"], joint)
        with TemporaryDirectory() as tmp:
            job, result = Path(tmp) / "job.pkl", Path(tmp) / "arm.json"
            job.write_bytes(pickle.dumps((data, first, joint, truth, row["history_sha256"])))
            for mode in ("B", "C"):
                result.unlink(missing_ok=True)
                timestamp = perf_counter()
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            "-m",
                            "ptid.runs.root_variants",
                            "--worker",
                            str(job),
                            str(result),
                            mode,
                        ],
                        check=True,
                        timeout=max(0.01, 120 - first_seconds),
                    )
                    row["arms"][mode] = json.loads(result.read_text(encoding="utf-8"))
                except subprocess.TimeoutExpired:
                    row["arms"][mode] = dict(status="wall_limit")
                except subprocess.CalledProcessError as exc:
                    row["arms"][mode] = dict(status="failed", process_exit=exc.returncode)
                row["arms"][mode]["wall_including_shared_solve_s"] = perf_counter() - timestamp + first_seconds
        print(
            name,
            row["screen_warm"]["present"],
            row["common_first"]["output"],
            {k: a.get("final", a["status"]) for k, a in row["arms"].items()},
            flush=True,
        )
    return dict(
        selection="One location step after the original comparison in none/head/middle/end cyclic order. "
        "All available SHA-verified small/medium feeders in size then name order; at most the first two "
        "feeders with a joint baseline miss still missing in the actual first-solve warm tree are traced.",
        root_definition="Union of true groups within two physical edges or two identifiable levels.",
        settings=asdict(st),
        elapsed_limit_s_per_arm=120,
        clock=CLOCK,
        summary=summary(rows),
        cases=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, help="Rebuild from the local verified development library")
    parser.add_argument(
        "--relax", action="store_true", help="Re-solve the two paired feeders with expanded time limits"
    )
    parser.add_argument("--out", type=Path, help="Working JSON file for a new run")
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        job, out, mode = args.worker
        arm(Path(job), Path(out), mode)
        return
    published = PUBLISHED.with_name("variants.json")
    if args.out and args.out.resolve() == published.resolve():
        parser.error("Choose a working output file")
    if args.cases_root:
        out = args.out or args.cases_root.resolve().parent / "topology_study_root_variants" / "variants.json"
        if args.relax:
            out = args.out or out.with_name("budget.json")
            if out.resolve() in (published.resolve(), published.with_name("budget.json").resolve()):
                parser.error("Choose a working output file")
            budget_replay(args.cases_root.resolve(), json.loads(published.read_text(encoding="utf8")), out)
            return
        report = compare(args.cases_root.resolve())
        from ptid.runs.four_offline import save

        save(out, report)
    else:
        report = json.loads(published.read_text(encoding="utf8"))
        if args.relax:
            budget = report["relaxed_budget"]
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(budget, indent=1, allow_nan=False) + "\n", encoding="utf8")
            for case in budget["cases"]:
                for cap, trial in case["caps"].items():
                    print(case["case"], cap, "s", "final target membership", trial["final"])
                for check in case["conditional_line_checks"]:
                    margin = check["forced_lower_minus_regular_upper_H"]
                    if margin is not None:
                        print("  forced-line lower bound minus feasible ordinary score (H):", round(margin, 3))
                verified = case.get("verify_changed_tree_until_closed", {})
                for change in verified.get("added_or_removed", ()):
                    if change["solver"] and not change["warm"] and change["edges"]:
                        print("  Tree MIQP recovered", len(change["group"]), "meters:", change["edges"])
            return
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes(published.read_bytes())
    if summary(report["cases"]) != report["summary"]:
        raise ValueError("Summary and case receipts differ")
    print(json.dumps(report["summary"], ensure_ascii=False))
    for row in report["cases"]:
        print(row["case"], row["status"], len(row.get("jointly_missed", [])))


if __name__ == "__main__":
    main()
