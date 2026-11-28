"""N3, Algorithm 1 with and without the node MIQP, end to end (experiments/node_miqp/README.md).

Four arms run from one common first tree solve (same data, builder pool, warm start and settings):
  A  tree MIQP only (no node search);
  B  node search by enumeration at nodes of at most K = 9 children (wider nodes are skipped);
  C  the full method: enumeration up to K children, the node MIQP beyond;
  D  C with the extended-BIC price for lines found at nodes (Settings.node_gamma = 0.5).
The feeders have one junction of 12 or 16 blocks (wider than K), with or without a hidden shared segment. Recorded per
arm: the final tree, whether it contains the segment's grouping and the segment's label, F1 of the meter groupings,
extra (false) lines, node MIQP calls and time.

Settings (all arms): the protocol's 2 anchored rounds and up to 10 lines added per round; solver limits shortened so
that the run fits a working day on a shared machine: tree MIQP 20 s, flip tests 10 s, node MIQP 30 s, local search 2 s
per start. --full uses the protocol's limits (120 s tree MIQP and flips, 60 s node MIQP, 20 s local search per start;
several hours per feeder).

Run from topology_study:
    python -m ptid.runs.node_pipeline run --workers 4     # experiments/node_miqp/pipeline.jsonl (resumable)
    python -m ptid.runs.node_pipeline summary
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from ptid.method.builders import builder_trees
from ptid.method.identify import Settings, identify
from ptid.runs.node_study import (Feeder, Simulator, draw_shared, feeder_dict, grouping_f1, model_of, read_rows,
                                  score, truth, write_row)

OUT = Path(__file__).resolve().parents[2]/"experiments"/"node_miqp"/"pipeline.jsonl"
SHORT = Settings(rounds=2, repairs=10, short_seconds=20, long_seconds=20, flip_seconds=10, node_seconds=30,
                 start_seconds=2, evidence=False)
FULL = Settings(rounds=2, repairs=10, evidence=False)          # the protocol's limits (identify.Settings defaults)


def arms(st):
    return {"C": st, "A": replace(st, node_refinement=False), "B": replace(st, wide_node_miqp=False),
            "D": replace(st, node_gamma=.5)}


ARMS = arms(SHORT)
WIDTHS = (12, 16)
R_LEVELS = (0., .006, .010)
EXCITATION = (.32,)
SEEDS = (41, 42)


def instances(widths, seeds):
    for d in widths:
        for e in EXCITATION:
            for r in R_LEVELS:
                for seed in seeds:
                    yield Feeder(blocks=d, per=2, shared=draw_shared(d, seed) if r > 0 else (), r_shared=r,
                                 excitation=e, hours=192, seed=seed)


def arm_record(out, true, target, seconds):
    tree = [tuple(g) for g in out["tree"]]
    labels = {tuple(x["grouping"]): x for x in out["labels"]}
    rounds = [x for x in out["rounds"] if "price" in x]
    wide = [w for x in rounds for s in x["price"].get("scans", [x["price"]]) for w in s.get("wide_nodes", [])]
    return dict(tree=tree, f1=grouping_f1(tree, true), extra=len(set(tree)-set(true)),
                missed=len(set(true)-set(tree)), target_in_tree=target is not None and target in set(tree),
                target_label=None if target is None or target not in labels else labels[target]["label"],
                added=[a for x in rounds for a in x["price"].get("added", [])],
                node_miqp_calls=sum(w["status"] != "skipped" for w in wide),
                node_miqp_status=[w["status"] for w in wide],
                stop=out["stop"], scope=out["scope"], crit=out["model"]["crit"], seconds=seconds)


def one(f, full=False):
    t0 = time.perf_counter()
    data = Simulator(f).history()
    true, flat, target = truth(f)
    model, load = model_of(data)
    trees = builder_trees(data["P"], data["Q"], data["Y"], data["V0sq"], None)
    pool = {C for t in trees for C in t}
    row = dict(feeder=feeder_dict(f), load_modelled=load, target=target,
               true_evidence=None if target is None else score(model, flat)-score(model, true),
               target_in_pool=None if target is None else target in pool, arms={},
               limits="full" if full else "short")
    first = None
    for name, st in arms(FULL if full else SHORT).items():
        s0 = time.perf_counter()
        out = identify(data["P"], data["Q"], data["Y"], data["V0sq"], data["Ppcc"], data["Qpcc"], st,
                       paired_first=first)
        if first is None:
            first = out["first"]
            row["first_tree"] = [list(C) for C in first["tree"]]
            row["target_in_first"] = target is not None and target in set(map(tuple, first["tree"]))
        row["arms"][name] = arm_record(out, true, target, time.perf_counter()-s0)
    row["seconds"] = time.perf_counter()-t0
    return row


def run(args):
    out = Path(args.out)
    done = {r["feeder"]["name"] for r in read_rows(out)}
    todo = [f for f in instances(args.widths, args.seeds) if f.name not in done]
    print(f"{len(todo)} feeders to run ({len(done)} done) -> {out}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(one, f, args.full): f for f in todo}
        for job in as_completed(jobs):
            row = job.result()
            write_row(out, row)
            print(f"{row['feeder']['name']}: evidence {row['true_evidence']}, pool {row['target_in_pool']}; "
                  + ", ".join(f"{k} F1 {v['f1']:.2f} seg {v['target_in_tree']} ({v['seconds']:.0f}s)"
                              for k, v in row["arms"].items()), flush=True)


def summary(args):
    rows = read_rows(args.out)
    print(f"{len(rows)} feeders from {args.out}\n")
    for title, g in (("segments", [r for r in rows if r["target"] is not None]),
                     ("controls (no segment)", [r for r in rows if r["target"] is None])):
        if not g:
            continue
        print(f"{title}: {len(g)} feeders" + ("" if title.startswith("controls") else
              f"; segment pays (true evidence > 0) in {sum(r['true_evidence'] > 0 for r in g)}, in the builder pool "
              f"{sum(bool(r['target_in_pool']) for r in g)}, in the first tree {sum(r['target_in_first'] for r in g)}"))
        print("| arm | mean F1 | segment in final tree | segment decided | extra lines (total) | node MIQP calls | "
              "median minutes |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for name in ("A", "B", "C", "D"):
            a = [r["arms"][name] for r in g]
            print(f"| {name} | {np.mean([x['f1'] for x in a]):.3f} | {sum(x['target_in_tree'] for x in a)} | "
                  f"{sum(x['target_label'] == 'decided' for x in a)} | {sum(x['extra'] for x in a)} | "
                  f"{sum(sum(s != 'skipped' for s in x['node_miqp_status']) for x in a)} | {np.median([x['seconds'] for x in a])/60:.1f} |")
        if not title.startswith("controls"):
            only = sum(r["arms"]["C"]["target_in_tree"] and not r["arms"]["B"]["target_in_tree"] for r in g)
            lost = sum(not r["arms"]["C"]["target_in_tree"] and r["arms"]["B"]["target_in_tree"] for r in g)
            print(f"segment found by C but not B: {only}; by B but not C: {lost}; "
                  f"found by C and absent from the builder pool: "
                  f"{sum(r['arms']['C']['target_in_tree'] and not r['target_in_pool'] for r in g)}")
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "summary"))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--widths", type=lambda s: tuple(int(x) for x in s.split(",")), default=WIDTHS)
    p.add_argument("--seeds", type=lambda s: tuple(int(x) for x in s.split(",")), default=SEEDS)
    p.add_argument("--full", action="store_true", help="the protocol's solver limits (slow)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    main()
