"""N2, recovery of hidden shared lines that the tree builders miss (experiments/node_miqp/README.md).

Each feeder has a wide junction of d blocks and, except in the controls, a hidden shared segment over a random subset
of them. For every feeder the run records
  - the true evidence of the segment: H(flat tree) - H(true tree), from the same data (what any method could see);
  - whether the segment's grouping is in the candidate pool of the nine tree builders (Appendix B), and in which;
  - the node MIQP and its flip test at the flat junction (the true tree with the segment contracted): the union it
    proposes, its gain, the runner-up gap and the decision at margin 4.6 (present / open / absent).
The segment's grouping can enter Algorithm 1's pool from the builders or from node search; at a junction wider than
K = 9 children only the node MIQP searches the unions. Controls (no segment) measure false "present" decisions.

Run from topology_study:
    python -m ptid.runs.node_recovery run --workers 8     # experiments/node_miqp/recovery.jsonl (resumable)
    python -m ptid.runs.node_recovery summary
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np

from ptid.method.builders import NAMES, builder_trees
from ptid.runs.node_study import (M, Feeder, Simulator, draw_shared, feeder_dict, model_of, node_scan, read_rows,
                                  score, truth, write_row)

OUT = Path(__file__).resolve().parents[2]/"experiments"/"node_miqp"/"recovery.jsonl"
WIDTHS = (10, 14, 18)
R_LEVELS = (0., .003, .005, .008, .012)
EXCITATION = (.32, .10)
SEEDS = (11, 12, 13, 14, 15, 16, 17, 18)


def instances(widths, seeds):
    for d in widths:
        for e in EXCITATION:
            for r in R_LEVELS:
                for seed in seeds:
                    yield Feeder(blocks=d, per=2, shared=draw_shared(d, seed) if r > 0 else (), r_shared=r,
                                 excitation=e, hours=192, seed=seed)


def one(f, seconds):
    t0 = time.perf_counter()
    data = Simulator(f).history()
    model, load = model_of(data)
    true, flat, target = truth(f)
    trees = builder_trees(data["P"], data["Q"], data["Y"], data["V0sq"], None)
    pool = {C for t in trees for C in t}
    row = dict(feeder=feeder_dict(f), load_modelled=load, p_s=model.p_s, N=model.N, beta=model.alpha*model.N/2,
               target=target, true_evidence=None if target is None else score(model, flat)-score(model, true),
               pool_size=len(pool), blocks_in_pool=sum(b in pool for b in flat),
               target_in_pool=None if target is None else target in pool,
               builders_with_target=[] if target is None else [n for n, t in zip(NAMES, trees) if target in t])
    row["node"] = node_scan(model, flat, seconds)
    row["seconds"] = time.perf_counter()-t0
    return row


def run(args):
    out = Path(args.out)
    done = {r["feeder"]["name"] for r in read_rows(out)}
    todo = [f for f in instances(args.widths, args.seeds) if f.name not in done]
    print(f"{len(todo)} feeders to run ({len(done)} done) -> {out}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(one, f, args.seconds): f for f in todo}
        for job in as_completed(jobs):
            row = job.result()
            write_row(out, row)
            n = row["node"]
            print(f"{row['feeder']['name']}: evidence {row['true_evidence']}, in pool {row['target_in_pool']}, "
                  f"node {n['decision']} {n['seconds']:.0f}s", flush=True)


def classify(row, flip=True):
    """right / wrong / open / absent for the node decision; flip=False uses the gain alone (no runner-up check)."""
    n = row["node"]
    if flip:
        present = n["decision"] == "present"
    else:
        present = n["gain"] is not None and n["gain"] >= M
    if present:
        return "right" if row["target"] is not None and n["line"] == row["target"] else "wrong"
    return n["decision"] if n["decision"] != "present" else "open"


def adds(row):
    """right / wrong / none: whether the node's best union pays (gain > 0) and is the hidden segment's grouping."""
    n = row["node"]
    if n["gain"] is None or n["gain"] <= 0:
        return "none"
    return "right" if row["target"] is not None and n["line"] == row["target"] else "wrong"


def band(ev):
    if ev is None:
        return "no segment"
    return "below 0" if ev < 0 else "0 to 4.6" if ev < M else "4.6 to 15" if ev < 15 else "15 to 50" if ev < 50 else \
        "50 or more"


BANDS = ("no segment", "below 0", "0 to 4.6", "4.6 to 15", "15 to 50", "50 or more")


def summary(args):
    rows = read_rows(args.out)
    print(f"{len(rows)} rows from {args.out}\n")
    print("By the true evidence of the segment (score units; it pays only above 0, and is decidable above 4.6):\n")
    print("Adds = the node's best union pays (gain > 0), so pricing in Algorithm 1 would add it to the tree.\n")
    print("| true evidence | feeders | in builder pool | node adds right union | adds it, not in pool | "
          "adds a wrong union | decided right | decided wrong | open | absent | decided wrong without flip test |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for b in BANDS:
        g = [r for r in rows if band(r["true_evidence"]) == b]
        if not g:
            continue
        c = [classify(r) for r in g]
        a = [adds(r) for r in g]
        seg = b != "no segment"
        pool = str(sum(bool(r["target_in_pool"]) for r in g)) if seg else "-"
        unique = str(sum(k == "right" and not r["target_in_pool"] for k, r in zip(a, g))) if seg else "-"
        print(f"| {b} | {len(g)} | {pool} | {a.count('right') if seg else '-'} | {unique} | {a.count('wrong')} | "
              f"{c.count('right') if seg else '-'} | {c.count('wrong')} | {c.count('open')} | {c.count('absent')} | "
              f"{[classify(r, False) for r in g].count('wrong')} |")
    print("\nBy junction width and load variation (segments only):\n")
    print("| d | variation | feeders | segment pays (evidence > 0) | in builder pool | node adds right | "
          "adds right, not in pool | decided right | decided wrong | node seconds (median / max) |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for d in sorted({r["feeder"]["blocks"] for r in rows}):
        for e in EXCITATION:
            g = [r for r in rows if r["feeder"]["blocks"] == d and r["feeder"]["excitation"] == e
                 and r["target"] is not None]
            if not g:
                continue
            c, a = [classify(r) for r in g], [adds(r) for r in g]
            s = [r["node"]["seconds"] for r in g]
            print(f"| {d} | {e:g} | {len(g)} | {sum(r['true_evidence'] > 0 for r in g)} | "
                  f"{sum(bool(r['target_in_pool']) for r in g)} | {a.count('right')} | "
                  f"{sum(k == 'right' and not r['target_in_pool'] for k, r in zip(a, g))} | {c.count('right')} | "
                  f"{c.count('wrong')} | {np.median(s):.0f} / {max(s):.0f} |")
    ctl = [r for r in rows if r["target"] is None]
    if ctl:
        print(f"\nControls (no segment): {len(ctl)} feeders, wrong 'present' decisions "
              f"{sum(classify(r) == 'wrong' for r in ctl)} (with the flip test), "
              f"{sum(classify(r, False) == 'wrong' for r in ctl)} (gain alone); 'absent' certified "
              f"{sum(classify(r) == 'absent' for r in ctl)}.")
    seg = [r for r in rows if r["target"] is not None]
    if seg:
        per_builder = {n: sum(n in r["builders_with_target"] for r in seg) for n in NAMES}
        print("Segments found by each builder:", per_builder)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "summary"))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--widths", type=lambda s: tuple(int(x) for x in s.split(",")), default=WIDTHS)
    p.add_argument("--seeds", type=lambda s: tuple(int(x) for x in s.split(",")), default=SEEDS)
    p.add_argument("--seconds", type=float, default=120., help="node MIQP time limit per solve")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    main()
