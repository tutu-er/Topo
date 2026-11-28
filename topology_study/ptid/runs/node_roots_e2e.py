"""N7, root lines end to end: the earlier EU LV comparison with a known truth (experiments/node_miqp/README.md).

The EU LV study compared a whole-map method with the tree builders on lines near the root. Here Algorithm 1 itself
runs on N5's feeders with short root lines (r = 0.002 and 0.003 ohm, about 0.03 to 0.04 of the root-to-meter
resistance, carrying 70 to 80% of the homes), in two arms from one common first solve (same data, builder pool, warm
start and settings):
  B  node search by enumeration at nodes of at most K = 9 children only (wider nodes are skipped);
  D  the method: B plus the node MIQP at wider nodes, with the extended-BIC price (gamma 0.5).
D minus B is what the node MIQP adds. Each builder's own tree is the baseline of the EU LV table.

Feeders (node_roots.layout): 12 or 16 laterals of 3 homes; "one" root line feeding 70% of the laterals, or a "chain"
of two nested ones (80% and 40%); controls without a root line. Solver limits: node_pipeline.SHORT (tree MIQP 20 s,
flip tests 10 s, node MIQP 30 s), so that the run fits a day; --full for the protocol's limits.

Recorded per feeder: each root line's true evidence, its presence in each builder's tree, in the pool, in the first
tree solve and in each arm's final tree with its label (decided or open); F1 of all meter groupings, extra lines.

Predictions written before the run (after N5's first 48 rows): the root lines that no builder proposes are recovered
by D and not by B; D's root-line recall exceeds every builder's; no decided line is wrong; D adds no line in controls.

Run from topology_study:
    python -m ptid.runs.node_roots_e2e run --workers 8    # experiments/node_miqp/roots_e2e.jsonl (resumable)
    python -m ptid.runs.node_roots_e2e summary
    python -m ptid.runs.node_roots_e2e run --widths 12 --r 0.003 --seeds 51 --out roots_e2e_try.jsonl
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from ptid.method.builders import NAMES, builder_trees
from ptid.method.identify import identify
from ptid.runs.node_pipeline import FULL, SHORT
from ptid.runs.node_roots import PER, layout, layout_of
from ptid.runs.node_study import (Feeder, Simulator, feeder_dict, grouping_f1, model_of, read_rows, root_feeder,
                                  root_lines, score, truth, write_row)

OUT = Path(__file__).resolve().parents[2]/"experiments"/"node_miqp"/"roots_e2e.jsonl"
WIDTHS = (12, 16)
R_LEVELS = (.002, .003)
SEEDS = (51, 52, 53, 54)
ARMS = ("B", "D")


def settings(st):
    return {"B": replace(st, wide_node_miqp=False), "D": replace(st, node_gamma=.5)}


def instances(widths, r_levels, seeds):
    for L in widths:
        for seed in seeds:
            yield Feeder(blocks=L, per=PER, seed=seed)                          # control
            for name in ("one", "chain"):
                for r in r_levels:
                    yield root_feeder(per=PER, r=r, seed=seed, **layout(name, L))


def one(f, full=False):
    t0 = time.perf_counter()
    data = Simulator(f).history()
    true, flat, _ = truth(f)
    roots = root_lines(f)
    model, load = model_of(data)
    trees = builder_trees(data["P"], data["Q"], data["Y"], data["V0sq"], None)
    pool = {C for t in trees for C in t}
    h_true = score(model, true)
    row = dict(feeder=feeder_dict(f), load_modelled=load, limits="full" if full else "short",
               roots=[dict(line=list(rl), share=len(rl)/f.n, evidence=score(model, [C for C in true if C != rl])-h_true,
                           in_pool=rl in pool, builders=[n for n, t in zip(NAMES, trees) if rl in t]) for rl in roots],
               builder_f1={n: grouping_f1(t, true) for n, t in zip(NAMES, trees)},
               builder_extra={n: len(set(t)-set(true)) for n, t in zip(NAMES, trees)}, arms={})
    first = None
    for name, st in settings(FULL if full else SHORT).items():
        s0 = time.perf_counter()
        out = identify(data["P"], data["Q"], data["Y"], data["V0sq"], data["Ppcc"], data["Qpcc"], st,
                       paired_first=first)
        if first is None:
            first = out["first"]
            row["roots_in_first"] = [rl in set(map(tuple, first["tree"])) for rl in roots]
        tree = [tuple(g) for g in out["tree"]]
        labels = {tuple(x["grouping"]): x["label"] for x in out["labels"]}
        rounds = [x for x in out["rounds"] if "price" in x]
        wide = [w for x in rounds for s in x["price"].get("scans", [x["price"]]) for w in s.get("wide_nodes", [])]
        row["arms"][name] = dict(tree=[list(C) for C in tree], f1=grouping_f1(tree, true),
                                 extra=[list(C) for C in set(tree)-set(true)],
                                 found=[rl in set(tree) for rl in roots], labels=[labels.get(rl) for rl in roots],
                                 decided_wrong=sum(labels.get(C) == "decided" for C in set(tree)-set(true)),
                                 node_miqp_calls=sum(w["status"] != "skipped" for w in wide),
                                 stop=out["stop"], seconds=time.perf_counter()-s0)
    row["seconds"] = time.perf_counter()-t0
    return row


def run(args):
    out = Path(args.out)
    done = {r["feeder"]["name"] for r in read_rows(out)}
    todo = [f for f in instances(args.widths, args.r, args.seeds) if f.name not in done]
    print(f"{len(todo)} feeders to run ({len(done)} done) -> {out}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(one, f, args.full): f for f in todo}
        for job in as_completed(jobs):
            row = job.result()
            write_row(out, row)
            a = row["arms"]
            print(f"{row['feeder']['name'][:60]}: evidence {[round(x['evidence'], 1) for x in row['roots']]}, "
                  f"pool {[x['in_pool'] for x in row['roots']]}, first {row['roots_in_first']}, "
                  + ", ".join(f"{k} {v['found']} F1 {v['f1']:.2f} ({v['seconds']/60:.0f} min)" for k, v in a.items()),
                  flush=True)


def recall_table(lines, title):
    """Root-line recall of every builder's tree, the pool, the first solve and both arms (EU LV table form)."""
    k = len(lines)
    print(f"{title}: {k} root lines")
    if not k:
        return
    pct = lambda c: f"{c} ({100*c/k:.0f}%)"
    print("| builder trees: " + " | ".join(NAMES) + " | pool | first solve | B final | D final | D decided |")
    print("| --- " * (len(NAMES)+6) + "|")
    cells = [pct(sum(n in x["builders"] for _, _, x in lines)) for n in NAMES]
    cells += [pct(sum(x["in_pool"] for _, _, x in lines)), pct(sum(r["roots_in_first"][i] for r, i, _ in lines))]
    cells += [pct(sum(r["arms"][a]["found"][i] for r, i, _ in lines)) for a in ARMS]
    cells += [pct(sum(r["arms"]["D"]["labels"][i] == "decided" for r, i, _ in lines))]
    print("| " + " | ".join(cells) + " |")
    only = sum(r["arms"]["D"]["found"][i] and not r["arms"]["B"]["found"][i] for r, i, _ in lines)
    lost = sum(r["arms"]["B"]["found"][i] and not r["arms"]["D"]["found"][i] for r, i, _ in lines)
    outside = sum(r["arms"]["D"]["found"][i] and not x["in_pool"] for r, i, x in lines)
    print(f"  found by D only {only}, by B only {lost}; found by D and in no builder's tree {outside}\n")


def summary(args):
    rows = read_rows(args.out)
    for r in rows:
        r["layout"] = layout_of(r["feeder"])
    seg = [r for r in rows if r["layout"] != "control"]
    ctl = [r for r in rows if r["layout"] == "control"]
    print(f"{len(rows)} feeders from {args.out} ({len(seg)} with root lines, {len(ctl)} controls)\n")
    lines = [(r, i, x) for r in seg for i, x in enumerate(r["roots"])]
    paying = [t for t in lines if t[2]["evidence"] > 0]
    recall_table(paying, "Root lines with positive true evidence, all feeders")
    recall_table([t for t in paying if t[2]["share"] > .5], "... of them feeding more than half of the meters "
                 "(the EU LV class)")
    for lay in ("one", "chain"):
        for rr in sorted({r["feeder"]["roots"][0][1] for r in seg}):
            recall_table([t for t in paying if t[0]["layout"] == lay and t[0]["feeder"]["roots"][0][1] == rr],
                         f"{lay}, r = {rr:g} ohm")
    print("| feeders | arm or builder | mean F1 | extra lines per feeder | decided wrong lines | median minutes |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for title, g in (("with root lines", seg), ("controls", ctl)):
        if not g:
            continue
        for a in ARMS:
            x = [r["arms"][a] for r in g]
            print(f"| {title} ({len(g)}) | {a} | {np.mean([y['f1'] for y in x]):.3f} | "
                  f"{np.mean([len(y['extra']) for y in x]):.2f} | {sum(y['decided_wrong'] for y in x)} | "
                  f"{np.median([y['seconds'] for y in x])/60:.1f} |")
        best = max(NAMES, key=lambda n: np.mean([r["builder_f1"][n] for r in g]))
        print(f"| {title} ({len(g)}) | best builder ({best}) | {np.mean([r['builder_f1'][best] for r in g]):.3f} | "
              f"{np.mean([r['builder_extra'][best] for r in g]):.2f} | - | - |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "summary"))
    p.add_argument("--out", default=str(OUT))
    ints = lambda s: tuple(int(x) for x in s.split(","))
    p.add_argument("--widths", type=ints, default=WIDTHS)
    p.add_argument("--r", type=lambda s: tuple(float(x) for x in s.split(",")), default=R_LEVELS)
    p.add_argument("--seeds", type=ints, default=SEEDS)
    p.add_argument("--full", action="store_true", help="the protocol's solver limits (slow)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    main()
