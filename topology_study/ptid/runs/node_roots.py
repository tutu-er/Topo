"""N5, root lines: the main case of the node MIQP (experiments/node_miqp/README.md).

A root line is a short line near the substation that carries many homes. When a search misses it, every lateral it
feeds hangs from the root junction, so that junction is the widest node of the tree: wider than enumeration's K = 9,
and only a search over its 2^c - c - 2 unions can put the line back. The tree builders work bottom up from noisy
pairwise statistics, so the top of the tree is where they err most; the node MIQP fits one more impedance with every
meter and hour, exactly, with a bound.

Feeders: L = 12, 16 or 20 laterals (0.03 to 0.05 ohm, 3 homes each) at the trunk end, and
  one:    a root line of r ohm (0.003 to 0.008, like the cable between a substation and its first cabinet) feeding 70%
          of the laterals; the rest leave the trunk end directly;
  chain:  the trunk continuing: a root line feeding 80% of the laterals and a second one inside it feeding 40% (two
          nested near-root lines, as on a radial trunk);
controls: the same laterals without a root line. The laterals behind each root line are drawn from the seed.

Per feeder:
  - each root line's true evidence (score of the true tree without it, minus the true tree's), its presence in the
    candidate pool of the nine builders and in each builder's tree;
  - node refinement from the laterals alone (the tree with the root lines removed): add the best paying line while
    one exists, at nodes of at most 9 children by enumeration and at wider ones by
      local0.5   a local search (the best pair of children, then one-child moves while one scores better);
      single0    the node MIQP (search.best_addition), the draft's price;
      single0.5  the node MIQP with the extended-BIC price, gamma 0.5;
    which root lines each arm recovers and which lines it adds wrongly;
  - the flip test of every added line at its node (node_study.node_label, node MIQP with the line excluded): decided
    or open. Only the node MIQP gives this bound; a local search can find a line but cannot certify it.

Predictions written before the run: root lines whose evidence is positive are found by the node MIQP more often than
by the local search and than any builder's tree, and the gap grows with the junction's width; some are in no builder's
tree; with gamma 0.5 the controls get no false line; no decided line is wrong.

Added after the first 37 rows (widths 12 and 16, r = 0.003 to 0.008): r = 0.0015 and 0.002 ohm. Those rows showed the
builders proposing almost every root line from 0.005 ohm on, and missing half of them at 0.003 ohm (relative length
about 0.04 of the root-to-meter resistance), where the node MIQP found all of them: the short near-root lines of the
earlier EU LV study (experiments/node_miqp/README.md, N5). The shorter levels test that regime; reported apart.

(An earlier version also had two sibling root lines side by side. There one line at a time adds their union first,
whatever the search, because the union of two lines whose loads move together stands in for both; that is a matter of
adding lines one at a time, not of finding them, and it is left out of the main case.)

Run from topology_study:
    python -m ptid.runs.node_roots run --workers 8      # experiments/node_miqp/roots.jsonl (resumable)
    python -m ptid.runs.node_roots summary
    python -m ptid.runs.node_roots run --widths 12 --seeds 51 --r 0.005 --out roots_try.jsonl    # a quick look
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np

from ptid.method.builders import NAMES, builder_trees
from ptid.runs.node_study import (Feeder, Simulator, feeder_dict, local_loop, model_of, node_label, price_loop,
                                  read_rows, root_feeder, root_lines, score, truth, write_row)

OUT = Path(__file__).resolve().parents[2]/"experiments"/"node_miqp"/"roots.jsonl"
WIDTHS = (12, 16, 20)
PER = 3
R_LEVELS = (.0015, .002, .003, .005, .008)    # .0015 and .002 added after the first 37 rows (see the docstring)
SEEDS = (51, 52, 53, 54)
ARMS = (("local0.5", local_loop, .5),             # a local search at nodes wider than 9 children
        ("single0", price_loop, 0.),              # the node MIQP there, the draft's price
        ("single0.5", price_loop, .5))            # the node MIQP there, extended-BIC price (gamma 0.5)
NAMES_ARMS = [name for name, _, _ in ARMS]
MAIN = "single0.5"


def layout(name, L):
    if name == "one":
        k = round(.7*L)
        return dict(k_root=1, per_root=k, trunk_laterals=L-k)
    k = round(.8*L)
    return dict(k_root=1, per_root=k, trunk_laterals=L-k, inner=round(.4*L))


def layout_of(f):
    if not f["roots"]:
        return "control"
    return "chain" if len(f["roots"]) > 1 else "one"


def instances(widths, r_levels, seeds):
    for L in widths:
        for seed in seeds:
            yield Feeder(blocks=L, per=PER, seed=seed)                          # control
            for name in ("one", "chain"):
                for r in r_levels:
                    yield root_feeder(per=PER, r=r, seed=seed, **layout(name, L))


def one(f, seconds, label_seconds):
    t0 = time.perf_counter()
    data = Simulator(f).history()
    model, load = model_of(data)
    true, flat, _ = truth(f)
    roots = root_lines(f)
    trees = builder_trees(data["P"], data["Q"], data["Y"], data["V0sq"], None)
    pool = {C for t in trees for C in t}
    h_true = score(model, true)
    row = dict(feeder=feeder_dict(f), load_modelled=load, N=model.N, beta=model.alpha*model.N/2,
               roots=[dict(line=list(rl), meters=len(rl),
                           evidence=score(model, [C for C in true if C != rl])-h_true,
                           in_pool=rl in pool, builders=[n for n, t in zip(NAMES, trees) if rl in t]) for rl in roots],
               laterals_in_pool=sum(b in pool for b in flat), pool_size=len(pool), arms={})
    labelled = {}                                   # the same line in the same tree has the same flip test
    for name, loop, gamma in ARMS:
        s0 = time.perf_counter()
        tree, added, info = loop(model, flat, seconds, gamma)
        labels = []
        for a in added:
            key = (tuple(tree), a)
            if key not in labelled:
                labelled[key] = node_label(model, tree, a, label_seconds)
            labels.append(labelled[key])
        row["arms"][name] = dict(
            added=[list(a) for a in added], found=[rl in set(tree) for rl in roots],
            wrong=[list(a) for a in added if a not in set(true)],
            labels=[dict(l, right=tuple(l["line"]) in set(true)) for l in labels], info=info,
            seconds=time.perf_counter()-s0)
    row["seconds"] = time.perf_counter()-t0
    return row


def run(args):
    out = Path(args.out)
    done = {r["feeder"]["name"] for r in read_rows(out)}
    todo = [f for f in instances(args.widths, args.r, args.seeds) if f.name not in done]
    print(f"{len(todo)} feeders to run ({len(done)} done) -> {out}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(one, f, args.seconds, args.label_seconds): f for f in todo}
        for job in as_completed(jobs):
            row = job.result()
            write_row(out, row)
            a = row["arms"]
            print(f"{row['feeder']['name'][:60]}: evidence {[round(x['evidence'], 1) for x in row['roots']]}, "
                  f"in pool {[x['in_pool'] for x in row['roots']]}, found local {a['local0.5']['found']}, "
                  f"node MIQP {a[MAIN]['found']}; {row['seconds']:.0f}s", flush=True)


def cells(g, arms):
    """One summary row over the feeders g: the root lines with positive true evidence (the ones a search should find)."""
    lines = [(r, k, x) for r in g for k, x in enumerate(r["roots"]) if x["evidence"] > 0]
    if not lines:
        return ["0"]+["-"]*8
    best = max(NAMES, key=lambda n: sum(n in x["builders"] for _, _, x in lines))
    found = lambda a, r, k: r["arms"][a]["found"][k]
    dec = sum(any(l["right"] and l["decision"] == "decided" and tuple(l["line"]) == tuple(x["line"])
                  for l in r["arms"][MAIN]["labels"]) for r, k, x in lines)
    return [str(len(lines)), f"{np.median([x['evidence'] for _, _, x in lines]):.1f}",
            str(sum(x["in_pool"] for _, _, x in lines)), f"{sum(best in x['builders'] for _, _, x in lines)} ({best})",
            " / ".join(str(sum(found(a, r, k) for r, k, _ in lines)) for a in arms),
            str(sum(found(MAIN, r, k) and not x["in_pool"] for r, k, x in lines)),
            str(sum(found(MAIN, r, k) and not found("local0.5", r, k) for r, k, _ in lines)) + " / " +
            str(sum(found("local0.5", r, k) and not found(MAIN, r, k) for r, k, _ in lines)),
            " / ".join(f"{np.mean([len(r['arms'][a]['wrong']) for r in g]):.2f}" for a in arms), str(dec)]


def summary(args):
    rows = read_rows(args.out)
    for r in rows:
        r["layout"] = layout_of(r["feeder"])
    ctl = [r for r in rows if r["layout"] == "control"]
    arms = [a for a in NAMES_ARMS if rows and a in rows[0]["arms"]]
    A = " / ".join(arms)
    print(f"{len(rows)} feeders from {args.out} ({len(rows)-len(ctl)} with root lines, {len(ctl)} controls)")
    print("Counted: root lines with positive true evidence. found = node refinement from the laterals alone ends with "
          "the root line in the tree; local = local search at nodes wider than 9 children, single = the node MIQP "
          "there; the number is gamma of the extended-BIC price; decided = its flip test at its node passes 4.6.\n")
    head = ("| root lines (evidence > 0) | evidence (median) | in builder pool | in best builder's tree | "
            f"found, {A} | found by {MAIN}, in no builder's tree | found by {MAIN} only / by local0.5 only | "
            f"wrong additions per feeder, {A} | decided right, {MAIN} |")
    for lay in [x for x in ("one", "chain") if any(r["layout"] == x for r in rows)]:
        seg = [r for r in rows if r["layout"] == lay]
        print(f"{lay}, by width and r ({len(seg)} feeders):\n")
        print("| laterals | r (ohm) "+head)
        print("| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |")
        for L in sorted({r["feeder"]["blocks"] for r in seg}):
            for rr in sorted({r["feeder"]["roots"][0][1] for r in seg}):
                g = [r for r in seg if r["feeder"]["blocks"] == L and r["feeder"]["roots"][0][1] == rr]
                if g:
                    print(f"| {L} | {rr:g} | "+" | ".join(cells(g, arms))+" |")
            g = [r for r in seg if r["feeder"]["blocks"] == L]
            print(f"| {L} | all | "+" | ".join(cells(g, arms))+" |")
        print(f"| all | all | "+" | ".join(cells(seg, arms))+" |\n")
    if ctl:
        print(f"Controls (no root line): {len(ctl)} feeders; wrong additions per feeder, {A}: "
              + " / ".join(f"{np.mean([len(r['arms'][a]['wrong']) for r in ctl]):.2f}" for a in arms))
    dw = " / ".join(str(sum(l["decision"] == "decided" and not l["right"] for r in rows
                            for l in r["arms"][a]["labels"])) for a in arms)
    print(f"Decided lines that are wrong (all feeders), {A}: {dw}")
    seg = [r for r in rows if r["layout"] != "control"]
    if seg:
        allx = [x for r in seg for x in r["roots"] if x["evidence"] > 0]
        print("Root lines (evidence > 0) in each builder's tree:", {n: sum(n in x["builders"] for x in allx)
                                                                   for n in NAMES}, f"of {len(allx)}")
        s = [r["seconds"] for r in rows]
        print(f"Seconds per feeder (median / max): {np.median(s):.0f} / {max(s):.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "summary"))
    p.add_argument("--out", default=str(OUT))
    ints = lambda s: tuple(int(x) for x in s.split(","))
    p.add_argument("--widths", type=ints, default=WIDTHS)
    p.add_argument("--r", type=lambda s: tuple(float(x) for x in s.split(",")), default=R_LEVELS)
    p.add_argument("--seeds", type=ints, default=SEEDS)
    p.add_argument("--seconds", type=float, default=120., help="node MIQP limit in node refinement")
    p.add_argument("--label-seconds", type=float, default=60., help="node MIQP limit of each flip test")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    main()
