"""N5, root lines: the main case of the node MIQP (experiments/node_miqp/README.md).

Twelve laterals (0.03 to 0.05 ohm, 2, 3 or 4 homes each) leave the trunk end through two short root lines of r ohm
(0.002 to 0.008, like the cable sections between a substation and its first cabinets), in two layouts:
  chain:    the trunk continues: a root line feeds 10 laterals, and a second root line inside it feeds 5 of them
            (nested near-root lines, as on a radial trunk);
  siblings: two root lines of 5 laterals each leave the trunk end side by side (two outgoing cables);
two laterals leave the trunk end directly. Controls: the same twelve laterals without root lines.

Mechanism tested: a root line is short but carries many homes. The tree builders look for its small step in a
pairwise matrix whose n(n+1)/2 entries are each noisy; the node MIQP fits one more impedance with every meter and hour.
When the builders miss the root lines, the estimated root junction has 12 children, wider than enumeration's K = 9,
so only a search over its 4082 unions can find them: exact (the node MIQP) or heuristic (a local search).

Per feeder:
  - each root line's true evidence (score of the true tree without it, minus the true tree's), its presence in the
    candidate pool of the nine builders and in each builder's tree;
  - node refinement from the laterals alone (the tree with every root line removed), five arms:
      local0.5           add the best paying line while one exists: a local search (best pair, then one-child moves)
                         at nodes wider than 9 children, enumeration at the others; extended-BIC price gamma 0.5;
      single0, single0.5 the same with the node MIQP at the wide nodes (search.best_addition), gamma 0 and 0.5;
      multi0, multi0.5   the node MIQP may also add two sibling lines jointly (search.node_miqp_multi);
    which root lines each arm recovers, which lines it adds wrongly;
  - the flip test of every added line at its node (node_study.node_label): decided or open.

Predictions written before the full run (after an 8-feeder trial of an earlier layout): the node MIQP arms find more
root lines than the local search and than any builder's tree; sibling root lines whose loads move together are the
hard case (their union can stand in for both), nested ones are not; with gamma 0.5 the controls get no false line; no
decided line is wrong; most root lines stay open on passive data (the case for probing, N6).

Run from topology_study:
    python -m ptid.runs.node_roots run --workers 6      # experiments/node_miqp/roots.jsonl (resumable)
    python -m ptid.runs.node_roots summary
    python -m ptid.runs.node_roots run --per 3 --seeds 51 --r 0.004 --out roots_try.jsonl    # a quick look
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np

from ptid.method.builders import NAMES, builder_trees
from ptid.runs.node_study import (Feeder, Simulator, feeder_dict, local_loop, model_of, node_label, price_loop,
                                  price_loop_multi, read_rows, root_feeder, root_lines, score, truth, write_row)

OUT = Path(__file__).resolve().parents[2]/"experiments"/"node_miqp"/"roots.jsonl"
LAYOUTS = {"chain": dict(k_root=1, per_root=10, trunk_laterals=2, inner=5),
           "siblings": dict(k_root=2, per_root=5, trunk_laterals=2)}
LATERALS = 12
PER = (2, 3, 4)
R_LEVELS = (.002, .004, .008)
SEEDS = (51, 52, 53, 54)
ARMS = (("local0.5", local_loop, .5),                                            # heuristic search at wide nodes
        ("single0", price_loop, 0.), ("single0.5", price_loop, .5),             # node MIQP, one line at a time
        ("multi0", price_loop_multi, 0.), ("multi0.5", price_loop_multi, .5))   # node MIQP, sibling lines jointly
NAMES_ARMS = [name for name, _, _ in ARMS]
MAIN = "multi0.5"


def layout_of(f):
    if not f["roots"]:
        return "control"
    return "chain" if any(p >= 0 for _, _, p in f["roots"]) else "siblings"


def instances(per_list, r_levels, seeds):
    for per in per_list:
        for seed in seeds:
            yield Feeder(blocks=LATERALS, per=per, seed=seed)                    # control
            for lay in LAYOUTS.values():
                for r in r_levels:
                    yield root_feeder(per=per, r=r, seed=seed, **lay)


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
                labelled[key] = node_label(model, tree, a, label_seconds, pairs=added)
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
    todo = [f for f in instances(args.per, args.r, args.seeds) if f.name not in done]
    print(f"{len(todo)} feeders to run ({len(done)} done) -> {out}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(one, f, args.seconds, args.label_seconds): f for f in todo}
        for job in as_completed(jobs):
            row = job.result()
            write_row(out, row)
            a = row["arms"][MAIN]
            print(f"{row['feeder']['name'][:60]}: roots in pool {[x['in_pool'] for x in row['roots']]}, "
                  f"found {a['found']}, wrong {len(a['wrong'])}; {row['seconds']:.0f}s", flush=True)


def summary(args):
    rows = read_rows(args.out)
    for r in rows:
        r["layout"] = layout_of(r["feeder"])
    ctl = [r for r in rows if r["layout"] == "control"]
    arms = [a for a in NAMES_ARMS if rows and a in rows[0]["arms"]]
    A = " / ".join(arms)
    print(f"{len(rows)} feeders from {args.out} ({len(rows)-len(ctl)} with root lines, {len(ctl)} controls)")
    print("Arms: local = local search at nodes wider than 9 children; single = node MIQP, one line at a time; multi = "
          "node MIQP, also two sibling lines jointly; the number is gamma of the extended-BIC price. 'found' = node "
          "refinement from the laterals alone ends with the root line in the tree; 'decided' = its flip test at its "
          "node passes 4.6.")
    for lay in [x for x in ("chain", "siblings") if any(r["layout"] == x for r in rows)]:
        seg = [r for r in rows if r["layout"] == lay]
        print(f"\n{lay} ({len(seg)} feeders):\n")
        print("| homes per lateral | r (ohm) | root lines | evidence (median) | in builder pool | in best builder's tree | "
              f"found, {A} | found by {MAIN}, not in pool | wrong additions per feeder, {A} | decided right, {MAIN} |")
        print("| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |")
        for per in sorted({r["feeder"]["per"] for r in seg}):
            for rr in sorted({r["feeder"]["roots"][0][1] for r in seg}):
                g = [r for r in seg if r["feeder"]["per"] == per and r["feeder"]["roots"][0][1] == rr]
                print("| "+" | ".join(table_row(g, arms, f"{per} | {rr:g}"))+" |")
        print("| all | all | "+" | ".join(table_row(seg, arms, "")[1:])+" |")
    if ctl:
        print(f"\nControls (no root lines): {len(ctl)} feeders; wrong additions per feeder, {A}: "
              + " / ".join(f"{np.mean([len(r['arms'][a]['wrong']) for r in ctl]):.2f}" for a in arms)
              + "; decided wrong lines " + " / ".join(
                  str(sum(l["decision"] == "decided" for r in ctl for l in r["arms"][a]["labels"])) for a in arms))
    seg = [r for r in rows if r["layout"] != "control"]
    if seg:
        dw = " / ".join(str(sum(l["decision"] == "decided" and not l["right"] for r in seg
                                for l in r["arms"][a]["labels"])) for a in arms)
        print(f"Decided lines that are wrong, feeders with root lines, {A}: {dw}")
        allx = [x for r in seg for x in r["roots"]]
        print("Root lines in each builder's tree:", {n: sum(n in x["builders"] for x in allx) for n in NAMES},
              f"of {len(allx)}")
        s = [r["seconds"] for r in rows]
        print(f"Seconds per feeder (median / max): {np.median(s):.0f} / {max(s):.0f}")


def table_row(g, arms, head):
    """Cells of one summary row over the feeders g (head: the first cells)."""
    lines = [(r, k, x) for r in g for k, x in enumerate(r["roots"])]
    if not lines:
        return [head]
    best = max(NAMES, key=lambda n: sum(n in x["builders"] for _, _, x in lines))
    found = " / ".join(str(sum(r["arms"][a]["found"][k] for r, k, _ in lines)) for a in arms)
    unique = sum(r["arms"][MAIN]["found"][k] and not x["in_pool"] for r, k, x in lines)
    wrong = " / ".join(f"{np.mean([len(r['arms'][a]['wrong']) for r in g]):.2f}" for a in arms)
    dec = sum(any(l["right"] and l["decision"] == "decided" and tuple(l["line"]) == tuple(x["line"])
                  for l in r["arms"][MAIN]["labels"]) for r, k, x in lines)
    return [head, str(len(lines)), f"{np.median([x['evidence'] for _, _, x in lines]):.1f}",
            str(sum(x["in_pool"] for _, _, x in lines)),
            f"{sum(best in x['builders'] for _, _, x in lines)} ({best})", found, str(unique), wrong, str(dec)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "summary"))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--per", type=lambda s: tuple(int(x) for x in s.split(",")), default=PER)
    p.add_argument("--r", type=lambda s: tuple(float(x) for x in s.split(",")), default=R_LEVELS)
    p.add_argument("--seeds", type=lambda s: tuple(int(x) for x in s.split(",")), default=SEEDS)
    p.add_argument("--seconds", type=float, default=120., help="node MIQP limit in node refinement")
    p.add_argument("--label-seconds", type=float, default=60., help="node MIQP limit of each flip test")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    main()
