"""N8, confirmation of the frozen method (experiments/node_miqp/README.md): N7 version 2 repeated on new seeds and a
second feeder layout, with nothing changed after the predictions below were written.

Registered before the run (2026-11-28, after N7 version 2): the method is identify.Settings with the line swap
(node_swap=True, swap_passes=3) and the extended-BIC price gamma = 0.5 at nodes wider than K = 9 children, which the
node MIQP solves (arm D); arm B is the same without the node MIQP (those nodes skipped). Solver limits
node_pipeline.SHORT, as in N7. The code per feeder is node_roots_e2e.one, unchanged. No setting is changed once the
run starts, and every row is reported.

Layouts (seeds 71-74, used by no earlier test):
  A  N7's layout: 12 or 16 laterals of 3 homes; one root line feeding 70% of the laterals, or a chain of two nested
     ones (80% and 40%); r = 0.002 and 0.003 ohm.
  B  a second layout: 14 or 18 laterals of 2 homes (without its root line the root holds up to 18 children, beyond
     N1's 16); one root line feeding 60% of the laterals, or a chain (75% and 35%); r = 0.003 and 0.0045 ohm, 1.5
     times A's, so that the root line's voltage drop (r times the homes behind it) is about A's. Fixed before any row.
  Controls without a root line in both (8 per layout).

Predictions (written before the run):
  1. Pooled over both layouts, D finds more root lines with positive true evidence than B (two-sided sign test on
     the lines found by one arm only, p < 0.05), and most lines found by D only come in by a swap the node MIQP made.
  2. In each layout, D's root-line recall exceeds every builder's tree and the pool of all nine builders' trees.
  3. No decided line is wrong: no extra line decided by the pool's labels, no extra line over half the meters in
     D's trees decided by the node flip test.
  4. No control tree of D has more lines than B's.
  5. Layout A repeats N7 version 2: D finds at least 60% of its root lines (76% on N7's fresh seeds), and in each
     layout D finds more lines than B on the lines found by one arm only.

Run from topology_study:
    python -m ptid.runs.node_confirm run --workers 8      # experiments/node_miqp/confirm.jsonl (resumable)
    python -m ptid.runs.node_confirm summary
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import comb
from pathlib import Path

import numpy as np

from ptid.method.builders import NAMES
from ptid.runs.node_roots import layout_of
from ptid.runs.node_roots_e2e import ARMS, one, recall_table
from ptid.runs.node_study import Feeder, read_rows, root_feeder, write_row

OUT = Path(__file__).resolve().parents[2]/"experiments"/"node_miqp"/"confirm.jsonl"
SEEDS = (71, 72, 73, 74)
LAYOUTS = {"A": dict(widths=(12, 16), per=3, r=(.002, .003), one=.7, chain=(.8, .4)),
           "B": dict(widths=(14, 18), per=2, r=(.003, .0045), one=.6, chain=(.75, .35))}


def shape(kind, L, lay):
    if kind == "one":
        k = round(lay["one"]*L)
        return dict(k_root=1, per_root=k, trunk_laterals=L-k)
    k = round(lay["chain"][0]*L)
    return dict(k_root=1, per_root=k, trunk_laterals=L-k, inner=round(lay["chain"][1]*L))


def instances(layouts, seeds):
    for seed in seeds:
        for name in layouts:
            lay = LAYOUTS[name]
            for L in lay["widths"]:
                yield Feeder(blocks=L, per=lay["per"], seed=seed)                      # control
                for kind in ("one", "chain"):
                    for r in lay["r"]:
                        yield root_feeder(per=lay["per"], r=r, seed=seed, **shape(kind, L, lay))


def layout_name(feeder):
    return next(k for k, lay in LAYOUTS.items() if lay["per"] == feeder["per"])


def run(args):
    out = Path(args.out)
    done = {r["feeder"]["name"] for r in read_rows(out)}
    todo = [f for f in instances(args.layouts, args.seeds) if f.name not in done]
    print(f"{len(todo)} feeders to run ({len(done)} done) -> {out}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(one, f): f for f in todo}
        for job in as_completed(jobs):
            row = job.result()
            write_row(out, row)
            a = row["arms"]
            print(f"{row['feeder']['name'][:60]}: evidence {[round(x['evidence'], 1) for x in row['roots']]}, "
                  + ", ".join(f"{k} {v['found']} F1 {v['f1']:.2f} swaps {len(v.get('swaps', []))} "
                              f"({v['seconds']/60:.0f} min)" for k, v in a.items()), flush=True)


def sign_p(a, b):
    n = a+b
    return min(1., 2*sum(comb(n, i) for i in range(min(a, b)+1))/2**n) if n else 1.


def paired(lines):
    """(found by D only, of them swapped in by the node MIQP, found by B only, sign test p)."""
    d = [(r, x) for r, i, x in lines if r["arms"]["D"]["found"][i] and not r["arms"]["B"]["found"][i]]
    b = sum(r["arms"]["B"]["found"][i] and not r["arms"]["D"]["found"][i] for r, i, _ in lines)
    miqp = sum(any(s["new"] == x["line"] and s["by"] == "miqp" for s in r["arms"]["D"]["swaps"]) for r, x in d)
    return len(d), miqp, b, sign_p(len(d), b)


def summary(args):
    rows = read_rows(args.out)
    for r in rows:
        r["layout"], r["name"] = layout_of(r["feeder"]), layout_name(r["feeder"])
    seg = [r for r in rows if r["layout"] != "control"]
    ctl = [r for r in rows if r["layout"] == "control"]
    print(f"{len(rows)} feeders from {args.out} ({len(seg)} with root lines, {len(ctl)} controls)\n")
    paying = [(r, i, x) for r in seg for i, x in enumerate(r["roots"]) if x["evidence"] > 0]
    groups = {f"layout {k}": [t for t in paying if t[0]["name"] == k] for k in LAYOUTS}
    groups["both layouts"] = paying
    for title, lines in groups.items():
        recall_table(lines, f"Root lines with positive true evidence, {title}")
    recall_table([t for t in paying if t[2]["share"] > .5], "... of them feeding more than half of the meters "
                 "(the EU LV class), both layouts")
    for k, lay in LAYOUTS.items():
        for kind in ("one", "chain"):
            for rr in lay["r"]:
                recall_table([t for t in paying if t[0]["name"] == k and t[0]["layout"] == kind
                              and t[0]["feeder"]["roots"][0][1] == rr], f"layout {k}, {kind}, r = {rr:g} ohm")
    print("| feeders | arm or builder | mean F1 | extra lines per feeder | decided wrong lines (pool labels) | "
          "swaps per feeder | node MIQP calls per feeder | median minutes |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for k in LAYOUTS:
        for title, g in (("with root lines", seg), ("controls", ctl)):
            g = [r for r in g if r["name"] == k]
            if not g:
                continue
            for a in ARMS:
                x = [r["arms"][a] for r in g]
                print(f"| layout {k}, {title} ({len(g)}) | {a} | {np.mean([y['f1'] for y in x]):.3f} | "
                      f"{np.mean([len(y['extra']) for y in x]):.2f} | {sum(y['decided_wrong'] for y in x)} | "
                      f"{np.mean([len(y.get('swaps', [])) for y in x]):.1f} | "
                      f"{np.mean([y['node_miqp_calls'] for y in x]):.1f} | "
                      f"{np.median([y['seconds'] for y in x])/60:.1f} |")
            best = max(NAMES, key=lambda n: np.mean([r["builder_f1"][n] for r in g]))
            print(f"| layout {k}, {title} ({len(g)}) | best builder ({best}) | "
                  f"{np.mean([r['builder_f1'][best] for r in g]):.3f} | "
                  f"{np.mean([r['builder_extra'][best] for r in g]):.2f} | - | - | - | - |")

    print("\nRegistered predictions:")
    d, miqp, b, p = paired(paying)
    print(f"  1. both layouts: found by D only {d} ({miqp} by node MIQP swaps), by B only {b}, sign test p = {p:.3g} "
          f"-> {'met' if d > b and p < .05 and 2*miqp > d else 'NOT met'}")
    ok2 = True
    for k in LAYOUTS:
        lines = groups[f"layout {k}"]
        if not lines:
            continue
        dk = sum(r["arms"]["D"]["found"][i] for r, i, _ in lines)
        best = max([*NAMES, "pool"], key=lambda n: sum((x["in_pool"] if n == "pool" else n in x["builders"])
                                                       for _, _, x in lines))
        bk = sum((x["in_pool"] if best == "pool" else best in x["builders"]) for _, _, x in lines)
        ok2 &= dk > bk
        print(f"  2. layout {k}: D {dk} of {len(lines)}, best builder tree or pool ({best}) {bk}")
    print(f"     -> {'met' if ok2 else 'NOT met'}")
    pool_wrong = sum(r["arms"][a]["decided_wrong"] for r in rows for a in ARMS)
    node_wrong = sum(r["arms"]["D"].get("extra_node_decided", 0) for r in rows)
    print(f"  3. decided wrong: pool labels {pool_wrong}, node flip test {node_wrong} "
          f"-> {'met' if pool_wrong == node_wrong == 0 else 'NOT met'}")
    more = [r["feeder"]["name"] for r in ctl if len(r["arms"]["D"]["tree"]) > len(r["arms"]["B"]["tree"])]
    print(f"  4. controls where D's tree has more lines than B's: {len(more)} of {len(ctl)} {more} "
          f"-> {'met' if not more else 'NOT met'}")
    ok5 = True
    for k in LAYOUTS:
        lines = groups[f"layout {k}"]
        if not lines:
            continue
        d, miqp, b, p = paired(lines)
        rec = sum(r["arms"]["D"]["found"][i] for r, i, _ in lines)/len(lines)
        ok5 &= d > b and (k != "A" or rec >= .6)
        print(f"  5. layout {k}: D recall {100*rec:.0f}%, found by D only {d} ({miqp} by node MIQP swaps), "
              f"by B only {b}, p = {p:.3g}")
    print(f"     -> {'met' if ok5 else 'NOT met'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "summary"))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--layouts", type=lambda s: tuple(s.split(",")), default=tuple(LAYOUTS))
    p.add_argument("--seeds", type=lambda s: tuple(int(x) for x in s.split(",")), default=SEEDS)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    main()
