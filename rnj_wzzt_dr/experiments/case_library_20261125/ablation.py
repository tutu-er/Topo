"""What does the exact MIQP add? Development cases of at most 25 meters only.

    python -m rnj_wzzt_dr.experiments.case_library_20261125.ablation run [--workers 2] [--boot 40]
    python -m rnj_wzzt_dr.experiments.case_library_20261125.ablation table

A. Same candidate lines (the nine sources of WZZT-C), three searches, the same score:
   exact (pool MIQP), local search (add, drop, swap; best of six starts), greedy (add the best line until none pays).
   Without a source on the plain design; on every twin on the S2 design (n + 1 leaves, tree and source path together).
B. S2 certificate by enumeration: for every node of the S2 tree, every union of its child blocks (the source leaf is a
   block too) is fitted; "certified" means that no union pays. At most 12 child blocks are enumerated per node.
C. Belief on the no-source cases: evidence margin of each line of the exact tree, against the bootstrap support of the
   lines of RNJ (0.08) and NJ (0.04), over blocks of 12 hours drawn with replacement.

Output: outputs/case_library_20261125/ablation.jsonl (resumable).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import math
import time
import traceback
from multiprocessing import get_context

import numpy as np

from . import build, compare_dev as cd

OUT = build.OUT/"ablation.jsonl"
MAX_BLOCKS = 12


# ------------------------------------------------ searches over one pool
def greedy(prob):
    """Add, one at a time, the line that lowers the score most, until none does."""
    S, best = frozenset(), prob.score(frozenset())
    while True:
        options = [(prob.score(S | {C}), C) for C in prob.addable(S)]
        if not options or min(options)[0] >= best-1e-12:
            return S, best
        best, C = min(options)
        S = S | {C}


def searches(design, lines, keep, alpha, bound, starts, seconds, warm=None, seed=7):
    """{search: (score, set of lines, seconds)} over the same lines and prices."""
    from rnj_wzzt_dr.identification import revised as rv
    out = {}
    fitter = rv.GramFitter(design, [*lines, *keep], keep, bound)
    prob = rv.Problem(fitter, lines, keep, alpha)
    t1 = time.perf_counter()
    S, s = greedy(prob)
    out["greedy"] = (s, S, time.perf_counter()-t1)
    t1 = time.perf_counter()
    best = None
    for start in starts:
        x = rv.local_search(prob, frozenset(C for C in start if C in prob.crosses), time.perf_counter()+20.)
        best = x if best is None or x[1] < best[1] else best
    out["local"] = (best[1], best[0], time.perf_counter()-t1)
    t1 = time.perf_counter()
    rec = rv.select_miqp(design, lines, keep, alpha, bound, seconds, warm=warm, seed=seed, gap=1e-5)
    fam = rec.get("family") or ()
    lines_set = frozenset(C for C in fam if C in set(prob.crosses))
    out["exact"] = (prob.score(lines_set), lines_set, time.perf_counter()-t1)
    out["status"] = rec["status"]
    return out


def certify_leaf(design, fam, alpha, bound, leaves):
    """Enumerate every union of child blocks of every node of a family over `leaves` leaves; the best improvement."""
    from rnj_wzzt_dr.identification import revised as rv
    from rnj_wzzt_dr.identification.trees import basis, canonical, children
    fam = canonical(fam)
    base = canonical([*basis(leaves)])
    start = frozenset(C for C in fam if C not in set(base))
    fitter = rv.GramFitter(design, [*base, *start], base, bound)
    current = fitter.fit([*base, *start])[0]+alpha*len(start)
    best, best_gain, skipped = None, 0., 0
    for node, blocks in children(fam, leaves).items():
        blocks = [tuple(b) for b in blocks]
        if len(blocks) < 3:
            continue
        if len(blocks) > MAX_BLOCKS:
            skipped += 1
            continue
        unions = [tuple(sorted(i for b in combo for i in b)) for k in range(2, len(blocks))
                  for combo in itertools.combinations(blocks, k)]
        unions = [U for U in unions if U not in start]
        for k in range(0, len(unions), 200):
            part = unions[k:k+200]
            fit2 = rv.GramFitter(design, [*base, *start, *part], base, bound)
            for U in part:
                gain = current-(fit2.fit([*base, *start, U])[0]+alpha*(len(start)+1))
                if gain > best_gain+1e-12:
                    best, best_gain = U, gain
    return dict(best=None if best is None else list(best), gain=float(best_gain), skipped=skipped,
                certified=best is None and skipped == 0)


# ------------------------------------------------ one case
def run_one(job):
    name, boots = job
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_confirm_c import WZZT_C
    from rnj_wzzt_dr.identification import combo, revised as rv, source as src, trees
    from rnj_wzzt_dr.identification.trees import basis, canonical
    try:
        rec = build.load("dev", name)
        inp = build.to_input(rec); n = inp.n
        tr, va = p2.series(inp)
        data = tr.join(va)
        bound, names = 2., WZZT_C["sources"]
        has = rec["source"] is not None
        out = dict(name=name, group=rec["group"], source=rec["source"] and rec["source"]["where"], n=n,
                   truth=cd.as_lists({tuple(sorted(C)) for C in rec["truth"] if 1 < len(C) < n}))
        inner = lambda S: [list(C) for C in S]
        half = None
        if not has:                                          # A on the plain design, and C
            S0 = combo.setup(tr, va)
            design, alpha, N = S0["design"], S0["alpha"], S0["N"]
            fams = [combo.source_family(S0, s) for s in names]
            lines = canonical(C for F in fams for C in F if 1 < len(C) < n)
            base = canonical(basis(n))
            res = searches(design, lines, base, alpha, bound, [[]]+[[C for C in F if 1 < len(C) < n] for F in fams[:3]], 240.)
            half = .5*N
            out["pool"] = {k: dict(score=half*(v[0]-res["exact"][0]), lines=inner(v[1]), seconds=v[2])
                           for k, v in res.items() if k in ("exact", "local", "greedy")}
            out["pool_status"], out["pool_size"] = res["status"], len(lines)
            prob = rv.Problem(rv.GramFitter(design, [*lines, *base], base, bound), lines, base, alpha)
            ev = combo.line_evidence(prob, list(res["exact"][1]), lines, N)
            out["margins"] = [[list(C), float(e["margin"])] for C, e in ev.items()]
            out["boot"] = bootstrap(inp, names, boots, n)
        # A and B on the S2 design
        up, uq = src.balance(inp.hist["Ppcc"], inp.hist["Qpcc"], data)
        S1 = src.setup_projected(tr, va, up, uq)
        T = data.p.shape[0]
        half = .5*S1["N"]
        design2 = src.LeafDesign(data, rv.Transform(T, n, S1["c"], True), math.sqrt(S1["sigma2"]), up, uq)
        fams = [combo.source_family(S1, s) for s in names]
        lines1 = canonical(C for F in fams for C in F if 1 < len(C) < n)
        pool2 = [C for C in src.with_source(lines1, n) if 1 < len(C) < n+1]
        keep2 = canonical(basis(n+1))
        tree1 = [C for C in fams[0] if 1 < len(C) < n]
        starts = [[]]+[[C for C in src.attach(tree1, n, D) if 1 < len(C) < n+1] for D in (None, *tree1[:3])]
        res2 = searches(design2, pool2, keep2, S1["alpha"], bound, starts, 240.)
        fam2 = canonical([*keep2, *res2["exact"][1]])
        out["s2"] = {k: dict(score=half*(v[0]-res2["exact"][0]), lines=inner(v[1]), seconds=v[2])
                     for k, v in res2.items() if k in ("exact", "local", "greedy")}
        out["s2_status"], out["s2_pool"] = res2["status"], len(pool2)
        out["s2_cert"] = certify_leaf(design2, fam2, S1["alpha"], bound, n+1)
        out["true_path"] = [list(C) for C in rec["source"]["path"]] if has else None
        return out
    except Exception as exc:
        return dict(name=name, error=repr(exc), trace=traceback.format_exc()[-1500:])


def bootstrap(inp, names, boots, n):
    """Bootstrap support of every line of RNJ (0.08) and NJ (0.04); blocks of 12 hours drawn with replacement."""
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    from rnj_wzzt_dr.identification import combo, revised as rv, trees
    tr, va = p2.series(inp)
    data = tr.join(va)
    T = data.p.shape[0]
    rng = np.random.default_rng(1000+n)
    counts, trees_full = {"rnj": {}, "nj": {}}, {}
    def build_lines(series):
        S = combo.setup(series)
        shared, depth, _ = trees.geometry(S["R"], S["R"], 1.)
        return {"rnj": [C for C in trees.rnj(shared, depth, .08) if 1 < len(C) < n],
                "nj": [C for C in trees.nj(shared, depth, .04) if 1 < len(C) < n]}
    full = build_lines(data)
    width = 12
    for _ in range(boots):
        starts = rng.integers(0, T-width+1, T//width)
        rows = np.concatenate([np.arange(s, s+width) for s in starts])
        s = rv.Series(data.p[rows], data.q[rows], data.y[rows], data.v0sq[rows])
        try:
            got = build_lines(s)
        except Exception:
            continue
        for k in counts:
            for C in got[k]:
                counts[k][tuple(C)] = counts[k].get(tuple(C), 0)+1
    return {k: [[list(C), counts[k].get(tuple(C), 0)/boots] for C in full[k]] for k in counts}


def run(workers, boots):
    manifest = json.loads((build.HERE/"manifest_dev.json").read_text(encoding="utf-8"))
    done = set()
    if OUT.exists():
        done = {r["name"] for r in map(json.loads, OUT.read_text().splitlines()) if "error" not in r}
    todo = [(r["name"], boots) for r in sorted(manifest, key=lambda r: (r["n"], r["name"]))
            if r["n"] <= 25 and r["name"] not in done]
    with open(OUT, "a") as fh, get_context("spawn").Pool(workers, maxtasksperchild=6) as pool:
        for k, row in enumerate(pool.imap_unordered(run_one, todo), 1):
            fh.write(json.dumps(row)+"\n"); fh.flush()
            print(f"[{k}/{len(todo)}] {row['name']} {row.get('error', '')}", flush=True)


# ------------------------------------------------ tables
def f1(tp, fp, fn):
    return 2*tp/max(2*tp+fp+fn, 1)


def table():
    rows = [r for r in map(json.loads, OUT.read_text().splitlines())]
    bad = [r["name"] for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    print(f"{len(rows)} cases; errors {bad[:5]}")
    plain = [r for r in rows if "pool" in r]
    print("\nA. Same candidate lines, no source (half-BIC units above the exact optimum)")
    for search in ("local", "greedy"):
        gaps = np.array([r["pool"][search]["score"] for r in plain])
        same = np.mean([sorted(map(tuple, r["pool"][search]["lines"])) == sorted(map(tuple, r["pool"]["exact"]["lines"])) for r in plain])
        print(f"  {search}: worse in {int((gaps > 1e-6).sum())}/{len(plain)}; mean gap {gaps.mean():.2f}, max {gaps.max():.2f}; same tree {100*same:.0f}%")
    print(f"  pool MIQP proved optimal: {sum(r['pool_status'] == 2 for r in plain)}/{len(plain)}; median pool size {np.median([r['pool_size'] for r in plain])}")
    for search in ("exact", "local", "greedy"):
        t = np.zeros(4)
        for r in plain:
            got, truth = {tuple(C) for C in r["pool"][search]["lines"]}, {tuple(C) for C in r["truth"]}
            t += [len(got & truth), len(got-truth), len(truth-got), 1]
        print(f"    {search:7s} recall {100*t[0]/(t[0]+t[2]):5.1f}  precision {100*t[0]/(t[0]+t[1]):5.1f}  F1 {100*f1(*t[:3]):5.1f}")
    print("\nA. S2 pool (tree and source path together), every twin")
    for search in ("local", "greedy"):
        gaps = np.array([r["s2"][search]["score"] for r in rows])
        print(f"  {search}: worse in {int((gaps > 1e-6).sum())}/{len(rows)}; mean gap {gaps.mean():.2f}, max {gaps.max():.2f}")
    print(f"  S2 MIQP proved optimal: {sum(r['s2_status'] == 2 for r in rows)}/{len(rows)}")
    by = [(s, sum(r["s2"][s]["seconds"] for r in rows)/len(rows)) for s in ("exact", "local", "greedy")]
    print("  mean seconds", {k: round(v, 2) for k, v in by})
    cert = [r["s2_cert"] for r in rows]
    print(f"\nB. S2 certificate by enumeration: certified {sum(c['certified'] for c in cert)}/{len(cert)};"
          f" improved by one more line in {sum(c['best'] is not None for c in cert)};"
          f" nodes wider than {MAX_BLOCKS} blocks in {sum(c['skipped'] > 0 for c in cert)} cases")
    print("\nC. Belief, no source (precision / recall of the lines kept at each threshold)")
    for label, key in (("WZZT-C margin", None), ("RNJ 0.08 bootstrap support", "rnj"), ("NJ 0.04 bootstrap support", "nj")):
        pts = []
        for r in plain:
            items = r["margins"] if key is None else r["boot"][key]
            for C, v in items:
                pts.append((v, tuple(C) in {tuple(x) for x in r["truth"]}))
        truth_total = sum(len(r["truth"]) for r in plain)
        pts.sort(reverse=True)
        print(f"  {label}: {len(pts)} lines, precision without threshold {100*np.mean([p[1] for p in pts]):.1f}%")
        for share in (.5, .7, .9, 1.):
            k = max(1, int(round(share*len(pts))))
            kept = pts[:k]
            print(f"     keep top {100*share:.0f}%: precision {100*np.mean([p[1] for p in kept]):5.1f}%, recall {100*sum(p[1] for p in kept)/truth_total:5.1f}%, threshold {kept[-1][0]:.2f}")
        # ranking quality: chance a true line ranks above a false one
        t = [v for v, o in pts if o]; f = [v for v, o in pts if not o]
        if t and f:
            auc = np.mean([[1. if a > b else .5 if a == b else 0. for b in f] for a in t])
            print(f"     chance that a true line ranks above a false one: {auc:.3f}  (false lines {len(f)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "table"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--boot", type=int, default=40)
    args = ap.parse_args()
    run(args.workers, args.boot) if args.what == "run" else table()


if __name__ == "__main__":
    main()
