"""WZZT-C development runs (prototype, not registered): run against WZZT-R on the 130 Phase 2 development inputs,
then tabulate and diagnose.

    python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_combo run [--tag dev] [--blocks builder|validated]
    python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_combo table [--tag dev]

Output: outputs/wzzt_evidence_20261124/combo/<tag>.jsonl. WZZT-R's families are read from the Phase 2 ladder
(variant R); both families are scored on the same design (same transform, noise level and price).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import traceback
from collections import Counter
from multiprocessing import get_context
from statistics import median

from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture

OUTC = capture.OUT/"combo"
P2 = capture.OUT/"phase2"


def _job(args):
    r, kwargs = args
    from rnj_wzzt_dr.identification import combo, revised as rv
    from rnj_wzzt_dr.identification.trees import basis, canonical
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    try:
        inp = capture.load_input(r["cond"], r["name"]); tr, va = p2.series(inp); n = inp.n
        out = combo.fit(tr, va, **kwargs)
        S = combo.setup(tr, va)
        crit = lambda f: rv.exact_fit(S["design"], f, 2.)["objective"]+S["alpha"]*rv.internal_count(f, n)
        fam_r = canonical([*basis(n), *map(tuple, r["selected"])])
        internal = lambda f: sorted({tuple(sorted(C)) for C in f if 1 < len(C) < n})
        return dict(cond=r["cond"], name=r["name"], n=n, combo=internal(out["selected"]), wzzt_r=internal(fam_r),
                    J_combo=crit(out["selected"]), J_wzzt_r=crit(fam_r), record=out["record"],
                    timing=out["timing"], wall_r=r["wall"], kwargs=kwargs)
    except Exception as exc:
        return dict(cond=r["cond"], name=r["name"], error=repr(exc), trace=traceback.format_exc()[-1500:])


def _sources_job(args):
    """Each source's terminal lines and blocks against the truth, and the three block rules over all sources."""
    r, limit, k_max = args
    import math
    from rnj_wzzt_dr.identification import combo, revised as rv
    from rnj_wzzt_dr.identification.trees import basis, canonical, compatible
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    try:
        inp = capture.load_input(r["cond"], r["name"]); tr, va = p2.series(inp); n = inp.n
        truth = {tuple(sorted(C)) for C in inp.truth_key if 1 < len(C) < n}
        S = combo.setup(tr, va)
        def judge_blocks(blocks):
            wrong = [B for B in blocks if len(B) > 1 and any(not compatible(B, T) for T in truth)]
            reach = [T for T in truth if all(compatible(T, B) for B in blocks)]
            return dict(k=len(blocks), multi=sum(len(B) > 1 for B in blocks), wrong=len(wrong),
                        reachable=len(reach), near_reachable=sum(len(T) > n/2 for T in reach))
        out = dict(cond=r["cond"], name=r["name"], n=n, truth=len(truth),
                   truth_terminal=sum(len(T) <= limit*n for T in truth), truth_near=sum(len(T) > n/2 for T in truth))
        lines = {}
        for name in combo.SOURCES:
            fam = combo.source_family(S, name)
            cand = [C for C in fam if 1 < len(C) <= limit*n]
            lines[name] = cand
            out[name] = dict(cand=len(cand), true=len(set(cand) & truth), **judge_blocks(combo.blocks_of(cand, n)))
        allc = canonical(C for v in lines.values() for C in v)
        out["union"] = dict(cand=len(allc), true=len(set(allc) & truth))
        base = canonical(basis(n))
        rec = rv.select_miqp(S["design"], allc, base, S["alpha"], 2., 240., seed=7, gap=1e-4)
        kept = [C for C in rec.get("family", ()) if 1 < len(C) < n]
        out["validated"] = dict(status=rec["status"], kept=len(kept), true=len(set(kept) & truth),
                                **judge_blocks(combo.blocks_of(kept, n)))
        prob = rv.Problem(rv.GramFitter(S["design"], [*allc, *base], base, 2.), allc, base, S["alpha"])
        margin_of = combo.line_margins(prob, kept, allc, S["N"])
        for m in (6., 4., 2., 0., -math.inf):
            blocks = combo.margin_blocks(kept, margin_of, m, n, limit)
            if m == 6.:
                out["margin6"] = judge_blocks(blocks)
            if len(blocks) <= k_max:
                break
        out["margin"] = dict(used=m, **judge_blocks(blocks))
        out["margin_lines"] = [dict(size=len(C), margin=v, true=C in truth) for C, v in margin_of.items()]
        return out
    except Exception as exc:
        return dict(cond=r["cond"], name=r["name"], error=repr(exc), trace=traceback.format_exc()[-1500:])


def sources(tag, limit, k_max, workers):
    OUTC.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (P2/"ladder.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["variant"] == "R"]
    path = OUTC/f"sources_{tag}.jsonl"
    with open(path, "w") as fh, get_context("spawn").Pool(workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(_sources_job, [(r, limit, k_max) for r in rows]), 1):
            fh.write(json.dumps(rec)+"\n"); fh.flush()
            if "error" in rec or k % 25 == 0:
                print(k, rec["name"], rec.get("error", ""), flush=True)


def sources_table(tag):
    from rnj_wzzt_dr.identification import combo
    rows = [json.loads(l) for l in (OUTC/f"sources_{tag}.jsonl").read_text().splitlines()]
    err = [r for r in rows if "error" in r]; rows = [r for r in rows if "error" not in r]
    print(f"sources_{tag}: {len(rows)} inputs, {len(err)} errors" + (f" e.g. {err[0]['error']}" if err else ""))
    tt = sum(r["truth_terminal"] for r in rows); tr = sum(r["truth"] for r in rows); tn = sum(r["truth_near"] for r in rows)
    print(f"true lines {tr} (terminal side {tt}, near root {tn}); per method: terminal candidates (recall of true terminal "
          f"lines, precision), blocks (median k, wrong multi-meter blocks, inputs with a wrong block), true lines left "
          f"reachable (all, near root)")
    def line(name, key, has_cand=True):
        cand = sum(r[key]["cand" if has_cand else "kept"] for r in rows); true = sum(r[key]["true"] for r in rows)
        ks = sorted(r[key]["k"] for r in rows) if "k" in rows[0][key] else None
        wrong = sum(r[key]["wrong"] for r in rows) if ks else None
        multi = sum(r[key]["multi"] for r in rows) if ks else None
        inp_wrong = sum(r[key]["wrong"] > 0 for r in rows) if ks else None
        reach = sum(r[key]["reachable"] for r in rows) if ks else None
        nreach = sum(r[key]["near_reachable"] for r in rows) if ks else None
        s = f"  {name:22s} cand {cand:5d} recall {100*true/tt:5.1f}% precision {100*true/max(cand, 1):5.1f}%"
        if ks:
            s += (f" | k median {ks[len(ks)//2]:2d} max {ks[-1]:2d} | wrong blocks {wrong}/{multi} on {inp_wrong} inputs"
                  f" | reachable {100*reach/tr:5.1f}% near root {100*nreach/max(tn, 1):5.1f}%")
        print(s)
    for name in combo.SOURCES:
        line(name, name)
    line("union of all sources", "union")
    line("validated (score)", "validated", has_cand=False)
    for key in ("margin6", "margin"):
        ks = sorted(r[key]["k"] for r in rows)
        print(f"  {key:22s} {'':45s} | k median {ks[len(ks)//2]:2d} max {ks[-1]:2d} | wrong blocks "
              f"{sum(r[key]['wrong'] for r in rows)}/{sum(r[key]['multi'] for r in rows)} on "
              f"{sum(r[key]['wrong'] > 0 for r in rows)} inputs | reachable "
              f"{100*sum(r[key]['reachable'] for r in rows)/tr:5.1f}% near root "
              f"{100*sum(r[key]['near_reachable'] for r in rows)/max(tn, 1):5.1f}%")
    used = Counter(str(r["margin"]["used"]) for r in rows)
    print("  margin actually used (to keep k <= k_max):", dict(used))
    ml = [l for r in rows for l in r["margin_lines"]]
    for lo, hi in ((-1e9, 0), (0, 2), (2, 6), (6, 20), (20, 1e12)):
        sub = [l for l in ml if lo <= l["margin"] < hi]
        if sub:
            print(f"  selected lines with margin in [{lo:g}, {hi:g}): {len(sub):4d}, true {100*sum(l['true'] for l in sub)/len(sub):5.1f}%")


def _evidence_job(r):
    """Kept lines (exact selection over all sources' candidates of any size) with their evidence, and the truth."""
    from rnj_wzzt_dr.identification import combo, revised as rv
    from rnj_wzzt_dr.identification.trees import basis, canonical
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    try:
        inp = capture.load_input(r["cond"], r["name"]); tr, va = p2.series(inp); n = inp.n
        truth = sorted({tuple(sorted(C)) for C in inp.truth_key if 1 < len(C) < n})
        S = combo.setup(tr, va)
        cand = canonical(C for name in combo.SOURCES for C in combo.source_family(S, name) if 1 < len(C) < n)
        base = canonical(basis(n))
        rec = rv.select_miqp(S["design"], cand, base, S["alpha"], 2., 240., seed=7, gap=1e-4)
        kept = [C for C in rec.get("family", ()) if 1 < len(C) < n]
        prob = rv.Problem(rv.GramFitter(S["design"], [*cand, *base], base, 2.), cand, base, S["alpha"])
        ev = combo.line_evidence(prob, kept, cand, S["N"])
        return dict(cond=r["cond"], name=r["name"], n=n, truth=[list(C) for C in truth], status=rec["status"],
                    candidates=len(cand), kept=[dict(line=list(C), **ev[C]) for C in kept])
    except Exception as exc:
        return dict(cond=r["cond"], name=r["name"], error=repr(exc), trace=traceback.format_exc()[-1500:])


def evidence(tag, workers):
    OUTC.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (P2/"ladder.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["variant"] == "R"]
    with open(OUTC/f"evidence_{tag}.jsonl", "w") as fh, get_context("spawn").Pool(workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(_evidence_job, rows), 1):
            fh.write(json.dumps(rec)+"\n"); fh.flush()
            if "error" in rec or k % 25 == 0:
                print(k, rec["name"], rec.get("error", ""), flush=True)


def evidence_sweep(tag, k_max=9):
    """Blocks as a function of the one parameter: the evidence margin m (or the posterior threshold)."""
    import math
    from rnj_wzzt_dr.identification import combo
    from rnj_wzzt_dr.identification.trees import compatible
    rows = [json.loads(l) for l in (OUTC/f"evidence_{tag}.jsonl").read_text().splitlines()]
    err = [r for r in rows if "error" in r]; rows = [r for r in rows if "error" not in r]
    print(f"evidence_{tag}: {len(rows)} inputs, {len(err)} errors")
    lines = [(l, tuple(l["line"]) in {tuple(t) for t in r["truth"]}, len(l["line"])/r["n"]) for r in rows for l in r["kept"]]
    print("\nCalibration of the kept lines (all sizes): margin band -> share true; posterior band -> share true")
    for lo, hi in ((-1e9, 0), (0, 2), (2, 4.6), (4.6, 6.9), (6.9, 10), (10, 20), (20, 50), (50, 1e12)):
        sub = [t for l, t, f in lines if lo <= l["margin"] < hi]
        if sub:
            print(f"  margin [{lo:g}, {hi:g}): {len(sub):4d} lines, true {100*sum(sub)/len(sub):5.1f}%")
    tt = sum(len(r["truth"]) for r in rows)
    tn = sum(sum(len(t) > r["n"]/2 for t in r["truth"]) for r in rows)
    for limit in (.5, 1.):
        print(f"\nBlocks from kept lines with at most {limit:g} n meters, as a function of the margin m (half-BIC units;"
              f" Bayes factor e^m against the strongest alternative)")
        print(f"  {'m':>6s} {'BF':>9s} | wrong blocks (inputs) | reachable all / near root | k median, max | k <= {k_max}")
        for m in (0., 2., 4.6, 6.9, 10., 15., 20., 30., 50.):
            wrong = multi = winp = reach = nreach = 0; ks = []
            for r in rows:
                n = r["n"]; truth = [tuple(t) for t in r["truth"]]
                kept = [tuple(l["line"]) for l in r["kept"]]
                mo = {tuple(l["line"]): l["margin"] for l in r["kept"]}
                blocks = combo.margin_blocks(kept, mo, m, n, limit)
                w = [B for B in blocks if len(B) > 1 and any(not compatible(B, T) for T in truth)]
                wrong += len(w); multi += sum(len(B) > 1 for B in blocks); winp += bool(w)
                rc = [T for T in truth if all(compatible(T, B) for B in blocks)]
                reach += len(rc); nreach += sum(len(T) > n/2 for T in rc); ks.append(len(blocks))
            ks.sort()
            print(f"  {m:6.1f} {math.exp(m):9.3g} | {wrong:4d} of {multi:4d} ({winp:3d}) | {100*reach/tt:5.1f}% / "
                  f"{100*nreach/max(tn, 1):5.1f}% | {ks[len(ks)//2]:3d}, {ks[-1]:3d} | {sum(k <= k_max for k in ks):3d}")


def run(tag, kwargs, workers):
    OUTC.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (P2/"ladder.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["variant"] == "R"]
    path = OUTC/f"{tag}.jsonl"
    done = {(d["cond"], d["name"]) for d in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    todo = [(r, kwargs) for r in rows if (r["cond"], r["name"]) not in done]
    with open(path, "a") as fh, get_context("spawn").Pool(workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(_job, todo), 1):
            fh.write(json.dumps(rec)+"\n"); fh.flush()
            print(k, rec["name"], rec.get("error", ""), flush=True)


def table(tag):
    from rnj_wzzt_dr.identification import combo
    from rnj_wzzt_dr.identification.trees import basis, canonical, compatible
    rows = [json.loads(l) for l in (OUTC/f"{tag}.jsonl").read_text().splitlines()]
    err = [r for r in rows if "error" in r]; rows = [r for r in rows if "error" not in r]
    print(f"{tag}: {len(rows)} inputs, {len(err)} errors")
    ceil = {}
    for line in (P2/"ceiling.jsonl").read_text().splitlines():
        c = json.loads(line)
        if "error" not in c:
            ceil[(c["cond"], c["name"])] = {tuple(s["segment"]) for s in c["segments"] if s["dJ"] > 0}
    agg = {m: Counter() for m in ("combo", "wzzt_r")}
    only, better, cause = Counter(), Counter(), Counter()
    for r in rows:
        inp = capture.load_input(r["cond"], r["name"]); n = inp.n
        truth = {tuple(sorted(C)) for C in inp.truth_key if 1 < len(C) < n}
        sup = ceil.get((r["cond"], r["name"]), set())
        exact = {}
        for m in agg:
            got = set(map(tuple, r[m])); a = agg[m]; exact[m] = got == truth
            if r["cond"] != "eulv":
                a["exact"] += got == truth
            else:
                a["sup_hit"] += len(got & sup); a["sup"] += len(sup)
            a["hit"] += len(got & truth); a["size"] += len(got)
            for C in truth:
                side = "near" if len(C) > n/2 else "term"
                a[side+"_t"] += 1; a[side+"_h"] += C in got
        if r["cond"] != "eulv" and exact["combo"] != exact["wzzt_r"]:
            only["combo" if exact["combo"] else "wzzt_r"] += 1
        tol = 1e-7*max(1, abs(r["J_wzzt_r"]))
        d = r["J_combo"]-r["J_wzzt_r"]
        better["WZZT-C" if d < -tol else ("WZZT-R" if d > tol else "tie")] += 1
        if d > tol:
            if "blocks" in r["record"]:
                blocks = [tuple(b) for b in r["record"]["blocks"]]
            else:
                tree = canonical([*basis(n), *map(tuple, r["record"]["builder_tree"])])
                blocks = combo.terminal_blocks(tree, n, r["record"]["limit"])[1]
            missing = [C for C in map(tuple, r["wzzt_r"]) if C not in set(map(tuple, r["combo"]))]
            cause["a WZZT-R line crosses a block" if any(not all(compatible(C, B) for B in blocks) for C in missing)
                  else "WZZT-R's extra lines fit the blocks (not proposed or not chosen)"] += 1
    for m, a in agg.items():
        print(f"  {m:7s} synthetic exact {a['exact']:3d} | real supported found {a['sup_hit']}/{a['sup']} | "
              f"precision {100*a['hit']/a['size']:.1f}% | recall near root {100*a['near_h']/max(a['near_t'], 1):.1f}% "
              f"({a['near_t']}), terminal {100*a['term_h']/max(a['term_t'], 1):.1f}% ({a['term_t']})")
    print("  synthetic exact recovery by only one method:", dict(only))
    print("  better (lower) score on the same design:", dict(better))
    print("  where WZZT-C scored worse, the cause:", dict(cause))
    print(f"  certified {sum(r['record']['certified'] for r in rows)}/{len(rows)}; root MIQP proved "
          f"{sum(r['record'].get('root_miqp', {}).get('status') == 2 for r in rows)}; k max {max(r['record']['k'] for r in rows)}")
    print(f"  median time WZZT-C {median(r['timing']['total'] for r in rows):.1f} s, WZZT-R {median(r['wall_r'] for r in rows):.1f} s")
    print(f"  repair added lines on {sum(any(c['added'] for c in r['record']['certificates']) for r in rows)} inputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "table", "sources", "sources_table", "evidence", "evidence_sweep"))
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--blocks", default="builder", choices=("builder", "validated", "margin"))
    ap.add_argument("--sources", default="rnj", help="comma-separated, from combo.SOURCES")
    ap.add_argument("--limit", type=float, default=None, help="block size limit as a share of n (default: 0.5 for "
                    "builder blocks, 1 for validated and margin)")
    ap.add_argument("--m", type=float, default=4.6, help="evidence margin for margin blocks (half-BIC units)")
    ap.add_argument("--k-max", type=int, default=9)
    ap.add_argument("--repairs", type=int, default=10)
    ap.add_argument("--pairs", action="store_true", help="pair moves inside a node after the single-line repair")
    ap.add_argument("--reselect", action="store_true", help="re-decide with the lines the repair added")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.stage == "run":
        run(args.tag, dict(repairs=args.repairs, blocks_from=args.blocks, sources=tuple(args.sources.split(",")),
                           limit=args.limit, k_max=args.k_max, m=args.m, pairs=args.pairs, reselect=args.reselect),
            args.workers)
    elif args.stage == "table":
        table(args.tag)
    elif args.stage == "sources":
        sources(args.tag, .5 if args.limit is None else args.limit, args.k_max, args.workers)
    elif args.stage == "sources_table":
        sources_table(args.tag)
    elif args.stage == "evidence":
        evidence(args.tag, args.workers)
    else:
        evidence_sweep(args.tag, args.k_max)


if __name__ == "__main__":
    main()
