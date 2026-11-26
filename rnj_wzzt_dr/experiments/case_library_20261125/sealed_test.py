"""The registered sealed identification test (PROTOCOL_ID.md): runner and analysis, fixed before the sealed set is opened.

    python -m rnj_wzzt_dr.experiments.case_library_20261125.sealed_test run --stream test [--workers 12]
    python -m rnj_wzzt_dr.experiments.case_library_20261125.sealed_test analyse --stream test

`--stream dev --names a,b,c` runs the same code on development cases (dry run; it never opens the sealed set).
Every case file is checked against the SHA-256 in its manifest before it is used.

Per case (the same readings for every method: `build.to_input` defaults, hourly, 144 + 48 values, meter seed 0):
  blind    the final method without the PCC (identification/final.py, default Settings, lam = 1.96)
  source   the final method with the PCC: S1 tree, S2 located tree, path, places, source evidence
  plain    the final method without the PCC and without the structural prior (lam = 0); no-source twins only (secondary)
  blind|<builder>|<tol>      RNJ, NJ, RG on the blind matrix (combo.setup's R), every tolerance
  projected|<builder>|<tol>  the same on the source-projected matrix (source.setup_projected's R)
A method that fails gets an empty tree for that case (failure rule); its trace is kept in `errors`.
Output: outputs/case_library_20261125/sealed_<stream>.jsonl (resumable: finished cases are skipped).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import pickle
import time
import traceback
from collections import defaultdict
from multiprocessing import get_context

import numpy as np

from . import build, compare_dev as cd

TOLERANCES = cd.TOLERANCES
BUILDERS = ("rnj", "nj", "rg")
# Development-best tolerances (PROTOCOL_ID.md section 3): per-case F1 against the identifiable truth, averaged within a feeder,
# equal weight per size class, on the random development feeders (the sealed set holds random feeders only).
TOL_BLIND = dict(rnj=.04, nj=.04, rg=.16)          # no-source development cases, blind matrix (compare_dev.jsonl)
TOL_PROJECTED = dict(rnj=.08, nj=.04, rg=.24)      # source development cases, projected matrix (compare_source.jsonl)
CLASSES = ("small", "medium", "large", "huge")
THRESHOLD = 4.6                                    # half-BIC units, Bayes factor 100
RESAMPLES, SEED, LEVEL = 10_000, 20261125, .05


def out_path(stream):
    return build.OUT/f"sealed_{stream}.jsonl"


def inner(f, n):
    return {tuple(sorted(C)) for C in f if 1 < len(C) < n}


def builders(R, n):
    from rnj_wzzt_dr.identification import trees
    shared, depth, _ = trees.geometry(R, R, 1.)
    make = dict(rnj=trees.rnj, nj=trees.nj, rg=trees.rg)
    return {f"{b}|{t}": cd.as_lists(inner(make[b](shared, depth, t), n)) for b in BUILDERS for t in TOLERANCES}


def run_one(job):
    stream, name, sha = job
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    from rnj_wzzt_dr.identification import final
    try:
        data = (build.OUT/stream/f"{name}.pkl").read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"{name}: the case file does not match its manifest hash")
        rec = pickle.loads(data)
        inp = build.to_input(rec); n = inp.n
        tr, va = p2.series(inp)
    except Exception as exc:
        return dict(name=name, error=repr(exc), trace=traceback.format_exc()[-1500:])
    src_ = rec["source"]
    out = dict(name=name, group=rec["group"], set=rec["set"], n=n, source=src_ and src_["where"], share=src_ and src_["share"],
               kind=src_ and src_["kind"], truth=cd.as_lists(inner(rec["truth"], n)),
               true_path=[list(C) for C in src_["path"]] if src_ else None, methods={}, errors={})
    try:                                                                            # blind
        t0 = time.perf_counter()
        r = final.fit(tr, va)
        out["blind"] = dict(lines=cd.as_lists(inner(r["tree"], n)), seconds=time.perf_counter()-t0,
                            certified=bool(r["record"]["certified"]), margins=r["record"].get("evidence"))
        out["methods"].update({f"blind|{k}": v for k, v in builders(r["S"]["R"], n).items()})
    except Exception:
        out["errors"]["blind"] = traceback.format_exc()[-1500:]
    try:                                                                            # with the PCC
        t0 = time.perf_counter()
        r = final.fit(tr, va, inp.hist["Ppcc"], inp.hist["Qpcc"])
        out["source_aware"] = dict(s1=cd.as_lists(inner(r["tree"], n)), s2=cd.as_lists(inner(r["located"], n)),
                                   seconds=time.perf_counter()-t0, certified=bool(r["record"]["certified"]),
                                   evidence=float(r["source_evidence"]), path=[list(C) for C in r["path"]],
                                   places=[[float(c), None if D is None else list(D)] for c, D in r["places"]], s2_record=r["s2"])
        out["methods"].update({f"projected|{k}": v for k, v in builders(r["S"]["R"], n).items()})
    except Exception:
        out["errors"]["source_aware"] = traceback.format_exc()[-1500:]
    if src_ is None:                                                                # secondary ablation
        try:
            t0 = time.perf_counter()
            r = final.fit(tr, va, st=final.Settings(lam=0.))
            out["plain"] = dict(lines=cd.as_lists(inner(r["tree"], n)), seconds=time.perf_counter()-t0)
        except Exception:
            out["errors"]["plain"] = traceback.format_exc()[-1500:]
    return json.loads(json.dumps(out, default=plain_value))


def plain_value(o):
    """JSON form of the solver records' NumPy numbers and sets."""
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset, tuple)):
        return sorted(map(plain_value, o), key=str) if isinstance(o, (set, frozenset)) else list(o)
    return str(o)


def run(stream, workers, names=None):
    manifest = json.loads((build.HERE/f"manifest_{stream}.json").read_text(encoding="utf-8"))
    if stream == "dev" and not names:
        raise SystemExit("a development dry run needs --names")
    path = out_path(stream)
    done = set()
    if path.exists():
        done = {r["name"] for r in map(json.loads, path.read_text().splitlines()) if "error" not in r}
    todo = [r for r in sorted(manifest, key=lambda r: (-r["n"], r["name"]))            # largest first: the long cases start early
            if r["name"] not in done and (names is None or r["name"] in names)]
    print(f"{stream}: {len(todo)} cases to run, {len(done)} done", flush=True)
    with open(path, "a") as fh, get_context("spawn").Pool(workers, maxtasksperchild=4) as pool:
        for k, row in enumerate(pool.imap_unordered(run_one, [(stream, r["name"], r["sha256"]) for r in todo]), 1):
            fh.write(json.dumps(row)+"\n"); fh.flush()
            print(f"[{k}/{len(todo)}] {row['name']} {row.get('error', '')} {sorted(row.get('errors', {}))}", flush=True)


# ------------------------------------------------ analysis (registered)
def size_class(row):
    return row["name"].split("_")[1]


def f1(row, lines):
    t, g = {tuple(C) for C in row["truth"]}, {tuple(C) for C in lines}
    return 1. if not t and not g else 2*len(t & g)/(len(t)+len(g))


def lines_of(row, method):
    """Returned lines of a method; an empty tree when the method failed (failure rule)."""
    if method == "blind":
        return row.get("blind", {}).get("lines", [])
    if method == "plain":
        return row.get("plain", {}).get("lines", [])
    if method in ("s1", "s2"):
        return row.get("source_aware", {}).get(method, [])
    return row["methods"].get(method, [])


def covered(row):
    """The true place is within 4.6 half-BIC units of the best: the smallest returned S2 line that holds the source's
    attachment (the smallest true path line), or the top when none does, has a place cost of at most 4.6."""
    sa = row.get("source_aware")
    if not sa:
        return 0.
    node = min((tuple(x) for x in row["true_path"]), key=len)
    chain = [tuple(x) for x in sa["s2"] if set(node) <= set(x)]
    D = min(chain, key=len) if chain else None
    places = {(None if d is None else tuple(d)): c for c, d in sa["places"]}
    c = places.get(D)
    return float(c is not None and c <= THRESHOLD)


def alarm(row):
    sa = row.get("source_aware")
    return float(bool(sa) and sa["evidence"] > THRESHOLD)


def feeder_means(rows, value):
    """{class: {feeder: mean of value over its cases}}."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[size_class(r)][r["group"]].append(value(r))
    return {c: {g: float(np.mean(v)) for g, v in d.items()} for c, d in out.items()}


def estimate(fm):
    return float(np.mean([np.mean(list(fm[c].values())) for c in CLASSES if c in fm]))


def bootstrap(fm, seed=SEED, resamples=RESAMPLES):
    """Feeders resampled with replacement within each size class; equal weight per class."""
    rng = np.random.default_rng(seed)
    means = []
    for c in CLASSES:
        if c not in fm:
            continue
        v = np.array(list(fm[c].values()))
        idx = rng.integers(0, len(v), size=(resamples, len(v)))
        means.append(v[idx].mean(axis=1))
    return np.mean(means, axis=0)


def two_sided(b, null):
    """Bootstrap p-value for 'the statistic differs from null'."""
    lo, hi = (1+np.sum(b <= null))/(len(b)+1), (1+np.sum(b >= null))/(len(b)+1)
    return float(min(1., 2*min(lo, hi)))


def holm(ps):
    """Holm step-down at LEVEL: {name: rejected}."""
    order = sorted(ps, key=ps.get)
    out, going = {}, True
    for k, h in enumerate(order):
        going = going and ps[h] <= LEVEL/(len(order)-k)
        out[h] = going
    return out


def hypotheses(rows):
    none = [r for r in rows if r["source"] is None]
    some = [r for r in rows if r["source"] is not None]
    res = {}
    for h, sub, a, b in (("H1", none, "blind", f"blind|rnj|{TOL_BLIND['rnj']}"),
                         ("H2", some, "s2", f"projected|rnj|{TOL_PROJECTED['rnj']}"),
                         ("H3", some, "s2", "blind")):
        fm = feeder_means(sub, lambda r, a=a, b=b: f1(r, lines_of(r, a))-f1(r, lines_of(r, b)))
        bs = bootstrap(fm)
        res[h] = dict(estimate=100*estimate(fm), ci=[100*float(np.percentile(bs, 2.5)), 100*float(np.percentile(bs, 97.5))],
                      p=two_sided(bs, 0.), cases=len(sub), sign_ok=estimate(fm) > 0)
    fc, fa = feeder_means(some, covered), feeder_means(none, alarm)
    bc, ba = bootstrap(fc), bootstrap(fa, seed=SEED+1)
    pc = min(1., 2*(1+np.sum(bc <= .85))/(len(bc)+1))
    pa = min(1., 2*(1+np.sum(ba >= .10))/(len(ba)+1))
    res["H4"] = dict(coverage=100*estimate(fc), coverage_ci=[100*float(np.percentile(bc, 2.5)), 100*float(np.percentile(bc, 97.5))],
                     false_alarms=100*estimate(fa), false_alarm_ci=[100*float(np.percentile(ba, 2.5)), 100*float(np.percentile(ba, 97.5))],
                     p=float(max(pc, pa)), cases=len(some)+len(none),
                     sign_ok=estimate(fc) >= .90 and estimate(fa) <= .05)
    rejected = holm({h: v["p"] for h, v in res.items()})
    for h, v in res.items():
        v["holds"] = bool(rejected[h] and v["sign_ok"])
    return res


def pooled(rows, method):
    tp = fp = fn = root = root_t = exact = 0
    for r in rows:
        t, g = {tuple(C) for C in r["truth"]}, {tuple(C) for C in lines_of(r, method)}
        tp += len(t & g); fp += len(g-t); fn += len(t-g); exact += g == t
        big = {C for C in t if len(C) > r["n"]/2}
        root += len(big & g); root_t += len(big)
    rec, pre = tp/max(tp+fn, 1), tp/max(tp+fp, 1)
    return dict(recall=100*rec, precision=100*pre, f1=200*rec*pre/max(rec+pre, 1e-9), root=f"{root}/{root_t}", exact=exact,
                false_per_case=fp/max(len(rows), 1), mean_case_f1=100*float(np.mean([f1(r, lines_of(r, method)) for r in rows])))


def analyse(stream):
    rows = [json.loads(x) for x in out_path(stream).read_text().splitlines()]
    broken = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    print(f"{stream}: {len(rows)} cases, {len(broken)} unreadable, method failures:",
          {k: sum(k in r["errors"] for r in rows) for k in ("blind", "source_aware", "plain")})
    print("\nPRIMARY (feeder-cluster bootstrap within size class, equal class weight, Holm over H1-H4 at 5%)")
    res = hypotheses(rows)
    for h, v in res.items():
        print(f"  {h}: " + ", ".join(f"{k} {v[k]:.4g}" if k == "p" else f"{k} {v[k]:.1f}" if isinstance(v[k], float) else f"{k} {v[k]}"
                                     for k in v if k not in ("ci", "coverage_ci", "false_alarm_ci"))
              + "".join(f", {k} [{v[k][0]:.1f}, {v[k][1]:.1f}]" for k in ("ci", "coverage_ci", "false_alarm_ci") if k in v))
    claim = all(res[h]["holds"] for h in ("H1", "H2", "H3"))
    print(f"  claim 'better than the distance methods, and better because it knows the unmetered load': {'HOLDS' if claim else 'does not hold'};"
          f" location claim (H4): {'HOLDS' if res['H4']['holds'] else 'does not hold'}")
    print("\nSECONDARY (pooled counts; no decision rests on them)")
    none = [r for r in rows if r["source"] is None]
    some = [r for r in rows if r["source"] is not None]
    for label, sub, methods in (
            ("no source", none, ["blind", "plain"]+[f"blind|{b}|{t}" for b in BUILDERS for t in TOLERANCES]),
            ("with a source", some, ["s2", "s1", "blind"]+[f"projected|{b}|{t}" for b in BUILDERS for t in TOLERANCES])):
        for c in CLASSES+("all",):
            rs = [r for r in sub if c == "all" or size_class(r) == c]
            if not rs:
                continue
            print(f"  {label}, {c} ({len(rs)})")
            for m in methods:
                tag = " (dev-best)" if m in (*(f"blind|{b}|{t}" for b, t in TOL_BLIND.items()), *(f"projected|{b}|{t}" for b, t in TOL_PROJECTED.items())) else ""
                if "|" in m and not tag and c != "all":
                    continue
                p = pooled(rs, m)
                print(f"    {m+tag:28s} F1 {p['f1']:5.1f}  prec {p['precision']:5.1f}  rec {p['recall']:5.1f}  case-F1 {p['mean_case_f1']:5.1f}"
                      f"  root {p['root']:>7s}  exact {p['exact']:3d}  false/case {p['false_per_case']:.1f}")
            for b in BUILDERS:
                pre = "blind" if label == "no source" else "projected"
                best = max(TOLERANCES, key=lambda t: pooled(rs, f"{pre}|{b}|{t}")["mean_case_f1"])
                print(f"    {b} best in hindsight: {best} (case-F1 {pooled(rs, f'{pre}|{b}|{best}')['mean_case_f1']:.1f})")
    print("\n  by source place / share / kind: S2 case-F1, blind case-F1, coverage, evidence > 4.6")
    for key in ("source", "share", "kind"):
        for v in sorted({r[key] for r in some}, key=str):
            rs = [r for r in some if r[key] == v]
            print(f"    {key}={v} ({len(rs)}): S2 {pooled(rs, 's2')['mean_case_f1']:.1f}  blind {pooled(rs, 'blind')['mean_case_f1']:.1f}"
                  f"  coverage {100*np.mean([covered(r) for r in rs]):.0f}%  evidence > 4.6 {100*np.mean([alarm(r) for r in rs]):.0f}%")
    print("\n  certified and time (seconds per case, one solver thread, several cases in parallel)")
    for c in CLASSES:
        rs = [r for r in rows if size_class(r) == c]
        for part in ("blind", "source_aware"):
            sec = [r[part]["seconds"] for r in rs if part in r]
            cert = sum(r[part]["certified"] for r in rs if part in r)
            if sec:
                print(f"    {c:6s} {part:12s} certified {cert}/{len(rs)}  median {np.median(sec):6.0f}  p90 {np.percentile(sec, 90):6.0f}  max {max(sec):6.0f}")
    print("\n  margin calibration (blind, pool-step lines): share true by margin class")
    bins = [(-np.inf, 0.), (0., 2.), (2., THRESHOLD), (THRESHOLD, 10.), (10., np.inf)]
    tally = np.zeros((len(bins), 2))
    for r in rows:
        t = {tuple(C) for C in r["truth"]}
        for e in (r.get("blind", {}).get("margins") or []):
            k = next(i for i, (a, b) in enumerate(bins) if a <= e["margin"] < b)
            tally[k] += (tuple(sorted(e["line"])) in t, 1)
    for (a, b), (hit, tot) in zip(bins, tally):
        if tot:
            print(f"    margin [{a}, {b}): {int(hit)}/{int(tot)} true ({100*hit/tot:.0f}%)")
    (build.OUT/f"sealed_{stream}_primary.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "analyse"))
    ap.add_argument("--stream", required=True, choices=("dev", "test"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--names", default="")
    args = ap.parse_args()
    names = set(args.names.split(",")) if args.names else None
    run(args.stream, args.workers, names) if args.what == "run" else analyse(args.stream)


if __name__ == "__main__":
    main()
