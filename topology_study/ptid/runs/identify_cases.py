"""Development runs of Algorithm 1 (PROTOCOL.md, section 4), with the diagnostics RESULTS.md and the tests of
PROPOSALS.md use. The sealed test set is never opened: development cases are hash-checked against their manifest.

    python -m ptid.runs.identify_cases run --set dev [--one-twin | --names a,b] [--max-n 25] [--blind] [--workers 8]
    python -m ptid.runs.identify_cases run --set real [--workers 4]
    python -m ptid.runs.identify_cases summary --set dev|real [--file other.jsonl]
    python -m ptid.runs.identify_cases compare OLD.jsonl NEW.jsonl

`run` is resumable: cases already in the output file are skipped. Each row is Algorithm 1's output (without the pool)
plus these measurements, which the method itself never sees:
  truth        the identifiable true meter groupings (lines of at least 3 milliohm)
  exact_truth  every true meter grouping, also of lines below 3 milliohm
  score        precision, recall and F1 of the tree's groupings against the identifiable truth
  score_exact  the same against the exact truth (its precision: is every grouping found a physical line?)
  label_truth  per label: is its grouping true (identifiable truth) and "exact" (a physical line at all); for a grouping
               outside the tree, "nested" (it could be added to the tree) or "crossing" (it crosses a line of the tree)
  adequacy     test S1 of PROPOSALS.md: F of the leader and of the true tree against the saturated model
  complete     test S2: the last pricing found no paying line and priced every node exactly
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import statistics
import time
import traceback
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import stats

from ptid.cases import library, real
from ptid.cases.paths import RESULTS
from ptid.method.identify import Settings, identify
from ptid.method.model import balance, gls_gamma, time_basis
from ptid.method.sets import basis, compatible

STATUS = {2: "optimal", 3: "infeasible", 6: "cut off", 9: "time limit", 12: "numerical failure", None: "local flip"}


def f1(est, truth):
    """Precision, recall and F1 of a set of groupings against the true set."""
    tp = len(est & truth)
    p = tp/len(est) if est else 0.
    r = tp/len(truth) if truth else 1.
    return dict(tp=tp, est=len(est), truth=len(truth), precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.)


# ------------------------------------------------------------------ S1: adequacy against the saturated model
def adequacy(P, Q, Y, V0sq, u, trees):
    """For each tree (name -> meter groupings): F = [(RSS_tree - RSS_sat) / df1] / [RSS_sat / df2], both unbounded
    least squares under the saturated model's projection (offsets and, with u, the load's profiles removed; S on the
    meters). A tree is nested in the saturated model (R_ij = sum of r_C over the lines C holding i and j), so
    F ~ F(df1, df2) when the tree is right and the noise is Gaussian. Returns ({name: dict(F, df1, df2, p)}, sigma2)."""
    T, n = P.shape
    gamma = gls_gamma(V0sq, Y)
    E = time_basis(T, u)
    drop = lambda A: A-E@(E.T@A)
    Pp, Qp, y = drop(P), drop(Q), drop(Y-gamma*Y.mean(1, keepdims=True)).ravel()
    after_S = lambda F: (F-gamma*F.mean(1, keepdims=True)).reshape(-1)

    def fit(columns):
        X = np.column_stack(columns)
        theta, _, rank, _ = np.linalg.lstsq(X, y, rcond=1e-10)
        return float(np.sum((y-X@theta)**2)), int(rank)

    ii, jj = np.triu_indices(n)
    sat = []
    for F0 in (Pp, Qp):                                    # entry (i, j): meter i sees flow j, meter j sees flow i
        for i, j in zip(ii, jj):
            F = np.zeros((T, n))
            F[:, i] += F0[:, j]
            if i != j:
                F[:, j] += F0[:, i]
            sat.append(after_S(F))
    rss_s, rank_s = fit(sat)
    del sat
    df2 = (T-E.shape[1])*(n-int(gamma == 1.))-rank_s
    out = {}
    for name, groupings in trees.items():
        cols = []
        for C in [*basis(n), *groupings]:                 # line C: each meter of C sees the summed flow of C
            Z = np.zeros(n)
            Z[list(C)] = 1.
            for F0 in (Pp, Qp):
                cols.append(after_S(F0[:, list(C)].sum(1)[:, None]*Z[None, :]))
        rss_t, rank_t = fit(cols)
        df1 = rank_s-rank_t
        F = ((rss_t-rss_s)/df1)/(rss_s/df2)
        out[name] = dict(F=F, df1=df1, df2=df2, p=float(stats.f.sf(F, df1, df2)))
    return out, rss_s/df2


# ------------------------------------------------------------------ one case
def open_any(name, blind):
    """(n, readings, truth, exact truth, where the load is, blind) of a development case or a real input (always
    blind)."""
    if name.startswith("eulv_"):
        r = real.open_real(name)
        return r["n"], r["hist"], r["truth"], r["exact"], None, True
    rec = library.open_case(name)
    return rec["n"], library.readings(rec), library.truth(rec), library.exact_truth(rec), library.load_where(rec), blind


def exact_of(name):
    """The exact truth of a case, for rows written before it was recorded."""
    if name.startswith("eulv_"):
        return real.open_real(name)["exact"]
    return library.exact_truth(library.open_case(name))


def run_one(task):
    """task = (name, blind, settings dict or None). Returns (name, row or None, error text or None, seconds)."""
    name, blind, settings = task
    started = time.perf_counter()
    try:
        st = Settings(**settings) if settings else Settings()
        n, h, truth, exact, where, blind = open_any(name, blind)
        pcc = () if blind else (h["Ppcc"], h["Qpcc"])
        r = identify(h["P"], h["Q"], h["Y"], h["V0sq"], *pcc, st=st)
        tree = {tuple(M) for M in r["tree"]}
        label_truth = []
        for x in r["labels"]:
            M = tuple(x["grouping"])
            relation = "inside" if x["inside"] else "nested" if all(compatible(M, C) for C in tree) else "crossing"
            label_truth.append(dict(grouping=list(M), label=x["label"], relation=relation, true=M in truth,
                                    exact=M in exact))
        u = balance(h["P"], h["Q"], h["Y"], h["V0sq"], h["Ppcc"], h["Qpcc"]) if r["gate"]["load"] else None
        fits, sigma2 = adequacy(h["P"], h["Q"], h["Y"], h["V0sq"], u, dict(leader=sorted(tree), truth=sorted(truth)))
        assert abs(sigma2/r["model"]["sigma2"]-1) < 1e-6, "the saturated fit differs from the identifier's"
        priced = [x["price"] for x in r["rounds"] if "price" in x]
        row = dict(name=name, kind="real" if name.startswith("eulv_") else library.kind(name), n=n, load=where,
                   blind=blind, score=f1(tree, truth), score_exact=f1(tree, exact), truth=[list(M) for M in sorted(truth)],
                   exact_truth=[list(M) for M in sorted(exact)], label_truth=label_truth,
                   adequacy=fits,
                   complete=bool(priced) and priced[-1]["still_paying"] is None and bool(priced[-1]["certified"]),
                   **{k: v for k, v in r.items() if k not in ("lines", "pool")})
        return name, row, None, time.perf_counter()-started
    except Exception:
        return name, None, traceback.format_exc(), time.perf_counter()-started


def out_path(which, blind=False):
    name = "real" if which == "real" else "dev_blind" if blind else "dev"
    return RESULTS/f"{name}.jsonl"


def run(args):
    if args.set == "real":
        names = real.names()
    else:
        rows = sorted(library.manifest(), key=lambda r: -r["n"])                  # largest first
        names = [r["name"] for r in rows if args.max_n is None or r["n"] <= args.max_n]
        if args.one_twin:
            keep = set(library.one_twin_per_feeder())
            names = [x for x in names if x in keep]
    if args.names:
        names = [x for x in names if x in set(args.names.split(","))]
    path = Path(args.file) if args.file else out_path(args.set, args.blind)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {json.loads(x)["name"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [x for x in names if x not in done]
    print(f"{len(todo)} cases to run, {len(done)} done; output {path}", flush=True)
    with open(path, "a", encoding="utf-8") as fh, open(path.with_suffix(".log"), "a", encoding="utf-8") as log, \
            get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for k, (name, row, error, seconds) in enumerate(pool.imap_unordered(run_one, [(x, args.blind, None)
                                                                                         for x in todo]), 1):
            if row is not None:
                fh.write(json.dumps(row, default=str)+"\n")
                fh.flush()
                line = f"[{k}/{len(todo)}] {name}: F1 {row['score']['f1']:.3f}, stop {row['stop'][:9]}, {seconds:.0f} s"
            else:
                line = f"[{k}/{len(todo)}] {name}: ERROR after {seconds:.0f} s\n{error}"
            print(line, flush=True)
            log.write(line+"\n")
            log.flush()


# ------------------------------------------------------------------ summary and comparison
def read_rows(path):
    """A run file's rows (the last row of each case). Rows written before the exact truth was recorded get it here,
    from the case files: a measurement only, the method's output is untouched."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = list({r["name"]: r for r in (json.loads(x) for x in lines if x.strip())}.values())
    for r in rows:
        r.setdefault("kind", "real" if r["name"].startswith("eulv_") else library.kind(r["name"]))
        if "exact_truth" not in r:
            exact = exact_of(r["name"])
            r["exact_truth"] = [list(M) for M in sorted(exact)]
            r["score_exact"] = f1({tuple(g) for g in r["tree"]}, exact)
            for x in r["label_truth"]:
                x["exact"] = tuple(x["grouping"]) in exact
    return rows


def statuses(r):
    """Solver outcomes in one row: the anchored MIQPs and the final flip tests."""
    solves = Counter(STATUS.get(x["solve"]["status"], x["solve"]["status"]) for x in r["rounds"] if x.get("solve"))
    flips = Counter(STATUS.get(x["status"], x["status"]) for x in r["labels"] if x["inside"])
    return solves, flips


def summary(args):
    path = Path(args.file) if args.file else out_path(args.set, args.blind)
    rows = read_rows(path)
    print(f"{len(rows)} cases in {path}")
    print("wrong: decided groupings not in the identifiable truth; 'not a line': not in the exact truth either")
    print(f"{'kind':8s} {'cases':>5s} {'F1':>6s} {'prec.':>6s} {'recall':>6s} {'p.exact':>7s} {'decided':>8s} {'wrong':>6s}"
          f" {'not a line':>10s} {'open':>5s} {'unsett.':>7s} {'median s':>9s} {'max s':>7s}")
    order = ["star", "street", "small", "medium", "large", "huge", "real"]
    for k in [x for x in order if any(r["kind"] == x for r in rows)]+["all"]:
        sel = [r for r in rows if k in ("all", r["kind"])]
        inside = [x for r in sel for x in r["label_truth"] if x["relation"] == "inside"]
        dec = [x for x in inside if x["label"] == "decided"]
        mean = lambda key: statistics.mean(r["score"][key] for r in sel)
        secs = [r["timing"]["total"] for r in sel]
        pe = statistics.mean(r["score_exact"]["precision"] for r in sel)
        print(f"{k:8s} {len(sel):5d} {mean('f1'):6.3f} {mean('precision'):6.3f} {mean('recall'):6.3f} {pe:7.3f}"
              f" {len(dec):8d} {sum(not x['true'] for x in dec):6d} {sum(not x['exact'] for x in dec):10d}"
              f" {sum(x['label'] == 'open' for x in inside):5d} {sum(x['label'] == 'unsettled' for x in inside):7d}"
              f" {statistics.median(secs):9.0f} {max(secs):7.0f}")
    solves, flips = Counter(), Counter()
    for r in rows:
        s, f = statuses(r)
        solves += s
        flips += f
    print("anchored MIQPs:", dict(solves))
    print("flip tests of the final labels:", dict(flips))
    print("stop:", dict(Counter(r["stop"][:9] for r in rows)))
    outside = Counter((x["relation"], x["true"]) for r in rows for x in r["label_truth"] if x["relation"] != "inside")
    print("open groupings outside the tree (relation, true):", dict(outside))
    for who in ("truth", "leader"):
        print(f"S1, {who}: flagged at the 1% level in {sum(r['adequacy'][who]['p'] < .01 for r in rows)} of {len(rows)}")
    print(f"S2: pricing complete in {sum(r['complete'] for r in rows)} of {len(rows)}")


def compare(args):
    """Two run files on their common cases: the same tree, F1, wrong decided groupings, time."""
    a, b = ({r["name"]: r for r in read_rows(p)} for p in (args.old, args.new))
    common = sorted(set(a) & set(b))
    print(f"{len(common)} common cases ({len(a)} old, {len(b)} new)")
    for k in ["star", "street", "small", "medium", "large", "huge", "real", "all"]:
        sel = [x for x in common if k in ("all", a[x]["kind"])]
        if not sel:
            continue
        same = sum({tuple(g) for g in a[x]["tree"]} == {tuple(g) for g in b[x]["tree"]} for x in sel)
        wrong = lambda r: sum(x["label"] == "decided" and x["relation"] == "inside" and not x["exact"]
                              for x in r["label_truth"])
        decided = lambda r: sum(x["label"] == "decided" and x["relation"] == "inside" for x in r["label_truth"])
        f1m = lambda d: statistics.mean(d[x]["score"]["f1"] for x in sel)
        t = lambda d: statistics.median(d[x]["timing"]["total"] for x in sel)
        print(f"{k:8s} {len(sel):3d} cases: same tree {same:3d}; F1 {f1m(a):.3f} -> {f1m(b):.3f}; "
              f"decided not a line {sum(wrong(a[x]) for x in sel)}/{sum(decided(a[x]) for x in sel)} -> "
              f"{sum(wrong(b[x]) for x in sel)}/{sum(decided(b[x]) for x in sel)}; median s {t(a):.0f} -> {t(b):.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--set", choices=("dev", "real"), default="dev")
        p.add_argument("--blind", action="store_true", help="development cases without the PCC power")
        p.add_argument("--file", default=None, help="another run file")
        if name == "run":
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--names", default=None, help="only these cases, comma-separated")
            p.add_argument("--one-twin", action="store_true", help="one twin per feeder (72 development cases)")
            p.add_argument("--max-n", type=int, default=None, help="only cases with at most this many meters")
    p = sub.add_parser("compare")
    p.add_argument("old")
    p.add_argument("new")
    args = ap.parse_args()
    {"run": run, "summary": summary, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
