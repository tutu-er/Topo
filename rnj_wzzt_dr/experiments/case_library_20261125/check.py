"""Quality check of the case library, without running any identifier.

    python -m rnj_wzzt_dr.experiments.case_library_20261125.check --stream dev [--evidence] [--workers 6]
    python -m rnj_wzzt_dr.experiments.case_library_20261125.check --stream test

Reports, for the old star cases, the new feeders and the real feeders side by side: size, shape (lines with more than
half of the meters, nested depth), line resistances, voltage drop, loading; how far the linear model is from the AC
data on the true tree; and how visible the unmetered load is at the PCC. --evidence (development only) adds how many
true lines the data support, by the same judge as the earlier studies. Held-out cases get only the physical part.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import experiments as hx
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc

from . import build, physics


def shape_of(network, labels):
    """Shape numbers of one network (works for every case format of the studies)."""
    from rnj_wzzt_dr.physical import backend
    order, parent, incoming = backend._radial_order(network)
    n = len(labels)
    below = {b: set() for b in order}
    for i, l in enumerate(labels):
        below[l].add(i)
    for b in reversed(order[1:]):
        below[parent[b]] |= below[b]
    inner = [b for b in order[1:] if 1 < len(below[b]) < n]
    r = {b: incoming[b]["r_ohm"] for b in order[1:]}

    def up(b, what):
        total = 0.
        while parent[b] is not None:
            total += what(b); b = parent[b]
        return total
    return dict(n=n, lines=len(inner), root_lines=sum(len(below[b])/n > .5 for b in inner),
                depth=max(up(l, lambda b: 1 < len(below[b]) < n) for l in labels),
                r_root=[r[b] for b in inner if len(below[b])/n > .5], r_far=[r[b] for b in inner if len(below[b])/n <= .5],
                r_leaf=[r[b] for b in order[1:] if len(below[b]) == 1], r_total=max(up(l, lambda b: r[b]) for l in labels),
                below_floor=sum(r[b] < build.R_FLOOR for b in inner))


def linear_misfit(rec):
    """Root mean square misfit of the best linear model (true tree, r and x free per line) to noiseless hourly data,
    divided by the standard deviation of the voltage readings' error in the same quantity."""
    f = physics.hourly
    p, q, v, v0 = (f(rec["true"][k])[:192] for k in ("P", "Q", "V", "V0"))
    y = v0[:, None]**2-v**2
    lines = [sorted(C) for C in rec["exact_truth"]]
    if [*range(rec["n"])] not in lines:
        lines.append([*range(rec["n"])])
    Z = np.zeros((rec["n"], len(lines)))
    for k, C in enumerate(lines):
        Z[C, k] = 1.
    flows = [p@Z, q@Z]
    # normal equations of the columns flow_k(t) Z[i,k]; meters' constants are removed first (the model's free offsets)
    flows = [a-a.mean(axis=0) for a in flows]; y = y-y.mean(axis=0)
    ZZ = Z.T@Z
    G = np.block([[(a.T@b)*ZZ for b in flows] for a in flows])
    h = np.concatenate([((a.T@y)*Z.T).sum(axis=1) for a in flows])
    theta = np.linalg.lstsq(G, h, rcond=1e-12)[0]
    K = len(lines)
    fit = (flows[0]*theta[:K])@Z.T+(flows[1]*theta[K:])@Z.T
    noise = 2*.0005*1.02**2
    return float(np.sqrt(np.mean((y-fit)**2))/noise)


def source_visibility(rec):
    """Mean of the PCC balance (PCC minus meters minus true losses) over its random error, at hourly readings."""
    if rec["source"] is None:
        return None
    f = physics.hourly
    p, s = f(rec["true"]["P"])[:192], f(rec["true"]["S_pcc"])[:192].real
    sd = .005*math.sqrt(np.mean(s**2)+np.mean((p**2).sum(axis=1)))
    loss = s-p.sum(axis=1)-f(rec["source"]["P"])[:192]
    return dict(mean_over_sd=float(f(rec["source"]["P"])[:192].mean()/sd), loss_share=float(loss.mean()/s.mean()),
                source_share=float(f(rec["source"]["P"])[:192].mean()/p.sum(axis=1).mean()))


def evidence(rec):
    """Change of the BIC score (half units) when one true line is removed; positive: the data support the line."""
    inp = build.to_input(rec)
    s = hx.settings_for("eulv")
    n = inp.n
    r_min = 2*s.r_min_ohm/(inp.network["base_kv"]**2/inp.network["base_mva"])
    truth = model.canonical_family(inp.truth_key, n)
    lines = sorted({tuple(sorted(C)) for C in truth if 1 < len(C) < n}, key=lambda c: (len(c), c))
    families = [truth]+[tuple(C for C in truth if C != line) for line in lines]
    fits = [bank.fit_one(F, n, inp.hist, r_min, s.upper_bound, 3, True, s.meter_offsets, s.root_whitening,
                         s.pq_sigma if s.identifiable_params else None) for F in families]
    N, extra = inp.hist["Y"].size, n if s.meter_offsets else 0
    sigma2 = fits[0].sse/(N-fits[0].c.p-extra)
    J = [f.sse/(2*sigma2)+.5*(f.c.p+extra)*math.log(N) for f in fits]
    return [(len(C)/n, float(J[k+1]-J[0])) for k, C in enumerate(lines)]


def one(job):
    stream, name, with_evidence = job
    rec = build.load(stream, name)
    out = dict(name=name, group=rec["group"], set=rec["set"], source=rec["source"] and rec["source"]["where"],
               style=rec["network"]["metadata"]["style"], rejected=rec["rejected"],
               **shape_of(rec["network"], rec["labels"]),
               **{k: rec[k] for k in ("max_drop", "min_voltage", "max_loading", "mean_kw", "peak_kw", "merged_lines")})
    out["misfit"] = linear_misfit(rec)
    out["visibility"] = source_visibility(rec)
    if with_evidence and rec["source"] is None:
        try:
            out["evidence"] = evidence(rec)
        except Exception as exc:                       # reported, not hidden
            out["evidence_error"] = repr(exc)
    return out


def show(title, rows):
    med = lambda key: np.median([r[key] for r in rows])
    rng = lambda a, d=1: f"{np.percentile(a, 10):.{d}f}/{np.median(a):.{d}f}/{np.percentile(a, 90):.{d}f}" if len(a) else "-"
    flat = lambda key: [v for r in rows for v in r[key]]
    print(f"\n{title}: {len(rows)} cases; meters {rng([r['n'] for r in rows], 0)} (10/50/90%)")
    print(f"  inner lines {rng([r['lines'] for r in rows], 0)}; with more than half of the meters {rng([r['root_lines'] for r in rows], 0)};"
          f" nested depth {rng([r['depth'] for r in rows], 0)}; below 0.003 ohm {rng([r['below_floor'] for r in rows], 0)}")
    print(f"  R (milliohm): root-side {rng([1000*v for v in flat('r_root')])}; other inner {rng([1000*v for v in flat('r_far')])};"
          f" service {rng([1000*v for v in flat('r_leaf')])}; to the far end {rng([1000*r['r_total'] for r in rows])}")
    if "max_drop" in rows[0]:
        print(f"  largest drop % {rng([100*r['max_drop'] for r in rows], 2)}; lowest voltage {min(r['min_voltage'] for r in rows):.3f};"
              f" mean kW {rng([r['mean_kw'] for r in rows])}; loading {rng([r['max_loading'] for r in rows], 2)};"
              f" linear misfit / noise {rng([r['misfit'] for r in rows], 3)} (max {max(r['misfit'] for r in rows):.3f})")
    ev = [e for r in rows for e in r.get("evidence", [])]
    if ev:
        for label, keep in (("root-side", lambda f: f > .5), ("other", lambda f: f <= .5)):
            d = np.array([x for f, x in ev if keep(f)])
            if len(d):
                print(f"  true lines supported by the data, {label}: {int((d > 0).sum())} of {len(d)} ({100*np.mean(d > 0):.0f}%);"
                      f" change {rng(d)}")
    bad = [r["name"] for r in rows if "evidence_error" in r]
    if bad:
        print("  evidence failed on", len(bad), bad[:3])
    vis = [r["visibility"] for r in rows if r.get("visibility")]
    if vis:
        print(f"  unmetered load: mean over reading error {rng([v['mean_over_sd'] for v in vis])}; loss share of PCC power"
              f" {rng([100*v['loss_share'] for v in vis], 2)}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, choices=("dev", "test"))
    ap.add_argument("--evidence", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.evidence and args.stream == "test":
        raise SystemExit("evidence is for development cases only; the held-out cases stay sealed")
    manifest = json.loads((build.HERE/f"manifest_{args.stream}.json").read_text(encoding="utf-8"))
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, [(args.stream, r["name"], args.evidence) for r in manifest]))
    (build.OUT/f"check_{args.stream}.json").write_text(json.dumps(rows))
    none = [r for r in rows if r["source"] is None]
    real = []
    for row in json.loads((sc.EULV_CASES/"index.json").read_text()):
        net = sc.load_network(row["case"])
        real.append(shape_of(net, sc.network_info(row["case"], net)[1]))
    show("Real EU LV feeders (33)", real)
    if args.stream == "dev":
        show("Named, star (old shape)", [r for r in none if r["group"].split("_l")[0].endswith("_star")])
        show("Named, street", [r for r in none if r["group"].split("_l")[0].endswith("_street")])
    for size in ("small", "medium", "large", "huge"):
        rows_ = [r for r in none if r["set"] == "random" and f"_{size}_" in r["name"]]
        show(f"Random, {size}", rows_)
        print("  styles", {s: sum(r["style"] == s for r in rows_) for s in ("urban", "suburban", "rural")},
              "rejected draws before acceptance: total", sum(r["rejected"] for r in rows_))
    show("With an unmetered load (all)", [r for r in rows if r["source"] is not None])


if __name__ == "__main__":
    main()
