"""Amendment D tables and tests (PROTOCOL_MIQP.md): WZZT-R against RNJ, NJ and RG on root and terminal lines.

    python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.analyse_builders [--sets eulv_16d,eulv_large,phase3]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import wilcoxon

from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.compare_builders import BUILDERS, DEFAULT, OUTB

BOOT, SEED, ALPHA = 2000, 20261125, .05
CLASSES = {
    "near root (f > 1/2)": lambda l: l["f"] > .5,
    "terminal side (f <= 1/2)": lambda l: l["f"] <= .5,
    "near root, short (rho < .08)": lambda l: l["f"] > .5 and l["rho"] < .08,
    "near root, long": lambda l: l["f"] > .5 and l["rho"] >= .08,
    "terminal side, short": lambda l: l["f"] <= .5 and l["rho"] < .08,
    "terminal side, long": lambda l: l["f"] <= .5 and l["rho"] >= .08,
    "all lines": lambda l: True,
}


def rows(name):
    path = OUTB/f"{name}.jsonl"
    if not path.exists():
        return []
    out = [json.loads(l) for l in path.read_text().splitlines()]
    bad = [r for r in out if "error" in r]
    if bad:
        print(f"{name}: {len(bad)} errors, e.g. {bad[0]['name']}: {bad[0]['error']}")
    return [r for r in out if "error" not in r]


def hits(r, method, cls):
    found = r["methods"][method]["found"]
    sel = [k for k, l in enumerate(r["lines"]) if cls(l)]
    return sum(found[k] == "1" for k in sel), len(sel)


def pooled(R, method, cls):
    h = t = 0
    for r in R:
        a, b = hits(r, method, cls); h += a; t += b
    return h/t if t else float("nan"), t


def best_setting(R, label):
    def f1(m):
        h = sum(r["methods"][m]["found"].count("1") for r in R)
        out = sum(r["methods"][m]["out"] for r in R)
        tru = sum(len(r["lines"]) for r in R)
        return 2*h/(out+tru) if out+tru else 0.
    keys = [k for k in R[0]["methods"] if k.startswith(label+"|")]
    return max(keys, key=f1)


def precision(R, method):
    out = sum(r["methods"][method]["out"] for r in R)
    true = sum(r["methods"][method]["out_true"] for r in R)
    return (true/out if out else float("nan")), (out-true)/len(R)


def boot_diff(R, a, b, cls, rng):
    """Pooled recall a - b with a bootstrap 95% interval over inputs."""
    per = [(hits(r, a, cls), hits(r, b, cls)) for r in R]
    per = [(x[0], y[0], x[1]) for x, y in per if x[1]]
    if not per:
        return float("nan"), (float("nan"), float("nan")), 0
    arr = np.array(per, float)
    est = (arr[:, 0].sum()-arr[:, 1].sum())/arr[:, 2].sum()
    idx = rng.integers(0, len(arr), (BOOT, len(arr)))
    s = arr[idx]
    d = (s[:, :, 0].sum(1)-s[:, :, 1].sum(1))/s[:, :, 2].sum(1)
    return est, tuple(np.quantile(d, [.025, .975])), len(arr)


def boot_short_long(R, method, band, rng):
    def recs(r):
        hs = ts = hl = tl = 0
        for k, l in enumerate(r["lines"]):
            if l["dJ"] is None or not band(l["dJ"]):
                continue
            hit = r["methods"][method]["found"][k] == "1"
            if l["rho"] < .08:
                hs += hit; ts += 1
            else:
                hl += hit; tl += 1
        return hs, ts, hl, tl
    arr = np.array([recs(r) for r in R], float)
    arr = arr[(arr[:, 1]+arr[:, 3]) > 0]
    if not len(arr) or arr[:, 1].sum() == 0 or arr[:, 3].sum() == 0:
        return float("nan"), (float("nan"), float("nan")), 0, 0
    est = arr[:, 0].sum()/arr[:, 1].sum()-arr[:, 2].sum()/arr[:, 3].sum()
    idx = rng.integers(0, len(arr), (BOOT, len(arr)))
    s = arr[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        d = s[:, :, 0].sum(1)/s[:, :, 1].sum(1)-s[:, :, 2].sum(1)/s[:, :, 3].sum(1)
    d = d[np.isfinite(d)]
    return est, tuple(np.quantile(d, [.025, .975])), int(arr[:, 1].sum()), int(arr[:, 3].sum())


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); running = 0.
    for rank, i in enumerate(order):
        running = max(running, (m-rank)*pvals[i]); adj[i] = min(1., running)
    return adj


def report(name, R):
    rng = np.random.default_rng(SEED)
    print(f"\n==================== {name}: {len(R)} inputs, {sum(len(r['lines']) for r in R)} true internal lines")
    default = {b: f"{b}|{DEFAULT[0]}|{DEFAULT[1]}" for b in BUILDERS}
    best = {b: best_setting(R, b) for b in BUILDERS}
    methods = [("WZZT-R", "wzzt_r")]+[(f"{b.upper()} default", default[b]) for b in BUILDERS] + \
              [(f"{b.upper()} best ({best[b].split('|', 1)[1].replace('|', ', tol ')})", best[b]) for b in BUILDERS]
    head = f"{'line class':30s} {'lines':>6s} " + " ".join(f"{m[0][:14]:>14s}" for m in methods)
    print("\nRecall by line class (pooled over inputs; best = setting with the best pooled F1, weight then tolerance)")
    print(head)
    for cname, cls in CLASSES.items():
        vals = [pooled(R, m, cls) for _, m in methods]
        print(f"{cname:30s} {vals[0][1]:6d} " + " ".join(f"{100*v[0]:14.1f}" for v in vals))
    print(f"{'precision':30s} {'':6s} " + " ".join(f"{100*precision(R, m)[0]:14.1f}" for _, m in methods))
    print(f"{'false lines per input':30s} {'':6s} " + " ".join(f"{precision(R, m)[1]:14.2f}" for _, m in methods))

    print("\nHR1, root lines: per-input recall difference WZZT-R minus builder, one-sided Wilcoxon, Holm over builders")
    near = CLASSES["near root (f > 1/2)"]
    for tag, choice in (("default", default), ("best", best)):
        stats = []
        for b in BUILDERS:
            diffs = []
            for r in R:
                a, t = hits(r, "wzzt_r", near)
                if t:
                    diffs.append((a-hits(r, choice[b], near)[0])/t)
            diffs = np.array(diffs)
            p = wilcoxon(diffs, alternative="greater").pvalue if np.any(diffs != 0) else 1.
            stats.append((b, len(diffs), float(diffs.mean()) if len(diffs) else float("nan"), p))
        adj = holm(np.array([s[3] for s in stats]))
        for (b, k, mean, p), pa in zip(stats, adj):
            print(f"  vs {b.upper():3s} {tag:7s}: inputs {k:3d}, mean difference {100*mean:6.1f} pt, p {p:.2e}, "
                  f"Holm {pa:.2e} -> {'supported' if pa < ALPHA else 'not supported'}")

    print("\nHR2, long terminal-side lines: best builder minus WZZT-R (pooled recall, bootstrap 95% over inputs)")
    term = CLASSES["terminal side, long"]
    top = max(BUILDERS, key=lambda b: pooled(R, best[b], term)[0])
    est, (lo, hi), k = boot_diff(R, best[top], "wzzt_r", term, rng)
    print(f"  best builder {top.upper()} ({best[top]}): {100*est:6.1f} pt [{100*lo:.1f}, {100*hi:.1f}], inputs {k} -> "
          f"{'supported (within 10 pt)' if lo > -.10 else 'not supported'}")
    for b in BUILDERS:
        e2, (l2, h2), _ = boot_diff(R, best[b], "wzzt_r", term, rng)
        print(f"    {b.upper():3s} best: {100*e2:6.1f} pt [{100*l2:.1f}, {100*h2:.1f}]")

    print("\nHR3, mechanism: short minus long recall within whole-map evidence tertiles (bootstrap 95% over inputs)")
    dj = np.array([l["dJ"] for r in R for l in r["lines"] if l["dJ"] is not None])
    if len(dj):
        q = np.quantile(dj, [1/3, 2/3])
        bands = [("low", lambda v: v <= q[0]), ("middle", lambda v: q[0] < v <= q[1]), ("high", lambda v: v > q[1])]
        print(f"  tertile edges dJ {q[0]:.1f}, {q[1]:.1f}")
        for label, m in methods:
            parts = []
            for bname, band in bands:
                est, (lo, hi), ns, nl = boot_short_long(R, m, band, rng)
                parts.append(f"{bname} {100*est:6.1f} [{100*lo:6.1f}, {100*hi:6.1f}] ({ns}/{nl})")
            print(f"  {label[:22]:22s} " + " | ".join(parts))

    print("\nWho finds which line (descriptive): WZZT-R vs the best distance-based builder per class")
    for cname in ("near root (f > 1/2)", "terminal side (f <= 1/2)"):
        cls = CLASSES[cname]
        b = max(BUILDERS, key=lambda x: pooled(R, best[x], cls)[0])
        both = only_w = only_b = neither = 0
        for r in R:
            fw, fb = r["methods"]["wzzt_r"]["found"], r["methods"][best[b]]["found"]
            for k, l in enumerate(r["lines"]):
                if cls(l):
                    w, d = fw[k] == "1", fb[k] == "1"
                    both += w and d; only_w += w and not d; only_b += d and not w; neither += not (w or d)
        tot = both+only_w+only_b+neither
        print(f"  {cname:26s} vs {b.upper()} best: both {both}, WZZT-R only {only_w}, {b.upper()} only {only_b}, "
              f"neither {neither} (of {tot})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="eulv_16d,eulv_large,phase3")
    args = ap.parse_args()
    for name in args.sets.split(","):
        R = rows(name)
        if not R:
            print(f"\n{name}: no rows yet"); continue
        if name.startswith("phase3"):
            report(f"{name} (EXPLORATORY), real feeders", [r for r in R if r["cond"] == "eulv"])
            report(f"{name} (EXPLORATORY), synthetic feeders", [r for r in R if r["cond"] != "eulv"])
        elif name.endswith("_ponly"):
            report(f"{name} (EXPLORATORY: builders on WZZT-R's P-only matrix, added after Amendment D results)", R)
        else:
            report(f"{name} (CONFIRMATORY, Amendment D)", R)


if __name__ == "__main__":
    main()
