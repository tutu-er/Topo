"""The registered analysis of the WZZT-C confirmation (PROTOCOL_WZZT_C.md §4).

    python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.analyse_confirm_c
"""
from __future__ import annotations

import json
from statistics import median

import numpy as np
from scipy.stats import binomtest, wilcoxon

from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_confirm_c import BUILDER_SETTINGS, CLEAN, OUTC, TOLERANCES

BOOT, SEED, MARGIN, ALPHA = 10_000, 20261125, -.05, .05


def load(which):
    """(rows, failed jobs). A failed job with a row in <which>_failed_rerun.jsonl (the comparators rerun alone, WZZT-C
    empty: section 4) is replaced by that row; failed jobs without one stay failed."""
    path = OUTC/f"{which}.jsonl"
    if not path.exists():
        return [], []
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    ok = [r for r in rows if "error" not in r]
    names = {r["name"] for r in ok}
    rerun_path = OUTC/f"{which}_failed_rerun.jsonl"
    rerun = {} if not rerun_path.exists() else {
        r["name"]: r for r in map(json.loads, rerun_path.read_text().splitlines()) if "error" not in r}
    failed = {r["name"]: r for r in rows if "error" in r and r["name"] not in names}
    ok += [rerun[k] for k in sorted(failed) if k in rerun]
    return ok, [r for k, r in sorted(failed.items()) if k not in rerun]


def found(r, m):
    """(per-line hits list, lines output, true lines output); a missing method (crash) is an empty output."""
    rec = r["methods"].get(m)
    if rec is None:
        return [False]*len(r["lines"]), 0, 0
    return [c == "1" for c in rec["found"]], rec["out"], rec["out_true"]


def exact(r, m):
    hits, out, true = found(r, m)
    return all(hits) and out == true == len(r["lines"])


def f1(r, m):
    hits, out, _ = found(r, m)
    t = len(r["lines"])
    return 1. if out+t == 0 else 2*sum(hits)/(out+t)


def holm(p):
    order = np.argsort(p); adj = np.empty(len(p)); run = 0.
    for k, i in enumerate(order):
        run = max(run, (len(p)-k)*p[i]); adj[i] = min(1., run)
    return adj


def network(name):
    return name.rsplit("_s", 1)[0]


def builder(m, kind):
    return f"{m}|{BUILDER_SETTINGS[kind][m]}"


def main():
    rng = np.random.default_rng(SEED)
    syn, syn_err = load("synthetic")
    big, big_err = load("large")
    print(f"synthetic: {len(syn)} inputs ({len(syn_err)} failed jobs); large: {len(big)} inputs ({len(big_err)} failed jobs)")
    for e in syn_err+big_err:
        print("  failed:", e["cond"], e["name"], e["error"][:160])
    for r in syn+big:
        if r.get("rerun"):
            print(f"  failed job, comparators rerun alone, WZZT-C empty: {r['name']}; crashes in the rerun: "
                  f"{sorted(r.get('errors', {})) or 'none'}")
    # failed jobs (section 4): with a rerun row the comparators keep their rerun results and WZZT-C is empty; a failed
    # synthetic job without one is scored, conservatively, as WZZT-C wrong and WZZT-R right in C1a

    print("\n## C1, non-inferiority to WZZT-R")
    a = np.array([exact(r, "wzzt_c") for r in syn]+[False]*len(syn_err), float)
    b = np.array([exact(r, "wzzt_r") for r in syn]+[True]*len(syn_err), float)
    d = a-b
    idx = rng.integers(0, len(d), (BOOT, len(d)))
    lo, hi = np.quantile(d[idx].mean(1), [.025, .975])
    c1a = lo > MARGIN
    print(f"C1a synthetic exact: WZZT-C {int(a.sum())}/{len(a)}, WZZT-R {int(b.sum())}/{len(b)}; difference "
          f"{100*d.mean():+.1f} pt [{100*lo:+.1f}, {100*hi:+.1f}] -> {'non-inferior' if c1a else 'NOT shown'}")

    def real_part(rows, label):
        """C1b on a set of large inputs (cluster bootstrap by network)."""
        nets = sorted({network(r["name"]) for r in rows})
        by_net = {k: [r for r in rows if network(r["name"]) == k] for k in nets}
        def pooled(rs, m):
            sh = st = out = tru = 0
            for r in rs:
                hits, o, t = found(r, m)
                for h, l in zip(hits, r["lines"]):
                    if l["dJ"] > 0:
                        st += 1; sh += h
                out += o; tru += t
            return sh, st, tru, out
        def stats(rs):
            c = pooled(rs, "wzzt_c"); w = pooled(rs, "wzzt_r")
            return c[0]/max(c[1], 1)-w[0]/max(w[1], 1), c[2]/max(c[3], 1)-w[2]/max(w[3], 1), c, w
        rec, prec, c, w = stats(rows)
        boots = []
        for _ in range(BOOT):
            pick = rng.integers(0, len(nets), len(nets))
            boots.append(stats([r for k in pick for r in by_net[nets[k]]])[:2])
        boots = np.array(boots)
        (rlo, rhi), (plo, phi) = np.quantile(boots[:, 0], [.025, .975]), np.quantile(boots[:, 1], [.025, .975])
        ok = bool(rlo > MARGIN and plo > MARGIN)
        print(f"C1b {label} ({len(rows)} inputs, {len(nets)} networks): supported recall WZZT-C {c[0]}/{c[1]}, WZZT-R "
              f"{w[0]}/{w[1]}, difference {100*rec:+.1f} pt [{100*rlo:+.1f}, {100*rhi:+.1f}]; precision difference "
              f"{100*prec:+.1f} pt [{100*plo:+.1f}, {100*phi:+.1f}] -> {'non-inferior' if ok else 'NOT shown'}")
        return ok
    clean = [r for r in big if r["name"] in CLEAN]
    c1b = real_part(big, "all large inputs, OVERLAPPING WITH DEVELOPMENT")
    clean_missing = sorted(CLEAN-{r["name"] for r in clean})
    c1b_clean = real_part(clean, "inputs without development meters") if clean else False
    if clean_missing:
        print(f"C1b clean: {len(clean_missing)} of the 13 inputs missing or failed -> not decided: {clean_missing}")
        c1b_clean = False

    print("\n## C2, better than the baselines")
    pairs = [(exact(r, "wzzt_c"), exact(r, "o_dm")) for r in syn]
    only_c = sum(x and not y for x, y in pairs); only_o = sum(y and not x for x, y in pairs)
    p = binomtest(only_c, only_c+only_o, .5, alternative="greater").pvalue if only_c+only_o else 1.
    c2a = p < ALPHA
    print(f"C2a synthetic exact against the original WZZT: WZZT-C only {only_c}, original only {only_o}, "
          f"one-sided p {p:.2e} -> {'supported' if c2a else 'not supported'}")
    allrows = [(r, "synthetic") for r in syn]+[(r, "real") for r in clean]
    ps, means = [], []
    for m in ("rnj", "nj", "rg"):
        diff = np.array([f1(r, "wzzt_c")-f1(r, builder(m, kind)) for r, kind in allrows])
        ps.append(wilcoxon(diff, alternative="greater").pvalue if np.any(diff != 0) else 1.); means.append(diff.mean())
    adj = holm(np.array(ps)); c2b = bool(np.all(adj < ALPHA))
    for m, mu, pa in zip(("rnj", "nj", "rg"), means, adj):
        print(f"C2b F1, WZZT-C against {m.upper()} (fixed setting): mean difference {100*mu:+.1f} pt, Holm p {pa:.2e}")
    print(f"C2b -> {'supported' if c2b else 'not supported'}")
    def root_lines(rows, label):
        """C2c on a set of large inputs: per-input recall of lines with more than half the meters downstream."""
        ps, means, ns = [], [], []
        for m in ("rnj", "nj", "rg"):
            diff = []
            for r in rows:
                near = [k for k, l in enumerate(r["lines"]) if l["f"] > .5]
                if near:
                    hc = found(r, "wzzt_c")[0]; hb = found(r, builder(m, "real"))[0]
                    diff.append((sum(hc[k] for k in near)-sum(hb[k] for k in near))/len(near))
            diff = np.array(diff); ns.append(len(diff))
            ps.append(wilcoxon(diff, alternative="greater").pvalue if np.any(diff != 0) else 1.)
            means.append(diff.mean() if len(diff) else float("nan"))
        adj = holm(np.array(ps)); ok = bool(np.all(adj < ALPHA))
        for m, mu, pa, k in zip(("rnj", "nj", "rg"), means, adj, ns):
            print(f"C2c {label}, root lines against {m.upper()}: {k} inputs, mean recall difference {100*mu:+.1f} pt, "
                  f"Holm p {pa:.2e}")
        print(f"C2c {label} -> {'supported' if ok else 'not supported'}")
        return ok
    c2c_all = root_lines(big, "all large inputs, OVERLAPPING WITH DEVELOPMENT")
    c2c = root_lines(clean, "inputs without development meters") if clean else False
    ok = c1a and c1b_clean and c2a and c2b and c2c
    complete = len(syn)+len(syn_err) == 192 and not clean_missing
    print(f"\n## Parts: C1a {c1a}; C1b clean {c1b_clean} (all 46, descriptive: {c1b}); C2a {c2a}; C2b {c2b} (synthetic + clean); "
          f"C2c clean {c2c} (all 46, descriptive: {c2c_all})")
    if complete:
        print(f"## Decision (user's rule: real part on the 13 clean inputs): {'CONFIRMED' if ok else 'NOT confirmed'}")
    else:
        print("## No decision yet: the registered runs are not complete")

    print("\n## Secondary")
    for label, rows, kind in (("synthetic", syn, "synthetic"), ("large", big, "real")):
        if not rows:
            continue
        ms = ["wzzt_c", "wzzt_r"]+(["o_dm"] if label == "synthetic" else [])+[builder(m, kind) for m in ("rnj", "nj", "rg")]
        print(f"\n{label}: recall by line class, precision, time")
        for m in ms:
            near = term = nh = th = out = tru = 0
            for r in rows:
                hits, o, t = found(r, m)
                for h, l in zip(hits, r["lines"]):
                    if l["f"] > .5: near += 1; nh += h
                    else: term += 1; th += h
                out += o; tru += t
            secs = [r["methods"][m]["seconds"] for r in rows if m in r["methods"] and r["methods"][m]["seconds"] is not None]
            tm = f"median {median(secs):.1f} s, max {max(secs):.0f} s" if secs else ""
            print(f"  {m:10s} near root {100*nh/max(near, 1):5.1f}% ({near}) terminal {100*th/max(term, 1):5.1f}% ({term}) "
                  f"precision {100*tru/max(out, 1):5.1f}% {tm}")
        cert = sum(bool(r["methods"]["wzzt_c"].get("certified")) for r in rows)
        better = worse = tie = 0
        scored = [r for r in rows if not r.get("rerun") and r["methods"]["wzzt_r"].get("criterion") is not None]
        for r in scored:
            x, y = r["methods"]["wzzt_c"]["criterion_same_design"], r["methods"]["wzzt_r"]["criterion"]
            tol = 1e-7*max(1., abs(y))
            better += x < y-tol; worse += x > y+tol; tie += abs(x-y) <= tol
        print(f"  WZZT-C certified {cert}/{len(rows)}; score against WZZT-R ({len(scored)} inputs with both scores; failed "
              f"jobs left out): better {better}, worse {worse}, tie {tie}")
        for m in ("rnj", "nj", "rg"):
            def f1pool(key):
                h = sum(sum(found(r, key)[0]) for r in rows); o = sum(found(r, key)[1] for r in rows)
                t = sum(len(r["lines"]) for r in rows)
                return 2*h/max(o+t, 1)
            best = max((f"{m}|{tol}" for tol in TOLERANCES), key=f1pool)
            print(f"  {m.upper()} pooled F1: fixed {builder(m, kind)} {f1pool(builder(m, kind)):.3f}; best after the fact "
                  f"{best} {f1pool(best):.3f}; WZZT-C {f1pool('wzzt_c'):.3f}")


if __name__ == "__main__":
    main()
