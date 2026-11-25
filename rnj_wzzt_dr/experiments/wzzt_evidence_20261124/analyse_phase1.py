"""Phase 1 summary and the registered decision rules D-E and D-W (PROTOCOL.md).

    python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.analyse_phase1
Writes outputs/wzzt_evidence_20261124/summary.json and prints the tables.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

from .capture import OUT

TOL = 1e-6
CONDS = ("eulv", "lock_short01", "lock_noisy")


def _load(name):
    path = OUT/name
    if not path.exists():
        return [], []
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if "error" not in r], [r for r in rows if "error" in r]


def _pct(a, b):
    return f"{a}/{b} ({100*a/b:.1f}%)" if b else f"{a}/0"


def experiment_a(summary):
    rows, errors = _load("a.jsonl")
    done = [r for r in rows if "scored" in r]
    skipped = [r for r in rows if "skipped" in r]
    cert = [r for r in done if r["certified"]]
    print(f"\n=== Experiment A (E stage): {len(done)} instances, {len(skipped)} skipped, {len(errors)} errors")
    print(f"certified: {_pct(len(cert), len(done))}")
    names = [k for k in (done[0]["scored"] if done else {}) if k.startswith(("H", "P", "MIQP"))]
    table = {}
    for k in names:
        reach = [r for r in cert if r["scored"][k]["objective"] <= r["optimum"]+TOL]
        gaps = [r["scored"][k]["objective"]-r["optimum"] for r in cert if r["scored"][k]["objective"] > r["optimum"]+TOL]
        secs = [r["heuristic_seconds"][k] for r in done if k in r.get("heuristic_seconds", {})]
        table[k] = dict(reach=len(reach), certified=len(cert), max_gap=max(gaps, default=0.),
                        median_gap_when_missed=float(np.median(gaps)) if gaps else 0.,
                        median_seconds=float(np.median(secs)) if secs else None)
        by = {c: _pct(sum(r["scored"][k]["objective"] <= r["optimum"]+TOL for r in cert if r["cond"] == c),
                      sum(r["cond"] == c for r in cert)) for c in CONDS}
        print(f"  {k:22s} reach {_pct(len(reach), len(cert)):18s} max gap {table[k]['max_gap']:.5f} "
              f"median s {table[k]['median_seconds'] if table[k]['median_seconds'] is None else round(table[k]['median_seconds'], 2)}  {by}")
    # decision D-E
    P_reach = table.get("P_portfolio", {}).get("reach", 0)
    lost_exact = [r for r in cert if r["scored"]["MIQP_certified"]["truth"]["exact"]
                  and not r["scored"]["P_portfolio"]["truth"]["exact"]]
    gained_exact = [r for r in cert if not r["scored"]["MIQP_certified"]["truth"]["exact"]
                    and r["scored"]["P_portfolio"]["truth"]["exact"]]
    cert_rate = len(cert)/max(len(done), 1)
    if cert_rate < .9:
        verdict = "undecided (certification below 90%)"
    elif P_reach >= .99*len(cert) and not lost_exact:
        verdict = "replaceable: E's MIQP moves to the certificate role"
    else:
        verdict = "not replaceable: the MIQP stays E's solver"
    print(f"D-E: P reach {_pct(P_reach, len(cert))}; MIQP exact but P not: {len(lost_exact)}; "
          f"P exact but MIQP not: {len(gained_exact)} -> {verdict}")
    misses = [r for r in cert if r["scored"]["P_portfolio"]["objective"] > r["optimum"]+TOL]
    for r in misses[:40]:
        P, M = r["scored"]["P_portfolio"], r["scored"]["MIQP_certified"]
        print(f"   miss {r['cond']}/{r['name']}/{r['mode']}: n={r['n']} pool={r['pool']} cross={r['crossing_pairs']} "
              f"gap {P['objective']-r['optimum']:.5f}; P hit {P['truth']['hit']}/{P['truth']['size']} vs MIQP "
              f"{M['truth']['hit']}/{M['truth']['size']} (truth {r['truth']})")
    # run's own MIQP against the certified optimum
    run_opt = sum(r["scored"]["MIQP_run"]["objective"] <= r["optimum"]+TOL for r in cert if "MIQP_run" in r["scored"])
    print(f"run's own E MIQP already optimal: {_pct(run_opt, len(cert))}")
    # F
    pairs = [r for r in done if r["run"]["seconds"] and r["F"]["seconds"]]
    ratio = np.array([r["run"]["seconds"]/max(r["F"]["seconds"], 1e-3) for r in pairs])
    same_status = sum(r["run"]["status"] == r["F"]["status"] for r in pairs)
    same_obj = sum(r["run"]["objective"] is not None and r["F"]["objective"] is not None and
                   abs(r["run"]["objective"]-r["F"]["objective"]) <= 1e-7*(1+abs(r["run"]["objective"])) for r in pairs)
    same_fam = sum(r["run"]["family"] == r["F"]["family"] for r in pairs)
    print(f"\n=== Experiment F (compressed E MIQP, same settings): {len(pairs)} instances")
    if len(pairs):
        print(f"speed-up median {np.median(ratio):.1f}x (range {ratio.min():.1f}-{ratio.max():.1f}); same status "
              f"{_pct(same_status, len(pairs))}, objective {_pct(same_obj, len(pairs))}, family {_pct(same_fam, len(pairs))}")
        for c in CONDS:
            rr = [r["run"]["seconds"]/max(r["F"]["seconds"], 1e-3) for r in pairs if r["cond"] == c]
            if rr:
                print(f"  {c}: median {np.median(rr):.1f}x, original median {np.median([r['run']['seconds'] for r in pairs if r['cond'] == c]):.1f}s")
    summary["A"] = dict(instances=len(done), certified=len(cert), table=table, lost_exact=len(lost_exact),
                        gained_exact=len(gained_exact), verdict=verdict, misses=len(misses),
                        run_optimal=run_opt, F=dict(median_speedup=float(np.median(ratio)) if len(pairs) else None,
                                                     same_family=same_fam, same_objective=same_obj, n=len(pairs)))
    return summary


def experiment_a_d1(summary):
    """Deviation D1: the optimum and the certificate from the SVD re-certification (a2.jsonl)."""
    rows = {(r["cond"], r["name"], r["mode"]): r for r in _load("a.jsonl")[0] if "scored" in r}
    a2, errors = _load("a2.jsonl")
    a2 = {(r["cond"], r["name"], r["mode"]): r for r in a2 if "certify" in r}
    if not a2:
        return summary
    keys = [k for k in rows if k in a2]
    print(f"\n=== Experiment A with deviation D1 (SVD re-certification): {len(keys)} instances, {len(errors)} errors")
    opt, cert, contradicted = {}, set(), []
    for k in keys:
        r, c = rows[k], a2[k]
        vals = [r["optimum"]]+[v for v in (c["certify"]["rescored"], c["F2"]["rescored"]) if v is not None]
        opt[k] = min(vals)
        if c["certify"]["status"] == 2:
            if c["certify"]["rescored"] is not None and c["certify"]["rescored"] <= opt[k]+TOL:
                cert.add(k)
            else:
                contradicted.append(k)
    print(f"certified and not contradicted: {_pct(len(cert), len(keys))}; status optimal but beaten by a known family: "
          f"{len(contradicted)} {contradicted[:5]}")
    false_cold = [k for k in keys if a2[k]["F2"]["status"] == 2 and a2[k]["F2"]["rescored"] is not None
                  and a2[k]["F2"]["rescored"] > opt[k]*(1+1e-4)+TOL]
    print(f"formulation errors (cold start claims optimal above optimum + MIPGap): {len(false_cold)} {false_cold[:8]}")
    improved = [k for k in keys if opt[k] < rows[k]["optimum"]-TOL]
    print(f"re-certification improved on the best known objective: {len(improved)}")
    for k in improved:
        print(f"   {k}: {rows[k]['optimum']:.6f} -> {opt[k]:.6f} (status {a2[k]['certify']['status']})")
    names = [n for n in rows[keys[0]]["scored"] if n.startswith(("H", "P", "MIQP"))]
    table = {}
    for nme in names:
        reach = [k for k in cert if rows[k]["scored"][nme]["objective"] <= opt[k]+TOL]
        table[nme] = dict(reach=len(reach), certified=len(cert))
        by = {c: _pct(sum(k[0] == c for k in reach), sum(k[0] == c for k in cert)) for c in CONDS}
        print(f"  {nme:22s} reach {_pct(len(reach), len(cert)):18s} {by}")
    lost = [k for k in cert if a2[k]["certify"]["truth"]["exact"] and not rows[k]["scored"]["P_portfolio"]["truth"]["exact"]]
    gained = [k for k in cert if not a2[k]["certify"]["truth"]["exact"] and rows[k]["scored"]["P_portfolio"]["truth"]["exact"]]
    rate = len(cert)/max(len(keys), 1)
    P_reach = table["P_portfolio"]["reach"]
    if rate < .9:
        verdict = "undecided (certification below 90%)"
    elif P_reach >= .99*len(cert) and not lost:
        verdict = "replaceable: E's MIQP moves to the certificate role"
    else:
        verdict = "not replaceable: the MIQP stays E's solver"
    print(f"D-E (D1): P reach {_pct(P_reach, len(cert))}; certified MIQP exact but P not: {len(lost)}; "
          f"P exact but MIQP not: {len(gained)} -> {verdict}")
    for k in cert:
        if rows[k]["scored"]["P_portfolio"]["objective"] > opt[k]+TOL:
            r = rows[k]
            print(f"   miss {k}: n={r['n']} pool={r['pool']} gap {r['scored']['P_portfolio']['objective']-opt[k]:.5f} "
                  f"(member {r['scored']['P_portfolio']['member']})")
    # F2: SVD formulation with the run's settings against the run's own MIQP
    pairs = [k for k in keys if rows[k]["run"]["seconds"] and a2[k]["F2"]["seconds"]]
    ratio = np.array([rows[k]["run"]["seconds"]/max(a2[k]["F2"]["seconds"], 1e-3) for k in pairs])
    run_obj = {k: rows[k]["scored"]["MIQP_run"]["objective"] for k in pairs if "MIQP_run" in rows[k]["scored"]}
    f2 = {k: a2[k]["F2"]["rescored"] for k in pairs}
    same = sum(abs(run_obj[k]-f2[k]) <= TOL for k in run_obj if f2[k] is not None)
    f2_better = sum(f2[k] < run_obj[k]-TOL for k in run_obj if f2[k] is not None)
    run_better = sum(run_obj[k] < f2[k]-TOL for k in run_obj if f2[k] is not None)
    f2_opt = sum(f2[k] is not None and f2[k] <= opt[k]+TOL for k in cert if k in f2)
    run_opt = sum(run_obj.get(k, 9e9) <= opt[k]+TOL for k in cert)
    print(f"\n=== F2 (SVD E MIQP, run settings) vs the run's own E MIQP: {len(pairs)} instances")
    if len(pairs):
        print(f"speed-up median {np.median(ratio):.1f}x (range {ratio.min():.1f}-{ratio.max():.1f}); same objective "
              f"{_pct(same, len(run_obj))}; SVD better {f2_better}, original better {run_better}; reaches the certified "
              f"optimum: SVD {_pct(f2_opt, len(cert))}, original {_pct(run_opt, len(cert))}")
    summary["A_D1"] = dict(instances=len(keys), certified=len(cert), contradicted=len(contradicted), false_cold=len(false_cold),
                           improved=len(improved), table=table, lost_exact=len(lost), gained_exact=len(gained),
                           verdict=verdict, F2=dict(n=len(pairs), median_speedup=float(np.median(ratio)) if len(pairs) else None,
                                                    same=same, svd_better=f2_better, original_better=run_better,
                                                    svd_optimal=f2_opt, original_optimal=run_opt))
    return summary


def experiment_a_d3(summary):
    """Deviation D3: confirmed certificates (warm-start D1 and cold-start Seed 7 agree), D-E on them."""
    rows = {(r["cond"], r["name"], r["mode"]): r for r in _load("a.jsonl")[0] if "scored" in r}
    a2 = {(r["cond"], r["name"], r["mode"]): r for r in _load("a2.jsonl")[0] if "certify" in r}
    a3, errors = _load("a3.jsonl")
    a3 = {(r["cond"], r["name"], r["mode"]): r for r in a3 if "certify2" in r}
    keys = [k for k in rows if k in a2 and k in a3]
    if not keys or "A_D1" not in summary:
        return summary
    print(f"\n=== Experiment A with deviation D3 (second certificate, cold start, Seed 7): {len(keys)} instances, "
          f"{len(errors)} errors")
    opt, confirmed, disagree, false2 = {}, set(), [], []
    for k in keys:
        c1, c2 = a2[k]["certify"], a3[k]["certify2"]
        vals = [rows[k]["optimum"]]+[v for v in (c1["rescored"], a2[k]["F2"]["rescored"], c2["rescored"]) if v is not None]
        opt[k] = min(vals)
        if c2["status"] == 2 and c2["rescored"] is not None and c2["rescored"] > opt[k]+TOL:
            false2.append(k)
        if (c1["status"] == 2 and c2["status"] == 2 and None not in (c1["rescored"], c2["rescored"])
                and abs(c1["rescored"]-c2["rescored"]) <= TOL and c1["rescored"] <= opt[k]+TOL):
            confirmed.add(k)
        elif c1["status"] == 2 or c2["status"] == 2:
            disagree.append(k)
    improved = [k for k in keys if opt[k] < min(rows[k]["optimum"], a2[k]["certify"]["rescored"] or 9e9)-TOL]
    secs = np.array([a3[k]["certify2"]["seconds"] for k in keys])
    print(f"confirmed: {_pct(len(confirmed), len(keys))}; one certificate only or disagreeing: {len(disagree)} {disagree[:6]}; "
          f"second certificate claims optimal above the optimum: {len(false2)} {false2[:6]}; "
          f"second certificate found a better objective: {len(improved)}; seconds median {np.median(secs):.1f}, max {secs.max():.0f}")
    cert = confirmed
    P_reach = sum(rows[k]["scored"]["P_portfolio"]["objective"] <= opt[k]+TOL for k in cert)
    lost = [k for k in cert if a2[k]["certify"]["truth"]["exact"] and not rows[k]["scored"]["P_portfolio"]["truth"]["exact"]]
    for nme in ("MIQP_run", "H5_beam20", "H6_rnj_multistart", "H7_random_multistart", "P_portfolio"):
        reach = sum(rows[k]["scored"][nme]["objective"] <= opt[k]+TOL for k in cert)
        print(f"  {nme:22s} reach {_pct(reach, len(cert))}")
    rate = len(cert)/max(len(keys), 1)
    if rate < .9:
        verdict = "undecided (confirmed certification below 90%)"
    elif P_reach >= .99*len(cert) and not lost:
        verdict = "replaceable: E's MIQP moves to the certificate role"
    else:
        verdict = "not replaceable: the MIQP stays E's solver"
    final = summary["A_D1"]["verdict"] if verdict == summary["A_D1"]["verdict"] else "undecided (D1 and D3 disagree)"
    print(f"D-E (D3, confirmed set): P reach {_pct(P_reach, len(cert))}; lost exact {len(lost)} -> {verdict}")
    print(f"E verdict (final, PROTOCOL D3): {final}")
    summary["A_D3"] = dict(instances=len(keys), confirmed=len(cert), disagree=len(disagree), false_second=len(false2),
                           improved=len(improved), P_reach=P_reach, lost_exact=len(lost), verdict=verdict, final=final)
    return summary


def experiment_b1(summary):
    rows, errors = _load("b1.jsonl")
    steps = [r for r in rows if "best" in r]
    cert = [r for r in steps if r["status"] == 2]
    print(f"\n=== Experiment B1 (W steps): {len(steps)} steps with an MIQP, {len(errors)} errors")
    print(f"certified: {_pct(len(cert), len(steps))}")
    names = list(steps[0]["best"]) if steps else []
    table = {}
    for k in names:
        avail = [r for r in cert if k in r["best"]]
        reach = sum(r["best"][k]["objective"] <= r["miqp_objective"]+TOL for r in avail)
        better = sum(r["best"][k]["objective"] < r["miqp_objective"]-TOL for r in avail)
        if k == "Gp_portfolio":
            insets = [any(r["miqp_in_set"].get(g) for g in r["miqp_in_set"] if g != "G6_brute")
                      for r in avail if r["miqp_support"]]
        else:
            insets = [r["miqp_in_set"].get(k) for r in avail if r["miqp_support"] and k in r["miqp_in_set"]]
        table[k] = dict(reach=reach, available=len(avail), better=better,
                        miqp_in_set=sum(bool(x) for x in insets), with_new=len(insets))
        by = {c: _pct(sum(r["best"][k]["objective"] <= r["miqp_objective"]+TOL for r in avail if r["cond"] == c),
                      sum(r["cond"] == c for r in avail)) for c in CONDS}
        print(f"  {k:18s} reach {_pct(reach, len(avail)):18s} better than MIQP {better:3d}  MIQP support in set "
              f"{_pct(table[k]['miqp_in_set'], len(insets))}  {by}")
    last = [r for r in cert if r["last"]]
    m_nogain = [r for r in last if r["miqp_closed"] is not None and r["miqp_closed"] >= r["current_closed"]-1e-10]
    agree = sum(r["portfolio_closed"] >= r["current_closed"]-1e-10 for r in m_nogain)
    print(f"final W steps where the MIQP found no gain: {len(m_nogain)}; portfolio also found none: {_pct(agree, len(m_nogain))}")
    gp = table.get("Gp_portfolio", {})
    summary["B1"] = dict(steps=len(steps), certified=len(cert), table=table, final_nogain=len(m_nogain), agree=agree)
    return summary, gp, len(cert)/max(len(steps), 1)


def experiment_b2(summary, gp, b1_cert_rate):
    rows, errors = _load("b2.jsonl")
    done = [r for r in rows if "skipped" not in r]
    print(f"\n=== Experiment B2 (truth-removed pools, demeaned run): {len(rows)} runs, {len(errors)} errors")
    by = defaultdict(dict)
    for r in done:
        by[(r["cond"], r["name"])][r["variant"]] = r
    full = [v for v in by.values() if len(v) == 3]
    removed = sum(v["miqp_W"]["removed"] for v in full)
    stats = {}
    for var in ("miqp_W", "pool_W", "generator_W"):
        sel = sum(v[var]["recovered_selected"] for v in full)
        out = sum(v[var]["recovered_output"] for v in full)
        ex = sum(v[var]["exact"] for v in full)
        secs = float(np.median([v[var]["seconds"] for v in full])) if full else None
        stats[var] = dict(selected=sel, output=out, exact=ex, median_seconds=secs)
        per = {c: f"{sum(v[var]['recovered_output'] for v in full if v[var]['cond'] == c)}/"
                  f"{sum(v[var]['removed'] for v in full if v[var]['cond'] == c)}" for c in CONDS}
        print(f"  {var:12s} removed supports recovered: selected {_pct(sel, removed)}, output {_pct(out, removed)}; "
              f"exact {ex}/{len(full)}; median {secs:.0f}s  output by condition {per}")
    tests = {}
    for metric in ("recovered_selected", "recovered_output"):
        d = np.array([v["generator_W"][metric]-v["miqp_W"][metric] for v in full])
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.
        tests[metric] = dict(deficit=int(-d.sum()), p=p, worse=int((d < 0).sum()), better=int((d > 0).sum()))
        print(f"  generator_W - miqp_W, {metric}: sum {int(d.sum())} (inputs worse {int((d < 0).sum())}, better "
              f"{int((d > 0).sum())}), Wilcoxon p = {p:.3g}")
    b = sum(v["miqp_W"]["exact"] and not v["generator_W"]["exact"] for v in full)
    c = sum(v["generator_W"]["exact"] and not v["miqp_W"]["exact"] for v in full)
    p_ex = float(binomtest(b, b+c).pvalue) if b+c else 1.
    print(f"  exact: MIQP W only {b}, generator W only {c}, McNemar p = {p_ex:.3g}")
    ok_b2 = all(t["deficit"] <= .02*removed and not (t["p"] < .05 and t["deficit"] > 0) for t in tests.values())
    ok_b1 = gp.get("available") and gp["reach"] >= .99*gp["available"]
    if b1_cert_rate < .9:
        verdict = "undecided (W certification below 90%)"
    elif ok_b1 and ok_b2:
        verdict = "replaceable: W's MIQP moves to a final no-gain certificate"
    else:
        verdict = "not replaceable: W's MIQP stays"
    print(f"D-W: B1 portfolio reach {_pct(gp.get('reach', 0), gp.get('available', 0))} (need 99%), "
          f"B2 within 2% and not significantly worse on both metrics: {ok_b2} -> {verdict}")
    summary["B2"] = dict(inputs=len(full), removed=removed, stats=stats, tests=tests, exact_mcnemar=dict(b=b, c=c, p=p_ex),
                         verdict=verdict)
    return summary


def main():
    summary = experiment_a_d3(experiment_a_d1(experiment_a({})))
    summary, gp, rate = experiment_b1(summary)
    summary = experiment_b2(summary, gp, rate)
    (OUT/"summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
