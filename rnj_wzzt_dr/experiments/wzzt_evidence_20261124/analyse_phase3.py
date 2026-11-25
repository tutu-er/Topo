"""Phase 3 tables and the registered decision (PROTOCOL_PHASE3.md).

Reads outputs/wzzt_evidence_20261124/phase3/{ceiling,orig,R,bank,judge}.jsonl. Primary endpoints P1 (synthetic exact
recovery, R vs O_dm, one-sided exact McNemar) and P2 (real supported recall, R vs O_dm, one-sided Wilcoxon, plus
precision non-inferiority of 5 points); the same tests against O_lv and the secondary endpoints S1-S4 are reported.
"""
from __future__ import annotations

import json
from collections import defaultdict
from statistics import median

from scipy.stats import binomtest, wilcoxon

from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase3 import OUT3, internal, load

CONDS = ("eulv", "lock_short01", "lock_noisy")
ALPHA = .05


def rows(stage):
    path = OUT3/f"{stage}.jsonl"
    if not path.exists():
        return []
    out = [json.loads(line) for line in path.read_text().splitlines()]
    errors = [r for r in out if "error" in r]
    if errors:
        print(f"{stage}: {len(errors)} errors, e.g. {errors[0]['error']}")
    return [r for r in out if "error" not in r]


def ceiling():
    return {(r["cond"], r["name"]): {tuple(s["segment"]) for s in r["segments"] if s["dJ"] > 0}
            for r in rows("ceiling")}


def truth_among(r, truth):
    if "truth_in_output" in r:
        return bool(r["truth_in_output"])
    return any(set(map(tuple, f)) == truth for f in r.get("families", []))


def mcnemar_greater(pairs):
    only_a = sum(1 for a, b in pairs if a and not b)
    only_b = sum(1 for a, b in pairs if b and not a)
    p = binomtest(only_a, only_a+only_b, .5, alternative="greater").pvalue if only_a+only_b else 1.
    return only_a, only_b, p


def pooled(recs, key_hit="hit", key_size="size"):
    return sum(r[key_hit] for r in recs)/max(sum(r[key_size] for r in recs), 1)


def main():
    ceil = ceiling()
    orig = rows("orig")
    Rrows = rows("R")
    M = defaultdict(dict)
    for r in orig+Rrows:
        M[(r["cond"], r["name"])][r["variant"]] = r
    complete = {k: v for k, v in M.items() if all(x in v for x in ("O_lv", "O_dm", "R"))}
    print(f"inputs with all three methods: {len(complete)} "
          f"({', '.join(f'{c} {sum(k[0] == c for k in complete)}' for c in CONDS)}); ceiling {len(ceil)}")
    truths, exact_truth = {}, {}
    for k in complete:
        inp = load(*k)
        truths[k] = internal(inp.truth_key, inp.n)
        if k[0] == "eulv":
            exact_truth[k] = internal(inp.meta["exact_truth_key"], inp.n)

    # ---------------------------------------------------------------- accuracy tables
    print("\n## Accuracy\n")
    for cond in CONDS:
        keys = [k for k in complete if k[0] == cond]
        if not keys:
            continue
        sup_total = sum(len(ceil[k]) for k in keys)
        full = sum(1 for k in keys if len(ceil[k]) == len(truths[k]))
        print(f"\n### {cond} ({len(keys)} inputs; {sup_total} of {sum(len(truths[k]) for k in keys)} true segments "
              f"supported; every segment supported on {full})\n")
        print("| Method | Exact | Recall | Precision | Supported recall | Output recall | Truth among output families "
              "| Median time | Confirmed |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for v in ("O_lv", "O_dm", "R"):
            rs = [complete[k][v] for k in keys]
            m = [r["metrics"] for r in rs]
            conf = ""
            if v == "R":
                cert = [r for r in rs if r["e"].get("certificates")]
                conf = f"{sum(bool(r['e'].get('confirmed')) for r in cert)}/{len(cert)}"
            print(f"| {v} | {sum(x['exact'] for x in m)} | {100*pooled(m, 'hit', 'truth'):.0f}% "
                  f"| {100*pooled(m):.0f}% | {100*sum(x['supported_hit'] for x in m)/max(sup_total, 1):.0f}% "
                  f"| {100*sum(r['output_hit'] for r in rs)/max(sum(x['truth'] for x in m), 1):.0f}% "
                  f"| {sum(truth_among(complete[k][v], truths[k]) for k in keys)} "
                  f"| {median(r['wall'] for r in rs):.0f} s | {conf or '–'} |")
        for label, src, mk in (("RNJ1 (original reference, demeaned)", "O_dm", "rnj1_metrics"),
                               ("RNJ1 (original reference, levels)", "O_lv", "rnj1_metrics"),
                               ("R0 reference (P-only RNJ, tol 0.16)", "R", "r0_metrics")):
            m = [complete[k][src][mk] for k in keys]
            print(f"| {label} | {sum(x['exact'] for x in m)} | {100*pooled(m, 'hit', 'truth'):.0f}% "
                  f"| {100*pooled(m):.0f}% | {100*sum(x['supported_hit'] for x in m)/max(sup_total, 1):.0f}% "
                  f"| | | | |")
        if cond == "eulv":
            print("\nAgainst the exact real truth (selected family):\n")
            print("| Method | Exact | Recall | Precision |")
            print("| --- | --- | --- | --- |")
            for v in ("O_lv", "O_dm", "R"):
                hit = sum(len(set(map(tuple, complete[k][v]["selected"])) & exact_truth[k]) for k in keys)
                size = sum(len(complete[k][v]["selected"]) for k in keys)
                ex = sum(set(map(tuple, complete[k][v]["selected"])) == exact_truth[k] for k in keys)
                print(f"| {v} | {ex} | {100*hit/sum(len(exact_truth[k]) for k in keys):.0f}% "
                      f"| {100*hit/max(size, 1):.0f}% |")

    # ---------------------------------------------------------------- primary endpoints
    print("\n## Primary endpoints (one-sided, alpha = 0.05)\n")
    verdict = {}
    for comp in ("O_dm", "O_lv"):
        tag = "primary" if comp == "O_dm" else "secondary"
        syn = [k for k in complete if k[0] != "eulv"]
        a, b, p = mcnemar_greater([(complete[k]["R"]["metrics"]["exact"], complete[k][comp]["metrics"]["exact"])
                                   for k in syn])
        ok1 = p < ALPHA
        print(f"- P1 ({tag}), synthetic exact, R vs {comp}: R only {a}, {comp} only {b}, "
              f"exact McNemar one-sided p = {p:.3g} -> {'holds' if ok1 else 'does NOT hold'} ({len(syn)} inputs)")
        for cond in ("lock_short01", "lock_noisy"):
            ks = [k for k in syn if k[0] == cond]
            ca, cb, cp = mcnemar_greater([(complete[k]["R"]["metrics"]["exact"],
                                           complete[k][comp]["metrics"]["exact"]) for k in ks])
            print(f"  - {cond}: R exact {sum(complete[k]['R']['metrics']['exact'] for k in ks)}, {comp} exact "
                  f"{sum(complete[k][comp]['metrics']['exact'] for k in ks)} of {len(ks)}; R only {ca}, {comp} only "
                  f"{cb}, p = {cp:.3g}")
        real = [k for k in complete if k[0] == "eulv"]
        diffs = [complete[k]["R"]["metrics"]["supported_hit"]-complete[k][comp]["metrics"]["supported_hit"]
                 for k in real]
        nz = [d for d in diffs if d]
        pw = wilcoxon(nz, alternative="greater").pvalue if nz else 1.
        rp = pooled([complete[k]["R"]["metrics"] for k in real])
        op = pooled([complete[k][comp]["metrics"] for k in real])
        rs = sum(complete[k]["R"]["metrics"]["supported_hit"] for k in real)
        os_ = sum(complete[k][comp]["metrics"]["supported_hit"] for k in real)
        ok2 = pw < ALPHA and rp >= op-.05
        print(f"- P2 ({tag}), real supported recall, R vs {comp}: {rs} vs {os_} of {sum(len(ceil[k]) for k in real)} "
              f"supported segments (R more on {sum(d > 0 for d in diffs)}, fewer on {sum(d < 0 for d in diffs)}, "
              f"equal on {sum(d == 0 for d in diffs)}); Wilcoxon one-sided p = {pw:.3g}; precision {100*rp:.1f}% vs "
              f"{100*op:.1f}% (margin -5 pt) -> {'holds' if ok2 else 'does NOT hold'} ({len(real)} inputs)")
        verdict[comp] = (ok1, ok2)
    p1, p2 = verdict["O_dm"]
    claim = "CONFIRMED" if p1 and p2 else "PARTIALLY CONFIRMED" if p1 or p2 else "NOT CONFIRMED"
    print(f"\nPhase 3 decision (P1 and P2 against O_dm): {claim} (P1 {'holds' if p1 else 'fails'}, "
          f"P2 {'holds' if p2 else 'fails'})")

    # ---------------------------------------------------------------- S1 judge
    judge = rows("judge")
    if judge:
        print("\n## S1: the study's own judge\n")
        print("| Condition | Inputs | O_lv J <= J(truth) | O_dm | R | R exact | R different and better |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        for cond in CONDS:
            rs = [r for r in judge if r["cond"] == cond]
            if not rs:
                continue
            at = {v: sum(r["dJ"][v] <= 1e-6 for r in rs) for v in ("O_lv", "O_dm", "R")}
            ex = sum(r["same_as_truth"]["R"] for r in rs)
            bt = sum(r["dJ"]["R"] < -1e-6 and not r["same_as_truth"]["R"] for r in rs)
            print(f"| {cond} | {len(rs)} | {at['O_lv']} | {at['O_dm']} | {at['R']} | {ex} | {bt} |")

    # ---------------------------------------------------------------- S2 downstream
    bank_rows = rows("bank")
    if bank_rows:
        print("\n## S2: stage 0 of the probing study (study closure)\n")
        print("| Source | Condition | Truth in bank | Closure finished | Leader correct | S1 / S2 / S3 | Median closure time |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        groups = defaultdict(list)
        for r in bank_rows:
            groups[(r["source"], r["cond"])].append(r)
        for (src, cond), rs in sorted(groups.items(), key=lambda kv: (CONDS.index(kv[0][1]), kv[0][0])):
            st = defaultdict(int)
            for r in rs:
                st[r["stratum"]] += 1
            print(f"| {src} | {cond} | {sum(r['truth_in_bank'] for r in rs)}/{len(rs)} "
                  f"| {sum(r['closure']['capped'] is None for r in rs)} | {sum(r['leader_correct'] for r in rs)} "
                  f"| {st['S1']} / {st['S2']} / {st['S3']} | {median(r['closure_seconds'] for r in rs):.0f} s |")
        B = defaultdict(dict)
        for r in bank_rows:
            B[(r["cond"], r["name"])][r["source"]] = r
        both = [k for k in B if "orig" in B[k] and "R" in B[k]]
        a, b, p = mcnemar_greater([(B[k]["R"]["truth_in_bank"], B[k]["orig"]["truth_in_bank"]) for k in both])
        print(f"\nTruth in bank, paired: R only {a}, original only {b} (one-sided p = {p:.3g}, {len(both)} inputs)")

    # ---------------------------------------------------------------- S3 time, S4 certificates
    print("\n## S3: time\n")
    for cond in (*CONDS, "all"):
        ks = [k for k in complete if cond == "all" or k[0] == cond]
        if not ks:
            continue
        pair = [complete[k]["O_lv"]["wall"]+complete[k]["O_dm"]["wall"] for k in ks]
        rw = [complete[k]["R"]["wall"] for k in ks]
        print(f"- {cond}: R median {median(rw):.0f} s (max {max(rw):.0f}) vs O_lv + O_dm median {median(pair):.0f} s; "
              f"O_lv {median(complete[k]['O_lv']['wall'] for k in ks):.0f} s, "
              f"O_dm {median(complete[k]['O_dm']['wall'] for k in ks):.0f} s")
    print("\n## S4: certificates\n")
    cert = [complete[k]["R"] for k in complete if complete[k]["R"]["e"].get("certificates")]
    conf = sum(bool(r["e"].get("confirmed")) for r in cert)
    contra = sum(bool(r["e"].get("contradicted")) for r in cert)
    esc = sum(bool(r["e"].get("escalated")) for r in cert)
    timed = sum(1 for r in cert if not r["e"].get("confirmed") and not r["e"].get("contradicted"))
    improved = sum(any(c.get("improved_search") for c in r["e"]["certificates"]) for r in cert)
    print(f"E confirmed {conf}/{len(cert)}; contradicted after escalation {contra}; escalated {esc}; time limit "
          f"{timed}; certificate found a better family than the search {improved}")
    by_n = defaultdict(lambda: [0, 0])
    for r in cert:
        g = "n <= 15" if r["n"] <= 15 else "16-20" if r["n"] <= 20 else "21+"
        by_n[g][0] += 1
        by_n[g][1] += bool(r["e"].get("confirmed"))
    print("by feeder size: " + "; ".join(f"{g}: {c}/{t}" for g, (t, c) in sorted(by_n.items())))


if __name__ == "__main__":
    main()
