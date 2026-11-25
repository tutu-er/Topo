"""Phase 2 tables and the decision rule D2 (PROTOCOL_PHASE2.md)."""
from __future__ import annotations

import json
from collections import defaultdict
from statistics import median

from scipy.stats import binomtest, wilcoxon

from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase2 import OUT, load_ceiling, internal

CONDS = ("eulv", "lock_short01", "lock_noisy")


def rows(name):
    path = OUT/f"{name}.jsonl"
    if not path.exists():
        return []
    out = [json.loads(line) for line in path.read_text().splitlines()]
    errors = [r for r in out if "error" in r]
    if errors:
        print(f"{name}: {len(errors)} errors, e.g. {errors[0]['error']}")
    return [r for r in out if "error" not in r]


def original(ceiling):
    """O_lv and O_dm from the Phase 1 capture, in the ladder's record format."""
    out = {}
    for cond, name in capture.select_inputs():
        sup = ceiling.get((cond, name), set())
        for mode in ("lv", "dm"):
            rec = capture.load_instance(capture.instance_path(cond, name, mode))
            sel, truth = set(map(tuple, rec["selected"])), set(map(tuple, rec["truth"]))
            outp = set(map(tuple, rec["output"]))
            paths = [internal(f, rec["n"]) for f in rec["path"]]
            out[(cond, name, "O_"+mode)] = dict(
                cond=cond, name=name, variant="O_"+mode, n=rec["n"], wall=rec["seconds"],
                selected=sorted(sel),
                metrics=dict(exact=sel == truth, hit=len(sel & truth), size=len(sel), truth=len(truth),
                             supported_hit=len(sel & sup)),
                output_metrics=dict(hit=len(outp & truth), size=len(outp), truth=len(truth),
                                    supported_hit=len(outp & sup)),
                truth_in_output=any(p == truth for p in paths) or sel == truth)
    return out


def table(recs, ceiling, exact_truth=None):
    """Aggregate per condition: exact, recall, precision, supported recall, output recall, truth in output, time."""
    agg = defaultdict(lambda: defaultdict(float))
    walls = defaultdict(list)
    for r in recs:
        a = agg[r["cond"]]
        m, o = r["metrics"], r["output_metrics"]
        a["inputs"] += 1; a["exact"] += m["exact"]; a["hit"] += m["hit"]; a["size"] += m["size"]
        a["truth"] += m["truth"]; a["sup_hit"] += m.get("supported_hit") or 0
        a["sup"] += len(ceiling.get((r["cond"], r["name"]), ()))
        a["out_hit"] += o["hit"]; a["out_sup_hit"] += o.get("supported_hit") or 0
        a["truth_in_output"] += bool(r.get("truth_in_output"))
        if exact_truth is not None and r["cond"] == "eulv":
            et = exact_truth[(r["cond"], r["name"])]
            sel = set(map(tuple, r["selected"]))
            a["x_exact"] += sel == et; a["x_hit"] += len(sel & et); a["x_truth"] += len(et)
        walls[r["cond"]].append(r["wall"])
        if "e" in r and r["e"].get("certificates"):
            a["certified"] += bool(r["e"].get("certified")); a["confirmed"] += bool(r["e"].get("confirmed"))
            a["cert_runs"] += 1
    return agg, walls


def fmt_row(label, a, walls):
    pct = lambda x, y: f"{100*x/y:.0f}%" if y else "–"
    s = (f"| {label} | {int(a['exact'])}/{int(a['inputs'])} | {pct(a['hit'], a['truth'])} | {pct(a['hit'], a['size'])} "
         f"| {pct(a['sup_hit'], a['sup'])} | {pct(a['out_hit'], a['truth'])} | {int(a['truth_in_output'])} "
         f"| {median(walls):.0f} s |")
    if a.get("cert_runs"):
        s += f" {int(a['confirmed'])}/{int(a['cert_runs'])} |"
    else:
        s += " – |"
    return s


def mcnemar(pairs):
    only_a = sum(1 for a, b in pairs if a and not b)
    only_b = sum(1 for a, b in pairs if b and not a)
    p = binomtest(only_a, only_a+only_b, .5).pvalue if only_a+only_b else 1.
    return only_a, only_b, p


def main():
    ceiling = load_ceiling()
    ladder = rows("ladder")
    orig = original(ceiling)
    exact_truth = {}
    for cond, name in capture.select_inputs():
        if cond == "eulv":
            inp = capture.load_input(cond, name)
            exact_truth[(cond, name)] = internal(inp.meta["exact_truth_key"], inp.n)
    by_variant = defaultdict(list)
    for r in ladder:
        by_variant[r["variant"]].append(r)
    for key, r in orig.items():
        by_variant[r["variant"]].append(r)
    order = ["O_lv", "O_dm", "R", *sorted(v for v in by_variant if v not in ("O_lv", "O_dm", "R"))]
    print("\n## Experiment C: component ladder\n")
    for cond in CONDS:
        print(f"\n### {cond}\n")
        print("| Variant | Exact | Recall | Precision | Supported recall | Output recall | Truth in output | Median time | Confirmed |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for v in order:
            recs = [r for r in by_variant.get(v, []) if r["cond"] == cond]
            if not recs:
                continue
            agg, walls = table(recs, ceiling, exact_truth)
            print(fmt_row(v, agg[cond], walls[cond]))
    print("\n### Real feeders against the exact truth (selected family)\n")
    print("| Variant | Exact | Recall |")
    print("| --- | --- | --- |")
    for v in order:
        recs = [r for r in by_variant.get(v, []) if r["cond"] == "eulv"]
        if recs:
            agg, _ = table(recs, ceiling, exact_truth)
            a = agg["eulv"]
            print(f"| {v} | {int(a['x_exact'])}/{int(a['inputs'])} | {100*a['x_hit']/a['x_truth']:.0f}% |")

    # ---------------------------------------------------------- D2
    print("\n## Decision rule D2\n")
    R = {(r["cond"], r["name"]): r for r in by_variant.get("R", [])}
    verdict = {}
    for mode in ("O_lv", "O_dm"):
        O = {(r["cond"], r["name"]): r for r in by_variant[mode]}
        syn = [k for k in R if k[0] != "eulv" and k in O]
        a, b, p = mcnemar([(R[k]["metrics"]["exact"], O[k]["metrics"]["exact"]) for k in syn])
        verdict[f"1_{mode}"] = a >= b
        print(f"- Rule 1, synthetic exact, R vs {mode}: R only {a}, {mode} only {b} (McNemar p = {p:.3g}) -> {'pass' if a >= b else 'FAIL'} ({len(syn)} inputs)")
        real = [k for k in R if k[0] == "eulv" and k in O]
        rs = sum(R[k]["metrics"]["supported_hit"] for k in real); os_ = sum(O[k]["metrics"]["supported_hit"] for k in real)
        rp = sum(R[k]["metrics"]["hit"] for k in real)/max(sum(R[k]["metrics"]["size"] for k in real), 1)
        op = sum(O[k]["metrics"]["hit"] for k in real)/max(sum(O[k]["metrics"]["size"] for k in real), 1)
        diffs = [R[k]["metrics"]["supported_hit"]-O[k]["metrics"]["supported_hit"] for k in real]
        pw = wilcoxon([d for d in diffs if d]).pvalue if any(diffs) else 1.
        ok = rs >= os_ and rp >= op-.05
        verdict[f"2_{mode}"] = ok
        print(f"- Rule 2, real supported recall, R vs {mode}: {rs} vs {os_} supported segments (Wilcoxon p = {pw:.3g}); "
              f"precision {100*rp:.1f}% vs {100*op:.1f}% -> {'pass' if ok else 'FAIL'} ({len(real)} inputs)")
    bank_rows = rows("bank")
    if bank_rows:
        B = defaultdict(dict)
        for r in bank_rows:
            B[(r["cond"], r["name"])][(r["source"], r["r3"])] = r
        keys = [k for k in B if all(s in B[k] for s in (("orig", False), ("orig", True), ("R", False), ("R", True)))]
        for r3 in (False, True):
            ro = sum(B[k][("orig", r3)]["truth_in_bank"] for k in keys)
            rr = sum(B[k][("R", r3)]["truth_in_bank"] for k in keys)
            fo = sum(B[k][("orig", r3)]["closure"]["capped"] is None for k in keys)
            fr = sum(B[k][("R", r3)]["closure"]["capped"] is None for k in keys)
            print(f"- Rule 3, closure {'R3' if r3 else 'study'}: truth in bank orig {ro} vs R {rr}; closure finished orig {fo} vs R {fr} ({len(keys)} inputs)")
        best = max((False, True), key=lambda r3: sum(B[k][("R", r3)]["truth_in_bank"] for k in keys))
        ok = len(keys) == 130 and all(sum(B[k][("R", r3)]["truth_in_bank"] for k in keys) >= sum(B[k][("orig", r3)]["truth_in_bank"] for k in keys)
                 and sum(B[k][("R", r3)]["closure"]["capped"] is None for k in keys) >= sum(B[k][("orig", r3)]["closure"]["capped"] is None for k in keys)
                 for r3 in (best,))
        verdict["3"] = ok
        print(f"  -> {'pass' if ok else 'FAIL'} (R closure {'R3' if best else 'study'})")
    cert = [r for r in R.values() if r["e"].get("certificates")]
    conf = sum(bool(r["e"].get("confirmed")) for r in cert)
    contra = sum(bool(r["e"].get("contradicted")) for r in cert)
    esc = sum(bool(r["e"].get("escalated")) for r in cert)
    timed = sum(1 for r in cert if not r["e"].get("confirmed") and not r["e"].get("contradicted"))
    improved = sum(any(c.get("improved_search") for c in r["e"]["certificates"]) for r in cert)
    verdict["4"] = conf >= .9*len(cert)
    print(f"- Rule 4, E confirmed: {conf}/{len(cert)} (still contradicted after escalation {contra}, escalated {esc}, "
          f"time limit {timed}; certificate found a better family than the search: {improved}) -> {'pass' if verdict['4'] else 'FAIL'}")
    for r in sorted(cert, key=lambda r: -r["n"]):
        if not r["e"].get("confirmed"):
            print(f"  - not confirmed: {r['cond']} {r['name']} (n = {r['n']}, pool {r['e']['pool']}): "
                  f"{'contradicted' if r['e'].get('contradicted') else 'time limit'}, gap {r['e'].get('gap')}")
    pair = [orig[(c, n, "O_lv")]["wall"]+orig[(c, n, "O_dm")]["wall"] for (c, n) in R]
    rw = [r["wall"] for r in R.values()]
    verdict["5"] = median(rw) <= median(pair)
    print(f"- Rule 5, time: R median {median(rw):.0f} s vs O_lv + O_dm median {median(pair):.0f} s -> {'pass' if verdict['5'] else 'FAIL'}")
    verdict.setdefault("3", False)
    print(f"\nD2: {'FREEZE' if verdict and all(verdict.values()) else 'not yet'} {verdict}")

    # ---------------------------------------------------------- downstream table
    if bank_rows:
        print("\n## Downstream: stage 0 of the probing study\n")
        print("| Source | Closure | Condition | Truth in bank | Closure finished | Leader correct | S1 / S2 / S3 | Median closure time |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        groups = defaultdict(list)
        for r in bank_rows:
            groups[(r["source"], r["r3"], r["cond"])].append(r)
        for (src, r3, cond), rs in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
            st = defaultdict(int)
            for r in rs:
                st[r["stratum"]] += 1
            print(f"| {src} | {'R3' if r3 else 'study'} | {cond} | {sum(r['truth_in_bank'] for r in rs)}/{len(rs)} "
                  f"| {sum(r['closure']['capped'] is None for r in rs)} | {sum(r['leader_correct'] for r in rs)} "
                  f"| {st['S1']} / {st['S2']} / {st['S3']} | {median(r['closure_seconds'] for r in rs):.0f} s |")
        stored = defaultdict(lambda: [0, 0, 0])
        for cond, name in capture.select_inputs():
            inp = capture.load_input(cond, name)
            s = stored[cond]
            s[0] += 1; s[1] += inp.truth_in_bank; s[2] += inp.meta["closure"]["capped"] is None
        for cond, (k, t, f) in stored.items():
            print(f"| stored inputs (study stage 0 as built) | study | {cond} | {t}/{k} | {f} | | | |")

    # ---------------------------------------------------------- root scenarios
    root = rows("root")
    if root:
        print("\n## Experiment E: root scenarios\n")
        print("| Variant | Scenario | Condition | Exact | Recall | Precision | Supported recall |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        groups = defaultdict(list)
        for r in root:
            groups[(r["variant"], r["scenario"], r["cond"])].append(r)
        for (v, sc, cond), rs in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
            agg, walls = table(rs, ceiling)
            a = agg[cond]
            print(f"| {v} | {sc} | {cond} | {int(a['exact'])}/{int(a['inputs'])} | {100*a['hit']/a['truth']:.0f}% "
                  f"| {100*a['hit']/max(a['size'], 1):.0f}% | {100*a['sup_hit']/max(a['sup'], 1):.0f}% |")


if __name__ == "__main__":
    main()
