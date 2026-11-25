"""Experiment M tables (PROTOCOL_MIQP.md): resolution certificates and their 16-day test (M3), the anatomy of W
(M1), pairs (M2), and WZZT-R on the transformer networks."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from statistics import median

from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_miqp import OUTM, internal, load
from rnj_wzzt_dr.identification.trees import basis, canonical, children


def rows(stage):
    path = OUTM/f"{stage}.jsonl"
    if not path.exists():
        return []
    out = [json.loads(line) for line in path.read_text().splitlines()]
    bad = [r for r in out if "error" in r]
    if bad:
        print(f"{stage}: {len(bad)} errors, e.g. {bad[0]['error']}")
    return [r for r in out if "error" not in r]


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > q)+.5*(p == q) for p in pos for q in neg)
    return wins/(len(pos)*len(neg))


def split_inside(node, blocks, family):
    """Does `family` hold a support strictly inside `node` that groups parts of two or more of its blocks?"""
    for S in family:
        if set(S) < set(node) and len(S) > 1 and not any(set(S) <= set(b) for b in blocks):
            return True
    return False


def resolution():
    res = rows("resolution")
    if not res:
        return
    print("\n## M3: certified resolution (node-restricted MIQP on WZZT-R's selected family)\n")
    print("| Inputs | Nodes | Proved optimal | Best split is a true segment | Nodes holding a missed true segment "
          "| AUC of certified gain for them | Median multiple (with / without) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for cond in ("eulv", "lock_short01", "lock_noisy", "eulv_large", "eulv_16d"):
        rs = [r for r in res if r["cond"] == cond]
        if not rs:
            continue
        nodes = [nd for r in rs for nd in r["nodes"] if "skipped" not in nd]
        opt = sum(nd["status"] == 2 for nd in nodes)
        true_best = sum(bool(nd["best_true"] and nd["best_true"][0]) for nd in nodes)
        with_m = [nd for nd in nodes if nd["missed_true"]]
        without = [nd for nd in nodes if not nd["missed_true"]]
        g = lambda nd: nd["max_gain_J"] if nd["max_gain_J"] is not None else -math.inf
        a = auc([g(nd) for nd in with_m], [g(nd) for nd in without])
        mult = lambda L: median(nd["multiple"] for nd in L if nd["multiple"] is not None) if L else None
        fmt = lambda v: "–" if v is None else (f"{v:.2f}" if v != math.inf else "inf")
        print(f"| {cond} ({len(rs)}) | {len(nodes)} | {opt} | {true_best} | {len(with_m)} "
              f"| {fmt(a)} | {fmt(mult(with_m))} / {fmt(mult(without))} |")
    # missed true segments: inside a certified node, or crossing the selected family
    for cond in ("eulv", "lock_short01", "lock_noisy"):
        rs = [r for r in res if r["cond"] == cond]
        if not rs:
            continue
        inside = sum(len(nd["missed_true"]) for r in rs for nd in r["nodes"] if "skipped" not in nd)
        missed = 0
        for r in rs:
            inp = load(r["cond"], r["name"])
            t = internal(inp.truth_key, inp.n)
            missed += len(t-set(map(tuple, r["family"])))
        print(f"- {cond}: {missed} true segments missed by WZZT-R; {inside} lie inside a node with a certificate "
              f"(the rest cross a selected support or sit in a node with fewer than three blocks)")
    # 16-day test of the data-requirement prediction
    r16 = {r["name"].rsplit("_s", 1)[0]: r for r in res if r["cond"] == "eulv_16d"}
    r8 = {r["name"].rsplit("_s", 1)[0]: r for r in res if r["cond"] == "eulv" and r["name"].endswith("_s0")}
    fit = {r["name"].rsplit("_s", 1)[0]: r for r in rows("fit") if r["cond"] == "eulv_16d" and r["variant"] == "R"}
    common = sorted(set(r8) & set(fit))
    if common:
        print("\n### Does the 8-day certificate predict what 16 days resolve? (33 feeders, start day 0)\n")
        table = defaultdict(lambda: [0, 0])
        detail = defaultdict(lambda: [0, 0])
        for case in common:
            r = r8[case]
            n = r["n"]
            fam8 = canonical([*map(tuple, r["family"]), *basis(n)])
            kids = children(fam8, n)
            fam16 = [tuple(S) for S in fit[case]["selected"]]
            for nd in [x for x in r["nodes"] if "skipped" not in x]:
                node = tuple(nd["node"])
                gmax = nd["max_gain_J"]
                if gmax is None:
                    continue
                n16 = 2*r["N"]
                price16 = .5*(r["price_J"]/(.5*math.log(r["N"])))*math.log(n16)     # p_s ln(2N) / 2
                predicted = 2*gmax > price16
                split = split_inside(node, kids[node], fam16)
                table[predicted][split] += 1
                best = tuple(nd["best"][0]) if nd["best"] else None
                detail[predicted][best in fam16 if best else False] += 1
        print("| 8-day certificate | Nodes | Split at 16 days | Not split | The certified best split itself selected |")
        print("| --- | --- | --- | --- | --- |")
        for pred in (False, True):
            ns, nn = table[pred][True], table[pred][False]
            print(f"| {'resolvable with 2x history' if pred else 'not resolvable with 2x history (certified)'} "
                  f"| {ns+nn} | {ns} | {nn} | {detail[pred][True]} |")


def anatomy():
    an = rows("anatomy")
    if not an:
        return
    print("\n## M1: anatomy of W (exact MIQP at every step of WZZT-R's path)\n")
    print("| Inputs | Steps | Root wider than 20 (screening off) | MIQP-critical steps | of which at nodes wider than 20 "
          "| MIQP support true (critical steps) | Generator support true (same steps) | MIQP time-limited |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    groups = defaultdict(list)
    for r in an:
        groups["development, n <= 20" if r["cond"] in ("eulv", "lock_short01", "lock_noisy") and r["n"] <= 20
               else "development, n 21-24" if r["cond"] in ("eulv", "lock_short01", "lock_noisy")
               else "transformer networks and large feeders, n 26-34"].append(r)
    for g, rs in groups.items():
        steps = [s for r in rs for s in r["steps"]]
        crit = [s for s in steps if s["critical"]]
        wide = sum(not s["screened"] for s in steps)
        print(f"| {g} ({len(rs)}) | {len(steps)} | {wide} | {len(crit)} | "
              f"{sum((s['miqp']['node_width'] or 0) > 20 for s in crit)} | {sum(bool(s['miqp']['true']) for s in crit)} "
              f"| {sum(bool(s['generator']['true']) for s in crit)} | {sum(s['miqp']['status'] == 9 for s in steps)} |")


def pairs():
    pr = rows("pairs")
    if not pr:
        return
    print("\n## M2: pairs (W2) after the single-support certificate\n")
    print("| Slot term | Inputs | Single certified | Pair solve proved | Improving pair | Interaction-critical "
          "| Pair supports true | Pair gain super-additive |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for v, label in (("R_c0", "c = 0"), ("R", "c = gamma (GLS)"), ("R_src", "c = 1 (free source)")):
        rs = [r for r in pr if r["variant"] == v]
        if not rs:
            continue
        sup = [r for r in rs if r["interaction_critical"]]
        superadd = sum(1 for r in sup if r.get("pair_alone_gain_J") and
                       r["pair"]["gain_J"] > sum(max(g, 0.) for g in r["pair_alone_gain_J"]))
        print(f"| {label} | {len(rs)} | {sum(r['single_certified'] for r in rs)} | "
              f"{sum(r['pair']['status'] == 2 for r in rs)} | {sum(r['pair_improves'] for r in rs)} | {len(sup)} "
              f"| {sum(sum(r['pair']['true']) for r in sup)} of {2*len(sup)} | {superadd} |")


def large():
    fit = rows("fit")
    if not fit:
        return
    print("\n## WZZT-R on the transformer networks and large feeders (generator W against MIQP W)\n")
    print("| Variant | Inputs | Exact | Recall | Precision | Recall (exact truth) | E confirmed | Median time | W MIQP time-limited steps |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cond in ("eulv_large", "eulv_16d"):
        for v in ("R", "R_wmiqp"):
            rs = [r for r in fit if r["cond"] == cond and r["variant"] == v]
            if not rs:
                continue
            m = [r["metrics"] for r in rs]
            conf = sum(bool(r["e"].get("confirmed")) for r in rs)
            tl = sum(1 for r in rs for ev in r["w_events"] if ev.get("miqp", {}).get("status") == 9)
            print(f"| {cond} {v} | {len(rs)} | {sum(x['exact'] for x in m)} | "
                  f"{100*sum(x['hit'] for x in m)/max(sum(x['truth'] for x in m), 1):.0f}% | "
                  f"{100*sum(x['hit'] for x in m)/max(sum(x['size'] for x in m), 1):.0f}% | "
                  f"{100*sum(x['exact_hit'] for x in m)/max(sum(x['exact_truth'] for x in m), 1):.0f}% | "
                  f"{conf}/{len(rs)} | {median(r['wall'] for r in rs):.0f} s | {tl} |")


def main():
    resolution()
    anatomy()
    pairs()
    large()


if __name__ == "__main__":
    main()
