"""Summaries of the development runs: outcomes, wrong confirmations by type, cost, b/J."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

from .experiments import ROOT

OUTCOMES = ("correct", "wrong", "undecided", "unresolved_no_progress", "unresolved_beyond_reach")


def load(condition, tag="runs"):
    path = ROOT/condition/f"{tag}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def pct(k, n):
    return f"{100*k/n:.0f}%" if n else "-"


def summarise(recs):
    rows = []
    groups = defaultdict(list)
    for r in recs:
        groups[(r["stratum"], r["arm"])].append(r)
    for (stratum, arm), rs in sorted(groups.items()):
        n = len(rs)
        oc = Counter(r["outcome"] for r in rs)
        wrong_types = Counter(t for r in rs for t in r["ever_wrong_types"])
        first = [r["first_correct_gate"] for r in rs if r["first_correct_gate"] is not None]
        blocks = [b for r in rs for b in r["blocks"] if any(abs(x) > 0 for x in b["u_kw"])]
        empty = sum(b["window_empty"] for b in blocks)
        flagged = Counter(); total_bias = Counter()
        for b in blocks:
            for key, v in (b.get("bias") or {}).items():
                if key.endswith("_loss"):
                    total_bias[v["type"]] += 1
                    flagged[v["type"]] += abs(v["whole"]) > .5
        rows.append(dict(
            stratum=stratum, arm=arm, n=n, **{o: oc.get(o, 0) for o in OUTCOMES},
            ever_wrong=sum(r["ever_confirmed_wrong"] for r in rs),
            wrong_extra=wrong_types.get("extra_line", 0), wrong_missing=wrong_types.get("missing_line", 0),
            wrong_placement=wrong_types.get("wrong_placement", 0),
            leader_correct_end=sum(r["final_leader_correct"] for r in rs),
            median_first_correct=float(np.median(first)) if first else None,
            probed_hours=float(np.mean([r["probed_hours"] for r in rs])),
            cost=float(np.mean([r["cost_cny"] for r in rs])),
            cost_max=float(np.max([r["cost_cny"] for r in rs])),
            blocks=len(blocks), window_empty=empty,
            bias_flagged={k: [flagged[k], total_bias[k]] for k in total_bias},
            ac_band_hours=sum(r["violations"]["ac_band_hours"] for r in rs),
            deadband_hours=sum(r["violations"]["deadband_hours"] for r in rs),
            gate_seconds_mean=float(np.mean([g["seconds"] for r in rs for g in r["gates"]])),
            gate_seconds_max=float(np.max([g["seconds"] for r in rs for g in r["gates"]])),
            beyond_types=dict(Counter(b["type"] for r in rs for b in r["beyond_at_last_gate"])),
        ))
    return rows


def paired(recs, a="t_opt_P", b="passive_bic", stratum=("S1",), field="outcome", value="correct"):
    """McNemar counts on paired (input, path): a yes / b no, and a no / b yes."""
    idx = {}
    for r in recs:
        if r["stratum"] in stratum:
            idx[(r["input"], r["path"], r["arm"])] = r
    keys = {(i, p) for (i, p, arm) in idx if arm == a and (i, p, b) in idx}
    gain = loss = 0
    for i, p in keys:
        ya = idx[(i, p, a)][field] == value if field == "outcome" else bool(idx[(i, p, a)][field])
        yb = idx[(i, p, b)][field] == value if field == "outcome" else bool(idx[(i, p, b)][field])
        gain += ya and not yb; loss += yb and not ya
    return dict(pairs=len(keys), gain=gain, loss=loss)


def table(rows):
    head = ("| Stratum | Arm | n | Correct | Wrong | Undecided | No progress | Beyond reach | Ever wrong "
            "(extra/missing/placement) | Median h to correct | Probed h | Cost CNY/day (max) | Empty windows |")
    lines = [head, "|"+" --- |"*13]
    for r in rows:
        mf = "-" if r["median_first_correct"] is None else f"{r['median_first_correct']:.0f}"
        lines.append(f"| {r['stratum']} | {r['arm']} | {r['n']} | {pct(r['correct'], r['n'])} | {pct(r['wrong'], r['n'])} | "
                     f"{pct(r['undecided'], r['n'])} | {pct(r['unresolved_no_progress'], r['n'])} | "
                     f"{pct(r['unresolved_beyond_reach'], r['n'])} | {r['ever_wrong']} ({r['wrong_extra']}/"
                     f"{r['wrong_missing']}/{r['wrong_placement']}) | {mf} | {r['probed_hours']:.1f} | "
                     f"{r['cost']:.2f} ({r['cost_max']:.1f}) | {r['window_empty']}/{r['blocks']} |")
    return "\n".join(lines)


def main(conditions):
    out = {}
    for c in conditions:
        recs = load(c)
        if not recs:
            continue
        rows = summarise(recs)
        out[c] = dict(rows=rows,
                      C1_correct_S1=paired(recs),
                      C1_wrong_S2S3=paired(recs, field="ever_confirmed_wrong", value=True, stratum=("S2", "S3")),
                      C2_correct_S1=paired(recs, b="random_matched_P"),
                      C3_wrong_S3=paired(recs, a="check_then_t_opt", field="ever_confirmed_wrong", value=True,
                                         stratum=("S3",)),
                      C4_wrong_S1S2=paired(recs, a="frozen_replay", field="ever_confirmed_wrong", value=True,
                                           stratum=("S1", "S2")))
        print(f"\n## {c}\n"); print(table(rows))
        for key in ("C1_correct_S1", "C1_wrong_S2S3", "C2_correct_S1", "C3_wrong_S3", "C4_wrong_S1S2"):
            print(key, out[c][key])
    (ROOT/"summary.json").write_text(json.dumps(out, indent=1))
    return out


def followup(conditions=("bias_typical", "v2_bias_typical", "v2_main", "v2_shortline", "v2_bias_stress")):
    """Second-round table: per condition, stratum and arm, outcomes plus how far the
    confirmed models are from the true voltages and the leader's adequacy ratio."""
    out = {}
    print("| Condition | Stratum | Arm | n | Correct | Ever wrong | Wrong >0.1% V | Max wrong V dev | Median h to correct "
          "| Probed h/day | CNY/day (max) | Leader adequacy, median (max) |")
    print("|"+" --- |"*12)
    for c in conditions:
        recs = load(c)
        if not recs:
            continue
        groups = defaultdict(list)
        for r in recs:
            groups[(r["stratum"], r["arm"])].append(r)
        out[c] = {}
        for (stratum, arm), rs in sorted(groups.items()):
            n = len(rs)
            wv = [r["wrong_vdev"] for r in rs if r["ever_confirmed_wrong"] and r.get("wrong_vdev") is not None]
            first = [r["first_correct_gate"] for r in rs if r["first_correct_gate"] is not None]
            adq = [g["adequacy"] for r in rs for g in r["gates"] if "adequacy" in g]
            row = dict(n=n, correct=sum(r["outcome"] == "correct" for r in rs),
                       ever_wrong=sum(r["ever_confirmed_wrong"] for r in rs),
                       wrong_over_1e3=sum(v > 1e-3 for v in wv), wrong_vdev_max=max(wv) if wv else None,
                       median_first_correct=float(np.median(first)) if first else None,
                       probed_hours=float(np.mean([r["probed_hours"] for r in rs])),
                       cost=float(np.mean([r["cost_cny"] for r in rs])), cost_max=float(np.max([r["cost_cny"] for r in rs])),
                       adequacy_median=float(np.median(adq)) if adq else None,
                       adequacy_max=float(np.max(adq)) if adq else None,
                       violations=sum(r["violations"]["ac_band_hours"]+r["violations"]["deadband_hours"] for r in rs))
            out[c][f"{stratum}/{arm}"] = row
            fmt = lambda v, f: "-" if v is None else format(v, f)
            print(f"| {c} | {stratum} | {arm} | {n} | {pct(row['correct'], n)} | {pct(row['ever_wrong'], n)} | "
                  f"{row['wrong_over_1e3']} | {fmt(row['wrong_vdev_max'] and 100*row['wrong_vdev_max'], '.2f')}% | "
                  f"{fmt(row['median_first_correct'], '.0f')} | {row['probed_hours']:.1f} | {row['cost']:.2f} ({row['cost_max']:.1f}) | "
                  f"{fmt(row['adequacy_median'], '.2f')} ({fmt(row['adequacy_max'], '.2f')}) |")
        out[c]["C1_correct_S1"] = paired(recs)
        out[c]["C3_wrong_S3"] = paired(recs, a="check_then_t_opt", field="ever_confirmed_wrong", value=True, stratum=("S3",))
        print(c, "C1 S1 correct t_opt vs passive", out[c]["C1_correct_S1"], "| C3 S3 wrong check vs passive", out[c]["C3_wrong_S3"])
    (ROOT/"summary_followup.json").write_text(json.dumps(out, indent=1))
    return out


def _mcnemar_p(gain, loss):
    """Exact two-sided McNemar p-value."""
    n = gain+loss
    if n == 0:
        return 1.
    k = min(gain, loss)
    return min(1., 2*sum(math.comb(n, i) for i in range(k+1))/2**n)


def compare(conditions=("cmp_main", "cmp_short01", "cmp_lowflex", "cmp_highflex"), ref="cert_P",
            arms=("passive_bic", "random_matched_P", "qbc_matched_P", "qbc_P", "greedy_P")):
    """Section 13: the certified planner against each arm, paired by input and path."""
    out, pooled = {}, defaultdict(lambda: [0, 0])
    for c in conditions:
        recs = load(c)
        if not recs:
            continue
        out[c] = {}
        for stratum in ("S1", "S3"):
            idx = {(r["input"], r["path"], r["arm"]): r for r in recs if r["stratum"] == stratum}
            arms_here = sorted({a for (_, _, a) in idx})
            if ref not in arms_here:
                continue
            print(f"\n## {c} {stratum}")
            print("| Arm | n | Correct | Ever wrong | Median h to correct | Probed h/day | CNY/day (max) | vs cert_P: cert gains / losses (p) |")
            print("|"+" --- |"*8)
            for arm in [ref, *arms]:
                rs = [r for (i, p, a), r in idx.items() if a == arm]
                if not rs:
                    continue
                first = [r["first_correct_gate"] for r in rs if r["first_correct_gate"] is not None]
                row = dict(n=len(rs), correct=sum(r["outcome"] == "correct" for r in rs),
                           ever_wrong=sum(r["ever_confirmed_wrong"] for r in rs),
                           median_first=float(np.median(first)) if first else None,
                           probed=float(np.mean([r["probed_hours"] for r in rs])),
                           cost=float(np.mean([r["cost_cny"] for r in rs])),
                           cost_max=float(np.max([r["cost_cny"] for r in rs])))
                if arm != ref:
                    keys = [(i, p) for (i, p, a) in idx if a == arm and (i, p, ref) in idx]
                    field = "outcome" if stratum == "S1" else "ever_confirmed_wrong"
                    good = (lambda r: r["outcome"] == "correct") if stratum == "S1" else (lambda r: not r["ever_confirmed_wrong"])
                    g = sum(good(idx[(i, p, ref)]) and not good(idx[(i, p, arm)]) for i, p in keys)
                    l = sum(good(idx[(i, p, arm)]) and not good(idx[(i, p, ref)]) for i, p in keys)
                    row.update(gain=g, loss=l, p=_mcnemar_p(g, l))
                    if stratum == "S1":
                        pooled[arm][0] += g; pooled[arm][1] += l
                out[c][f"{stratum}/{arm}"] = row
                mf = "-" if row["median_first"] is None else f"{row['median_first']:.0f}"
                vs = "" if arm == ref else f"{row['gain']} / {row['loss']} ({row['p']:.3f})"
                print(f"| {arm} | {row['n']} | {pct(row['correct'], row['n'])} | {pct(row['ever_wrong'], row['n'])} | "
                      f"{mf} | {row['probed']:.1f} | {row['cost']:.2f} ({row['cost_max']:.1f}) | {vs} |")
    out["pooled_S1"] = {a: dict(gain=g, loss=l, p=_mcnemar_p(g, l)) for a, (g, l) in pooled.items()}
    print("\npooled S1, cert_P gains / losses:", {a: (v["gain"], v["loss"], round(v["p"], 4)) for a, v in out["pooled_S1"].items()})
    (ROOT/"summary_compare.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["compare"]:
        compare(*([sys.argv[2:]] if sys.argv[2:] else []))
    elif sys.argv[1:2] == ["followup"]:
        followup(*([sys.argv[2:]] if sys.argv[2:] else []))
    else:
        main(sys.argv[1:] or ["main"])
