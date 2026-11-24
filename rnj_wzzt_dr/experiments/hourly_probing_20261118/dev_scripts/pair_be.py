import json, sys, collections, numpy as np
R = r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120"
be_cond, base_cond = sys.argv[1], sys.argv[2]
load = lambda c: [json.loads(l) for l in open(rf"{R}\{c}\runs.jsonl")]
base = {(r["input"], r["path"]): r for r in load(base_cond) if r["arm"] == "passive_bic"}
be = load(be_cond)
for stratum in ("S1", "S2", "S3"):
    for arm in ("t_opt_P", "random_matched_P"):
        rs = [r for r in be if r["arm"] == arm and r["stratum"] == stratum and (r["input"], r["path"]) in base]
        if not rs: continue
        ps = [base[(r["input"], r["path"])] for r in rs]
        corr = lambda x: x["outcome"] == "correct"
        gain = sum(corr(a) and not corr(b) for a, b in zip(rs, ps)); loss = sum(corr(b) and not corr(a) for a, b in zip(rs, ps))
        wg = sum(a["ever_confirmed_wrong"] and not b["ever_confirmed_wrong"] for a, b in zip(rs, ps))
        wl = sum(b["ever_confirmed_wrong"] and not a["ever_confirmed_wrong"] for a, b in zip(rs, ps))
        fa = [r["first_correct_gate"] for r in rs if r["first_correct_gate"] is not None]
        fb = [r["first_correct_gate"] for r in ps if r["first_correct_gate"] is not None]
        wv = [r["wrong_vdev"] for r in rs if r["ever_confirmed_wrong"] and r.get("wrong_vdev") is not None]
        reasons = collections.Counter(b.get("reason") for r in rs for b in r["blocks"])
        print(f"{stratum} {arm:16s} n={len(rs)} correct {sum(map(corr, rs))} vs passive {sum(map(corr, ps))} (gain {gain}, loss {loss}) | "
              f"ever-wrong {sum(r['ever_confirmed_wrong'] for r in rs)} vs {sum(p['ever_confirmed_wrong'] for p in ps)} (+{wg}/-{wl}) | "
              f"median h {np.median(fa) if fa else '-'} vs {np.median(fb) if fb else '-'} | probed h {np.mean([r['probed_hours'] for r in rs]):.1f} "
              f"| CNY {np.mean([r['cost_cny'] for r in rs]):.2f} (max {max(r['cost_cny'] for r in rs):.1f}) | wrong>0.1% {sum(v > 1e-3 for v in wv)} max {100*max(wv) if wv else 0:.2f}% "
              f"| violations {sum(r['violations']['ac_band_hours']+r['violations']['deadband_hours'] for r in rs)} | reasons {dict(reasons.most_common(4))}")
