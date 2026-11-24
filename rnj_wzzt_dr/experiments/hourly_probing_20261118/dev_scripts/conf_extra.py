import json, numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118.report import _mcnemar_p
R = r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120"
recs = {c: [json.loads(l) for l in open(rf"{R}\{c}\runs.jsonl")] for c in ("conf_main", "conf_short01", "conf_base_main", "conf_base_short01")}
nat = {c: [r for r in v if not r["input"].endswith("_missing")] for c, v in recs.items()}
# pooled costs and accuracy per selector on the refined pipeline
for arm in ("qbc_P", "bgreedy_P", "cert_P", "random_matched_P", "passive_bic"):
    rs = [r for c in ("conf_main", "conf_short01") for r in nat[c] if r["arm"] == arm]
    print(f"{arm:16s} runs {len(rs)} correct {sum(r['outcome']=='correct' for r in rs)} cost/day {np.mean([r['cost_cny'] for r in rs]):.3f} "
          f"probed h/day {np.mean([r['probed_hours'] for r in rs]):.2f} max day {max(r['cost_cny'] for r in rs):.1f}")
def paired(a, b):
    A = {(r["input"], r["path"]): r for c in ("conf_main", "conf_short01") for r in nat[c] if r["arm"] == a}
    B = {(r["input"], r["path"]): r for c in ("conf_main", "conf_short01") for r in nat[c] if r["arm"] == b}
    k = set(A) & set(B); ok = lambda r: r["outcome"] == "correct"
    g = sum(ok(A[x]) and not ok(B[x]) for x in k); l = sum(ok(B[x]) and not ok(A[x]) for x in k)
    return g, l, round(_mcnemar_p(g, l), 3)
print("bgreedy vs cert", paired("bgreedy_P", "cert_P"), "bgreedy vs qbc", paired("bgreedy_P", "qbc_P"), "bgreedy vs passive", paired("bgreedy_P", "passive_bic"))
allr = [r for v in recs.values() for r in v]
viol = sum(r["violations"]["ac_band_hours"]+r["violations"]["deadband_hours"] for r in allr)
gs = [g["seconds"] for r in allr for g in r["gates"]]
print("confirmation days", len(allr), "violation hours", viol, "V", round(min(r["violations"]["min_v"] for r in allr), 4), round(max(r["violations"]["max_v"] for r in allr), 4),
      "gate s mean", round(np.mean(gs), 2), "max", round(max(gs), 1))
for c in ("conf_base_main", "conf_base_short01"):
    inp = json.load(open(rf"{R}\{c}\inputs.json"))
    print(c, "identifier validation-MSE pick correct", sum(r.get("validation_mse_correct", False) for r in inp), "of", len(inp))
