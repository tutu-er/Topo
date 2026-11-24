import json, sys, numpy as np
R = r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120"
for cond in sys.argv[1].split(","):
    recs = [json.loads(l) for l in open(rf"{R}\{cond}\runs.jsonl") if "_missing" in l]
    for arm in ("passive_bic", "qbc_P", "bgreedy_P", "cert_P", "random_matched_P"):
        rs = [r for r in recs if r["arm"] == arm and r["input"].endswith("_missing")]
        if not rs:
            continue
        wv = [r["wrong_vdev"] for r in rs if r["ever_confirmed_wrong"] and r.get("wrong_vdev") is not None]
        print(f"{cond} S3 {arm:16s} n={len(rs)} ever wrong {sum(r['ever_confirmed_wrong'] for r in rs)} "
              f"({100*np.mean([r['ever_confirmed_wrong'] for r in rs]):.0f}%) max V dev {100*max(wv) if wv else 0:.2f}% "
              f">0.1%: {sum(v > 1e-3 for v in wv)} >0.2%: {sum(v > 2e-3 for v in wv)} cost {np.mean([r['cost_cny'] for r in rs]):.2f}")
