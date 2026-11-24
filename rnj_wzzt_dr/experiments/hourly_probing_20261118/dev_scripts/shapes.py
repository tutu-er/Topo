import json, sys, collections, numpy as np
R = r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120"
for c in sys.argv[1:]:
    recs = [json.loads(l) for l in open(rf"{R}\{c}\runs.jsonl")]
    for arm in ("cert_P", "qbc_P"):
        bl = [b for r in recs if r["arm"] == arm and r["stratum"] == "S1" for b in r["blocks"] if any(abs(x) > 0 for x in b["u_kw"])]
        kinds = collections.Counter(("group" if (b.get("shape") or "").startswith("group") else "node") for b in bl)
        kw = [sum(abs(x) for x in b["u_kw"]) for b in bl]
        B = [b["cert_B"] for b in bl if b.get("cert_B") is not None]
        print(c, arm, "blocks", len(bl), dict(kinds), "mean |u| kW %.2f" % np.mean(kw) if kw else "", 
              "B median %.2f min %.2f share<1 %.2f" % (np.median(B), min(B), np.mean(np.array(B) < 1)) if B else "")
