import json, numpy as np
r = [json.loads(l) for l in open(r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120\coverage\coverage.jsonl")]
r = [x for x in r if "error" not in x and (x["history"] == "1d" or x["seed"] <= 4)]
for lab in ("D", "D0"):
    s = [x[lab]["seconds"] for x in r]; z = [x[lab]["size"] for x in r]; p = [x[lab]["pruned_size"] for x in r]
    cap = [x[lab]["capped"] for x in r if x[lab]["capped"]]
    print(lab, "n", len(r), "sec min/med/max", round(min(s), 1), round(np.median(s), 1), round(max(s), 1),
          "size min/max", min(z), max(z), "pruned min/max", min(p), max(p), "capped", cap)
