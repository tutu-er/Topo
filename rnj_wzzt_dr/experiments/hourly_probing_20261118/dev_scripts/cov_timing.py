import json, numpy as np
r = [json.loads(l) for l in open(r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120\coverage\coverage.jsonl")]
for h in ("1d", "3d"):
    for c in ("soumalas11", "pengwah18", "flynn16", "paper15"):
        s = [x["seconds"] for x in r if x["history"] == h and x["case"] == c]
        i = [sum(x["id_seconds"]) for x in r if x["history"] == h and x["case"] == c]
        if s:
            print(h, c, len(s), "job median", round(np.median(s)), "max", round(max(s)), "id median", round(np.median(i)))
