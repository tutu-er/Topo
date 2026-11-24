import json, collections
base = r"C:/0-workspace/Topo/repo/outputs/hourly_probing_dev_20261120/"
for cond in ["conf_main", "conf_short01", "dev_short01"]:
    c = collections.Counter(); runs = collections.Counter(); out = collections.Counter()
    with open(base+cond+"/runs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["arm"] != "bgreedy_P": continue
            reasons = [b["reason"] for b in r["blocks"]]
            for x in reasons: c[x] += 1
            st = "S3" if r["stratum"] == "S3" else "nat"
            if "one_class" in reasons:
                runs[st] += 1; out[(st, r["outcome"])] += 1
    print(cond, dict(c), "runs with one_class:", dict(runs), dict(out))
