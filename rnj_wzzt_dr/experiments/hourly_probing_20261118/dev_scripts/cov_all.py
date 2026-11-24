import json
r = [json.loads(l) for l in open(r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120\coverage\coverage.jsonl")]
r = [x for x in r if "error" not in x]
print(len(r), "D cover", sum(x["D"]["cover"] for x in r), "D leader", sum(x["D"]["leader_correct"] for x in r))
