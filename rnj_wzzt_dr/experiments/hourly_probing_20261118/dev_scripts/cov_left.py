import json
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import coverage as cv
recs = [json.loads(l) for l in open(cv.OUT/"coverage.jsonl") if l.strip()]
done = {r["name"] for r in recs if "error" not in r and ("neighbour_max_children" in r or r["case"] not in cv.REDO_PREFIX)}
left = [cv.job_name(*j) for j in cv.jobs(range(1, 9)) if cv.job_name(*j) not in done]
print(len(left), left[:20])
