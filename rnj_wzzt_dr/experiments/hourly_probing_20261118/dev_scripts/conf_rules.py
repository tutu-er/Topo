"""Offline re-evaluation of the confirmation rule from logged gates (exact for passive_bic)."""
import json, sys, numpy as np
R = r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120"

def replay(gates, alpha, hours):
    t = alpha/(1-alpha); leader, count = None, 0
    first, ever_wrong, conf = None, False, False
    for g in gates:
        ok = g["mass"] <= t
        if ok and leader == g["leader"]:
            count += 1
        elif ok:
            leader, count = g["leader"], 1
        else:
            leader, count = None, 0
        conf = count >= hours
        if conf and not g["leader_correct"]:
            ever_wrong = True
        if conf and g["leader_correct"] and first is None:
            first = g["g"]
    return conf and gates[-1]["leader_correct"], ever_wrong, first

for cond in sys.argv[1].split(","):
    recs = [json.loads(l) for l in open(rf"{R}\{cond}\runs.jsonl")]
    arms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["passive_bic"]
    for arm in arms:
        for stratum in ("S1", "S3"):
            rs = [r for r in recs if r["arm"] == arm and r["stratum"] == stratum]
            if not rs:
                continue
            line = []
            for alpha, hours in ((.05, 2), (.05, 3), (.02, 2), (.01, 2), (.01, 3)):
                out = [replay(r["gates"], alpha, hours) for r in rs]
                corr = sum(o[0] for o in out); wrong = sum(o[1] for o in out)
                fs = [o[2] for o in out if o[2] is not None]
                line.append(f"a={alpha} h={hours}: correct {corr}/{len(rs)} wrong {wrong} med {np.median(fs) if fs else '-'}")
            print(cond, arm, stratum, " | ".join(line))
