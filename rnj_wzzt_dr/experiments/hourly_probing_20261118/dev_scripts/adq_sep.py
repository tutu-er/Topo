import json, sys, numpy as np
R = r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120"
for cond in sys.argv[1].split(","):
    recs = [json.loads(l) for l in open(rf"{R}\{cond}\runs.jsonl")]
    for stratum in ("S1", "S2", "S3"):
        a_conf, vdev = [], []
        for r in recs:
            if r["arm"] != "passive_bic" or r["stratum"] != stratum:
                continue
            g = next((g for g in r["gates"] if g["confirmed"]), None)
            if g is not None and "adequacy" in g:
                a_conf.append((g["adequacy"], g["leader_correct"]))
            if r["ever_confirmed_wrong"] and r.get("wrong_vdev") is not None:
                vdev.append(r["wrong_vdev"])
        if not a_conf:
            continue
        c = np.array([a for a, ok in a_conf if ok]); w = np.array([a for a, ok in a_conf if not ok])
        q = lambda x: f"n={len(x)} med {np.median(x):.3f} p90 {np.percentile(x, 90):.3f} max {x.max():.3f}" if len(x) else "n=0"
        print(f"{cond} {stratum}: adequacy at first confirmation | correct {q(c)} | wrong {q(w)} | wrong vdev max {100*max(vdev) if vdev else 0:.2f}% med {100*np.median(vdev) if vdev else 0:.2f}%")
