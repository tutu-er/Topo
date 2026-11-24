"""Rerun random_matched_P with the fixed headroom cap on the section 18 natural S1 inputs (tag rfix).

The random arm replays cert_P's hours and costs; cert_P is deterministic and unaffected by the fix, so its
logged schedule from runs.jsonl is replayed directly instead of re-running cert_P.
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, run
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import ROOT, settings_for, _json

CONDS = ("lock_short01", "lock_short005", "lock_noisy")


def one(cond, name, items):
    s = settings_for(cond)
    inp = bank.load(ROOT/cond/"inputs"/f"{name}.pkl")
    out = []
    for path, sched in items:
        rec = run.run(inp, "random_matched_P", path, s, replay=sched)
        rec["condition"] = cond
        out.append(rec)
    return out


def main(workers):
    for cond in CONDS:
        jobs = {}
        for line in open(ROOT/cond/"runs.jsonl"):
            r = json.loads(line)
            if r["arm"] == "cert_P" and r["stratum"] == "S1":
                jobs.setdefault(r["input"], []).append((r["path"], r["schedule"]))
        dst = ROOT/cond/"rfix.jsonl"
        done = {json.loads(l)["input"] for l in open(dst)} if dst.exists() else set()
        t0 = time.time()
        with ProcessPoolExecutor(workers) as pool, open(dst, "a") as fh:
            fut = {pool.submit(one, cond, n, items): n for n, items in jobs.items() if n not in done}
            for f in as_completed(fut):
                for rec in f.result():
                    fh.write(json.dumps(rec, default=_json)+"\n")
                fh.flush()
                print(f"{cond} {fut[f]} done {time.time()-t0:.0f}s", flush=True)
        print(f"ran rfix {cond}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
