"""Development run for a missing-truth check (after section 18; exploratory, not registered).

The strong greedy at V = 2,000 and passive, path 0, on the section 18 natural S1 inputs and their
truth-removed copies, with the evaluation-only prequential check switched on (tag preq). Decisions are
unchanged (the check only logs), so outcomes reproduce the logged section 18 runs.
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, run
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import ROOT, settings_for, _json

CONDS = ("lock_short01", "lock_short005", "lock_noisy")
ARMS = ("bgreedy_V2000_P", "passive_bic")


def one(cond, name):
    s = settings_for(cond).with_(prequential_check=True)
    inp = bank.load(ROOT/cond/"inputs"/f"{name}.pkl")
    out = []
    for arm in ARMS:
        rec = run.run(inp, arm, 0, s)
        rec["condition"] = cond
        out.append(rec)
    return out


def main(workers):
    for cond in CONDS:
        s1 = sorted({json.loads(l)["input"] for l in open(ROOT/cond/"runs.jsonl") if json.loads(l)["stratum"] == "S1"})
        names = s1+[n+"_missing" for n in s1]
        dst = ROOT/cond/"preq.jsonl"
        done = {json.loads(l)["input"] for l in open(dst)} if dst.exists() else set()
        t0 = time.time()
        with ProcessPoolExecutor(workers) as pool, open(dst, "a") as fh:
            fut = {pool.submit(one, cond, n): n for n in names if n not in done}
            for f in as_completed(fut):
                try:
                    for rec in f.result():
                        fh.write(json.dumps(rec, default=_json)+"\n")
                    fh.flush()
                    print(f"{cond} {fut[f]} done {time.time()-t0:.0f}s", flush=True)
                except Exception as e:
                    print(f"{cond} {fut[f]} FAILED {e!r}", flush=True)
        print(f"ran preq {cond}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
