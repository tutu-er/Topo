"""Real-feeder pilot (development, not evidence): 33 feeders x start days 0 and 6, 8-day hourly history,
identifiable-parameter BIC and identifiable-truth target; passive and the strong greedy, 1 path (tag pilot)."""
import json, sys, time
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import experiments as ex, config

if __name__ == "__main__":
    t = time.time()
    seeds = {c: [(d, h) for d, h in lst if d in (0, 6)] for c, lst in ex._seeds_eulv().items()}
    rows = ex.build("eulv", workers=14, seeds=seeds)
    print(f"built {len(rows)} inputs in {time.time()-t:.0f}s; errors {sum('error' in r for r in rows)}", flush=True)
    names = [r["name"] for r in rows if "error" not in r]
    ex.run_condition("eulv", ["passive_bic", config.LOCKED_ARM], paths=1, workers=14, names=names, tag="pilot")
    print(f"ran pilot in {time.time()-t:.0f}s", flush=True)
