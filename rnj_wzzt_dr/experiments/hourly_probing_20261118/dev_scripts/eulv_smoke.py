"""Smoke test of the real-feeder condition: build 4 inputs, run passive and the strong greedy on each."""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, run, config
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import ROOT, settings_for, _build_one

JOBS = [("eulv_td009116_feeder106", 0), ("eulv_td009116_feeder106", 9),
        ("eulv_td008690_feeder21", 0), ("eulv_td008690_feeder21", 9)]


def build(job):
    t = time.time()
    row = _build_one("eulv", job[0], job[1], (24, 12))
    row["build_s"] = round(time.time()-t)
    return row


def go(name):
    s = settings_for("eulv")
    inp = bank.load(ROOT/"eulv"/"inputs"/f"{name}.pkl")
    out = []
    for arm in ("passive_bic", config.LOCKED_ARM):
        t = time.time()
        r = run.run(inp, arm, 0, s)
        out.append(dict(input=name, arm=arm, outcome=r["outcome"], first_correct=r["first_correct_gate"],
                        ever_wrong=r["ever_confirmed_wrong"], probed=r["probed_hours"], cost=round(r["cost_cny"], 1),
                        K=r["K"], truth_in_bank=r["truth_in_bank"], viol=r["violations"]["ac_band_hours"],
                        excess=r["headroom_excess"]["hours"], seconds=round(time.time()-t)))
    return out


if __name__ == "__main__":
    s = settings_for("eulv")
    print("settings: slots_per_hour", s.slots_per_hour, "V", s.cert_value_cny, "cost", s.cost_model, "bias", s.meter_bias_sigma)
    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(build, JOBS))
    for r in rows:
        print(json.dumps({k: r.get(k) for k in ("name", "stratum", "truth_in_bank", "bank", "leader_correct",
                                                 "posterior_ok", "build_s", "error")}))
        if r.get("error"):
            print(r.get("trace"))
    names = [r["name"] for r in rows if not r.get("error")]
    with ProcessPoolExecutor(4) as pool:
        for out in pool.map(go, names):
            for o in out:
                print(json.dumps(o))
