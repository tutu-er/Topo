"""Profile WZZT (method 111) on real-feeder histories: time per stage, MIQP size and status, and how close its
output is to the identifiable truth (support recall / precision), for 8- and 18-day training data (demeaned run)."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys, time
from multiprocessing import Pool
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
sys.path.insert(0, r"C:/0-workspace/Topo/repo/rnj_wzzt_dr/experiments/hourly_probing_20261118/dev_scripts")
import numpy as np
from wzzt_criterion import history, S
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import model, experiments as ex
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc
from rnj_wzzt_dr.identification import algorithm
from rnj_wzzt_dr.identification.algorithm import SearchConfig
from rnj_wzzt_dr.physical.backend import Observation

CASES = ["eulv_td000653_feeder15", "eulv_td006144_feeder67", "eulv_td008020_feeder43", "eulv_td400067_feeder148",
         "eulv_td008016_feeder92", "eulv_td004454_feeder159", "eulv_td008690_feeder19"]


def one(job):
    case, days = job
    network, labels, hist, v_m, r_m = history(case, 0, days)
    n = len(labels)
    ident, _ = sc.identifiable_truth_family(network, labels, S.r_min_ohm)
    truth = {C for C in ident if 1 < len(C) < n}
    T = 24*days; tr, va = slice(0, T*3//4), slice(T*3//4, T)
    dm = lambda s: Observation(*(a[s]-a[s].mean(0) for a in (hist["P"], hist["Q"], hist["Y"])), labels)
    cfg = SearchConfig(total_seconds=900, per_mip_seconds=300, output_reserve_seconds=120)
    t0 = time.perf_counter()
    common = algorithm.prepare_common(dm(tr), cfg)
    t_common = time.perf_counter()-t0
    res = algorithm.fit(dm(tr), dm(va), "111", config=cfg, common=common)
    stages = {}
    for e in res["events"]:
        st = e["stage"]; sv = e.get("solver") or {}
        stages.setdefault(st, []).append(dict(seconds=sv.get("seconds"), status=sv.get("status"),
                                             variables=sv.get("variables"), binaries=sv.get("binary_variables"),
                                             gap=sv.get("gap"), accepted=e["accepted"],
                                             pretrial=e.get("pretrial_seconds")))
    qp = sum(r["seconds"] for r in res["QP"])
    sel = {tuple(C) for C in res["selected"]["point"]["supports"] if 1 < len(C) < n}
    best = max(([{tuple(C) for C in c["point"]["supports"] if 1 < len(C) < n} for c in res["frozen_candidates"]]
                + [{tuple(C) for C in p["supports"] if 1 < len(C) < n} for p in res.get("uncontracted_output_path", [])]),
               key=lambda s: len(s & truth)-.5*len(s-truth))
    return dict(case=case, days=days, n=n, truth=len(truth), total=res["total_seconds"], common=t_common, qp_seconds=qp,
                qp_count=len(res["QP"]), stages=stages, stop=res["stop_reason"],
                selected=len(sel), selected_hit=len(sel & truth), best_hit=len(best & truth), best_size=len(best))


if __name__ == "__main__":
    jobs = [(c, d) for c in CASES for d in (8, 18)]
    with Pool(14) as pool:
        out = pool.map(one, jobs)
    (ex.ROOT/"eulv"/"wzzt_profile.json").write_text(json.dumps(out, default=str))
    for o in out:
        st = {k: [(round(x["seconds"] or 0, 1), x["status"], x["variables"]) for x in v] for k, v in o["stages"].items()}
        print(f"{o['case'][5:]} n={o['n']} days={o['days']}: total {o['total']:.0f}s (common {o['common']:.0f}s, "
              f"fixed QPs {o['qp_seconds']:.0f}s in {o['qp_count']}), stop {o['stop']}; truth {o['truth']} segments, "
              f"selected {o['selected']} with {o['selected_hit']} true, best output {o['best_size']} with {o['best_hit']} true")
        print("    MIQP (seconds, status, variables):", st)
