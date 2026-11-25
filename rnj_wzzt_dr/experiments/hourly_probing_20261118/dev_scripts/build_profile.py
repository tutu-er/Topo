"""Where does building one real-feeder input spend its time? cProfile of bank.build for two pilot inputs (one core each)."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import cProfile, io, pstats, sys, time
from multiprocessing import Pool
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, experiments as ex

S = ex.settings_for("eulv")
FUNCS = ("identifier_families", "closure", "fit_one", "passive_fit", "classify", "fit_sensitivity", "rnj_guidance",
         "optimize", "lsq_linear", "loss_term", "voltages", "neighbours", "identifiable_p")


def one(case):
    pr = cProfile.Profile(); t = time.perf_counter()
    pr.enable()
    inp = bank.build(case, 0, S, history=(144, 48), step1="v2")
    pr.disable()
    total = time.perf_counter()-t
    st = pstats.Stats(pr)
    rows = {}
    for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
        if name in FUNCS:
            r = rows.setdefault(name, [0, 0.])
            r[0] += nc; r[1] = max(r[1], ct)
    return case, total, inp.meta.get("closure"), inp.meta.get("identifier_seconds"), rows


if __name__ == "__main__":
    with Pool(2) as pool:
        for case, total, cl, ids, rows in pool.map(one, ["eulv_td004454_feeder159", "eulv_td008016_feeder92"]):
            print(f"{case}: build {total:.0f}s; identifier {ids:.0f}s; closure {cl}")
            for name, (calls, cum) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
                print(f"    {name}: {calls} calls, cumulative {cum:.0f}s")
