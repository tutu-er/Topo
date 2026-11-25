"""How much does each support's downstream Q vary independently of its P (residual of b on a, relative to the
level of b)? Real feeders (fixed power factor) against synthetic inputs."""
import sys, glob
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
import numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import ROOT


def ratios(inp):
    out = []
    for f in inp.families[:40]:
        c = model.Candidate(f, inp.n)
        a, b = c.flows(inp.hist["P"], inp.hist["Q"])
        ad, bd = a-a.mean(0), b-b.mean(0)
        for s in range(c.S):
            coef = ad[:, s]@bd[:, s]/max(ad[:, s]@ad[:, s], 1e-300)
            nb = np.linalg.norm(b[:, s])
            if nb > 0:
                out.append(np.linalg.norm(bd[:, s]-coef*ad[:, s])/nb)
    return np.array(out)


for label, pat in (("real (eulv)", "eulv/inputs/*_s*.pkl"), ("synthetic lock_short01", "lock_short01/inputs/*_s6[5-9].pkl"),
                   ("synthetic lock_noisy", "lock_noisy/inputs/*_s6[5-9].pkl")):
    rs = np.concatenate([ratios(bank.load(p)) for p in sorted(glob.glob(str(ROOT/pat)))[:6] if "_missing" not in p])
    print(f"{label:24s}: supports {rs.size}, ratio min {rs.min():.4f}, q01 {np.quantile(rs,.01):.4f}, median "
          f"{np.median(rs):.4f}, q99 {np.quantile(rs,.99):.4f}, max {rs.max():.4f}")
print("threshold 3*sqrt(2)*pq_sigma =", 3*np.sqrt(2)*.005)
