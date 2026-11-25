"""Acceleration test: WZZT's E-stage MIQP with the data rows compressed by QR (exact: same objective up to a constant)
against the original (one residual variable per data row). Same pool, keep, warm start and time limit."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys, time
from itertools import combinations
from multiprocessing import Pool
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
sys.path.insert(0, r"C:/0-workspace/Topo/repo/rnj_wzzt_dr/experiments/hourly_probing_20261118/dev_scripts")
import numpy as np
from wzzt_criterion import history
from rnj_wzzt_dr.identification import algorithm, trees
from rnj_wzzt_dr.identification.algorithm import SearchConfig
from rnj_wzzt_dr.identification.optimization import QPFitter, StructuralSolver, feature_matrices, predict, compatible
from rnj_wzzt_dr.identification.trees import canonical, basis
from rnj_wzzt_dr.physical.backend import Observation


class CompressedSolver(StructuralSolver):
    def select(self, pool, keep, warm, *, deadline):
        started = time.perf_counter()
        n, bound = self.data.n, self.config.response_bound
        family = canonical([*pool, *keep, *basis(n)])
        fixed = set(canonical([*keep, *basis(n)]))
        proper = [C for C in family if 1 < len(C) < n]
        with self._model("RTE_laminar_selection_qr", deadline-started) as m:
            r = m.addMVar(len(family), lb=0., ub=bound); x = m.addMVar(len(family), lb=0., ub=bound)
            a = m.addMVar(len(proper), vtype="B")
            positions = {C: i for i, C in enumerate(family)}; selected = {C: i for i, C in enumerate(proper)}
            for C, j in selected.items():
                i = positions[C]
                m.addConstr(r[i] <= bound*a[j]); m.addConstr(x[i] <= bound*a[j])
                if C in fixed:
                    a[j].LB = 1.
            for C, D in combinations(proper, 2):
                if not compatible(C, D):
                    m.addConstr(a[selected[C]]+a[selected[D]] <= 1.)
            for W in [*proper, tuple(range(n))]:
                inside = [selected[C] for C in proper if set(C) < set(W)]
                if inside:
                    m.addConstr(a[inside].sum() <= len(W)-2)
            Fr, Fx = feature_matrices(self.data, family)
            F, y = np.column_stack((Fr, Fx)), self.data.y.ravel()
            Qm, R = np.linalg.qr(F)                      # F = Q R, Q'Q = I
            qy = Qm.T@y
            const = float(y@y-qy@qy)
            scale = self.fitter.residual_scale
            theta = [r, x]
            res = m.addMVar(len(qy), lb=-float("inf"))
            m.addConstr(res == (R[:, :len(family)]@r+R[:, len(family):]@x-qy)/scale)
            N = y.size
            m.setObjective((res@res+const/scale**2)/N+self.config.alpha*a.sum())
            lookup = {tuple(C): (rr, xx) for C, rr, xx in zip(warm["supports"], warm["r"], warm["x"], strict=True)}
            r0 = np.array([lookup.get(C, (0., 0.))[0] for C in family]); x0 = np.array([lookup.get(C, (0., 0.))[1] for C in family])
            r.Start, x.Start = r0, x0
            a.Start = [float(C in lookup) for C in proper]
            res.Start = (R@np.r_[r0, x0]-qy)/scale
            start_residual = ((predict(self.data, warm)-self.data.y)/scale).ravel()
            initial = float(np.mean(start_residual**2))+self.config.alpha*sum(C in lookup for C in proper)

            def decoded():
                idx = [i for i, C in enumerate(family) if C in fixed or a.X[selected[C]] > .5]
                return dict(supports=[list(family[i]) for i in idx],
                            r=np.clip(r.X[idx], 0., bound).tolist(), x=np.clip(x.X[idx], 0., bound).tolist())
            return self._finish(m, started, deadline, scope="finite laminar support pool (QR)",
                                point_factory=decoded, initial_objective=initial)


def one(job):
    case, days = job
    network, labels, hist, v_m, r_m = history(case, 0, days)
    n = len(labels); T = 24*days; tr = slice(0, T*3//4)
    obs = Observation(*(a[tr]-a[tr].mean(0) for a in (hist["P"], hist["Q"], hist["Y"])), labels)
    cfg = SearchConfig(total_seconds=900, per_mip_seconds=300, output_reserve_seconds=120)
    common = algorithm.prepare_common(obs, cfg)
    fitter = QPFitter(obs, reference_mse=common["reference_mse"], bound=2.)
    pool = canonical([*common["guidance"]["pool"], *combinations(range(n), 2)])
    keep = canonical(basis(n)); warm = fitter.fit(basis(n))
    out = dict(case=case, days=days, n=n, pool=len(pool))
    for label, cls in (("original", StructuralSolver), ("qr", CompressedSolver)):
        s = cls(fitter, cfg)
        t = time.perf_counter()
        pt, rec = s.select(pool, keep, warm, deadline=time.perf_counter()+600)
        out[label] = dict(seconds=time.perf_counter()-t, status=rec["status"], objective=rec["objective"],
                          variables=rec["variables"], supports=sorted(tuple(C) for C in pt["supports"] if 1 < len(C) < n))
        s.close()
    out["same_family"] = out["original"]["supports"] == out["qr"]["supports"]
    return out


if __name__ == "__main__":
    jobs = [(c, d) for c in ("eulv_td004454_feeder159", "eulv_td008016_feeder92", "eulv_td008690_feeder19") for d in (8, 18)]
    with Pool(6) as p:
        for o in p.map(one, jobs):
            a, b = o["original"], o["qr"]
            print(f"{o['case'][5:]} n={o['n']} days={o['days']} pool={o['pool']}: original {a['seconds']:.1f}s "
                  f"({a['variables']} vars, status {a['status']}, obj {a['objective']:.8f}) | QR {b['seconds']:.1f}s "
                  f"({b['variables']} vars, status {b['status']}, obj {b['objective']:.8f}) | same family {o['same_family']}")
