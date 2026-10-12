"""Save compact independent AC/Jacobian implementation evidence in staging."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = Path(r"C:\Users\23761\.codex\worktrees\36a1\Topo")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np
import scipy
from research_experiments.mobile_ev_mle.likelihood import ac_predict
from research_experiments.mobile_ev_mle.candidates import RadialCandidate
from research_experiments.mobile_ev_mle.continuous_ac_inverse import inverse_ac, forward_ac, source_logpdf


def main():
    started = perf_counter()
    rng = np.random.default_rng(1310301)
    families = ((), ((0, 1),), ((0, 2),), ((1, 2),))
    rows = []
    count = 37
    c = rng.uniform(-.012, .012, (count, 1))
    p = .12+c+rng.uniform(-.004, .004, (count, 3))
    q = .04+.25*c+rng.uniform(-.004, .004, (count, 3))
    p[np.arange(count), rng.integers(0, 3, count)] += rng.uniform(0, .3, count)
    sp, sq = rng.uniform(.065, .135, count), rng.uniform(.02, .05, count)
    for family in families:
        candidate = RadialCandidate.from_supports("validation", 3, family)
        for case in ("strong", "weak"):
            r0 = np.array([.005 if len(s) == 3 else .028+.003*s[0] if len(s) == 1
                           else .00003 if case == "weak" else .012 for s in candidate.supports])
            x0 = np.array([.0035 if len(s) == 3 else .011+.002*s[0] if len(s) == 1
                           else .00002 if case == "weak" else .007 for s in candidate.supports])
            r, x = r0*rng.uniform(.8, 1.2, len(r0)), x0*rng.uniform(.8, 1.2, len(x0))
            independent = ac_predict(candidate, r, x, p, q, sp, sq)
            v = inverse_ac(candidate, r, x, p, q, independent["master_p"], independent["master_q"])
            f = forward_ac(candidate, r, x, p, q, sp, sq)
            eps = 2e-6
            finite_jac = np.empty((count, 2, 2))
            f_jac = np.empty_like(finite_jac)
            for j in range(2):
                a = inverse_ac(candidate, r, x, p, q, independent["master_p"]+eps*(j == 0),
                               independent["master_q"]+eps*(j == 1))
                b = inverse_ac(candidate, r, x, p, q, independent["master_p"]-eps*(j == 0),
                               independent["master_q"]-eps*(j == 1))
                finite_jac[:, 0, j] = (a["source_p"]-b["source_p"])/(2*eps)
                finite_jac[:, 1, j] = (a["source_q"]-b["source_q"])/(2*eps)
                aa = ac_predict(candidate, r, x, p, q, sp+eps*(j == 0), sq+eps*(j == 1))
                bb = ac_predict(candidate, r, x, p, q, sp-eps*(j == 0), sq-eps*(j == 1))
                f_jac[:, 0, j] = (aa["master_p"]-bb["master_p"])/(2*eps)
                f_jac[:, 1, j] = (aa["master_q"]-bb["master_q"])/(2*eps)
            reciprocal = np.einsum("tij,tjk->tik", v["source_jacobian"], f_jac)-np.eye(2)
            rows.append(dict(family=[list(s) for s in family], case=case, states=count,
                             all_valid=bool(v["valid"].all() and f["valid"].all()),
                             all_high_branch=bool(v["high_branch"].all() and f["high_branch"].all()),
                             all_certified=bool(v["certified_domain"].all()),
                             inverse_voltage_error=float(np.max(abs(v["v"]-independent["v"]))),
                             inverse_source_error=float(max(np.max(abs(v["source_p"]-sp)), np.max(abs(v["source_q"]-sq)))),
                             forward_voltage_error=float(np.max(abs(f["v"]-independent["v"]))),
                             forward_master_error=float(max(np.max(abs(f["master_p"]-independent["master_p"])),
                                                            np.max(abs(f["master_q"]-independent["master_q"])))),
                             jacobian_central_difference_error=float(np.max(abs(v["source_jacobian"]-finite_jac))),
                             jacobian_forward_reciprocal_error=float(np.max(abs(reciprocal))),
                             determinant_min=float(v["determinant"].min()), vmin=float(v["v"].min()),
                             shared_residual_max=float(np.max(abs(v["shared_residual"])))))

    # Midpoint quadrature over a conservative master rectangle for one fixed
    # hypothetical background/action. This is a normalization sanity check;
    # the uniform proof, not a finite grid, certifies properness.
    mass_rows = []
    Pp = np.array([.30, .12, .12])
    Qq = np.full(3, .04)
    for family in ((), ((0, 1),)):
        candidate = RadialCandidate.from_supports("mass_check", 3, family)
        r = np.array([.006 if len(s) == 3 else .0216 if len(s) == 2 else .0408 for s in candidate.supports])
        x = np.array([.0042 if len(s) == 3 else .0108 if len(s) == 2 else .018 for s in candidate.supports])
        for grid in (400, 800):
            Plo, Phi = Pp.sum()+.065, Pp.sum()+.135+.031
            Qlo, Qhi = Qq.sum()+.02, Qq.sum()+.05+.014
            dP, dQ = (Phi-Plo)/grid, (Qhi-Qlo)/grid
            pp, qq = np.meshgrid(Plo+(np.arange(grid)+.5)*dP, Qlo+(np.arange(grid)+.5)*dQ, indexing="ij")
            inv = inverse_ac(candidate, r, x, Pp, Qq, pp, qq)
            density = np.exp(source_logpdf(inv["source_p"], inv["source_q"]))*inv["jacobian_abs"]
            mass = float(np.sum(density)*dP*dQ)
            mass_rows.append(dict(family=[list(s) for s in family], midpoint_grid=grid,
                                  mass=mass, absolute_error=abs(mass-1),
                                  all_valid=bool(inv["valid"].all() and inv["high_branch"].all()),
                                  source_support_captured_by_uniform_forward_loss_bounds=True))
    files = [HERE/name for name in ("continuous_ac_inverse.py", "continuous_ac_inverse_math.md",
                                   "test_continuous_ac_inverse.py", "validate_continuous_ac_inverse.py")]
    files += [PROJECT/"research_experiments"/"mobile_ev_mle"/name for name in
              ("likelihood.py", "simulation.py", "continuous_predictive_eig_protocol.md")]
    hashes = {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    result = dict(seed=1310301, runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  actual_future_or_evaluation_truth_used=False, prior_model="declared nominal synthetic continuous prior",
                  proof_status="uniform high-voltage existence, prior image inclusion and rectangular inverse injectivity proved",
                  proof_source="independent derivation in continuous_ac_inverse_math.md",
                  proof_constants=dict(forward_W_upper=.041, existence_ball_radius=.05, master_P_upper=.867,
                                       master_Q_upper=.204, inverse_junction_tau_lower=.94,
                                       inverse_Gprime_lower=.9991, inverse_terminal_V_lower=.949,
                                       inverse_J_minus_identity_inf_upper=.019),
                  finite_tests_are_global_proof=False, regression_pytest=dict(executed_by_this_driver=False, expected_tests=20, command="python -m pytest research_experiments/mobile_ev_mle/test_continuous_ac_inverse.py -q -p no:cacheprovider"),
                  comparison_records=rows, normalization_sanity=mass_rows, source_sha256=hashes,
                  seconds=perf_counter()-started)
    result["all_comparisons_pass"] = all(row["all_valid"] and row["all_high_branch"] and row["all_certified"]
                                         and row["inverse_voltage_error"] < 2e-12 and row["inverse_source_error"] < 2e-12
                                         and row["forward_master_error"] < 2e-12 and row["jacobian_central_difference_error"] < 2e-9
                                         and row["jacobian_forward_reciprocal_error"] < 2e-9 for row in rows)
    (HERE/"evidence"/"continuous_ac_inverse_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(all_comparisons_pass=result["all_comparisons_pass"], comparisons=len(rows),
                         max_voltage_error=max(row["inverse_voltage_error"] for row in rows),
                         max_source_error=max(row["inverse_source_error"] for row in rows),
                         max_finite_jac_error=max(row["jacobian_central_difference_error"] for row in rows),
                         max_reciprocal_jac_error=max(row["jacobian_forward_reciprocal_error"] for row in rows),
                         normalization_sanity=mass_rows, seconds=result["seconds"])))
    if not result["all_comparisons_pass"]:
        raise SystemExit("independent AC/Jacobian acceptance failed")


if __name__ == "__main__":
    main()
