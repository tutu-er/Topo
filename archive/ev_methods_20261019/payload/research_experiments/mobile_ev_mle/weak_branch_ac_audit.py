"""Small physical AC audit of the rational weak-branch construction.

Only fixed samples/schedules are evaluated. No controller, optimizer, fit,
calibration, or route enumeration is run. Project code is imported read-only
with bytecode disabled; the explicit workspace is recorded and hashed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
import numpy as np
import scipy

from research_experiments.mobile_ev_mle.weak_branch_information import exact_certificate, serialise_rationals

DEFAULT_WORKSPACE = Path(r"C:\Users\23761\.codex\worktrees\36a1\Topo")


def project_imports(workspace):
    workspace = Path(workspace).resolve()
    for directory in (workspace, workspace / "rnj_wzzt_core",
                      workspace / "research_experiments" / "mobile_ev_probing"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from ac_general_v9 import RadialCandidate
    from research_experiments.mobile_ev_mle.likelihood import ac_predict
    from research_experiments.mobile_ev_mle.simulation import model
    return RadialCandidate, ac_predict, model


def paired_networks(pair, r_scale, x_scale, candidate_type):
    cherry = candidate_type.from_supports("weak_cherry", 3, (tuple(pair),))
    star = candidate_type.from_supports("matched_star", 3, ())
    r = np.array([.005 if len(s) == 3 else .028 + .003 * s[0]
                  if len(s) == 1 else .00003 for s in cherry.supports]) * r_scale
    x = np.array([.0035 if len(s) == 3 else .011 + .002 * s[0]
                  if len(s) == 1 else .00002 for s in cherry.supports]) * x_scale
    shared = cherry.supports.index(tuple(pair))
    rs, xs = np.empty(star.m), np.empty(star.m)
    for e, support in enumerate(star.supports):
        old = cherry.supports.index(support)
        rs[e], xs[e] = r[old], x[old]
        if len(support) == 1 and support[0] in pair:
            rs[e] += 2 * r[shared]
            xs[e] += 2 * x[shared]
    return cherry, star, r, x, rs, xs, r[shared] + 1j * x[shared]


def reduced_indices(candidate):
    return [0, *candidate.terminal_edges]


def reduced_ac(zbus, loads):
    """Independent small fixed point on source1 plus three loaded terminals."""
    voltage = np.ones_like(loads)
    for iteration in range(1, 2001):
        new = 1 - np.conj(loads / voltage) @ zbus.T
        if np.max(abs(new - voltage)) < 5e-15:
            voltage = new
            break
        voltage = new
    else:
        raise AssertionError("independent reduced AC iteration failed")
    residual = float(np.max(abs(voltage - 1 + np.conj(loads / voltage) @ zbus.T)))
    return voltage, residual, iteration


def run_audit(workspace=DEFAULT_WORKSPACE):
    candidate_type, ac_predict, model = project_imports(workspace)
    c, checks, margins = exact_certificate()
    vmin, a, b = (float(c[key]) for key in ("vmin", "a", "b"))
    meter = model(3)
    if not (meter.v_sigma == float(c["sigma_v"])
            and meter.master_p_sigma == meter.master_q_sigma == float(c["sigma_master"])):
        raise AssertionError("project Gaussian voltage/master noise differs from proof constants")
    rng = np.random.default_rng(2026101301)
    records = []
    for pair in ((0, 1), (0, 2), (1, 2)):
        nominal = candidate_type.from_supports("template", 3, (pair,))
        patterns = [(.8 * np.ones(nominal.m), .8 * np.ones(nominal.m)),
                    (1.2 * np.ones(nominal.m), 1.2 * np.ones(nominal.m)),
                    (np.where(np.arange(nominal.m) % 2, .8, 1.2),
                     np.where(np.arange(nominal.m) % 2, 1.2, .8)),
                    (np.where(np.arange(nominal.m) % 2, 1.2, .8),
                     np.where(np.arange(nominal.m) % 2, .8, 1.2)),
                    (rng.uniform(.8, 1.2, nominal.m), rng.uniform(.8, 1.2, nominal.m)),
                    (rng.uniform(.8, 1.2, nominal.m), rng.uniform(.8, 1.2, nominal.m))]
        for pattern, (rscale, xscale) in enumerate(patterns):
            cherry, star, r, x, rs, xs, zs = paired_networks(pair, rscale, xscale, candidate_type)
            common = rng.uniform(-.012, .012, (20, 1))
            p0 = .12 + common + rng.uniform(-.004, .004, (20, 3))
            q0 = .04 + .25 * common + rng.uniform(-.004, .004, (20, 3))
            # Include both common-load endpoints and the largest pair contrast.
            for t, sign in ((0, -1), (1, 1), (12, 1), (19, -1)):
                offsets = np.full(3, sign * .004)
                offsets[pair[0]], offsets[pair[1]] = .004, -.004
                p0[t] = .12 + sign * .012 + offsets
                q0[t] = .04 + sign * .003 + offsets
            sp, sq = .10 + rng.uniform(-.035, .035, 20), .035 + rng.uniform(-.015, .015, 20)
            sp[[0, 12]], sq[[0, 12]] = [.065, .135], [.02, .05]
            for schedule in ("none", "constant", "pulse", "changing_sites"):
                u = np.zeros((20, 3))
                if schedule == "constant":
                    u[12:, pair[0]] = .15
                elif schedule == "pulse":
                    u[12:16, pair[1]] = .3
                elif schedule == "changing_sites":
                    u[12 + np.arange(8), np.arange(8) % 3] = .15
                power = u.sum(axis=1)
                p, q = p0 + u, q0
                ch = ac_predict(cherry, r, x, p, q, sp, sq, root_v=1.)
                st = ac_predict(star, rs, xs, p, q, sp, sq, root_v=1.)
                ich, ist = reduced_indices(cherry), reduced_indices(star)
                zch, zst = ch["zbus"][np.ix_(ich, ich)], st["zbus"][np.ix_(ist, ist)]
                direction = np.zeros(4); direction[1 + pair[0]], direction[1 + pair[1]] = 1, -1
                identity_error = float(np.max(abs(zst - zch - zs * np.outer(direction, direction))))
                loads = np.c_[sp + 1j * sq, p + 1j * q]
                vr_ch, residual_ch, _ = reduced_ac(zch, loads)
                vr_st, residual_st, _ = reduced_ac(zst, loads)
                reduced_error = max(float(np.max(abs(vr_ch - ch["voltage"][:, ich]))),
                                    float(np.max(abs(vr_st - st["voltage"][:, ist]))))
                delta = np.max(abs(vr_st - vr_ch), axis=1)
                bound = a + b * power
                current_ch = np.conj(loads / vr_ch)
                lactual = abs(current_ch[:, 1 + pair[0]] - current_ch[:, 1 + pair[1]])
                lbound = float(c["L0"]) + float(c["L1"]) * power
                master_ch = ch["master_p"] + 1j * ch["master_q"]
                master_st = st["master_p"] + 1j * st["master_q"]
                delta_master = abs(master_st - master_ch)
                master_bound = float(c["master_gain"]) * delta
                actual_kl = .5 * float(np.sum((st["v"] - ch["v"])**2 / float(c["sigma_v"])**2)
                                        + np.sum(delta_master**2 / float(c["sigma_master"])**2))
                pointwise_kl_bound = .5 * float(c["gaussian_weight"]) * float(np.sum(bound**2))
                c_actual = max(float(np.max(abs(zch) @ abs(loads).T)),
                               float(np.max(abs(zst) @ abs(loads).T)))
                assert identity_error < 5e-17
                assert reduced_error < 5e-13 and max(residual_ch, residual_st) < 5e-14
                assert min(abs(ch["voltage"]).min(), abs(st["voltage"]).min()) >= vmin
                assert c_actual <= float(c["C"]) + 1e-14
                assert np.all(lactual <= lbound + 1e-12)
                assert np.all(delta <= bound + 1e-12)
                assert np.all(delta_master <= master_bound + 1e-12)
                assert actual_kl <= pointwise_kl_bound + 1e-11
                assert pointwise_kl_bound <= float(c["kl_exact"]) + 1e-12
                records.append(dict(pair=list(pair), impedance_pattern=pattern, schedule=schedule,
                    samples=20, sum_u=float(power.sum()), sum_u2=float(power @ power),
                    matrix_identity_error=identity_error, reduced_full_voltage_error=reduced_error,
                    min_full_voltage=float(min(abs(ch["voltage"]).min(), abs(st["voltage"]).min())),
                    max_c_actual=c_actual, max_voltage_difference=float(delta.max()),
                    max_voltage_bound_ratio=float(np.max(delta / bound)),
                    max_current_contrast_ratio=float(np.max(lactual / lbound)),
                    max_master_difference=float(delta_master.max()), actual_gaussian_kl=actual_kl,
                    schedule_gaussian_kl_bound=pointwise_kl_bound,
                    ac_equation_residual=max(ch["ac_equation_residual"], st["ac_equation_residual"]),
                    power_balance_residual=max(ch["power_balance_residual"], st["power_balance_residual"])))
    names = ("research_experiments/mobile_ev_mle/simulation.py",
             "research_experiments/mobile_ev_mle/adaptive_milp_study.py",
             "research_experiments/mobile_ev_mle/likelihood.py",
             "research_experiments/mobile_ev_mle/ev_operations_milp.py",
             "research_experiments/mobile_ev_probing/ac_general_v9.py")
    workspace = Path(workspace).resolve()
    return dict(status="passed", kind="conditional_full_AC_weak_pair_audit",
        scope="72 fixed parameter/schedule paths; no fits, controllers, calibration, or route enumeration",
        control_relaxation="changing_sites satisfies only pointwise power and grid energy; no route feasibility claim",
        source_workspace=str(workspace), random_seed=2026101301,
        source_sha256={name:hashlib.sha256((workspace / name).read_bytes()).hexdigest() for name in names},
        audit_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ("weak_branch_information.py", "weak_branch_ac_audit.py")},
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        rational_constants=serialise_rationals(c), squared_nominal_checks=serialise_rationals(checks),
        positive_proof_margins=serialise_rationals(margins), path_count=len(records),
        time_sample_count=sum(row["samples"] for row in records),
        max_actual_gaussian_kl=max(row["actual_gaussian_kl"] for row in records),
        max_voltage_bound_ratio=max(row["max_voltage_bound_ratio"] for row in records),
        max_reduced_full_voltage_error=max(row["reduced_full_voltage_error"] for row in records),
        max_power_balance_residual=max(row["power_balance_residual"] for row in records), records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("weak_branch_information_validation.json"))
    args = parser.parse_args()
    result = run_audit(args.workspace)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("status", "path_count", "time_sample_count",
                       "max_actual_gaussian_kl", "max_voltage_bound_ratio",
                       "max_reduced_full_voltage_error", "max_power_balance_residual")}, ensure_ascii=False))

