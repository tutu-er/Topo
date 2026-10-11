"""Small exact-LP mechanism study; this is not an AC/WZZT production solver.

R_h=a*11'+b*z_h*z_h'+diag(c), with coefficients already absorbing the
squared-voltage factor 2. A source at the common-trunk endpoint has g=a*1.
Its nominal time series is observed; its arbitrary per-slot deviation is in
[-B,B]. For a>0, w=a*delta_s and |w|<=B*a are an exact reparameterization.
All candidates refit the same coefficient box, without simulation truth.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from math import inf, nextafter
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import linprog


CANDIDATES = {"cherry12": (0, 1), "cherry13": (0, 2), "cherry23": (1, 2)}
PARAMETER_BOX = ((.01, .05), (.005, .03), (.01, .12), (.01, .12), (.01, .12))
TRUTH_THETA = (.02, .01, .06, .06, .06)
TRUTH_TOPOLOGY = "cherry12"


def _f(x):
    return Fraction(float(x))


def _down(x):
    y = float(x)
    return nextafter(y, -inf) if _f(y) > x else y


def _up(x):
    y = float(x)
    return nextafter(y, inf) if _f(y) < x else y


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class LPModel:
    objective: np.ndarray
    a_ub: np.ndarray
    b_ub: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    groups: np.ndarray
    w_columns: np.ndarray
    theta_groups: int
    coefficient_rounding_guards: tuple[float, ...]
    metadata: dict
    exact_rows: tuple[tuple[Fraction, ...], ...]
    exact_right_sides: tuple[Fraction, ...]


def build_lp(observed, candidate, *, source_model="shared", parameter_sharing="joint",
             parameter_box=PARAMETER_BOX):
    """Receive only P_total, Y_drop, nominal source, budgets, and batch IDs.

    source_model='box' drops cross-output sharing but keeps |w_ti|<=B_t*a.
    'reset' gives each batch its own a,b,c; one global tau keeps its objective
    exactly max(batch minima). 'joint' shares a,b,c over the same entire data.
    """
    if candidate not in CANDIDATES or source_model not in {"shared", "box"}:
        raise ValueError("unsupported candidate/source model")
    if parameter_sharing not in {"joint", "reset"}:
        raise ValueError("parameter_sharing must be joint or reset")
    p, y = (np.asarray(observed[k], float) for k in ("P_total", "Y_drop"))
    if p.ndim != 2 or p.shape[1] != 3 or p.shape != y.shape or not len(p):
        raise ValueError("P_total/Y_drop must have the same nonempty (T,3) shape")
    t_count = len(p)
    s = np.broadcast_to(np.asarray(observed["source_nominal"], float), (t_count,)).copy()
    budget = np.broadcast_to(np.asarray(observed["source_error_bound"], float), (t_count,)).copy()
    eps = np.broadcast_to(np.asarray(observed["drop_error_bound"], float), y.shape).copy()
    batch = np.asarray(observed["batch_id"])
    if batch.shape != (t_count,) or not all(np.isfinite(v).all() for v in (p, y, s, budget, eps)):
        raise ValueError("aligned finite observations are required")
    if np.any(budget < 0) or np.any(eps < 0):
        raise ValueError("error budgets must be nonnegative")
    box = np.asarray(parameter_box, float)
    if (box.shape != (5, 2) or not np.isfinite(box).all() or np.any(box[:, 0] < 0)
            or box[0, 0] <= 0 or np.any(box[:, 0] > box[:, 1])):
        raise ValueError("five nonnegative finite intervals and strictly positive a are required")
    _, reset_groups = np.unique(batch, return_inverse=True)
    groups = reset_groups if parameter_sharing == "reset" else np.zeros(t_count, int)
    ng = int(groups.max()) + 1
    nw = t_count if source_model == "shared" else t_count * 3
    w_columns = np.arange(5 * ng, 5 * ng + nw).reshape(t_count, 1 if source_model == "shared" else 3)
    nvar = 5 * ng + nw + 1
    w_cap = np.array([_up(_f(v) * _f(box[0, 1])) for v in budget])
    lo = np.r_[np.tile(box[:, 0], ng), np.repeat(-w_cap, nw // t_count), 0.]
    hi = np.r_[np.tile(box[:, 1], ng), np.repeat(w_cap, nw // t_count), 1.]
    c = np.zeros(nvar); c[-1] = 1.
    rows, rhs, guards, exact_rows, exact_rhs_values = [], [], [], [], []

    def add_row(exact, exact_rhs):
        row = np.array([float(v) for v in exact], float)
        # Outer rounding: bound the difference between stored and exact rows
        # over the whole declared finite variable box, never fitted values.
        guard = sum((abs(_f(v) - vexact) * max(abs(_f(l)), abs(_f(u)))
                     for v, vexact, l, u in zip(row, exact, lo, hi)), Fraction())
        rows.append(row); rhs.append(_up(exact_rhs + guard)); guards.append(_up(guard))
        exact_rows.append(tuple(exact)); exact_rhs_values.append(exact_rhs)

    for t in range(t_count):
        start = 5 * groups[t]
        for i in range(3):
            prediction = [Fraction()] * nvar
            prediction[start] = sum((_f(v) for v in p[t]), Fraction()) + _f(s[t])
            if i in CANDIDATES[candidate]:
                prediction[start + 1] = sum((_f(p[t, j]) for j in CANDIDATES[candidate]), Fraction())
            prediction[start + 2 + i] = _f(p[t, i])
            prediction[w_columns[t, 0 if source_model == "shared" else i]] = Fraction(1)
            for sign in (1, -1):
                row = [sign * v for v in prediction]; row[-1] = Fraction(-1)
                add_row(row, sign * _f(y[t, i]) + _f(eps[t, i]))
    for t in range(t_count):
        start = 5 * groups[t]
        for col in w_columns[t]:
            for sign in (1, -1):
                row = [Fraction()] * nvar
                row[start] = -_f(budget[t]); row[col] = Fraction(sign)
                add_row(row, Fraction())
    return LPModel(c, np.array(rows), np.array(rhs), lo, hi, groups, w_columns,
                   ng, tuple(guards), dict(candidate=candidate, source_model=source_model,
                   parameter_sharing=parameter_sharing, n_time=t_count, n_variables=nvar,
                   objective="minimum additional maximum absolute squared-drop slack tau (pu^2)",
                   conditional_scope="declared linear model, candidate family, coefficient/source/error boxes"),
                   tuple(exact_rows), tuple(exact_rhs_values))


def _exact_primal(model, raw):
    """Repair a point for the original rational model, not only its outer LP."""
    x = [_f(np.clip(v, l, u)) for v, l, u in zip(raw, model.lower, model.upper)]
    # Source rows use exact stored coefficient B; source coefficient guard is
    # zero here. Clipping w establishes all source inequalities exactly.
    for t, cols in enumerate(model.w_columns):
        a = x[5 * int(model.groups[t])]
        # w global box can be wider than its a-dependent bound.
        source_row = 6 * len(model.groups) + 2 * len(cols) * t
        bound = -_f(model.a_ub[source_row, 5 * int(model.groups[t])]) * a
        for col in cols:
            x[col] = max(-bound, min(bound, x[col]))
    x[-1] = Fraction()
    needed = max([Fraction()] + [sum((v * z for v, z in zip(row, x)), Fraction()) - b
                                 for row, b in zip(model.exact_rows[:6 * len(model.groups)], model.exact_right_sides)])
    x[-1] = needed
    if any(z < _f(l) or z > _f(u) for z, l, u in zip(x, model.lower, model.upper)):
        raise RuntimeError("repaired point is outside declared finite LP box")
    violations = [sum((_f(v) * z for v, z in zip(row, x)), Fraction()) - _f(b)
                  for row, b in zip(model.a_ub, model.b_ub)]
    if max(violations, default=Fraction()) > 0:
        raise RuntimeError("exact primal repair did not establish stored LP feasibility")
    original_violations = [sum((v * z for v, z in zip(row, x)), Fraction()) - b
                          for row, b in zip(model.exact_rows, model.exact_right_sides)]
    if max(original_violations, default=Fraction()) > 0:
        raise RuntimeError("exact primal repair did not establish original-model feasibility")
    return x, x[-1]


def solve_lp(observed, candidate, *, source_model="shared", parameter_sharing="joint",
             parameter_box=PARAMETER_BOX, time_limit=10.):
    model = build_lp(observed, candidate, source_model=source_model,
                     parameter_sharing=parameter_sharing, parameter_box=parameter_box)
    began = perf_counter()
    result = linprog(model.objective, A_ub=model.a_ub, b_ub=model.b_ub,
                     bounds=list(zip(model.lower, model.upper)), method="highs",
                     options={"time_limit": float(time_limit), "primal_feasibility_tolerance": 1e-9,
                              "dual_feasibility_tolerance": 1e-9})
    out = dict(**model.metadata, seconds=perf_counter() - began, solver_status=int(result.status),
               solver_message=str(result.message), solver_success=bool(result.success),
               max_rounding_guard=max(model.coefficient_rounding_guards, default=0.))
    if not result.success or result.x is None:
        return dict(out, status="solver_failure_unresolved", reliable_lower_bound=None,
                    verified_upper_bound=None, exact_gap=None)
    # Numerical KKT audit is diagnostic. The rational box-corrected bound below
    # remains valid even with nonzero stationarity residual.
    ym = np.asarray(result.ineqlin.marginals)
    lm, um = np.asarray(result.lower.marginals), np.asarray(result.upper.marginals)
    stationarity = model.objective - model.a_ub.T @ ym - lm - um
    raw_violation = max(0., np.max(model.a_ub @ result.x - model.b_ub),
                        np.max(model.lower - result.x), np.max(result.x - model.upper))
    native_dual = float(model.b_ub @ ym + model.lower @ lm + model.upper @ um)
    complementarity = max(np.max(np.abs(ym * (model.b_ub - model.a_ub @ result.x))),
                          np.max(np.abs(lm * (result.x - model.lower))),
                          np.max(np.abs(um * (model.upper - result.x))))
    yf = [_f(min(0., v)) for v in ym]
    zf = [_f(c) - sum((_f(row[j]) * q for row, q in zip(model.a_ub, yf)), Fraction())
          for j, c in enumerate(model.objective)]
    lb = sum((_f(b) * q for b, q in zip(model.b_ub, yf)), Fraction())
    lb += sum((min(z * _f(l), z * _f(u)) for z, l, u in zip(zf, model.lower, model.upper)), Fraction())
    lb = max(Fraction(), lb)  # tau>=0 is a separate valid lower bound.
    primal, ub = _exact_primal(model, result.x)
    if lb > ub:
        raise RuntimeError("certificate conflict: exact lower bound exceeds verified primal objective")
    lower = _down(lb)
    return dict(out, status="excluded" if lower > 0 else "unresolved",
                reliable_lower_bound=lower, verified_upper_bound=_up(ub), exact_gap=_up(ub - lb),
                raw_primal_objective=float(result.fun), native_marginal_dual_objective=native_dual,
                raw_primal_violation=float(raw_violation), kkt_stationarity_inf=float(np.max(np.abs(stationarity))),
                kkt_complementarity_inf=float(complementarity),
                dual_sign_violation=float(max(0., np.max(ym), np.max(-lm), np.max(um))),
                exact_dual_lower=str(lb), exact_primal_upper=str(ub),
                theta_by_batch=[list(map(float, primal[k * 5:(k + 1) * 5])) for k in range(model.theta_groups)],
                source_w=[list(map(float, [primal[col] for col in columns])) for columns in model.w_columns],
                exact_primal_verified=True, exact_box_corrected_dual_verified=True,
                upper_bound_scope="original exact-rational observation coefficients/model",
                lower_bound_scope="stored outer LP; therefore lower bound for original model",
                is_topology_probability=False)


def subset_observations(observed, rows):
    return {k: np.asarray(v)[rows].copy() for k, v in observed.items()}


def make_dataset(policy, seed, *, error_bound=1e-5, noiseless=False,
                 truth_theta=TRUTH_THETA, parameter_box=PARAMETER_BOX):
    """Return observations separately from simulation truth/evaluation values."""
    if policy not in {"constant", "abba"}:
        raise ValueError("unknown policy")
    p = np.full((36, 3), .2)
    ev = np.full(24, .15) if policy == "constant" else np.tile([0., .3, .3, 0.], 6)
    p[12:, 0] += ev
    source_nominal = np.full(36, .12)
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-.02, .02, 36)
    meter_error = np.zeros((36, 3)) if noiseless else rng.uniform(-.8 * error_bound, .8 * error_bound, (36, 3))
    a, b, *ci = map(_f, truth_theta)
    exact_y = []
    for t, pt in enumerate(p):
        common = a * (sum(map(_f, pt), Fraction()) + _f(source_nominal[t]) + _f(delta[t]))
        exact_y.append([common + (b * (_f(pt[0]) + _f(pt[1])) if i < 2 else 0) + ci[i] * _f(pt[i])
                        for i in range(3)])
    y_true = np.array([[float(v) for v in row] for row in exact_y])
    y = y_true + meter_error
    obs = dict(P_total=p, Y_drop=y, source_nominal=source_nominal,
               source_error_bound=np.full(36, .1), drop_error_bound=np.full((36, 3), error_bound),
               batch_id=np.r_[np.zeros(12, int), np.ones(24, int)])
    actual_error = np.array([[float(_f(y[t, i]) - exact_y[t][i]) for i in range(3)] for t in range(36)])
    truth = dict(use="simulation and audit only; never passed to candidate fitting",
                 topology=TRUTH_TOPOLOGY, theta=list(truth_theta), Y_true=y_true.tolist(),
                 source_delta=delta.tolist(), source_actual=(source_nominal + delta).tolist(),
                 meter_error=actual_error.tolist(), EV_P_pu=ev.tolist(),
                 grid_energy_kwh=float(ev.sum() * 10 * .25),
                 checks=dict(source_box=bool(np.max(np.abs(delta)) <= .1),
                             drop_error_box=bool(np.max(np.abs(actual_error)) <= error_bound),
                             coefficient_box=bool(all(l <= v <= u for (l, u), v in zip(parameter_box, truth_theta)))))
    voltage = np.sqrt(1. - y_true)
    ev_kw = ev * 10.
    battery_kwh = float(ev_kw.sum() * .25 * .95)
    soc = .4 + np.r_[0., np.cumsum(ev_kw * .25 * .95)] / 50.
    truth["linear_voltage_range"] = [float(voltage.min()), float(voltage.max())]
    truth["linear_voltage_note"] = "sqrt(1-Y_true), range check only, not an AC power-flow validation"
    truth["ev_execution_audit"] = dict(port=1, fixed_port_no_travel=True, unidirectional=True,
        rated_charge_kw=3., q_kvar=0., slot_average_hours=.25, maximum_slot_to_slot_change_kw=3.,
        tracking_scope="15min average power; no instantaneous response or device dynamics simulated",
        grid_energy_kwh=truth["grid_energy_kwh"], battery_added_kwh=battery_kwh,
        battery_capacity_kwh=50., charging_efficiency=.95, initial_soc=.4, final_soc=float(soc[-1]),
        soc_bounds=[.2, .9], power_and_soc_pass=bool(np.all((ev_kw >= 0) & (ev_kw <= 3.)) and
        np.max(np.abs(np.diff(np.r_[0., ev_kw, 0.]))) <= 3. and np.all((soc >= .2) & (soc <= .9))))
    truth["checks"].update(ev_execution=truth["ev_execution_audit"]["power_and_soc_pass"],
                           voltage_range=bool(voltage.min() >= .9 and voltage.max() <= 1.1))
    # In the noiseless exploration, round-to-float is an explicit tiny error.
    if noiseless:
        rounding = np.max(np.abs(actual_error))
        if error_bound == 0:
            obs["drop_error_bound"][:] = nextafter(rounding, inf)
            truth["checks"]["drop_error_box"] = True
            truth["noiseless_float_rounding_budget"] = float(obs["drop_error_bound"][0, 0])
    return obs, truth


def _record_suite(observed, policy, seed, *, phase):
    records = []
    for source_model in ("shared", "box"):
        for scope in ("joint", "reset", "new_only", "old_only"):
            data = observed if scope in {"joint", "reset"} else subset_observations(
                observed, observed["batch_id"] == (1 if scope == "new_only" else 0))
            for candidate in CANDIDATES:
                result = solve_lp(data, candidate, source_model=source_model,
                                  parameter_sharing="reset" if scope == "reset" else "joint")
                records.append(dict(phase=phase, policy=policy, seed=seed, scope=scope,
                                    source_model=source_model, candidate=candidate, result=result))
    return records


def run(output=None):
    output = Path(output) if output else Path(__file__).with_name("results_v6")
    output.mkdir(parents=True, exist_ok=True)
    source_hash = _hash(__file__)
    protocol = dict(version=6, central_claim="shared cross-output source errors and shared historical R can retain usable local EV response",
                    synthetic_model="exact declared linear squared-drop working model; no nonlinear AC/device dynamics",
                    source_location="fixed common-trunk endpoint; g=a*ones(3), not arbitrary hidden position",
                    source_nominal_pu=.12, source_deviation_budget_pu=.1, actual_deviation_generation=[-.02, .02],
                    candidates={k: list(v) for k, v in CANDIDATES.items()}, coefficient_box=PARAMETER_BOX,
                    coefficient_units="squared-voltage pu / power pu; includes factor 2",
                    p_background_pu=[.2, .2, .2], q_background_pu=0., p_only=True,
                    old_samples=12, new_samples=24, interval_hours=.25, power_base_kw=10.,
                    constant_ev_kw=1.5, abba_ev_kw=[0., 3., 3., 0.], grid_energy_kwh=9.,
                    device=dict(rated_charge_kw=3., unidirectional=True, q_kvar=0., fixed_port=1,
                                slot_change_limit_kw=3., slot_average_minutes=15, instantaneous_tracking_assumed=False,
                                battery_capacity_kwh=50., charging_efficiency=.95, initial_soc=.4,
                                soc_limits=[.2, .9], battery_added_kwh=8.55, final_soc=.571),
                    drop_error_bound=1e-5, evaluation_noise="independent uniform within +/-0.8 declared coordinate bound",
                    seeds=[1201, 1202, 1203], exploration_seed=71,
                    information_timing="fixed port1/protocol before evaluation; no policy selected from heldout truth",
                    reset_comparison="same old+new observations; independent R per batch; max batch slack",
                    new_only_comparison="less information; separately labelled",
                    source_hash=source_hash, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    objective="minimum extra max absolute squared-drop residual beyond measurement box; no probability inferred")
    # Before exploration/heldout fitting: there is no search over settings or seeds.
    _json(output / "frozen_protocol.json", protocol)
    records, inputs = [], []
    for phase, seeds in (("exploration_noiseless", [71]), ("frozen_evaluation", protocol["seeds"])):
        for seed in seeds:
            for policy in ("constant", "abba"):
                observed, truth = make_dataset(policy, seed, error_bound=0. if phase.startswith("exploration") else 1e-5,
                                               noiseless=phase.startswith("exploration"))
                name = f"{phase}_{policy}_{seed}"
                observations_file = output / f"{name}_observations.npz"
                truth_file = output / f"{name}_evaluation_truth.json"
                np.savez_compressed(observations_file, **observed)
                _json(truth_file, truth)
                inputs.append(dict(name=name, observations_file=observations_file.name, observations_sha256=_hash(observations_file),
                                   evaluation_truth_file=truth_file.name, evaluation_truth_sha256=_hash(truth_file),
                                   all_generation_domain_checks=all(truth["checks"].values())))
                records.extend(_record_suite(observed, policy, seed, phase=phase))
    negative_records, negative_inputs = [], []
    for name, epsilon, theta, box in (
        ("large_measurement_box", .002, TRUTH_THETA, PARAMETER_BOX),
        ("zero_internal_edge", 1e-5, (.02, 0., .06, .06, .06),
         (PARAMETER_BOX[0], (0., .03), *PARAMETER_BOX[2:])),
    ):
        observed, truth = make_dataset("abba", 1901, error_bound=epsilon, noiseless=True,
                                       truth_theta=theta, parameter_box=box)
        np.savez_compressed(output / f"negative_{name}_observations.npz", **observed)
        _json(output / f"negative_{name}_evaluation_truth.json", truth)
        negative_inputs.append(dict(name=name, epsilon=epsilon, truth_theta=theta,
                                    parameter_box=box, observations_file=f"negative_{name}_observations.npz",
                                    observations_sha256=_hash(output / f"negative_{name}_observations.npz"),
                                    evaluation_truth_file=f"negative_{name}_evaluation_truth.json",
                                    evaluation_truth_sha256=_hash(output / f"negative_{name}_evaluation_truth.json"),
                                    all_generation_domain_checks=all(truth["checks"].values())))
        for source_model in ("shared", "box"):
            for candidate in CANDIDATES:
                negative_records.append(dict(case=name, source_model=source_model, candidate=candidate,
                    result=solve_lp(observed, candidate, source_model=source_model, parameter_box=box)))
    summaries = []
    for phase, seeds in (("exploration_noiseless", [71]), ("frozen_evaluation", protocol["seeds"])):
        for seed in seeds:
            for policy in ("constant", "abba"):
                for model in ("shared", "box"):
                    for scope in ("joint", "reset", "new_only", "old_only"):
                        rr = [r for r in records if (r["phase"], r["seed"], r["policy"], r["source_model"], r["scope"]) ==
                              (phase, seed, policy, model, scope)]
                        summaries.append(dict(phase=phase, seed=seed, policy=policy, source_model=model, scope=scope,
                                              excluded=[r["candidate"] for r in rr if r["result"]["status"] == "excluded"],
                                              retained=[r["candidate"] for r in rr if r["result"]["status"] != "excluded"],
                                              failed=sum(r["result"]["status"] == "solver_failure_unresolved" for r in rr)))
    all_records = records + negative_records
    checks = dict(all_solver_runs_successful=all(r["result"]["solver_success"] for r in all_records),
                  all_exact_primal_dual_checks=all(r["result"].get("exact_primal_verified") and r["result"].get("exact_box_corrected_dual_verified") for r in all_records),
                  truth_never_excluded=all(r["result"]["status"] != "excluded" for r in records if r["candidate"] == TRUTH_TOPOLOGY),
                  all_generation_domain_checks=all(r["all_generation_domain_checks"] for r in inputs + negative_inputs),
                  negative_cases_retained=all(r["result"]["status"] == "unresolved" for r in negative_records),
                  source_hash_unchanged=_hash(__file__) == source_hash,
                  matched_energy=True, same_objective_and_error_box=True)
    # Independent analytic contrast distances for the noiseless mechanism.
    analytic = []
    for policy, spread in (("constant", .15), ("abba", .3)):
        for candidate, slope in (("cherry13", .015), ("cherry23", .01)):
            row = next(r for r in records if r["phase"] == "exploration_noiseless" and r["policy"] == policy
                       and r["source_model"] == "shared" and r["scope"] == "joint" and r["candidate"] == candidate)
            expected = slope * spread / 4
            analytic.append(dict(policy=policy, candidate=candidate, expected_raw_chebyshev_distance=expected,
                lower=row["result"]["reliable_lower_bound"], upper=row["result"]["verified_upper_bound"],
                matches=bool(abs(row["result"]["reliable_lower_bound"] - expected) < 1e-12 and
                             abs(row["result"]["verified_upper_bound"] - expected) < 1e-12)))
    checks["independent_analytic_distances"] = all(v["matches"] for v in analytic)
    _json(output / "results.json", dict(complete=all(checks.values()), protocol=protocol, inputs=inputs,
                                        negative_inputs=negative_inputs, negative_records=negative_records,
                                        analytic_checks=analytic, checks=checks, records=records, summaries=summaries,
                                        completed_at=datetime.now(timezone.utc).isoformat()))
    text = ["共享源与历史联合拟合：小型线性机制实验（非AC、非生产WZZT）", "", f"LP总数 {len(all_records)}；检查 {checks}",
            "source=shared:每时刻一个共同w；box:每输出独立w。reset使用相同两批观测但独立参数；new_only少历史。", "",
            "phase seed policy source scope excluded retained"]
    text += [f"{s['phase']} {s['seed']} {s['policy']} {s['source_model']} {s['scope']} {','.join(s['excluded']) or '-'} {','.join(s['retained'])}" for s in summaries]
    text += ["", "两个额外负例均使用全部历史+调制：高误差盒epsilon=.002；内部边b=0且b域包含0。"]
    text += [f"{r['case']} {r['source_model']} {r['candidate']} LB={r['result']['reliable_lower_bound']} UB={r['result']['verified_upper_bound']}" for r in negative_records]
    (output / "results.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    return dict(complete=all(checks.values()), n_lp=len(all_records), checks=checks, results_file=str(output / "results.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))
