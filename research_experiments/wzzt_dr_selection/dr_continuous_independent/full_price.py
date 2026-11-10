"""Full resource-by-hour conditional-probability pricing for frozen paper15.

All currently available resource/hour requests are decision variables. The old
contract is a warm start only. The independent-voltage probability is evaluated
without sampling; full stochastic execution and AC are separate evaluations.
Run directories are private because the saved plans contain resource schedules.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import platform
import sys
import time
import traceback
import zipfile

for option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[option] = "1"

import numpy as np
import scipy
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as bounded
from continuous import IndependentResiduals
from full_adjoint import probability_with_adjoint
from gauss_frequency import gauss_probability


TARGET = .90
FEASIBILITY_TOLERANCE = 1e-7
SEARCH_OPTIONS = dict(cutoff=512., order=32, refinements=0, tolerance=1e-7, chunk=64)


def write_pickle(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(pickle.dumps(value, protocol=4))
    temporary.replace(path)


class FullPriceProblem:
    """Compose existing device derivatives with scalar residual adjoints."""

    def __init__(self, problem):
        self.data = problem
        self.device = problem["device"]
        self.shape = (self.device.R, 24)
        self.base = {
            k: problem["c"] + problem["mean_power"] @ (problem["h0"] - problem["matrices"][k]).T
            for k in ("C", "F")
        }
        hourly_capacity = self.device.scale.reshape(self.device.R, 24, 4).max(axis=2)
        self.active = np.flatnonzero(hourly_capacity.ravel() > 0)
        self.lower = np.full(self.active.size, -1.)
        self.upper = np.where(self.device.kind[self.active // 24] == 1, 0., 1.)
        self.bounds = list(zip(self.lower, self.upper))
        self.old = np.clip(np.asarray(problem["frozen_x"]).ravel()[self.active], self.lower, self.upper)
        self.zero = np.zeros_like(self.old)
        old_point = self.nominal(self.old)
        old_model = IndependentResiduals(old_point["a_C"], old_point["a_F"], problem["sigma"].ravel())
        self.normalization = (old_model.sum_mean, old_model.sum_sd)
        self.last = None
        self.last_point = None
        self.evaluations = 0

    def expand(self, values):
        values = np.asarray(values, float)
        if values.shape != self.active.shape or not np.isfinite(values).all():
            raise ValueError("Finite active-hour request vector required")
        full = np.zeros(self.device.R * 24)
        full[self.active] = values
        return full.reshape(self.shape)

    def nominal(self, values):
        x = self.expand(values)
        u, cache = self.device.nominal(x, cache=True)
        coefficients = {
            k: (self.base[k] + np.einsum("ir,rt->ti", self.data["packet"]["effect_" + k], u)).ravel()
            for k in ("C", "F")
        }
        fee, fee_gradient = self.device.fees(u, gradient=True)
        return dict(x=x, mean_u=u, cache=cache, fee=fee, fee_gradient=fee_gradient,
                    a_C=coefficients["C"], a_F=coefficients["F"])

    def evaluate(self, values):
        values = np.asarray(values, float)
        if self.last_point is not None and np.array_equal(values, self.last_point):
            return self.last
        began = time.perf_counter()
        nominal = self.nominal(values)
        model = IndependentResiduals(nominal["a_C"], nominal["a_F"], self.data["sigma"].ravel())
        probability = probability_with_adjoint(
            model, self.data["packet"]["threshold"], normalization=self.normalization,
            **SEARCH_OPTIONS,
        )
        coarse = np.asarray(probability["gradient_coarse"]).reshape(96, 15)
        fine = np.asarray(probability["gradient_fine"]).reshape(96, 15)
        gu = (coarse @ np.asarray(self.data["packet"]["effect_C"])
              + fine @ np.asarray(self.data["packet"]["effect_F"])).T
        probability_gradient = self.device.hourly_vjp(gu, nominal["cache"]).ravel()[self.active]
        fee_gradient = self.device.hourly_vjp(nominal["fee_gradient"], nominal["cache"]).ravel()[self.active]
        p = float(probability["probability"])
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise FloatingPointError("Probability outside the unit interval")
        if not np.isfinite(probability_gradient).all() or not np.isfinite(fee_gradient).all():
            raise FloatingPointError("Nonfinite full-variable gradient")
        self.evaluations += 1
        self.last_point = values.copy()
        self.last = dict(x=nominal["x"], mean_u=nominal["mean_u"], values=values.copy(),
                         probability=p, fee=float(nominal["fee"]), gradient=probability_gradient,
                         fee_gradient=fee_gradient, seconds=time.perf_counter()-began,
                         moments=probability.get("moments", model.moments()),
                         analytic_tail_bound=probability.get("analytic_tail_bound"),
                         nominal_audit=self.device.audit(nominal["mean_u"]))
        if not self.last["nominal_audit"]["passed"]:
            raise ArithmeticError("Existing device projection failed its resource audit")
        return self.last


def directional_check(problem):
    """Check a full-dimensional interior direction against independent values."""
    indices = np.arange(problem.active.size)
    point = np.clip(.8 * problem.old + .002 * np.sin(indices + 1),
                    problem.lower + .001, problem.upper - .001)
    direction = np.cos(indices + 1)
    direction /= np.linalg.norm(direction)
    analytic = problem.evaluate(point)
    expected = float(analytic["gradient"] @ direction)
    expected_fee = float(analytic["fee_gradient"] @ direction)
    base_cache = problem.nominal(point)["cache"]
    rows = []
    for step in (1e-4, 1e-5, 1e-6):
        left, right = point - step * direction, point + step * direction
        low, high = problem.evaluate(left), problem.evaluate(right)
        estimate = (high["probability"] - low["probability"]) / (2 * step)
        fee_estimate = (high["fee"] - low["fee"]) / (2 * step)
        changed = {}
        for label, trial in (("left", left), ("right", right)):
            cache = problem.nominal(trial)["cache"]
            changed[label] = int(sum(np.count_nonzero(a != b) for a, b in zip(base_cache[:2], cache[:2])))
        rows.append(dict(step=step, finite_difference=estimate, analytic=expected,
                         absolute_difference=abs(estimate-expected), fee_finite_difference=fee_estimate,
                         fee_analytic=expected_fee, fee_absolute_difference=abs(fee_estimate-expected_fee),
                         changed_device_faces=changed))
    passed = (min(row["absolute_difference"] for row in rows[-2:]) < 1e-4
              and min(row["fee_absolute_difference"] for row in rows[-2:]) < 1e-4)
    return dict(passed=bool(passed), point_scope="deterministic interior full-dimensional direction",
                probability=analytic["probability"], rows=rows,
                coefficient_gradient_scope="analytic CF adjoints plus existing device VJP")


def optimize(problem, out, initial=None, *, maxiter=60, seconds=300.,
             restore_evaluations=10, restore_seconds=30.):
    """Minimize the original fee, then restore the same chance constraint.

    One trace and checkpoint cover both stages. SLSQP may start infeasibly.
    Its objective and derivative use the same positive fixed cost scale.
    Every returned point is selected by actual probability and original fee.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records, points, statuses = [], [], []
    stage, deadline = "starts", float("inf")

    def observe(values, label):
        if time.perf_counter() > deadline:
            raise TimeoutError(stage + " wall limit")
        q = problem.evaluate(values)
        if not points or not np.array_equal(q["values"], points[-1]["values"]):
            row = {key:q[key] for key in
                   ("probability", "fee", "seconds", "moments", "analytic_tail_bound")}
            row.update(index=len(records), stage=stage, label=label,
                       elapsed=time.perf_counter()-started)
            records.append(row)
            points.append(q)
            bounded.write(out/"optimization_trace.json", records)
            write_pickle(out/"checkpoint_private.pkl", dict(records=records, points=points))
        return q

    zero = observe(problem.zero, "zero")
    old = observe(problem.old, "old_warm_start")
    if initial is not None:
        initial = np.asarray(initial, float)
        if (initial.shape != problem.old.shape or not np.isfinite(initial).all()
                or np.any(initial < problem.lower) or np.any(initial > problem.upper)):
            raise ValueError("Warm request must satisfy the original active-hour bounds")
        warm = observe(initial, "explicit_warm_start")
    else:
        warm = old
    cost_scale = old["fee"] if old["fee"] > 0 else 1.
    if not np.isfinite(cost_scale):
        raise ValueError("Fixed cost scale must be finite and positive")

    def feasible():
        return [q for q in points if q["probability"] >= TARGET]

    qualified = feasible()
    # Original fees are nonnegative, so an already feasible zero fee needs no
    # numerical search. Otherwise the explicit warm start is honored.
    if not any(q["fee"] == 0 for q in qualified):
        start = min(qualified, key=lambda q:q["fee"]) if qualified else warm
        stage = "cost"
        began = time.perf_counter()
        deadline = began + seconds
        iterations = None
        try:
            fit = minimize(
                lambda z:observe(z, "objective")["fee"] / cost_scale,
                start["values"],
                jac=lambda z:observe(z, "objective")["fee_gradient"] / cost_scale,
                constraints=[dict(
                    type="ineq",
                    fun=lambda z:observe(z, "constraint")["probability"]-TARGET,
                    jac=lambda z:observe(z, "constraint")["gradient"])],
                bounds=problem.bounds, method="SLSQP",
                options=dict(maxiter=maxiter, ftol=1e-8))
            iterations = int(fit.nit)
            observe(fit.x, "solver_return")
            status, success = str(fit.message), bool(fit.success)
        except Exception as exc:
            status, success = repr(exc), False
            bounded.write(out/"cost_exception.json",
                          dict(error=status, traceback=traceback.format_exc()))
        statuses.append(dict(stage=stage, status=status, solver_success=success,
                             iterations=iterations, seconds=time.perf_counter()-began))

    if not feasible() and restore_evaluations > 0:
        stage = "restore"
        began = time.perf_counter()
        deadline = began + restore_seconds
        current = max(points, key=lambda q:q["probability"])
        scale, accepted, count = 1., 0, 0
        status = "maximum_evaluations"
        try:
            while count < restore_evaluations:
                if current["probability"] >= TARGET:
                    status = "target_reached"
                    break
                values = current["values"]
                direction = current["gradient"].copy()
                direction[(values <= problem.lower+1e-8) & (direction < 0)] = 0.
                direction[(values >= problem.upper-1e-8) & (direction > 0)] = 0.
                norm_squared = float(direction@direction)
                if norm_squared <= 1e-24:
                    status = "zero_projected_gradient"
                    break
                alpha = scale*(1.2*(TARGET-current["probability"])+1e-6)/norm_squared
                trial = np.clip(values+alpha*direction, problem.lower, problem.upper)
                if np.array_equal(trial, values):
                    status = "zero_projected_step"
                    break
                candidate = observe(trial, "probability_gradient_trial")
                count += 1
                if candidate["probability"] > current["probability"]+1e-12:
                    current, scale = candidate, 1.
                    accepted += 1
                else:
                    scale *= .5
            if current["probability"] >= TARGET:
                status = "target_reached"
        except Exception as exc:
            status = repr(exc)
            bounded.write(out/"restore_exception.json",
                          dict(error=status, traceback=traceback.format_exc()))
        statuses.append(dict(stage=stage, status=status, accepted_steps=accepted,
                             trial_evaluations=count, seconds=time.perf_counter()-began))

    qualified = feasible()
    best = (min(qualified, key=lambda q:q["fee"]) if qualified
            else max(points, key=lambda q:q["probability"]))
    return best, dict(statuses=statuses, qualified_by_search=bool(qualified), target=TARGET,
        fixed_cost_scale=cost_scale, seconds=time.perf_counter()-started,
        price_evaluations=len(records), feasible_price_evaluations=len(qualified),
        global_optimal=False, target_scope="fixed predicted powers and nominal delivery, independent voltage innovations",
        no_true_topology_or_future_used=True)


def finalize_quote(problem, best, solution, out, sources):
    """Audit one saved point at 16/32/64 orders and freeze the quote once."""
    out = Path(out)
    point = problem.nominal(best["values"])
    model = IndependentResiduals(point["a_C"], point["a_F"], problem.data["sigma"].ravel())
    checks = [gauss_probability(model, problem.data["packet"]["threshold"],
        cutoff=512., order=order, refinements=0, tolerance=FEASIBILITY_TOLERANCE,
        normalization=problem.normalization) for order in (16, 32, 64)]
    probabilities = [q["probability"] for q in checks]
    spread = max(probabilities)-min(probabilities)
    refined = checks[-1]
    refined["forced_orders_audit"] = dict(orders=[16, 32, 64], probabilities=probabilities,
                                          maximum_probability_difference=spread)
    refined["numerical_pass"] = bool(spread <= FEASIBILITY_TOLERANCE and all(
        0 <= q["probability"] <= 1 and q["analytic_tail_bound"] <= FEASIBILITY_TOLERANCE
        for q in checks))
    device = problem.device
    audit = device.audit(point["mean_u"])
    qualified = bool(refined["numerical_pass"] and audit["passed"]
                     and refined["probability"] >= TARGET)
    headroom = np.where(point["mean_u"] >= 0, device.delta_upper_kw, -device.delta_lower_kw)
    reward = np.divide(device.cost_full_activation*np.abs(point["mean_u"]), headroom,
                       out=np.zeros_like(headroom), where=headroom > 0)
    if abs(device.dt*np.sum(reward*np.abs(point["mean_u"]))-point["fee"]) > 1e-8:
        raise ArithmeticError("Frozen reward differs from original contract fee")
    plan = dict(x=point["x"], mean_u=point["mean_u"], fee=point["fee"], cost=point["fee"],
        unit_reward=reward, reward_units="contract fee units per kWh",
        probability=refined["probability"], qualified=qualified, nominal_audit=audit,
        target_scope=solution["target_scope"], planning_inputs=problem.data["inputs"],
        planning_sources=sources, source_request_sha256=bounded.sha(out/"request.json"))
    write_pickle(out/"quote.pkl", plan)
    bounded.write(out/"price_freeze.json", dict(plan=bounded.reference(out/"quote.pkl"),
        request=bounded.reference(out/"request.json"), qualified=qualified,
        target_scope=solution["target_scope"], true_future_used=False))
    summary = dict(case=bounded.CASE, active_variable_count=int(problem.active.size),
        raw_variable_count=device.R*24, qualified=qualified, best_fee=point["fee"],
        refined_probability=refined["probability"], nominal_audit=audit, refined=refined,
        search=solution, global_optimal=False, no_new_AC=True,
        probability_scope=solution["target_scope"], plan_sha256=bounded.sha(out/"quote.pkl"))
    bounded.write(out/"summary.json", summary)
    bounded.write(out/"complete.json", dict(status="complete", qualified=qualified,
        summary_sha256=bounded.sha(out/"summary.json"),
        price_freeze=bounded.reference(out/"price_freeze.json")))
    return summary


def main(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"complete.json").exists():
        print("Already complete", flush=True)
        return
    if (out/"request.json").exists():
        raise ValueError("Existing incomplete stage is preserved; choose a new output directory")
    lock = out/"running.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        data = bounded.build_problem(args.archive.resolve(), args.public.resolve())
        problem = FullPriceProblem(data)
        sources = dict(data["sources"])
        for file in (Path(__file__), HERE/"full_adjoint.py", HERE/"gauss_frequency.py"):
            sources[str(file.relative_to(bounded.TASK))] = bounded.sha(file)
        warm_path = getattr(args, "warm_quote", None)
        initial = None
        if warm_path is not None:
            warm_path = Path(warm_path).resolve()
            plan = bounded.unpickle(warm_path)
            if plan.get("planning_inputs") != data["inputs"]:
                raise ValueError("Warm quote has a different historical or issued-forecast information set")
            requests = np.asarray(plan["x"], float)
            if requests.shape != problem.shape or not np.isfinite(requests).all():
                raise ValueError("Warm quote differs from the original full request domain")
            initial = requests.ravel()[problem.active]
            if (np.any(initial < problem.lower) or np.any(initial > problem.upper)
                    or not np.array_equal(problem.expand(initial), requests)):
                raise ValueError("Warm quote violates active-hour bounds or uses a zero-capacity hour")
        check_only = bool(getattr(args, "check_only", False))
        if check_only and initial is None:
            raise ValueError("--check-only requires an explicit --warm-quote")
        controls = dict(maxiter=getattr(args, "maxiter", 60), seconds=getattr(args, "seconds", 300.),
                        restore_evaluations=10, restore_seconds=30.)
        if controls["maxiter"] <= 0 or not np.isfinite(controls["seconds"]) or controls["seconds"] <= 0:
            raise ValueError("Positive finite solver limits required")
        request = dict(case=bounded.CASE, protocol=bounded.reference(args.protocol.resolve()),
            inputs=data["inputs"], sources=sources, active_variable_count=int(problem.active.size),
            raw_variable_count=problem.device.R*24, normalization=problem.normalization,
            target=TARGET, search_options=SEARCH_OPTIONS, solver_controls=controls,
            warm_quote=None if warm_path is None else bounded.reference(warm_path), check_only=check_only,
            python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            scope="conditional independent voltage chance constraint; no new AC here")
        bounded.write(out/"request.json", request)
        with zipfile.ZipFile(out/"source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as snapshot:
            for filename in sources:
                snapshot.write(bounded.TASK/filename, filename)
        if check_only:
            best = problem.evaluate(initial)
            delivery_error = float(np.max(np.abs(best["mean_u"]-np.asarray(plan["mean_u"]))))
            fee_error = abs(best["fee"]-float(plan["fee"]))
            if delivery_error > 1e-10 or fee_error > 1e-8:
                raise ArithmeticError("Frozen quote does not replay its original device and fee")
            solution = dict(mode="check_only", optimization_performed=False,
                target_scope=plan["target_scope"], global_optimal=False,
                input_delivery_replay_error=delivery_error, input_fee_replay_error=fee_error)
        else:
            check = directional_check(problem)
            bounded.write(out/"gradient_check.json", check)
            if not check["passed"]:
                raise ArithmeticError("Full-variable device/probability gradient check unresolved")
            best, solution = optimize(problem, out, initial, **controls)
        summary = finalize_quote(problem, best, solution, out, sources)
        print(__import__("json").dumps(bounded.plain(summary), ensure_ascii=True), flush=True)
    except Exception as exc:
        bounded.write(out/"failure.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "public", "protocol", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--warm-quote", type=Path,
                        help="Optional frozen quote with the same historical and forecast inputs")
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--seconds", type=float, default=300.)
    parser.add_argument("--check-only", action="store_true",
                        help="Replay --warm-quote at 16/32/64 orders; no price optimization or AC")
    main(parser.parse_args())
