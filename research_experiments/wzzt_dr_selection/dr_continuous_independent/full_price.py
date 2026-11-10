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


def optimize(problem, out):
    """Find a feasible point then minimize fees over every available request."""
    records = []
    points = []
    started = time.perf_counter()
    phase = "starts"
    phase_started = started

    def observe(values, label):
        nonlocal phase_started
        limit = 240 if phase == "feasibility" else 300
        if phase != "starts" and time.perf_counter()-phase_started > limit:
            raise TimeoutError(phase + " wall limit")
        q = problem.evaluate(values)
        if not records or not np.array_equal(q["values"], points[-1]["values"]):
            row = {k:q[k] for k in ("probability", "fee", "seconds", "moments", "analytic_tail_bound")}
            row.update(index=len(records), label=label, phase=phase,
                       elapsed=time.perf_counter()-started,
                       gradient_norm=float(np.linalg.norm(q["gradient"])),
                       nonzero_resource_hours=int(np.count_nonzero(np.abs(q["mean_u"]) > 1e-8)))
            records.append(row)
            points.append(q)
            bounded.write(out/"optimization_trace.json", records)
            write_pickle(out/"checkpoint_private.pkl", dict(records=records, points=points))
        return q

    statuses = []
    zero = observe(problem.zero, "zero")
    old = observe(problem.old, "old_warm_start")
    if not any(q["probability"] >= TARGET-FEASIBILITY_TOLERANCE for q in points):
        phase = "feasibility"
        phase_started = time.perf_counter()
        try:
            fit = minimize(lambda z:-observe(z,"feasibility")["probability"], problem.old,
                           jac=lambda z:-observe(z,"feasibility")["gradient"],
                           bounds=problem.bounds, method="L-BFGS-B",
                           options=dict(maxiter=50, ftol=1e-11, gtol=1e-7, maxls=20))
            observe(fit.x,"feasibility_return")
            statuses.append(dict(phase=phase,status=str(fit.message),success=bool(fit.success),
                                 iterations=int(fit.nit),seconds=time.perf_counter()-phase_started))
        except Exception as exc:
            statuses.append(dict(phase=phase,status=repr(exc),success=False,
                                 seconds=time.perf_counter()-phase_started))
    feasible = [q for q in points if q["probability"] >= TARGET-FEASIBILITY_TOLERANCE]
    if feasible:
        phase = "cost"
        phase_started = time.perf_counter()
        initial = min(feasible,key=lambda q:q["fee"])["values"]
        try:
            fit = minimize(lambda z:observe(z,"cost")["fee"], initial,
                           jac=lambda z:observe(z,"cost")["fee_gradient"],
                           constraints=[dict(type="ineq",fun=lambda z:observe(z,"constraint")["probability"]-TARGET,
                                             jac=lambda z:observe(z,"constraint")["gradient"])],
                           bounds=problem.bounds,method="SLSQP",
                           options=dict(maxiter=60,ftol=1e-8))
            observe(fit.x,"cost_return")
            statuses.append(dict(phase=phase,status=str(fit.message),success=bool(fit.success),
                                 iterations=int(fit.nit),seconds=time.perf_counter()-phase_started))
        except Exception as exc:
            statuses.append(dict(phase=phase,status=repr(exc),success=False,
                                 seconds=time.perf_counter()-phase_started))
    feasible = [q for q in points if q["probability"] >= TARGET-FEASIBILITY_TOLERANCE]
    best = min(feasible,key=lambda q:q["fee"]) if feasible else max(points,key=lambda q:q["probability"])
    return best, dict(statuses=statuses,qualified_by_search=bool(feasible), target=TARGET,
                      feasibility_tolerance=FEASIBILITY_TOLERANCE,global_optimal=False,
                      seconds=time.perf_counter()-started,price_evaluations=len(records),
                      target_scope="fixed predicted powers and nominal delivery, independent voltage innovations",
                      no_true_topology_or_future_used=True,trace=records)


def main(args):
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():
        print("Already complete",flush=True)
        return
    lock = out/"running.lock"
    with lock.open("x",encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        data = bounded.build_problem(args.archive.resolve(),args.public.resolve())
        problem = FullPriceProblem(data)
        sources = dict(data["sources"])
        for file in (Path(__file__),HERE/"full_adjoint.py",HERE/"gauss_frequency.py"):
            sources[str(file.relative_to(bounded.TASK))] = bounded.sha(file)
        request = dict(case=bounded.CASE,protocol=bounded.reference(args.protocol.resolve()),
                       inputs=data["inputs"],sources=sources,active_variable_count=problem.active.size,
                       raw_variable_count=problem.device.R*24,normalization=problem.normalization,
                       target=TARGET,feasibility_tolerance=FEASIBILITY_TOLERANCE,
                       search_options=SEARCH_OPTIONS,phase_one=dict(maxiter=50,seconds=240),
                       cost_phase=dict(maxiter=60,seconds=300),full_resource_hour_variables=True,
                       old_quote_is_warm_start_only=True,python=platform.python_version(),
                       numpy=np.__version__,scipy=scipy.__version__,
                       scope="conditional independent voltage chance constraint; no new AC here")
        if (out/"request.json").exists():
            raise ValueError("Incomplete existing stage must be resumed explicitly; sources are preserved")
        bounded.write(out/"request.json",request)
        with zipfile.ZipFile(out/"source_snapshot.zip","x",zipfile.ZIP_DEFLATED) as snapshot:
            for filename in sources:
                snapshot.write(bounded.TASK/filename,filename)
        check = directional_check(problem)
        bounded.write(out/"gradient_check.json",check)
        if not check["passed"]:
            raise ArithmeticError("Full-variable device/probability gradient check unresolved")
        best, solution = optimize(problem,out)
        point = problem.nominal(best["values"])
        model = IndependentResiduals(point["a_C"],point["a_F"],data["sigma"].ravel())
        refined = gauss_probability(model,data["packet"]["threshold"],cutoff=512.,order=16,
                                    refinements=2,tolerance=1e-7,normalization=problem.normalization)
        qualified = bool(refined["numerical_pass"] and refined["probability"] >= TARGET-FEASIBILITY_TOLERANCE)
        direction_headroom = np.where(point["mean_u"]>=0,problem.device.delta_upper_kw,
                                      -problem.device.delta_lower_kw)
        unit_reward = np.divide(problem.device.cost_full_activation*np.abs(point["mean_u"]),
                                direction_headroom,out=np.zeros_like(point["mean_u"]),
                                where=direction_headroom>0)
        if abs(.25*np.sum(unit_reward*np.abs(point["mean_u"]))-point["fee"])>1e-8:
            raise ArithmeticError("Resource reward schedule does not replay contract fee")
        plan = dict(x=point["x"],mean_u=point["mean_u"],fee=point["fee"],cost=point["fee"],
                    probability=refined["probability"],qualified=qualified,
                    target_scope=solution["target_scope"],nominal_audit=problem.device.audit(point["mean_u"]),
                    unit_reward=unit_reward,reward_units="contract fee units per kWh",
                    planning_inputs=data["inputs"],planning_sources=sources,
                    source_request_sha256=bounded.sha(out/"request.json"))
        write_pickle(out/"quote.pkl",plan)
        bounded.write(out/"price_freeze.json",dict(plan=bounded.reference(out/"quote.pkl"),
                      request=bounded.reference(out/"request.json"),qualified=qualified,
                      target_scope=solution["target_scope"],true_future_used=False))
        old_u=problem.device.nominal(data["frozen_x"])
        projected_old=np.zeros_like(point["x"])
        best_old_scales=[]
        for start,end in ((0,12),(12,24)):
            original=np.asarray(data["frozen_x"])[:,start:end]
            current=point["x"][:,start:end]
            denominator=float(np.sum(original*original))
            scale=float(np.sum(current*original)/denominator) if denominator else 0.
            best_old_scales.append(scale)
            projected_old[:,start:end]=scale*original
        relative_ray_error=float(np.linalg.norm(point["x"]-projected_old)
                                 /max(np.linalg.norm(point["x"]),1e-12))
        summary = dict(case=bounded.CASE,active_variable_count=int(problem.active.size),
                       raw_variable_count=problem.device.R*24,target=TARGET,
                       old_fee=float(problem.device.fees(problem.device.nominal(data["frozen_x"]))),
                       best_fee=point["fee"],best_search_probability=best["probability"],
                       refined_probability=refined["probability"],qualified=qualified,
                       full_resource_hour_variables=True,old_quote_is_warm_start_only=True,
                       active_nodes=len(set(problem.device.node[problem.active//24])),
                       delivered_active_nodes=int(np.unique(problem.device.node[np.any(np.abs(point["mean_u"])>1e-8,axis=1)]).size),
                       nonzero_resource_slots=int(np.count_nonzero(np.abs(point["mean_u"])>1e-8)),
                       changed_resource_slots=int(np.count_nonzero(np.abs(point["mean_u"]-old_u)>1e-6)),
                       activated_from_old_zero=int(np.count_nonzero((np.abs(old_u)<=1e-8)&(np.abs(point["mean_u"])>1e-6))),
                       old_two_scale_fit=best_old_scales,old_two_scale_relative_error=relative_ray_error,
                       absolute_DR_kwh=float(.25*np.abs(point["mean_u"]).sum()),
                       statuses=solution["statuses"],global_optimal=False,
                       seconds=solution["seconds"],price_evaluations=solution["price_evaluations"],
                       gradient_check=check,refined=refined,nominal_audit=plan["nominal_audit"],
                       probability_scope=solution["target_scope"],no_new_AC=True,
                       plan_sha256=bounded.sha(out/"quote.pkl"))
        bounded.write(out/"summary.json",summary)
        bounded.write(out/"complete.json",dict(status="complete",qualified=qualified,
                      summary_sha256=bounded.sha(out/"summary.json"),price_freeze=bounded.reference(out/"price_freeze.json")))
        print(__import__("json").dumps(bounded.plain(summary),ensure_ascii=True),flush=True)
    except Exception as exc:
        bounded.write(out/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive","public","protocol","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    main(parser.parse_args())
