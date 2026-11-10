"""Continue the same constrained price problem with fixed objective scaling.

SLSQP accepts an infeasible initial point. The conditional probability, device
map, all request variables and cost remain exactly those of full_price.py.
Dividing the cost and its gradient by a fixed positive historical cost changes
only numerical conditioning, preserving every minimizer and feasible point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import platform
import time
import traceback
import zipfile

for option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[option] = "1"

import numpy as np
import scipy
from scipy.optimize import minimize

import full_price as full
import run as bounded
from continuous import IndependentResiduals
from gauss_frequency import gauss_probability


def optimize(problem, initial, out):
    cost_scale = float(problem.evaluate(problem.old)["fee"])
    if not np.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("Positive fixed historical cost required for scaling")
    started = time.perf_counter()
    records, points = [], []
    status = "not_started"
    success = False
    iterations = 0

    def observe(values, label):
        if time.perf_counter()-started > 300:
            raise TimeoutError("Full constrained pricing wall limit")
        q = problem.evaluate(values)
        if not points or not np.array_equal(q["values"],points[-1]["values"]):
            projected = q["gradient"].copy()
            projected[(values<=problem.lower+1e-8)&(projected<0)] = 0.
            projected[(values>=problem.upper-1e-8)&(projected>0)] = 0.
            row = {k:q[k] for k in ("probability","fee","seconds","moments","analytic_tail_bound")}
            row.update(index=len(records),label=label,elapsed=time.perf_counter()-started,
                       probability_gradient_norm=float(np.linalg.norm(q["gradient"])),
                       probability_projected_gradient_norm=float(np.linalg.norm(projected)),
                       resource_slots_nonzero=int(np.count_nonzero(np.abs(q["mean_u"])>1e-8)))
            records.append(row)
            points.append(q)
            bounded.write(out/"optimization_trace.json",records)
            full.write_pickle(out/"checkpoint_private.pkl",dict(records=records,points=points))
        return q

    observe(initial,"frozen_infeasible_start")
    try:
        fit = minimize(lambda z:observe(z,"objective")["fee"]/cost_scale,initial,
                       jac=lambda z:observe(z,"objective")["fee_gradient"]/cost_scale,
                       constraints=[dict(type="ineq",fun=lambda z:observe(z,"constraint")["probability"]-full.TARGET,
                                         jac=lambda z:observe(z,"constraint")["gradient"])],
                       bounds=problem.bounds,method="SLSQP",options=dict(maxiter=60,ftol=1e-8))
        observe(fit.x,"return")
        status,success,iterations = str(fit.message),bool(fit.success),int(fit.nit)
    except Exception as exc:
        status = repr(exc)
        bounded.write(out/"solver_exception.json",dict(error=repr(exc),traceback=traceback.format_exc()))
    feasible = [q for q in points if q["probability"]>=full.TARGET-full.FEASIBILITY_TOLERANCE]
    best = min(feasible,key=lambda q:q["fee"]) if feasible else max(points,key=lambda q:q["probability"])
    summary = dict(status=status,solver_success=success,iterations=iterations,
                   qualified_by_search=bool(feasible),feasible_price_evaluations=len(feasible),
                   price_evaluations=len(records),seconds=time.perf_counter()-started,
                   global_optimal=False,conditional_target=full.TARGET,
                   fixed_objective_scale=cost_scale,objective="original_fee / fixed_old_fee")
    return best,summary


def main(args):
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():
        print("Already complete",flush=True)
        return
    with (out/"running.lock").open("x",encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        previous=args.previous.resolve()
        if not (previous/"complete.json").exists():
            raise ValueError("Initial full-price stage must be complete")
        old_plan=bounded.unpickle(previous/"quote.pkl")
        prior_request=bounded.read(previous/"request.json")
        gradient_path=previous/"gradient_check.json"
        if not gradient_path.exists():
            gradient_ref=prior_request["previous_gradient"]
            gradient_path=Path(gradient_ref["path"])
            if bounded.sha(gradient_path)!=gradient_ref["sha256"]:
                raise ValueError("Inherited gradient check hash differs")
        prior_gradient=bounded.read(gradient_path)
        if not prior_gradient["passed"]:
            raise ValueError("Initial full-variable derivative check is not qualified")
        data=bounded.build_problem(args.archive.resolve(),args.public.resolve())
        problem=full.FullPriceProblem(data)
        if tuple(problem.normalization)!=tuple(prior_request["normalization"]):
            raise ValueError("Normalization differs from frozen stage")
        initial=np.asarray(old_plan["x"]).ravel()[problem.active]
        if np.any(initial<problem.lower-1e-8) or np.any(initial>problem.upper+1e-8):
            raise ValueError("Previous checkpoint violates original request bounds")
        initial=np.clip(initial,problem.lower,problem.upper)
        sources=dict(data["sources"])
        for file in (Path(__file__),full.HERE/"full_price_continue.py",full.HERE/"full_price.py",full.HERE/"full_adjoint.py",full.HERE/"gauss_frequency.py"):
            sources[str(file.relative_to(bounded.TASK))]=bounded.sha(file)
        request=dict(case=bounded.CASE,protocol=bounded.reference(args.protocol.resolve()),
                     previous_plan=bounded.reference(previous/"quote.pkl"),
                     previous_complete=bounded.reference(previous/"complete.json"),
                     previous_gradient=bounded.reference(gradient_path),
                     inputs=data["inputs"],sources=sources,active_variable_count=problem.active.size,
                     raw_variable_count=problem.device.R*24,normalization=problem.normalization,
                     solver="SLSQP",maxiter=60,seconds=300,target=full.TARGET,
                     objective_scale=float(problem.evaluate(problem.old)["fee"]),
                     objective_scale_scope="fixed positive old cost; minimizers and feasibility unchanged",
                     feasibility_tolerance=full.FEASIBILITY_TOLERANCE,
                     allowed_infeasible_initial_point=True,all_original_resource_hour_variables=True,
                     search_options=full.SEARCH_OPTIONS,python=platform.python_version(),
                     numpy=np.__version__,scipy=scipy.__version__)
        if (out/"request.json").exists():
            raise ValueError("Incomplete continuation stage is preserved; use an explicit resume")
        bounded.write(out/"request.json",request)
        with zipfile.ZipFile(out/"source_snapshot.zip","x",zipfile.ZIP_DEFLATED) as snapshot:
            for filename in sources:
                snapshot.write(bounded.TASK/filename,filename)
        best,search=optimize(problem,initial,out)
        point=problem.nominal(best["values"])
        model=IndependentResiduals(point["a_C"],point["a_F"],data["sigma"].ravel())
        low_order=gauss_probability(model,data["packet"]["threshold"],cutoff=512.,order=16,
                                    refinements=0,tolerance=1e-7,normalization=problem.normalization)
        refined=gauss_probability(model,data["packet"]["threshold"],cutoff=512.,order=32,
                                  refinements=1,tolerance=1e-7,normalization=problem.normalization)
        refined["forced_orders_audit"] = dict(orders=[16,32,64], probability16=low_order["probability"],
            probability32=refined["trace"][0]["probability"], probability64=refined["trace"][1]["probability"],
            maximum_probability_difference=max(abs(low_order["probability"]-q["probability"]) for q in refined["trace"]))
        refined["numerical_pass"] = bool(refined["numerical_pass"] and refined["forced_orders_audit"]["maximum_probability_difference"]<=1e-7)
        qualified=bool(refined["numerical_pass"] and refined["probability"]>=full.TARGET-full.FEASIBILITY_TOLERANCE)
        headroom=np.where(point["mean_u"]>=0,problem.device.delta_upper_kw,-problem.device.delta_lower_kw)
        unit_reward=np.divide(problem.device.cost_full_activation*np.abs(point["mean_u"]),headroom,
                              out=np.zeros_like(point["mean_u"]),where=headroom>0)
        if abs(.25*np.sum(unit_reward*np.abs(point["mean_u"]))-point["fee"])>1e-8:
            raise ArithmeticError("Reward schedule differs from original fee")
        plan=dict(x=point["x"],mean_u=point["mean_u"],fee=point["fee"],cost=point["fee"],
                  probability=refined["probability"],qualified=qualified,
                  target_scope="fixed predicted powers and nominal delivery, independent voltage innovations",
                  nominal_audit=problem.device.audit(point["mean_u"]),unit_reward=unit_reward,
                  reward_units="contract fee units per kWh",planning_inputs=data["inputs"],planning_sources=sources,
                  source_request_sha256=bounded.sha(out/"request.json"))
        full.write_pickle(out/"quote.pkl",plan)
        bounded.write(out/"price_freeze.json",dict(plan=bounded.reference(out/"quote.pkl"),
                      request=bounded.reference(out/"request.json"),qualified=qualified,
                      target_scope=plan["target_scope"],true_future_used=False))
        old_u=problem.device.nominal(data["frozen_x"])
        projection=np.zeros_like(point["x"])
        scales=[]
        for start,end in ((0,12),(12,24)):
            original=np.asarray(data["frozen_x"])[:,start:end]
            current=point["x"][:,start:end]
            denominator=float(np.sum(original*original))
            scale=float(np.sum(current*original)/denominator) if denominator else 0.
            scales.append(scale)
            projection[:,start:end]=scale*original
        summary=dict(case=bounded.CASE,active_variable_count=int(problem.active.size),raw_variable_count=problem.device.R*24,
                     previous_probability=old_plan["probability"],previous_fee=old_plan["fee"],
                     best_fee=point["fee"],best_search_probability=best["probability"],
                     refined_probability=refined["probability"],qualified=qualified,
                     active_nodes=len(set(problem.device.node[problem.active//24])),
                     delivered_active_nodes=int(np.unique(problem.device.node[np.any(np.abs(point["mean_u"])>1e-8,axis=1)]).size),
                     nonzero_resource_slots=int(np.count_nonzero(np.abs(point["mean_u"])>1e-8)),
                     changed_resource_slots=int(np.count_nonzero(np.abs(point["mean_u"]-old_u)>1e-6)),
                     activated_from_old_zero=int(np.count_nonzero((np.abs(old_u)<=1e-8)&(np.abs(point["mean_u"])>1e-6))),
                     old_two_scale_fit=scales,old_two_scale_relative_error=float(np.linalg.norm(point["x"]-projection)/max(np.linalg.norm(point["x"]),1e-12)),
                     absolute_DR_kwh=float(.25*np.abs(point["mean_u"]).sum()),
                     nominal_audit=plan["nominal_audit"],refined=refined,search=search,
                     probability_scope=plan["target_scope"],plan_sha256=bounded.sha(out/"quote.pkl"),
                     no_new_AC=True,global_optimal=False)
        bounded.write(out/"summary.json",summary)
        bounded.write(out/"complete.json",dict(status="complete",qualified=qualified,
                      summary_sha256=bounded.sha(out/"summary.json"),price_freeze=bounded.reference(out/"price_freeze.json")))
        print(json.dumps(bounded.plain(summary),ensure_ascii=True),flush=True)
    except Exception as exc:
        bounded.write(out/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        (out/"running.lock").unlink()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","public","previous","protocol","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    main(parser.parse_args())
