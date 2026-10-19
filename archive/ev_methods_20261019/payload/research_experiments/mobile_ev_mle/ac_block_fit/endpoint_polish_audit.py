"""Budget-preserving independent polish of saved multi-candidate endpoints.

All original results and failures remain immutable.  Each of 16 candidates has
two source-method endpoints (32 in total).  At most three polish trial points
can use the *remaining* part of its original 250-evaluation budget.  Independent
entry/final audits are reported separately; an original failed fit is never
promoted into a scientific success.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import sys
from time import perf_counter
import traceback

from . import benchmark as b
from . import candidate_audit as candidates


def prepare_plan(source, whitelist, output, *, max_steps=3):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("retain previous endpoint audit: "+str(output))
    if max_steps < 0 or max_steps > 3:
        raise ValueError("frozen polish permits zero to three trial steps")
    package = b.load_history_package(whitelist)
    source_plan = b._read(source/"plan.json")
    if not (source/"completed.json").exists():
        raise ValueError("finish the old pilot before declaring an endpoint audit")
    tasks, retained = [], []
    for old_task in source_plan["tasks"]:
        path = source/"tasks"/old_task["task_id"]/"worker_result.json"
        envelope_path = path.with_name("result.json")
        if not path.exists():
            retained.append(dict(task=old_task, reason="missing original endpoint; retain original failure/timeout",
                original_envelope=str(envelope_path.resolve()),
                original_status=b._read(envelope_path).get("status") if envelope_path.exists() else "missing"))
            continue
        original = b._read(path)
        if original.get("status") != "returned" or "attained_state" not in original:
            retained.append(dict(task=old_task, reason="original worker exception retained",
                                 original_endpoint=str(path.resolve())))
            continue
        nfev = int(original["fit"]["nfev"])
        budget = int(original["fit_options"]["max_nfev"])
        task = {key: value for key, value in old_task.items() if key != "task_id"}
        task.update(endpoint_file=str(path.resolve()), endpoint_sha256=b._digest(path),
            original_task_id=old_task["task_id"], original_nfev=nfev,
            original_success=bool(original.get("solver_success")),
            original_budget=budget, remaining_nfev=max(0, budget-nfev),
            polish_max_steps=int(max_steps), common_pg_target=candidates.THRESHOLDS["common_projected_gradient"],
            audit_kind="independent_saved_endpoint_polish")
        if task["package_sha256"] != package["payload_sha256"]:
            raise ValueError("original endpoint package differs from supplied historical whitelist")
        task["task_id"] = sha256(b._bytes(task)).hexdigest()[:20]
        tasks.append(task)
    result = dict(schema="saved_endpoint_polish_plan_v1", source_pilot=str(source.resolve()),
        source_plan_sha256=b._digest(source/"plan.json"),
        historical_whitelist=str(Path(whitelist).resolve()), historical_whitelist_sha256=b._digest(whitelist),
        created_utc=b._utc(), tasks=tasks, retained_unavailable=retained,
        original_task_denominator=len(source_plan["tasks"]),
        source_candidate_denominator=len({(task["network"], task["candidate_name"], task["prefix"], task["repeat"])
                                          for task in source_plan["tasks"]}),
        thresholds=candidates.THRESHOLDS, max_steps=int(max_steps),
        optimizer_budget_rule="max(0, original max_nfev minus original nfev), never replenished",
        original_failure_can_be_promoted=False, historical_or_future_samples_added=0,
        declared_before_polish=True)
    b._write(output, result)
    return result


def _context(whitelist, task, original):
    import numpy as np
    from ..ev_rx_fit import _RxInputProblem
    package = b.load_history_package(whitelist)
    item = package["items"][task["network"]]
    candidate, data, meter, options, _, _ = b._reconstruct(item, task)
    state = {key: np.asarray(original["attained_state"][key], float)
             for key in ("r", "x", "source_p", "source_q")}
    domain = {key: options[key] for key in ("r_bounds", "x_bounds", "ratio_bounds",
        "source_p_bounds", "source_q_bounds", "voltage_bounds")}
    counter = {}
    problem = _RxInputProblem(candidate, data, meter, initial=state, call_counter=counter, **domain)
    r, alpha = problem.mapping.inverse(state["r"], state["x"])
    full = np.r_[r, alpha, np.c_[state["source_p"], state["source_q"]].ravel()]
    vector = full[problem.free]
    rebuilt_r, rebuilt_x, _ = problem.unpack(vector)
    reconstruction = float(max(np.max(abs(rebuilt_r-state["r"]), initial=0.),
                               np.max(abs(rebuilt_x-state["x"]), initial=0.)))
    if reconstruction > 1e-12:
        raise ValueError("saved endpoint cannot be rebuilt in the same physical domain")
    return candidate, data, meter, options, problem, vector, counter, item, reconstruction


def worker(whitelist, task, *, polisher=None):
    import numpy as np
    from .. import likelihood as lk
    from ..ev_fixed_train_fit import _compact_bias
    started = perf_counter()
    result = dict(task=task, task_sha256=b._task_identity(task), status="started",
                  future_used=False, truth_used=False, audit_source_sha256=b._digest(__file__))
    try:
        endpoint = Path(task["endpoint_file"])
        if b._digest(endpoint) != task["endpoint_sha256"]:
            raise ValueError("original saved endpoint hash changed")
        original = b._read(endpoint)
        result["original_endpoint"] = dict(file=str(endpoint), sha256=task["endpoint_sha256"],
            success=bool(original.get("solver_success")), loss=original["fit"]["loss"],
            status=original["fit"]["status"], nfev=original["fit"]["nfev"],
            common_pg=(original.get("audit") or {}).get("reference_scaled_projected_gradient"))
        if task["remaining_nfev"] != max(0, task["original_budget"]-task["original_nfev"]):
            raise ValueError("declared remaining evaluation budget differs from old task")
        candidate, data, meter, options, problem, vector, counter, item, reconstruction = \
            _context(whitelist, task, original)
        if polisher is None:
            from .polished_fit import polish_stationarity as polisher
        began = perf_counter()
        outcome = polisher(problem, vector, target=float(task["common_pg_target"]),
            max_steps=int(task["polish_max_steps"]), remaining_nfev=int(task["remaining_nfev"]),
            damping=1e-4, inner_tol=float(task["inner_tol"]), inner_maxiter=2000)
        polish_seconds = perf_counter()-began
        if outcome["trial_nfev"] > task["remaining_nfev"]:
            raise RuntimeError("polisher exceeded remaining original evaluation budget")
        vector = np.asarray(outcome["x"], float)
        residual = problem.evaluate(vector); physical = problem.base.cache["physical"]
        r, x, source = problem.unpack(vector); base = problem.base
        loss = float(residual@residual)
        # Independent endpoint audit can classify a numerically useful state,
        # but cannot erase or promote a failed original optimizer.
        proposed_success = bool(task["original_success"] and outcome["success"])
        diagnostics = dict(original_success=bool(task["original_success"]),
            original_failure_retained=not bool(task["original_success"]),
            independent_endpoint_polish=True, common_stationarity_pass=bool(outcome["success"]),
            polish={key: value for key, value in outcome.items() if key not in ("x", "fun")},
            total_nfev=int(task["original_nfev"])+int(outcome["trial_nfev"]),
            optimization_budget=int(task["original_budget"]),
            entry_and_final_audit_calls_are_separate=True,
            RX_reconstruction_max_abs=reconstruction, global_optimality_certified=False)
        predicted = dict(physical, p=base.p+base.bias["p"], q=base.q+base.bias["q"],
                         v=physical["v"]+base.bias["v"], physical_v=physical["v"])
        biases = [_compact_bias(base.bias[key]) for key in ("p", "q", "v")]
        fit = lk.FitResult(candidate.name, r.copy(), x.copy(), base.p.copy(), base.q.copy(),
            source[:, 0].copy(), source[:, 1].copy(), *biases, loss,
            -.5*(loss+base.normalizer), proposed_success, 1 if proposed_success else -7,
            "budgeted saved-endpoint stationarity audit; original failure flag retained",
            int(task["original_nfev"])+int(outcome["trial_nfev"]),
            float(outcome["final_common_pg"]), predicted, diagnostics)
        audit_start = perf_counter()
        audit = b.audit_fit(candidate, data, meter, options, fit, fixed_lines=False)
        final_success = bool(proposed_success and audit.get("available") and
            audit.get("physical_voltage_feasible") and audit.get("line_domain_feasible") and
            audit.get("source_and_optimizer_box_feasible") and
            audit.get("reference_scaled_projected_gradient", np.inf) <= task["common_pg_target"])
        fit.success = final_success
        local = candidates.local_response(candidate, data, meter,
            dict(r=r, x=x, source_p=source[:, 0], source_q=source[:, 1]), base_kw=item["base_kw"])
        result.update(status="returned", solver_success=final_success, fit=fit.summary(),
            attained_state=dict(r=r, x=x, source_p=source[:, 0], source_q=source[:, 1]),
            audit=audit, local_response=local, counters=counter,
            original_failure_retained=not bool(task["original_success"]),
            remaining_nfev_at_entry=task["remaining_nfev"], polish_trial_nfev=outcome["trial_nfev"],
            total_optimization_nfev=fit.nfev, fixed_lines=False,
            timings=dict(polish_seconds=polish_seconds, audit_seconds=perf_counter()-audit_start),
            source_hashes=b._source_fingerprints())
    except Exception as error:
        result.update(status="exception", exception=type(error).__name__+": "+str(error),
                      traceback=traceback.format_exc())
    result.update(wall_seconds=perf_counter()-started, finished_utc=b._utc())
    return b._serial(result)


@contextmanager
def _dispatch():
    original = b._worker_command
    def command(whitelist, task_file, result_file):
        return [sys.executable, "-m", "research_experiments.mobile_ev_mle.ac_block_fit.endpoint_polish_audit",
                "worker", "--whitelist", str(whitelist), "--task", str(task_file), "--result", str(result_file)]
    b._worker_command = command
    try:
        yield
    finally:
        b._worker_command = original


def execute(plan_file, output, *, timeout=60.):
    plan = b._read(plan_file)
    whitelist = Path(plan["historical_whitelist"])
    if b._digest(whitelist) != plan["historical_whitelist_sha256"]:
        raise ValueError("historical whitelist changed after endpoint plan")
    output = Path(output)
    b._write(output/"endpoint_protocol.json", dict(plan_file=str(Path(plan_file).resolve()),
        plan_sha256=b._digest(plan_file), original_task_denominator=plan["original_task_denominator"],
        retained_unavailable=plan["retained_unavailable"], declared_before_polish=True,
        remaining_optimizer_budget_only=True, source_sha256=b._digest(__file__)))
    with _dispatch():
        result = b.run_matrix(whitelist, plan["tasks"], output,
                              joint_timeout=timeout, fixed_timeout=timeout)
    candidates.summarize(output, whitelist)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", required=True, type=Path)
    prepare.add_argument("--whitelist", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--max-steps", default=3, type=int)
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--timeout", default=60., type=float)
    single = sub.add_parser("worker")
    single.add_argument("--whitelist", required=True, type=Path)
    single.add_argument("--task", required=True, type=Path)
    single.add_argument("--result", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        plan = prepare_plan(args.source, args.whitelist, args.output, max_steps=args.max_steps)
        print(len(plan["tasks"]), "retained endpoints", len(plan["retained_unavailable"]), "unavailable")
    elif args.command == "worker":
        b._write(args.result, worker(args.whitelist, b._read(args.task)))
    else:
        execute(args.plan, args.output, timeout=args.timeout)


if __name__ == "__main__":
    main()
