"""Eight declared case33 cross-warm numerical fits with strict common PG.

Two controlled case33 candidates each supply saved ref/QR starts.  Both solvers
run from each start with their adaptive globalization, a separate declared
250-evaluation diagnostic budget and 60-second child budget.  Old fit results
and the original scientific batch remain immutable.  No future data are used.
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
from . import candidate_audit as ca
from . import endpoint_polish_audit as ep


NAMES = ("controlled_case33_00", "controlled_case33_01")
METHODS = ca.METHODS
TARGET = ca.THRESHOLDS["common_projected_gradient"]


def _kernel_files():
    local = Path(__file__).parent
    files = [local/name for name in ("crosswarm_audit.py", "candidate_audit.py", "benchmark.py",
        "precision_audit.py", "blocks.py", "linear.py", "nonlinear.py")]
    files.extend(local.parent/name for name in ("likelihood.py", "ev_rx_fit.py", "ev_fixed_train_fit.py",
        "reference_benchmark/fitting.py", "reference_benchmark/derivatives.py"))
    return files


def prepare(source, whitelist, output, *, max_nfev=250, timeout=60.):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("retain previous cross-warm audit: "+str(output))
    if max_nfev != 250 or timeout <= 0:
        raise ValueError("this declared audit uses max_nfev=250 and a positive timeout")
    if not (source/"completed.json").exists():
        raise ValueError("old numerical pilot has not completed")
    package = b.load_history_package(whitelist)
    original_tasks = b._read(source/"plan.json")["tasks"]
    old = {(task["candidate_name"], task["algorithm"]): task for task in original_tasks
           if task["network"] == "case33" and task["candidate_name"] in NAMES}
    output.mkdir(parents=True)
    snapshot = output/"source_snapshot"; snapshot.mkdir()
    hashes = {}
    for path in _kernel_files():
        relative = path.relative_to(b.STAGE)
        target = snapshot/relative; target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes(); target.write_bytes(data)
        hashes[relative.as_posix()] = sha256(data).hexdigest()
    tasks = []
    for name in NAMES:
        for repeat, warm_method in enumerate(METHODS):
            original_task = old[(name, warm_method)]
            endpoint = source/"tasks"/original_task["task_id"]/"worker_result.json"
            if not endpoint.exists():
                raise ValueError("declared warm endpoint missing; do not replace its identity")
            saved = b._read(endpoint)
            if saved.get("status") != "returned" or "attained_state" not in saved:
                raise ValueError("declared warm endpoint has no finite attained state")
            for method in METHODS:
                task = {key: value for key, value in original_task.items() if key != "task_id"}
                task.update(algorithm=method, repeat=repeat, warm_method=warm_method,
                    endpoint_file=str(endpoint.resolve()), endpoint_sha256=b._digest(endpoint),
                    original_task_id=original_task["task_id"],
                    diagnostic_max_nfev=max_nfev, common_pg_target=TARGET,
                    kernel_sha256=hashes, source_stage=str(b.STAGE),
                    independent_diagnostic_budget=True, original_budget_not_rewritten=True)
                task["task_id"] = sha256(b._bytes(task)).hexdigest()[:20]
                tasks.append(task)
    plan = dict(schema="case33_eight_crosswarm_numerical_audit_v1", tasks=tasks,
        created_utc=b._utc(), declared_before_first_fit=True, expected_fit_denominator=8,
        source_pilot=str(source.resolve()), historical_whitelist=str(Path(whitelist).resolve()),
        historical_whitelist_sha256=b._digest(whitelist), kernel_sha256=hashes,
        common_scale="original problem.scale, fixed and shared between solvers at each start",
        reference_settings=dict(x_scale="jac", ftol=None, xtol=None, gtol=TARGET),
        block_settings=dict(scale_mode="jac", stationarity_tol=TARGET,
            inner_acceptance="certified_gap", adaptive_damping=True,
            stop_hook="fixed original scale for PG only; step uses dynamic Jac scale"),
        max_nfev=max_nfev, child_wall_timeout=timeout,
        original_failed_samples_reclassified=False, scientific_samples_added=0,
        purpose="separate initialization/globalization paths from small-step floating-point polish")
    b._write(snapshot/"manifest.json", hashes)
    b._write(output/"crosswarm_plan.json", plan)
    return plan


def worker(whitelist, task):
    import numpy as np
    from ..reference_benchmark import fitting
    from . import nonlinear
    from .precision_audit import _reference_precision, _common_gradient_stop
    result = dict(task=task, task_sha256=b._task_identity(task), status="started",
                  future_used=False, truth_used=False, started_utc=b._utc())
    began_all = perf_counter()
    try:
        for relative, expected in task["kernel_sha256"].items():
            if b._digest(b.STAGE/relative) != expected:
                raise ValueError("cross-warm kernel changed after snapshot: "+relative)
        if b._digest(task["endpoint_file"]) != task["endpoint_sha256"]:
            raise ValueError("original warm endpoint bytes changed")
        original = b._read(task["endpoint_file"])
        candidate, data, meter, options, problem, vector, _, item, reconstruction = ep._context(whitelist, task, original)
        initial = {key: np.asarray(original["attained_state"][key], float)
                   for key in ("r", "x", "source_p", "source_q")}
        options.update(max_nfev=int(task["diagnostic_max_nfev"]), multistart=1)
        common_scale = problem.scale.copy()
        result.update(initial_state_sha256=sha256(b._bytes(initial)).hexdigest(),
            effective_initial_state_sha256=sha256(b._bytes(problem.state_record(problem.start))).hexdigest(),
            common_scale=common_scale.tolist(), requested_RX_reconstruction_max_abs=reconstruction,
            initialization="saved H24 endpoint from "+task["warm_method"],
            original_warm_success=bool(original.get("solver_success")),
            fit_options=options, source_hashes=task["kernel_sha256"])
        actual_calls, counter = [], {}
        began = perf_counter()
        if task["algorithm"] == "ref_jac":
            with _reference_precision(TARGET, actual_calls):
                fitted = fitting.fit_rx_inputs(candidate, data, meter, initial=initial,
                    seed=task["seed"], call_counter=counter, **options)
        elif task["algorithm"] == "block_jac_gap":
            with _common_gradient_stop(common_scale):
                fitted = nonlinear.fit_rx_blocks(candidate, data, meter, initial=initial,
                    seed=task["seed"], call_counter=counter, scale_mode="jac",
                    inner_acceptance="certified_gap", stationarity_tol=TARGET, **options)
        else:
            raise ValueError("undeclared cross-warm method")
        seconds = perf_counter()-began
        audit = b.audit_fit(candidate, data, meter, options, fitted, fixed_lines=False)
        state = dict(r=fitted.r, x=fitted.x, source_p=fitted.source_p, source_q=fitted.source_q)
        local = ca.local_response(candidate, data, meter, state, base_kw=item["base_kw"])
        result.update(status="returned", solver_success=bool(fitted.success), fit=fitted.summary(),
            attained_state=state, audit=audit, local_response=local, counters=counter,
            actual_scipy_calls=actual_calls, timings=dict(fit_seconds=seconds),
            independent_diagnostic_budget=True, old_endpoint_bytes_unchanged=True)
    except Exception as error:
        result.update(status="exception", exception=type(error).__name__+": "+str(error),
                      traceback=traceback.format_exc())
    result.update(wall_seconds=perf_counter()-began_all, finished_utc=b._utc())
    return b._serial(result)


@contextmanager
def _dispatch():
    original = b._worker_command
    def command(whitelist, task_file, result_file):
        return [sys.executable, "-m", "research_experiments.mobile_ev_mle.ac_block_fit.crosswarm_audit",
                "worker", "--whitelist", str(whitelist), "--task", str(task_file), "--result", str(result_file)]
    b._worker_command = command
    try:
        yield
    finally:
        b._worker_command = original


def summarize(output):
    output = Path(output); plan = b._read(output/"crosswarm_plan.json")
    jobs, grouped = [], {}
    for task in plan["tasks"]:
        path = output/"tasks"/task["task_id"]/"result.json"
        if not path.exists():
            jobs.append(dict(task=task, status="missing")); continue
        envelope = b._read(path); jobs.append(envelope)
        if envelope.get("worker"):
            grouped[(task["candidate_name"], task["warm_method"], task["algorithm"])] = envelope["worker"]
    same_start, across_starts = [], []
    for name in NAMES:
        for warm in METHODS:
            pair = [grouped.get((name, warm, method)) for method in METHODS]
            entry = dict(candidate=name, warm_method=warm)
            if all(pair):
                entry.update(same_requested_initial_state=pair[0].get("initial_state_sha256") == pair[1].get("initial_state_sha256"),
                    same_effective_initial_state=pair[0].get("effective_initial_state_sha256") == pair[1].get("effective_initial_state_sha256"),
                    same_common_scale=pair[0].get("common_scale") == pair[1].get("common_scale"),
                    comparison=ca.compare_candidate(*pair))
            else:
                entry["reason"] = "missing returned endpoint; retain timeout/failure"
            same_start.append(entry)
        for method in METHODS:
            pair = [grouped.get((name, warm, method)) for warm in METHODS]
            entry = dict(candidate=name, method=method)
            if all(pair):
                entry["comparison"] = ca.compare_candidate(*pair)
            else:
                entry["reason"] = "missing returned endpoint; retain timeout/failure"
            across_starts.append(entry)
    indicators = []
    for warm in METHODS:
        record = dict(warm_method=warm)
        for method in METHODS:
            pair = [grouped.get((name, warm, method)) for name in NAMES]
            if all(row and "whitened_mu" in row.get("local_response", {}) for row in pair):
                record[method] = ca.pair_metric(*pair)
        if all(method in record for method in METHODS):
            for key in ("G_SQ_nonzero_power_gradient", "G_AQ_nonzero_power_gradient",
                        "local_square_separation_coefficient"):
                record[key+"_comparison"] = ca._relative(record[METHODS[1]][key], record[METHODS[0]][key])
        indicators.append(record)
    qualified = sum(ca._numerical_usable(job["worker"]) for job in jobs if job.get("worker"))
    result = dict(status="crosswarm_numerical_review", expected_fit_denominator=8,
        recorded_job_count=len(jobs), precision_qualified=qualified,
        timeouts=sum(job["status"] == "timeout" for job in jobs),
        exceptions=sum(job["status"] in ("exception", "worker_failed") for job in jobs),
        incomplete=sum(job["status"] == "missing" for job in jobs),
        same_start_solver_comparisons=same_start, same_solver_across_start_comparisons=across_starts,
        production_proxy_nonzero_indicator_comparison=indicators,
        common_pg_target=TARGET, old_failures_retained=True,
        H192_expansion_allowed=False, scientific_EV_efficacy_evaluated=False)
    b._write(output/"crosswarm_summary.json", result)
    return result


def execute(output):
    output = Path(output); plan = b._read(output/"crosswarm_plan.json")
    with _dispatch():
        b.run_matrix(plan["historical_whitelist"], plan["tasks"], output,
                     joint_timeout=plan["child_wall_timeout"], fixed_timeout=plan["child_wall_timeout"])
    return summarize(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_cli = sub.add_parser("prepare")
    prepare_cli.add_argument("--source", required=True, type=Path)
    prepare_cli.add_argument("--whitelist", required=True, type=Path)
    prepare_cli.add_argument("--output", required=True, type=Path)
    prepare_cli.add_argument("--timeout", type=float, default=60.)
    for name in ("run", "summarize"):
        command = sub.add_parser(name); command.add_argument("--output", required=True, type=Path)
    single = sub.add_parser("worker")
    single.add_argument("--whitelist", required=True, type=Path)
    single.add_argument("--task", required=True, type=Path)
    single.add_argument("--result", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.source, args.whitelist, args.output, timeout=args.timeout)
        print(len(value["tasks"]), "declared fits; no fits executed")
    elif args.command == "worker":
        b._write(args.result, worker(args.whitelist, b._read(args.task)))
    elif args.command == "run":
        value = execute(args.output)
        print(value["precision_qualified"], "/8 precision qualified")
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
