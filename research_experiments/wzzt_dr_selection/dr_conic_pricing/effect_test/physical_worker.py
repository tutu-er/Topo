"""Frozen continuous-DR physical execution, reusing the original AC producer.

No topology identifier is imported. Request and source hashes are checked before
the physical producer is loaded. Only scalar/short excitation receipts are
public-safe; P/Q/V, actual factors and static truth remain in private files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import traceback


PRODUCER_MODULES = (
    "rnj_wzzt.scenario.simulation", "rnj_wzzt.scenario.profiles",
    "rnj_wzzt.scenario.settings", "rnj_wzzt.data.paper_style_case_bank",
    "rnj_wzzt.data.paper_style_terminal_lv", "rnj_wzzt.models.network",
    "rnj_wzzt.models.ac_powerflow",
)
HELPER_FILES = (
    "research_experiments/wzzt_dr_selection/dr_day_modules/day_physical_worker.py",
    "research_experiments/wzzt_dr_selection/dr_day_modules/day_schedule.py",
    "research_experiments/wzzt_dr_selection/dr_competition_bridge/coded_shapes.py",
    "research_experiments/wzzt_dr_selection/dr_competition_bridge/physical_bridge.py",
    "research_experiments/wzzt_dr_selection/end_to_end/physical_case.py",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def write_pickle(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".partial")
    temp.write_bytes(pickle.dumps(value, protocol=4))
    temp.replace(path)


def resolve(path, parent):
    p = Path(path)
    return (p if p.is_absolute() else parent/p).resolve()


def reference(value, parent, name):
    if not isinstance(value, dict) or not {"path", "sha256"} <= set(value):
        raise ValueError(name+" requires an explicit path/hash reference")
    path = resolve(value["path"], parent)
    if sha(path) != value["sha256"]:
        raise RuntimeError(name+" hash mismatch")
    return path


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name+" requires an integer >= "+str(minimum))
    return value


def load_request(request_path, expected_sha):
    """Hash gates run before plan construction or importing physical modules."""
    request_path = Path(request_path).resolve()
    if not expected_sha or sha(request_path) != expected_sha:
        raise RuntimeError("request differs from the predeclared hash")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("status") != "frozen" or request.get("phase", "test") != "test":
        raise ValueError("a frozen independent-test request is required")
    parent = request_path.parent
    for key in ("case", "record_id"):
        if not isinstance(request.get(key), str) or not request[key]:
            raise ValueError(key+" must be a nonempty string")
    request["path"] = integer(request.get("path"), "path")
    request["replicate_base"] = integer(request.get("replicate_base"), "replicate_base", 20000)
    labels = request.get("labels")
    if (not isinstance(labels, list) or not labels or len(set(labels)) != len(labels)
            or any(isinstance(x, bool) or not isinstance(x, int) for x in labels)):
        raise ValueError("explicit unique integer terminal labels required")
    root = integer(request.get("root_label"), "root_label")
    if root in labels:
        raise ValueError("root cannot be a terminal")
    base_kw = float(request.get("base_kw", float("nan")))
    if not math.isfinite(base_kw) or base_kw <= 0:
        raise ValueError("positive finite power base required")
    request["base_kw"] = base_kw
    profiles = request.get("event_profiles")
    if (not isinstance(profiles, list) or len(profiles) != 4
            or any(isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1, 2) for v in profiles)):
        raise ValueError("four existing natural source-profile indices required")
    factors = request.get("factors_by_event")
    if (not isinstance(factors, list) or len(factors) != 4
            or any(not isinstance(z, list) or len(z) != 2 for z in factors)
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or abs(v) > 1 for z in factors for v in z)):
        raise ValueError("four frozen two-factor realizations in [-1,1] required")
    sigma_v = float(request.get("voltage_sigma", float("nan")))
    sigma_pq = float(request.get("pq_sigma", .005))
    if sigma_v not in (.0002, .0005) or sigma_pq != .005:
        raise ValueError("keep the predeclared original .0002/.0005 V and .005 P/Q noise")
    request["voltage_sigma"], request["pq_sigma"] = sigma_v, sigma_pq
    budget = float(request.get("incentive_budget", 1.6))
    if budget != 1.6:
        raise ValueError("the frozen four-event hard budget must remain 1.6")
    request["incentive_budget"] = budget
    protocol_path = reference(request.get("protocol"), parent, "protocol")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise ValueError("protocol must be frozen before actual data generation")
    request["plan_path"] = str(reference(request.get("plan_packet"), parent, "price plan"))
    request["problem_path"] = str(reference(request.get("problem"), parent, "planning problem"))
    source_root = resolve(request["source_root"], parent)
    repo_root = resolve(request["repo_root"], parent)
    source_hashes = request.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("source hashes must be frozen explicitly")
    checked = {}
    for file, digest in source_hashes.items():
        path = resolve(file, parent)
        if str(path) in checked and checked[str(path)] != digest:
            raise ValueError("conflicting source aliases")
        if sha(path) != digest:
            raise RuntimeError("source hash mismatch: "+str(path))
        checked[str(path)] = digest
    mandatory = [Path(__file__).resolve()]+[repo_root/file for file in HELPER_FILES]
    mandatory += [source_root.joinpath(*name.split(".")).with_suffix(".py") for name in PRODUCER_MODULES]
    if any(str(path.resolve()) not in checked for path in mandatory):
        raise ValueError("frozen sources omit a physical/resource/worker dependency")
    if isinstance(protocol.get("source_sha256"), dict):
        protocol_sources = {str(resolve(p, protocol_path.parent)): h for p, h in protocol["source_sha256"].items()}
        if any(protocol_sources.get(file) != digest for file, digest in checked.items()):
            raise ValueError("request sources differ from protocol-bound sources")
    metadata = dict(request.get("source_metadata", {}))
    metadata["path"] = str(reference(metadata, parent, "source metadata"))
    request.update(source_root=str(source_root), repo_root=str(repo_root), source_metadata=metadata,
                   checked_source_sha256=checked, protocol_path=str(protocol_path), request_path=str(request_path))
    return request


def load_legacy(repo_root):
    path = Path(repo_root)/HELPER_FILES[3]
    spec = importlib.util.spec_from_file_location("_continuous_effects_original_physics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("original physical bridge unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_plan(request):
    """Resolve continuous actual amplitudes and audit device/payment constraints.

    This is a pure resource audit and reads no future, voltage, or true graph.
    It never passes a response multiplier larger than one to the old rho API.
    """
    import numpy as np
    sys.path.insert(0, request["repo_root"])
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import (
        Battery, DayModule, ResourceDay, evaluate_schedule,
    )
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_physical_worker import node_increment_arrays
    plan = json.loads(Path(request["plan_path"]).read_text(encoding="utf-8"))
    keys = ("resources_by_event", "modules_by_event", "expected_amplitudes_by_event",
            "prices_by_event", "response_factors_by_event")
    if any(not isinstance(plan.get(key), list) or len(plan[key]) != 4 for key in keys):
        raise ValueError("continuous plan requires four resource/module/amplitude/price/factor lists")
    events, payment, safe_payment, expected_payment = [], 0., 0., 0.
    mapping = None
    for event in range(4):
        modules = [DayModule(**source) for source in plan["modules_by_event"][event]]
        n = len(modules)
        x = np.asarray(plan["expected_amplitudes_by_event"][event], float)
        prices = np.asarray(plan["prices_by_event"][event], float)
        d = np.asarray(plan["response_factors_by_event"][event], float).reshape(n, 2)
        if (x.shape != (n,) or prices.shape != (n,) or not np.isfinite(x).all()
                or not np.isfinite(prices).all() or not np.isfinite(d).all()
                or (x < 0).any() or (prices < 0).any() or (prices > 2+1e-10).any()
                or (np.abs(d).sum(axis=1) > 1).any()):
            raise ValueError("invalid frozen continuous amplitudes, prices or response factors")
        if not np.allclose(x, .1*prices, rtol=0, atol=1e-8):
            raise ValueError("mean response and price must use the declared elasticity .1")
        z = np.asarray(request["factors_by_event"][event], float)
        actual = x*(1+d@z)
        e = np.zeros(n)
        quotes = {}
        for j, module in enumerate(modules):
            if len(module.profiles) != 1:
                raise ValueError("each continuous-price contract must have one independently metered resource")
            resource_id, profile = next(iter(module.profiles.items()))
            e[j] = .25*float(np.maximum(-np.asarray(profile, float), 0.).sum())
            if e[j] <= 0:
                raise ValueError("continuous price contract must have nonzero compensated reduction")
            if x[j] > 0:
                if resource_id in quotes:
                    raise ValueError("multiple positive contract types for one physical resource")
                quotes[resource_id] = float(prices[j])
        resources = []
        for source in plan["resources_by_event"][event]:
            source = dict(source)
            if source.get("battery") is not None:
                source["battery"] = Battery(**source["battery"])
            source["incentive_per_kwh"] = quotes.get(source["resource_id"], 0.)
            resources.append(ResourceDay(**source))
        pairs = {r.resource_id: int(r.node_label) for r in resources}
        if not resources or len(pairs) != len(resources) or set(quotes)-set(pairs):
            raise ValueError("resource declarations do not match contract resources")
        if mapping is not None and pairs != mapping:
            raise ValueError("resource positions change across events")
        mapping = pairs
        # Passing actual amplitudes (with rho=None) preserves the generic
        # resource evaluator. Actual response may exceed the expected amplitude.
        audit = evaluate_schedule(resources, modules, actual.tolist(), rho=None)
        mean_audit = evaluate_schedule(resources, modules, x.tolist(), rho=None)
        node_increment_arrays(audit, request["labels"])
        actual_cost = float((e*prices)@actual)
        mean_cost = float((e*prices)@x)
        safe_cost = float((e*prices*x)@(1+np.abs(d).sum(axis=1)))
        if not math.isclose(actual_cost, audit["total_incentive_payment"], rel_tol=0, abs_tol=1e-9):
            raise AssertionError("per-contract price/energy billing differs from net-resource settlement")
        if not math.isclose(mean_cost, mean_audit["total_incentive_payment"], rel_tol=0, abs_tol=1e-9):
            raise AssertionError("mean-plan billing mismatch")
        if actual_cost < -1e-12 or actual_cost > safe_cost+1e-9:
            raise AssertionError("actual continuous payment exceeds the analytical hard bound")
        payment += actual_cost
        expected_payment += mean_cost
        safe_payment += safe_cost
        events.append(dict(audit=audit, expected_audit=mean_audit, resources=resources,
                           modules=modules, expected_amplitudes=x, actual_amplitudes=actual,
                           prices=prices, factors=z, response_factors=d,
                           actual_payment=actual_cost, expected_payment=mean_cost,
                           worst_payment=safe_cost))
    if safe_payment > request["incentive_budget"]+1e-8 or payment > request["incentive_budget"]+1e-8:
        raise ValueError("whole-path continuous-price payment exceeds frozen hard budget")
    declared = plan.get("hard_budget_bound")
    if declared is not None and abs(float(declared)-safe_payment) > 1e-7:
        raise AssertionError("plan hard budget bound disagrees with independent contract replay")
    return plan, events, dict(actual_payment=payment, expected_payment=expected_payment,
                             worst_payment=safe_payment, budget_pass=True)


def excitation(request, plan, physical_events, observed_events):
    """Evaluate explicit frozen excitation coefficients, never true topology."""
    import numpy as np
    problem = json.loads(Path(request["problem_path"]).read_text(encoding="utf-8"))
    g = np.asarray(problem["benefit_directions"], float)
    beta = np.asarray(problem["benefit_offsets"], float)
    physical, observed, natural = [], [], []
    for truth, scene in zip(physical_events, observed_events):
        physical.extend((np.asarray(truth["P_pu"]).ravel()*request["base_kw"],
                         np.asarray(truth["Q_pu"]).ravel()*request["base_kw"]))
        observed.extend((scene["P_terminal"].to_numpy().ravel()*request["base_kw"],
                         scene["Q_terminal"].to_numpy().ravel()*request["base_kw"]))
        natural.extend((np.asarray(truth["natural_P_pu"]).ravel()*request["base_kw"],
                        np.asarray(truth["natural_Q_pu"]).ravel()*request["base_kw"]))
    y, ym, y0 = np.concatenate(physical), np.concatenate(observed), np.concatenate(natural)
    if g.ndim != 2 or g.shape[1] != y.size or beta.shape != (g.shape[0],):
        raise ValueError("frozen excitation coefficients disagree with event-block P/Q order")
    mode = problem.get("benefit_mode", "total")
    if mode == "incremental":
        physical_value, observed_value = g@(y-y0), g@(ym-y0)
        observed_reference = "evaluator-only paired noiseless natural P/Q"
    elif mode == "total":
        physical_value, observed_value = beta+g@y, beta+g@ym
        observed_reference = "direct measured P/Q"
    else:
        raise ValueError("unsupported excitation mode")
    threshold = request.get("excitation_threshold", plan.get("excitation_threshold",
                            plan.get("achieved_benefit_lower_bound")))
    result = dict(excitation_physical=physical_value.tolist(), excitation_observed=observed_value.tolist(),
                  excitation_mode=mode, excitation_threshold=None,
                  excitation_observed_reference=observed_reference,
                  excitation_event_order="four event blocks, each P then Q time-major [slot,node]",
                  excitation_probability_scope="declared excitation, not final topology recovery")
    if threshold is not None:
        threshold = float(threshold)
        if not math.isfinite(threshold):
            raise ValueError("excitation threshold must be finite")
        result.update(excitation_threshold=threshold,
                      excitation_physical_joint_pass=bool(np.all(physical_value >= threshold)),
                      excitation_observed_joint_pass=bool(np.all(observed_value >= threshold)))
    return result


def run(request, output, receipt):
    import numpy as np
    import pandas as pd
    plan, audits, payments = audit_plan(request)
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_physical_worker import (
        node_increment_arrays, scalar_resource_report,
    )
    legacy = load_legacy(request["repo_root"])
    metadata = legacy.frozen_source_metadata(request, request["case"])
    if metadata["root_label"] != request["root_label"] or abs(metadata["base_kw"]-request["base_kw"]) > 1e-12:
        raise ValueError("request root/power base differs from frozen metadata")
    receipt.update(**payments, phase="test", case=request["case"], record_id=request["record_id"], path=request["path"],
                   voltage_sigma=request["voltage_sigma"], pq_sigma=request["pq_sigma"],
                   base_kw=metadata["base_kw"], base_kv=metadata["base_kv"], base_mva=metadata["base_mva"],
                   source_sha256=request["checked_source_sha256"], protocol_sha256=request["protocol"]["sha256"],
                   plan_packet_sha256=request["plan_packet"]["sha256"], problem_sha256=request["problem"]["sha256"],
                   quote_qualification_pass=all(a["audit"]["resource_feasible"] and a["expected_audit"]["resource_feasible"] for a in audits),
                   response_model="frozen continuous correlated whole-resource-day amplitude x*(1+Dz)",
                   response_calibrated=False, no_clipping=True, no_resampling=True,
                   actual_payment_scope="contracted response payment retained even if AC/measurement execution fails",
                   physical_background_scope="unchanged original source producer; forecast-law mismatch is evaluated separately")
    write_json(output/"events_receipt.json", receipt)
    sim, runtime = legacy._simulation_runtime(Path(request["source_root"]))
    for name in PRODUCER_MODULES:
        source = Path(request["source_root"]).joinpath(*name.split(".")).with_suffix(".py").resolve()
        if runtime[name] != request["checked_source_sha256"][str(source)]:
            raise RuntimeError("loaded physical producer hash differs from protocol")
    checks_function, checks_source = legacy._physical_check_function(Path(request["repo_root"]))
    if sha(checks_source) != request["checked_source_sha256"][str(checks_source.resolve())]:
        raise RuntimeError("loaded physical checks differ from protocol")
    context = sim._prepare_case_context(request["case"], 96, scenario_suite="reference",
        root_observation="noisy", root_meter_noise_rel=request["voltage_sigma"],
        root_sigma=.003, impedance_scale=1.)
    legacy.validate_context_metadata(context, metadata, request["labels"])
    context.net.check_terminal_load_only(strict=True)
    receipt["runtime_sha256"] = runtime
    training, validation, truth, observed_all = [], [], [], []
    for event, detail in enumerate(audits):
        audit = detail["audit"]
        record = receipt["event_records"][event]
        record.update(status="started", role="training" if event < 2 else "validation",
                      replicate=request["replicate_base"]+10*request["path"]+event,
                      profile_index=request["event_profiles"][event],
                      resource_reports=scalar_resource_report(audit),
                      resource_feasible=bool(audit["resource_feasible"]),
                      maximum_resource_violation=float(audit["maximum_violation"]),
                      actual_payment=detail["actual_payment"], worst_payment=detail["worst_payment"],
                      expected_payment=detail["expected_payment"],
                      independent_resource_billing_pass=True)
        private_event = dict(event=event, status="started", resource_audit=audit,
                             actual_amplitudes=detail["actual_amplitudes"],
                             expected_amplitudes=detail["expected_amplitudes"],
                             prices=detail["prices"], factors=detail["factors"])
        truth.append(private_event)
        try:
            if not audit["resource_feasible"] or not detail["expected_audit"]["resource_feasible"]:
                raise ValueError("expected or actual continuous response violates declared resource constraints")
            base = sim._build_base_trajectory(context, record["profile_index"], record["replicate"])
            p0, q0 = base.p_base.to_numpy(), base.q_base.to_numpy()
            dp, dq = node_increment_arrays(audit, request["labels"])
            if p0.shape != dp.shape or q0.shape != dq.shape:
                raise ValueError("natural power order differs from frozen terminal increments")
            physical_p, physical_q = p0+dp/metadata["base_kw"], q0+dq/metadata["base_kw"]
            p_error = float(np.max(np.abs((physical_p-p0)*metadata["base_kw"]-dp), initial=0.))
            q_error = float(np.max(np.abs((physical_q-q0)*metadata["base_kw"]-dq), initial=0.))
            if not np.isfinite(physical_p).all() or not np.isfinite(physical_q).all() or max(p_error, q_error) > 1e-9:
                raise AssertionError("physical increment identity failed; baseline may have been added twice")
            record.update(increment_only_bridge_pass=True, max_p_increment_error_kw=p_error,
                          max_q_increment_error_kvar=q_error, profile_seed=int(base.shifted_seed))
            pframe = pd.DataFrame(physical_p, index=base.p_base.index, columns=request["labels"])
            qframe = pd.DataFrame(physical_q, index=base.q_base.index, columns=request["labels"])
            ac, voltage = sim._solve_physical_ac(context, pframe, qframe, base.root_true)
            measured = sim._observe_terminal_channels(context, base, pframe, qframe, voltage,
                pq_noise_rel=request["pq_sigma"], v_noise_rel=request["voltage_sigma"])
            scene = legacy.scene(measured["p_measured"].to_numpy(), measured["q_measured"].to_numpy(),
                measured["voltage_measured"].to_numpy(), measured["root_observed"].to_numpy(),
                request["labels"], "continuous_"+request["record_id"]+"_e"+str(event), 6+event)
            scene["actual_event_id"] = event
            (training if event < 2 else validation).append(scene)
            observed_all.append(scene)
            equations = checks_function(context.net, pframe, qframe, ac, (.9, 1.1), 1e-8)
            converged = int(ac["converged"].sum())
            checks = dict(electrical_equations_pass=bool(equations["electrical_equations_pass"]),
                          voltage_limits_pass=bool(equations["voltage_limits_pass"]),
                          voltage_violation_cells=int(equations["voltage_violation_cells"]),
                          ac_converged_samples=converged, ac_all_96_converged=converged == 96,
                          voltage_min_pu=float(voltage.to_numpy().min()), voltage_max_pu=float(voltage.to_numpy().max()),
                          max_nodal_p_residual_pu=float(equations["max_nodal_p_residual_pu"]),
                          max_nodal_q_residual_pu=float(equations["max_nodal_q_residual_pu"]),
                          max_distflow_identity_error_pu2=float(equations["max_distflow_identity_error_pu2"]),
                          branch_ampacity_checked=False)
            passed = checks["electrical_equations_pass"] and checks["voltage_limits_pass"] and checks["ac_all_96_converged"]
            record.update(checks=checks, status="complete" if passed else "physical_checks_failed",
                          delivery_status="complete" if passed else "unresolved")
            private_event.update(P_pu=physical_p, Q_pu=physical_q, root_pu=base.root_true.to_numpy(),
                                 natural_P_pu=p0, natural_Q_pu=q0, delta_kw=dp, delta_kvar=dq,
                                 V_pu=voltage.to_numpy(), ac=ac)
        except Exception as error:
            record.update(status="physical_failed", delivery_status="unresolved",
                          error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        private_event["status"] = record["status"]
        write_pickle(output/"physical_truth.pkl", dict(events=truth, partial=len(truth) != 4))
        write_json(output/"events_receipt.json", receipt)
    complete = all(r["status"] == "complete" for r in receipt["event_records"])
    receipt.update(status="complete" if complete else "physical_failed", completed_utc=now(),
                   physical_events_complete=sum(r["status"] == "complete" for r in receipt["event_records"]),
                   observed_training_days=len(training), observed_validation_days=len(validation),
                   training_event_ids=[s["actual_event_id"] for s in training],
                   validation_event_ids=[s["actual_event_id"] for s in validation],
                   all_operation_checks_pass=complete,
                   known_actual_payment=payments["actual_payment"], unknown_payment_events=0)
    if complete:
        receipt.update(excitation(request, plan, truth, observed_all))
    packet = dict(training=training, validation=validation, metadata=dict(
        terminal_labels=request["labels"], base_kw=metadata["base_kw"], base_kv=metadata["base_kv"],
        base_mva=metadata["base_mva"], root_label=metadata["root_label"], case=request["case"],
        interval_hours=.25, synthetic_dates_are_order_labels=True, observed_only=True,
        request_sha256=receipt["request_sha256_expected"], protocol_sha256=request["protocol"]["sha256"],
        phase="test", record_id=request["record_id"], path=request["path"], status=receipt["status"],
        training_event_ids=receipt["training_event_ids"], validation_event_ids=receipt["validation_event_ids"]))
    write_pickle(output/"packet.pkl", packet)
    # This evaluator-only artifact is not part of the observation packet and
    # must only be opened by the root evaluator after identifier outputs freeze.
    evaluation = legacy.evaluate_supports([], context.net)
    write_json(output/"truth_evaluation.json", dict(true_supports=evaluation["missed_blocks"],
        terminal_labels=request["labels"], full_effective_truth_blocks=evaluation["full_effective_truth_blocks"],
        zero_impedance_merged=True, arbitrarily_weak_positive_lines_retained=True,
        use="evaluator only after final identifier result freeze; never planning/fit input"))
    write_pickle(output/"physical_truth.pkl", dict(events=truth))
    for filename in ("packet.pkl", "physical_truth.pkl", "truth_evaluation.json"):
        p = output/filename
        receipt["private_files"][filename] = dict(sha256=sha(p), bytes=p.stat().st_size)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        print(json.dumps(dict(status="existing_output_retained")), flush=True)
        return 3
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(status="started", started_utc=now(), request_sha256_expected=args.expected_sha,
                   physical_events_expected=4, physical_event_rows_expected=96,
                   event_records=[dict(event=e, status="not_attempted", actual_payment=None) for e in range(4)],
                   private_files={}, actual_payment=None, known_actual_payment=0., unknown_payment_events=4,
                   no_clipping=True, no_resampling=True, automatic_retry=False,
                   identifier_run=False, old_test_or_truth_files_read=False,
                   public_receipt_contains_trajectories=False)
    write_json(output/"events_receipt.json", receipt)
    try:
        request = load_request(args.request, args.expected_sha)
        receipt = run(request, output, receipt)
    except Exception as error:
        receipt.update(status="worker_failed", completed_utc=now(), error_type=type(error).__name__,
                       message=str(error), traceback=traceback.format_exc(), failure_retained=True)
    write_json(output/"events_receipt.json", receipt)
    print(json.dumps({key: receipt.get(key) for key in ("status", "physical_events_complete", "actual_payment",
        "known_actual_payment", "unknown_payment_events", "worst_payment", "budget_pass", "all_operation_checks_pass")},
        allow_nan=False), flush=True)
    return 0 if receipt["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
