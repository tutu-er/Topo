"""Saved full-flow comparison: standard-library reads only, no model execution.

The independent unit is one history.  Calibration thresholds are persisted
before an evaluation directory is opened.  Planned, failed, and pending
histories stay visible in all method and paired-comparison denominators.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from . import ev_dynamic_collect as collector

DOMAINS = ("v2g_fixed", "v2g_mobile")
PREFIXES = ("square", "nearest", "gradac", "gradi2", "pairac")
METHODS = ("none", "constant", "highlow", *(p+"_"+d for d in DOMAINS for p in PREFIXES))
RUNTIMES = ("common_history_pipeline_seconds", "planning_actual_seconds",
            "planning_standalone_seconds", "operational_future_fit_seconds",
            "end_to_end_standalone_seconds", "dependency_seconds", "soft_budget_overrun_seconds")
TOLERANCE = 1e-6


def stats(values, denominator, *, tolerance=TOLERANCE):
    valid = [float(v) for v in values if collector._finite(v)]
    if len(valid) > denominator:
        raise ValueError("numeric values exceed independent-history denominator")
    return dict(total=denominator, valid=len(valid), unavailable=denominator-len(valid),
        mean=statistics.fmean(valid) if valid else None,
        median=statistics.median(valid) if valid else None,
        minimum=min(valid) if valid else None, maximum=max(valid) if valid else None,
        positive=sum(v > tolerance for v in valid), negative=sum(v < -tolerance for v in valid),
        ties=sum(abs(v) <= tolerance for v in valid), tolerance=tolerance)


def contrasts():
    result = {}
    for domain in DOMAINS:
        for left, right in (("gradac", "square"), ("gradi2", "square"),
                            ("nearest", "square"), ("gradi2", "gradac"),
                            ("pairac", "gradac"), ("gradac", "nearest")):
            result[left+"_vs_"+right+"_"+domain] = (left+"_"+domain, right+"_"+domain)
    for prefix in PREFIXES:
        result[prefix+"_mobile_vs_fixed"] = (prefix+"_v2g_mobile", prefix+"_v2g_fixed")
    return result


def _iterations(log):
    """Keep scalar iteration evidence; raw source-profile arrays stay in trials."""
    compact = []
    for row in log.get("iterations", []) or []:
        if not isinstance(row, dict):
            compact.append(dict(iteration=row))
            continue
        item = {k:row[k] for k in ("start_index", "start_origin", "iteration", "before_score",
            "candidate_score", "after_score", "accepted", "gradient_method", "trust_kw", "failure",
            "rejection_reason", "stop_reason") if k in row}
        step = row.get("MILP", {})
        if step:
            item["MILP"] = {k:step[k] for k in ("success", "predicted_directional_gain",
                "attained_directional_gain", "linear_best_bound", "linear_mip_gap", "seconds", "status") if k in step}
        compact.append(item)
    return compact, dict(count=len(compact), accepted=sum(r.get("accepted") is True for r in compact),
        gradient_failures=sum("gradient source fit" in str(r.get("failure", "")) for r in compact),
        rejection_reasons=dict(Counter(r["rejection_reason"] for r in compact if r.get("rejection_reason"))),
        stop_reasons=dict(Counter(r["stop_reason"] for r in compact if r.get("stop_reason"))))


def _counters(log):
    counters = dict(log.get("counters", {}))
    events = counters.pop("cache_payment_log", [])
    if events:
        counters["cache_payment_summary"] = dict(events=len(events),
            cache_hits=sum(e.get("cache_hit") is True for e in events),
            paid_pair_fits=sum(e.get("paid_pair_fits", 0) for e in events),
            actual_seconds=sum(e.get("actual_seconds", 0.) for e in events))
    return counters


def timing_audit(trial):
    """Rebuild standalone costs from saved own costs and ancestor sets."""
    selection = trial.get("selection", {})
    timing = selection.get("timing", {})
    common = trial.get("common_runtime", {})
    if not timing or not common:
        return dict(available=False, checks=0, failures=[], reason="saved timing components unavailable")
    failures, checks = [], 0
    def equal(actual, expected, label):
        nonlocal checks
        checks += 1
        if not (collector._finite(actual) and collector._finite(expected)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10)):
            failures.append(dict(label=label, actual=actual, expected=expected))
    def ancestry(name, stack=()):
        if name in stack:
            raise ValueError("cycle in planning dependencies")
        answer = set()
        for parent in timing[name].get("dependencies", []):
            if parent not in timing:
                raise ValueError("missing planning ancestor: "+parent)
            answer.add(parent)
            answer.update(ancestry(parent, stack+(name,)))
        return answer
    setup = selection.get("public_setup_seconds", 0.)
    for name, row in timing.items():
        if name in ("none", "constant", "highlow"):
            continue
        ancestors = ancestry(name)
        expected_dep = setup+sum(timing[a]["planning_actual_seconds"] for a in ancestors)+row.get("context_seconds", 0.)
        equal(row.get("dependency_seconds"), expected_dep, name+"/dependency")
        equal(row.get("planning_standalone_seconds"), row["planning_actual_seconds"]+expected_dep, name+"/standalone")
        if "charged_ancestors" in row:
            checks += 1
            if set(row["charged_ancestors"]) != ancestors:
                failures.append(dict(label=name+"/ancestor_names", actual=row["charged_ancestors"], expected=sorted(ancestors)))
            expected_calls = dict(row.get("actual_calls", {}))
            for ancestor in ancestors:
                for key, value in timing[ancestor].get("actual_calls", {}).items():
                    expected_calls[key] = expected_calls.get(key, 0)+value
            for key, value in row.get("context_calls", {}).items():
                expected_calls[key] = expected_calls.get(key, 0)+value
            for key, value in expected_calls.items():
                equal(row.get("standalone_calls", {}).get(key, 0), value, name+"/calls/"+key)
    for method in trial.get("methods", []):
        runtime = method.get("runtime")
        if not runtime:
            continue
        name = method["name"]
        common_seconds = sum(common[k] for k in ("training_and_candidates_seconds", "validation_seconds", "forecast_seconds"))
        equal(runtime.get("common_history_pipeline_seconds"), common_seconds, name+"/common")
        equal(runtime.get("operational_future_fit_seconds"), method.get("noisy", {}).get("seconds", 0.), name+"/future")
        expected_total = (common_seconds+runtime["planning_standalone_seconds"]
            +runtime["operational_future_fit_seconds"]+runtime["final_operation_audit_seconds"])
        equal(runtime.get("end_to_end_standalone_seconds"), expected_total, name+"/end_to_end")
        expected_ac = (common["training_calls"].get("AC_batches", 0)+common["validation_calls"].get("AC_batches", 0)
            +runtime.get("standalone_calls", runtime.get("actual_calls", {})).get("AC_batches", 0)
            +method.get("noisy", {}).get("cost", {}).get("ac_calls", 0))
        equal(runtime.get("standalone_AC_batches"), expected_ac, name+"/AC_batches")
    return dict(available=True, checks=checks, failures=failures, success=not failures,
        scope="saved own costs, dependencies and arithmetic; source fitting is not rerun")


def _detail(trial, parsed, methods):
    raw = {r["name"]: r for r in trial.get("methods", [])}
    design = trial.get("selection", {}).get("methods", {})
    result = dict(seed=parsed["seed"], family=parsed["family"], status=parsed["status"],
        candidate_miss=parsed["candidate_miss"], training_top2=trial.get("training_top2", {}),
        candidates=parsed["candidate_names"], seconds=trial.get("seconds"),
        actual_solver_calls=trial.get("actual_solver_calls", {}), timing_audit=timing_audit(trial), methods={})
    for name in methods:
        row, saved = parsed["methods"][name], raw.get(name, {})
        log = design.get(name, {})
        runtime = saved.get("runtime", {})
        for field in RUNTIMES:
            value = runtime.get(field)
            if value is not None and (not collector._finite(value) or value < 0):
                raise ValueError("invalid runtime: "+name+"/"+field)
        failures = list(log.get("failures", []))
        if log.get("failure"):
            failures.append(log["failure"])
        iterations, iteration_summary = _iterations(log)
        result["methods"][name] = dict(status=row["status"], raw_status=row["raw_status"],
            M=row["M"], clean_future_M=saved.get("clean_future_M")
                if saved.get("clean_future_complete") is True else None,
            failure_reasons=row["failure_reasons"], runtime=runtime, energy=row["energy"],
            selected_origin=log.get("selected_origin"), iterations=iterations, iteration_summary=iteration_summary,
            planning_success=saved.get("planning_success"), planning_failures=failures,
            planning_counters=_counters(log), selected_score=log.get("selected_score"),
            operational_fit_cost=row["logical_future_fit_cost"],
            source_failures=row["source_failures"],
            pair_scope="training top two for planning; complete bank for evaluation"
                if name.startswith("pairac_") else "complete candidate bank")
    return result


def summarize_phase(directory, phase, *, collected=None):
    directory = Path(directory)
    freeze = collector.read_json(directory/"freeze.json")
    identity = collector._identity(freeze)
    methods = collector.expected_methods(freeze["config"])
    if set(methods) != set(METHODS) or len(methods) != len(METHODS):
        raise ValueError("full-flow freeze must declare exactly the 13 comparison methods")
    planned = [str(s) for s in freeze["config"].get(phase+"_seeds", [])]
    if not planned or len(planned) != len(set(planned)):
        raise ValueError("phase requires distinct planned history seeds")
    paths = sorted(directory.glob("trial*.json.gz"))
    if collected is None and paths:
        collected = collector.collect(directory, mode="summarize")
    records = {} if collected is None else {r["seed"]: r for r in collected["records"]}
    if not set(records) <= set(planned):
        raise ValueError("observed history seed not in frozen phase plan")
    details = {}
    for path in paths:
        trial = collector.read_json(path)
        seed = str(trial["seed"])
        if seed not in records:
            raise ValueError("raw history is absent from parsed records")
        details[seed] = _detail(trial, records[seed], methods)
        del trial
    if set(details) != set(records):
        raise ValueError("raw and parsed independent-history coverage differs")
    n = len(planned)
    pending = [s for s in planned if s not in records]
    summary = dict(directory=str(directory.resolve()), phase=phase, **identity,
        planned_seeds=planned, observed_seeds=sorted(records), pending_seeds=pending,
        planned_histories=n, observed_histories=len(records), pending_histories=len(pending),
        candidate_omissions=sum(r["candidate_miss"] for r in records.values()),
        training_top2_miss=sum(r.get("training_top2", {}).get("miss_true") is True for r in details.values()),
        training_top2_available=sum(isinstance(r.get("training_top2", {}).get("miss_true"), bool)
            for r in details.values()),
        family_counts=dict(Counter(r["family"] for r in records.values())),
        candidate_counts=dict(Counter(str(len(r["candidate_names"])) for r in records.values())),
        methods={}, paired_contrasts={}, records=details,
        freeze_sha256=hashlib.sha256((directory/"freeze.json").read_bytes()).hexdigest(),
        input_files=[] if collected is None else collected["trial_files"])
    summary["actual_history_runtime_seconds"] = stats([r["seconds"] for r in details.values()], n)
    summary["timing_audit"] = dict(available_histories=sum(r["timing_audit"]["available"] for r in details.values()),
        planned_histories=n, checks=sum(r["timing_audit"]["checks"] for r in details.values()),
        failures=[dict(seed=seed, **failure) for seed, row in details.items() for failure in row["timing_audit"]["failures"]])
    call_keys = sorted({k for r in details.values() for k in r["actual_solver_calls"]})
    summary["actual_solver_calls"] = {k:stats([r["actual_solver_calls"].get(k) for r in details.values()], n)
        for k in call_keys}
    for name in methods:
        rows = [r["methods"][name] for r in details.values()]
        summary["methods"][name] = dict(planned=n, observed=len(rows), pending=len(pending),
            complete=sum(r["status"] == "complete" for r in rows),
            failed=sum(r["status"] != "complete" for r in rows),
            raw_status_counts=dict(Counter(r["raw_status"] for r in rows)),
            M=stats([r["M"] for r in rows], n),
            clean_future_M=stats([r["clean_future_M"] for r in rows], n),
            failure_reasons=dict(Counter(v for r in rows for v in set(r["failure_reasons"]))),
            planning_failures=sum(bool(r["planning_failures"]) for r in rows),
            planning_unsuccessful=sum(r["planning_success"] is False for r in rows),
            future_source_fit_failures=sum(not r["source_failures"]["future"]["all_complete"] for r in rows),
            validation_source_fit_failures=sum(not r["source_failures"]["validation"]["all_complete"] for r in rows),
            selected_origin=dict(Counter(str(r["selected_origin"]) for r in rows)),
            runtime={k:stats([r["runtime"].get(k) for r in rows], n) for k in RUNTIMES},
            standalone_AC_batches=stats([r["runtime"].get("standalone_AC_batches") for r in rows], n),
            planning_counters={k:stats([r["planning_counters"].get(k) for r in rows], n)
                for k in ("profile_evaluations", "pair_fit_calls", "source_solver_calls", "source_total_nfev")},
            response_calls={k:stats([r["planning_counters"].get("response_calls", {}).get(k) for r in rows], n)
                for k in ("ac_value_calls", "ac_jacobian_calls", "approximate_value_calls", "approximate_jacobian_calls")},
            iterations=dict(total=sum(r["iteration_summary"]["count"] for r in rows),
                accepted=sum(r["iteration_summary"]["accepted"] for r in rows),
                gradient_failures=sum(r["iteration_summary"]["gradient_failures"] for r in rows)),
            source_kkt_by_method={backend:dict(
                successful_fits=sum(r["planning_counters"].get("source_kkt_by_method", {}).get(backend, {}).get("successful_fits", 0) for r in rows),
                scaled_max=max((r["planning_counters"].get("source_kkt_by_method", {}).get(backend, {}).get("scaled_max") for r in rows
                    if collector._finite(r["planning_counters"].get("source_kkt_by_method", {}).get(backend, {}).get("scaled_max"))), default=None),
                scaled_count_gt_1e_minus_5=sum(r["planning_counters"].get("source_kkt_by_method", {}).get(backend, {}).get("scaled_count_gt_1e_minus_5", 0) for r in rows))
                for backend in ("ac", "i2")},
            energy={k:stats([r["energy"].get(k) for r in rows], n)
                for k in ("grid_energy_kwh", "charging_kwh", "discharging_kwh", "throughput_kwh",
                          "driving_kwh", "worker_minutes", "moves")})
    for label, (left, right) in contrasts().items():
        paired = []
        for seed in planned:
            history = details.get(seed)
            a, b = (history["methods"][m] for m in (left, right)) if history else ({}, {})
            complete = (a.get("status") == b.get("status") == "complete"
                and collector._finite(a.get("M")) and collector._finite(b.get("M")))
            clean = complete and all(collector._finite(r.get("clean_future_M")) for r in (a, b))
            paired.append(dict(seed=seed, complete=complete, pending=history is None,
                delta_M=a["M"]-b["M"] if complete else None,
                delta_clean_future_M=a["clean_future_M"]-b["clean_future_M"] if clean else None,
                left_status=a.get("status"), right_status=b.get("status")))
        summary["paired_contrasts"][label] = dict(left=left, right=right, rows=paired,
            delta_M=stats([r["delta_M"] for r in paired], n),
            delta_clean_future_M=stats([r["delta_clean_future_M"] for r in paired], n))
    summary["planned_method_slots"] = n*len(methods)
    summary["complete_method_slots"] = sum(r["complete"] for r in summary["methods"].values())
    summary["failed_method_slots"] = sum(r["failed"] for r in summary["methods"].values())
    summary["pending_method_slots"] = len(pending)*len(methods)
    summary["status"] = ("partial" if pending else "complete_with_failures"
        if summary["failed_method_slots"] else "complete")
    return summary


def _save_thresholds(path, thresholds):
    encoded = (json.dumps(thresholds, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")
    path = Path(path)
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("existing output thresholds differ; preserve prior analysis")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(encoded)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                thresholds_id=thresholds["thresholds_id"])


def render(report):
    lines = ["# EV完整流程比较", "", "统计单位是一份独立历史；模型对、方法和未来重复噪声不增加样本量。",
        "固定每拓扑一组训练R/X，评价保留全部四候选。训练前两名仅用于pairac规划对照。", ""]
    for phase in ("development", "calibration", "evaluation"):
        if phase not in report:
            continue
        block = report[phase]
        lines.extend(["## "+phase, "", f"历史：{block['observed_histories']}/{block['planned_histories']}；"
            f"候选漏真：{block['candidate_omissions']}；训练前二漏真："
            f"{block['training_top2_miss']}/{block['training_top2_available']}。", "",
            "|方法|完整/计划|执行或拟合失败|规划未成功|M均值|M中位数|M范围|独立运行总秒数均值|",
            "|---|---:|---:|---:|---:|---:|---|---:|"])
        def fmt(v): return "—" if v is None else format(v, ".6g")
        for name in METHODS:
            row = block["methods"][name]
            m = row["M"]
            lines.append(f"|{name}|{row['complete']}/{row['planned']}|{row['failed']}|{row['planning_unsuccessful']}|{fmt(m['mean'])}|"
                f"{fmt(m['median'])}|{fmt(m['minimum'])} .. {fmt(m['maximum'])}|"
                f"{fmt(row['runtime']['end_to_end_standalone_seconds']['mean'])}|")
        lines.extend(["", "差值为左方法减右方法；胜/平/负容差为1e-6。", "",
            "|配对比较|有效/计划|差值均值|中位数|胜/平/负|", "|---|---:|---:|---:|---:|"])
        for name, row in block["paired_contrasts"].items():
            s = row["delta_M"]
            lines.append(f"|{name}|{s['valid']}/{s['total']}|{fmt(s['mean'])}|{fmt(s['median'])}|"
                f"{s['positive']}/{s['ties']}/{s['negative']}|")
        lines.append("")
    if "calibration_thresholds" in report:
        lines.extend(["## 方法内校准与独立评价", "", "阈值在读取评价记录前保存。校准误差水平为"
            +str(report["alpha"])+"，逐方法、逐真实结构族计算；未对多种策略同时作错误率保证。", "",
            "|方法|star|clade_01|clade_02|clade_12|", "|---|---:|---:|---:|---:|"])
        for name in METHODS:
            values = [report["calibration_thresholds"][name][f]["threshold"] for f in collector.FAMILIES]
            lines.append("|"+name+"|"+"|".join(str(v) for v in values)+"|")
        if report.get("decisions_by_method"):
            lines.extend(["", "|方法|评价历史|误拒绝|唯一正确|唯一错误|未决|计算失败保留全部|",
                "|---|---:|---:|---:|---:|---:|---:|"])
            for name in METHODS:
                row = report["decisions_by_method"][name]["overall"]
                lines.append("|"+name+"|"+"|".join(str(row[k]) for k in
                    ("total", "false_reject", "unique_correct", "unique_incorrect", "unresolved", "failed_keep_all"))+"|")
    lines.extend(["", "M为实际验证段与未来段SSE差，不是拓扑正确概率。失败、负例和逐历史差值保存在JSON中。",
        "规划失败后执行可行基线的记录，仍可得到完整实际损失；规划失败另列，不被成功回代覆盖。",
        "运行时间分别保存公共历史处理、规划实际时间、依赖成本、独立运行规划与完整运行时间；缓存共享不重复记为实际开销。", ""])
    return "\n".join(lines)


def analyze(output_directory, *, development=None, calibration=None, evaluation=None,
            alpha=.1, expected_per_family=9):
    out = Path(output_directory)
    directories = [Path(p) for p in (development, calibration, evaluation) if p is not None]
    if not directories or (development is not None and (calibration is not None or evaluation is not None)):
        raise ValueError("choose development or calibration/evaluation")
    if any(out.resolve() == p.resolve() for p in directories):
        raise ValueError("analysis output must differ from raw trial directories")
    report = dict(schema="ev_gradient_full_flow_saved_analysis_v1", methods=list(METHODS),
        independent_unit="one history", two_model_planner_uses_all_candidates=False,
        actual_evaluation_uses_complete_bank=True,
        topology_probability_reported=False, tolerance=TOLERANCE,
        analyzer=dict(new_AC_calls=0, new_fit_calls=0, new_selector_calls=0,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            collector=collector.collector_manifest()))
    if development is not None:
        report["development"] = summarize_phase(development, "development")
        report["status"] = report["development"]["status"]
    else:
        if calibration is None:
            raise ValueError("formal analysis requires calibration directory")
        # This collection call and planned-seed audit precede every evaluation read.
        thresholds = collector.collect(calibration, mode="calibrate", alpha=alpha,
            expected_per_family=expected_per_family)
        cal = summarize_phase(calibration, "calibration", collected=thresholds)
        if cal["pending_seeds"]:
            raise ValueError("calibration is missing planned independent histories")
        saved = _save_thresholds(out/"thresholds.json", thresholds)
        report.update(calibration=cal, alpha=alpha, threshold_file=saved,
            thresholds_id=thresholds["thresholds_id"], calibration_thresholds=thresholds["calibration"],
            thresholds_persisted_before_evaluation_read=True)
        if evaluation is not None:
            report["evaluation"] = summarize_phase(evaluation, "evaluation")
            if any(report["evaluation"][k] != cal[k] for k in
                   ("policy_id", "config_sha256", "source_manifest_sha256")):
                raise ValueError("calibration/evaluation frozen policy differs")
            if report["evaluation"]["observed_histories"]:
                evaluated = collector.collect(evaluation, mode="evaluate", thresholds=out/"thresholds.json")
                collector.write_json(out/"evaluation.json", evaluated)
                report["decisions_by_method"] = evaluated["evaluation"]
                report["decisions"] = evaluated["decisions"]
            if hashlib.sha256((out/"thresholds.json").read_bytes()).hexdigest() != saved["sha256"]:
                raise ValueError("calibration thresholds changed during evaluation")
            report["status"] = report["evaluation"]["status"]
            if report["status"] == "complete" and cal["status"] != "complete":
                report["status"] = "complete_with_failures"
        else:
            report["status"] = "calibration_only"
    collector.write_json(out/"summary.json", report)
    (out/"summary.md").write_text(render(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--expected-per-family", type=int, default=9)
    args = parser.parse_args()
    report = analyze(args.output, development=args.development, calibration=args.calibration,
        evaluation=args.evaluation, alpha=args.alpha, expected_per_family=args.expected_per_family)
    print(json.dumps(dict(status=report["status"], output=str(args.output.resolve())), ensure_ascii=False))


if __name__ == "__main__":
    main()
