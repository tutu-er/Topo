"""Independent descriptive analysis of frozen gradient-price trial records.

Reads raw or compact records without re-fitting, selecting, or changing gates.
Signed margins use full SSE. Prediction scores retain their half-SSE units.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .likelihood import RadialCandidate

METHODS = ("none", "constant_fixed", "optimized_fixed", "optimized_random", "optimized_designed", "pq_only_designed")
PLAN_ARRAYS = ("p", "q", "sites", "soc")
EDGE_CLASSES = ("common_full", "leaf", "internal")
MATERIAL_GAIN = 1e-6


def _finite(value):
    if isinstance(value, (int, float, np.number)) and np.isfinite(value):
        return float(value)
    return None


def _stats(values):
    finite = [v for x in values if (v := _finite(x)) is not None]
    return dict(n=len(finite), mean=float(np.mean(finite)) if finite else None,
                median=float(np.median(finite)) if finite else None,
                minimum=min(finite) if finite else None, maximum=max(finite) if finite else None)


def _record_map(trial):
    records = trial.get("records", [])
    if isinstance(records, dict):
        return records
    return {r.get("method"):r for r in records if isinstance(r, dict) and r.get("method") is not None}


def _fits(row):
    values = row.get("fits", [])
    if isinstance(values, dict):
        return {name:value for name, value in values.items() if isinstance(value, dict)}
    return {f.get("name", f.get("candidate")):f for f in values if isinstance(f, dict)}


def signed_actual_margin(trial, row):
    """min(wrong full-SSE loss)-true loss; preserve negative/positive values."""
    if row is None or row.get("status") != "complete":
        return None
    truth = row.get("true_name", trial.get("true_name"))
    if truth is None or row.get("true_covered", trial.get("true_covered")) is False:
        return None
    losses = row.get("losses", {})
    if not isinstance(losses, dict) or truth not in losses:
        return None
    expected = trial.get("candidate_names", list(losses))
    if any(name not in losses or _finite(losses[name]) is None for name in expected):
        return None
    true_loss = _finite(losses[truth])
    wrong = [v for name, loss in losses.items() if name != truth and (v := _finite(loss)) is not None]
    return None if true_loss is None or not wrong else float(min(wrong)-true_loss)


def paired_correlation(x, y):
    """Descriptive correlations only; constant inputs return NA with a reason."""
    pairs = [(a, b) for xv, yv in zip(x, y)
             if (a := _finite(xv)) is not None and (b := _finite(yv)) is not None]
    n = len(pairs)
    result = dict(n=n, pearson=None, spearman=None, unavailable_reason=None)
    if n < 2:
        result["unavailable_reason"] = "fewer_than_two_paired_histories"
        return result
    a, b = np.asarray(pairs, float).T
    constant = lambda z: np.ptp(z) <= 1e-12*max(1., float(np.max(abs(z))))
    if constant(a) or constant(b):
        result["unavailable_reason"] = "constant_prediction" if constant(a) else "constant_observed_effect"
        return result
    result["pearson"] = float(np.corrcoef(a, b)[0, 1])
    result["spearman"] = float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
    return result


def compare_plans_exact(full, pq):
    """Compare decoded physical arrays exactly, not solver diagnostics."""
    result = dict(comparable=False, all_arrays_equal=None, same_site=None,
                  field_equal={k:None for k in PLAN_ARRAYS}, missing_fields=[])
    if not isinstance(full, dict) or not isinstance(pq, dict):
        result["missing_fields"] = ["full_or_pq_plan"]
        return result
    for field in PLAN_ARRAYS:
        if field not in full or field not in pq:
            result["missing_fields"].append(field)
        else:
            a, b = np.asarray(full[field]), np.asarray(pq[field])
            result["field_equal"][field] = bool(a.shape == b.shape and np.array_equal(a, b))
    if "site" in full and "site" in pq:
        result["same_site"] = full["site"] == pq["site"]
    result["comparable"] = not result["missing_fields"]
    if result["comparable"]:
        result["all_arrays_equal"] = bool(all(result["field_equal"].values()))
    return result


def _joint_fit_seconds(row):
    if row is None:
        return None
    explicit = _finite(row.get("fit_seconds"))
    if explicit is not None:
        return explicit
    fits = list(_fits(row).values())
    if not fits:
        return 0. if row.get("fit_calls") == 0 else None
    values = [_finite(f.get("diagnostics", {}).get("seconds")) for f in fits]
    return float(sum(values)) if all(v is not None for v in values) else None


def true_impedance_errors(trial, row):
    """Align saved true-topology estimates by support; return squared errors."""
    if row is None:
        return None, "missing_method_record"
    name = row.get("true_name", trial.get("true_name"))
    fitted = _fits(row).get(name)
    if name is None or fitted is None:
        return None, "true_fit_missing_or_candidate_uncovered"
    if not fitted.get("success", False):
        return None, "true_fit_failed"
    truth = trial.get("truth", {})
    tr = truth.get("r_pu", truth.get("r")); tx = truth.get("x_pu", truth.get("x"))
    if tr is None or tx is None or "supports" not in fitted or "r" not in fitted or "x" not in fitted:
        return None, "impedance_or_support_fields_missing"
    try:
        supports = [tuple(sorted(map(int, s))) for s in fitted["supports"]]
        n = max(map(len, supports))
        rt, xt, rf, xf = (np.asarray(v, float) for v in (tr, tx, fitted["r"], fitted["x"]))
        if any(a.ndim != 1 or not np.isfinite(a).all() for a in (rt, xt, rf, xf)):
            return None, "nonfinite_impedance_arrays"
        given = [tuple(sorted(map(int, s))) for s in truth.get("supports", [])]
        if len(given) == len(rt) and any(len(s) == n for s in given):
            truth_supports = given
        else:
            family = [s for s in given if 1 < len(s) < n]
            truth_supports = list(RadialCandidate.from_supports("analysis_truth", n, family).supports)
        if len(truth_supports) != len(rt) or len(xt) != len(rt) or len(supports) != len(rf) or len(xf) != len(rf):
            return None, "impedance_support_dimension_mismatch"
        target = {tuple(s):(r, x) for s, r, x in zip(truth_supports, rt, xt)}
        if len(target) != len(truth_supports) or set(target) != set(supports):
            return None, "true_fit_supports_do_not_match_truth"
        edges = []
        for support, r, x in zip(supports, rf, xf):
            true_r, true_x = target[support]
            kind = "common_full" if len(support) == n else "leaf" if len(support) == 1 else "internal"
            edges.append(dict(edge_class=kind, support=list(support),
                r_squared_error=float((r-true_r)**2), x_squared_error=float((x-true_x)**2),
                r_relative_squared_error=None if true_r == 0 else float(((r-true_r)/true_r)**2),
                x_relative_squared_error=None if true_x == 0 else float(((x-true_x)/true_x)**2),
                true_r_pu=float(true_r), true_x_pu=float(true_x)))
        return edges, None
    except (ValueError, TypeError, IndexError):
        return None, "invalid_impedance_or_support_fields"


def _coverage(trial):
    value = trial.get("true_covered")
    if value is not None:
        return bool(value)
    name = trial.get("true_name")
    return bool(name in trial["candidate_names"]) if name is not None and "candidate_names" in trial else None


def _case(trial):
    value = trial.get("case", trial.get("truth", {}).get("case", "unknown"))
    return value if value in ("star", "strong", "weak") else "unknown"


def trial_diagnostic(trial):
    selection = trial.get("selection", {})
    searches = selection.get("searches", {})
    full, pq = searches.get("full", {}), searches.get("pq_only", {})
    records = _record_map(trial)
    truth = trial.get("true_name")
    design_names = selection.get("design_candidates")
    activated = selection.get("activated")
    design_truth = bool(truth in design_names) if truth is not None and isinstance(design_names, list) else None
    plan_comparison = compare_plans_exact(full.get("plan"), pq.get("plan"))
    site0 = next((r for r in full.get("site_records", []) if r.get("site") == 0), {})
    score, constant_score = _finite(full.get("score")), _finite(site0.get("constant_score"))
    predicted_gain = score-constant_score if score is not None and constant_score is not None else None
    seed_gain = _finite(full.get("gradient_gain_over_best_seed"))
    if seed_gain is None and score is not None and _finite(full.get("best_seed_score")) is not None:
        seed_gain = score-float(full["best_seed_score"])
    accepted = [s for r in full.get("site_records", []) for s in r.get("steps", []) if s.get("accepted")]
    material_steps = [s for s in accepted if (_finite(s.get("actual_local_score_gain")) or 0.) > MATERIAL_GAIN]
    margins = {m:signed_actual_margin(trial, records.get(m)) for m in METHODS}
    full_margin, constant_margin = margins["optimized_designed"], margins["constant_fixed"]
    effect = full_margin-constant_margin if full_margin is not None and constant_margin is not None else None
    method_rows = {}
    for method in METHODS:
        row = records.get(method)
        edges, error = true_impedance_errors(trial, row)
        method_rows[method] = dict(status=None if row is None else row.get("status"),
            signed_margin_full_SSE=margins[method],
            method_seconds=None if row is None else _finite(row.get("seconds")),
            joint_fit_seconds=_joint_fit_seconds(row),
            ac_calls=None if row is None else _finite(row.get("ac_calls")),
            fit_calls=None if row is None else _finite(row.get("fit_calls")),
            parameter_edges=edges, parameter_unavailable_reason=error,
            false_rejection=None if row is None else row.get("false_rejection"),
            only_truth_retained=None if row is None else row.get("only_truth_retained"),
            unresolved=None if row is None else row.get("unresolved"))
    return dict(history_id=str(trial.get("history_id", trial.get("seed", "unknown"))),
        case=_case(trial), true_name=truth, candidate_covered=_coverage(trial),
        complete=trial.get("complete"), activated=activated,
        design_truth_in_reference_set=design_truth,
        design_truth_check_is_evaluation_only=True,
        full_search_success=full.get("success"), pq_search_success=pq.get("success"),
        exact_full_pq_plan_comparison=plan_comparison,
        accepted_steps=len(accepted), material_accepted_steps=len(material_steps),
        gradient_gain_over_best_seed_half_SSE=seed_gain,
        material_gradient_gain_over_best_seed=None if seed_gain is None else seed_gain > MATERIAL_GAIN,
        predicted_gain_vs_fixed0_half_SSE=predicted_gain,
        actual_design_minus_constant_margin_full_SSE=effect,
        selection_seconds=_finite(selection.get("seconds")),
        selector_ac_calls=_finite(selection.get("selector_ac_calls")),
        total_ac_calls=_finite(trial.get("total_ac_calls")),
        full_search_seconds=_finite(full.get("seconds")),
        full_search_ac_calls=_finite(full.get("total_ac_calls")),
        historical_fit_seconds=_finite(trial.get("historical_cost", {}).get("seconds")),
        methods=method_rows)


def _aggregate_impedance(rows, method):
    successful = [r["methods"][method]["parameter_edges"] for r in rows
                  if r["methods"][method]["parameter_edges"] is not None]
    reasons = {}
    for row in rows:
        reason = row["methods"][method]["parameter_unavailable_reason"]
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0)+1
    classes = {}
    for kind in EDGE_CLASSES:
        edges = [e for group in successful for e in group if e["edge_class"] == kind]
        result = dict(edge_count=len(edges), history_count=sum(any(e["edge_class"] == kind for e in group) for group in successful))
        for channel in ("r", "x"):
            absolute = [e[channel+"_squared_error"] for e in edges]
            relative = [e[channel+"_relative_squared_error"] for e in edges
                        if e[channel+"_relative_squared_error"] is not None]
            result[channel] = dict(absolute_rmse_pu=float(np.sqrt(np.mean(absolute))) if absolute else None,
                relative_rmse=float(np.sqrt(np.mean(relative))) if relative else None,
                absolute_edge_count=len(absolute), relative_edge_count=len(relative),
                zero_true_parameter_count=len(absolute)-len(relative),
                true_parameter_min_pu=min((e["true_"+channel+"_pu"] for e in edges), default=None))
        classes[kind] = result
    return dict(successful_true_fit_history_count=len(successful), unavailable_count=len(rows)-len(successful),
                unavailable_reasons=reasons, by_edge_class=classes)


def _aggregate(rows):
    paired_plans = [r for r in rows if r["exact_full_pq_plan_comparison"]["comparable"]]
    known_active_truth = [r for r in rows if r["activated"] and r["design_truth_in_reference_set"] is not None]
    correlation_rows = [r for r in rows if r["predicted_gain_vs_fixed0_half_SSE"] is not None
                        and r["actual_design_minus_constant_margin_full_SSE"] is not None]
    output = dict(all_history_count=len(rows), candidate_covered_count=sum(r["candidate_covered"] is True for r in rows),
        candidate_coverage_failure_count=sum(r["candidate_covered"] is False for r in rows),
        candidate_coverage_unknown_count=sum(r["candidate_covered"] is None for r in rows),
        complete_count=sum(r["complete"] is True for r in rows),
        incomplete_count=sum(r["complete"] is False for r in rows),
        completeness_unknown_count=sum(r["complete"] is None for r in rows),
        activated_count=sum(bool(r["activated"]) for r in rows),
        design_truth_reference_denominator=len(known_active_truth),
        design_truth_reference_covered_count=sum(r["design_truth_in_reference_set"] for r in known_active_truth),
        design_truth_reference_excluded_count=sum(not r["design_truth_in_reference_set"] for r in known_active_truth),
        full_pq_plan_comparison_denominator=len(paired_plans),
        full_pq_all_arrays_exact_match_count=sum(r["exact_full_pq_plan_comparison"]["all_arrays_equal"] for r in paired_plans),
        full_pq_each_array_exact_match_count={k:sum(r["exact_full_pq_plan_comparison"]["field_equal"][k] for r in paired_plans) for k in PLAN_ARRAYS},
        accepted_steps=sum(r["accepted_steps"] for r in rows),
        material_accepted_steps=sum(r["material_accepted_steps"] for r in rows),
        gradient_gain_over_best_seed=_stats(r["gradient_gain_over_best_seed_half_SSE"] for r in rows),
        material_gradient_gain_over_best_seed_count=sum(r["material_gradient_gain_over_best_seed"] is True for r in rows),
        material_gain_threshold_half_SSE=MATERIAL_GAIN,
        predicted_gain_vs_fixed0=_stats(r["predicted_gain_vs_fixed0_half_SSE"] for r in rows),
        actual_design_minus_constant_margin=_stats(r["actual_design_minus_constant_margin_full_SSE"] for r in rows),
        predicted_vs_observed_gain_correlation=paired_correlation(
            [r["predicted_gain_vs_fixed0_half_SSE"] for r in correlation_rows],
            [r["actual_design_minus_constant_margin_full_SSE"] for r in correlation_rows]),
        correlation_history_ids=[r["history_id"] for r in correlation_rows],
        timing={k:_stats(r[k] for r in rows) for k in ("selection_seconds", "selector_ac_calls", "total_ac_calls",
            "full_search_seconds", "full_search_ac_calls", "historical_fit_seconds")}, methods={})
    for method in METHODS:
        mr = [r["methods"][method] for r in rows]
        statuses = {}
        for r in mr:
            status = r["status"] or "record_or_status_missing"
            statuses[status] = statuses.get(status, 0)+1
        gate_rows = [r for r in mr if r["false_rejection"] is not None]
        output["methods"][method] = dict(all_history_count=len(rows), status_counts=statuses,
            complete_count=statuses.get("complete", 0),
            failed_count=sum(v for k, v in statuses.items() if k not in ("complete", "record_or_status_missing")),
            missing_count=statuses.get("record_or_status_missing", 0),
            signed_actual_margin_full_SSE=_stats(r["signed_margin_full_SSE"] for r in mr),
            joint_fit_seconds=_stats(r["joint_fit_seconds"] for r in mr),
            total_method_seconds=_stats(r["method_seconds"] for r in mr),
            ac_calls=_stats(r["ac_calls"] for r in mr),
            recorded_gate_count=len(gate_rows),
            false_rejections=sum(bool(r["false_rejection"]) for r in gate_rows) if gate_rows else None,
            only_truth_retained=sum(bool(r["only_truth_retained"]) for r in gate_rows) if gate_rows else None,
            unresolved=sum(bool(r["unresolved"]) for r in gate_rows) if gate_rows else None,
            impedance=_aggregate_impedance(rows, method))
    return output


def analyze_trials(trials, *, phase="pilot", calibration=None, freeze=None):
    diagnostics = [trial_diagnostic(t) for t in trials]
    ids = [d["history_id"] for d in diagnostics]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate independent history identifiers")
    policy_ids = sorted({str(t["policy_id"]) for t in trials if t.get("policy_id") is not None})
    source_manifests = {}
    for t in trials:
        if t.get("policy_id") is not None and t.get("source_sha256") is not None:
            source_manifests[str(t["policy_id"])] = t["source_sha256"]
    cases = {case:_aggregate([d for d in diagnostics if d["case"] == case])
             for case in ("star", "strong", "weak", "unknown") if any(d["case"] == case for d in diagnostics)}
    return dict(phase=phase, created_utc=datetime.now(timezone.utc).isoformat(),
        policy_ids=policy_ids, mixed_policy_hashes=len(policy_ids)>1,
        frozen_source_sha256=source_manifests, freeze_metadata=freeze,
        analysis_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        calibration_metadata=calibration,
        calibration_scope=None if calibration is None else calibration.get("scope"),
        calibration_threshold=None if calibration is None else calibration.get("relative", {}).get("threshold"),
        all=_aggregate(diagnostics), by_case=cases, histories=diagnostics,
        definitions=dict(actual_margin="min wrong full-SSE loss minus true full-SSE loss after H+N refit; signed",
            actual_effect="design signed margin minus fixed-site constant signed margin, paired same history",
            predicted_gain="full selected half-SSE score minus site-zero constant score",
            correlation="descriptive Pearson/Spearman over valid paired independent histories; constant inputs NA",
            plan_match="exact equality of p,q,sites,soc arrays; array matches counted separately",
            accepted_steps="all accepted full-search steps across sites and starts; material gain exceeds 1e-6",
            impedance="successful true-topology fits; support-aligned pooled squared errors by edge class",
            relative_rmse="divide each error by its own true parameter; small weak branches amplify this ratio",
            fit_timing="historical fit-bank seconds; joint sum of saved per-fit diagnostic seconds, excluding selector",
            design_truth_coverage="evaluation-only check among activated histories with known true candidate",
            risk="existing frozen threshold and declared synthetic mixture only; no new confidence claim"),
        alters_policy_data_or_thresholds=False)


def _read_json(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def _optional_json(base, name):
    for path in (base/name, base/(name+".gz")):
        if path.exists():
            return _read_json(path)
    return None


def load_completed_phase(output, phase):
    """Raw and compact share one trial schema; require completed batch."""
    output = Path(output); directory = output/phase
    config = _optional_json(directory, "config.json")
    paths = sorted({*directory.glob("trial_*.json"), *directory.glob("trial_*.json.gz")})
    if config is not None and config.get("count") is not None and len(paths) != config["count"]:
        raise ValueError("phase is incomplete: stored trials differ from planned count")
    if _optional_json(directory, "summary.json") is None:
        raise ValueError("phase has no completed summary; analysis output was not generated")
    if not paths:
        raise ValueError("no saved trials to analyze")
    trials = [_read_json(p) for p in paths]
    return trials, dict(batch_config=config, input_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in paths}),\
        _optional_json(output/"calibration", "thresholds.json"), _optional_json(output, "freeze.json")


def analysis_markdown(analysis):
    all_rows = analysis["all"]
    lines = [f"EV 梯度配置独立分析：{analysis['phase']}", "",
        f"独立历史 {all_rows['all_history_count']} 条；完整 {all_rows['complete_count']} 条，"
        f"不完整 {all_rows['incomplete_count']} 条，候选漏真 {all_rows['candidate_coverage_failure_count']} 条。",
        "", "| 场景 | 全分母 | 完整 | 漏真 | 启动 EV | 参考集含真/可检查 | 设计真唯一/未决/误拒 | 相关 n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case, row in analysis["by_case"].items():
        gate = row["methods"]["optimized_designed"]
        counts = "/".join("NA" if gate[k] is None else str(gate[k])
                          for k in ("only_truth_retained", "unresolved", "false_rejections"))
        lines.append(f"| {case} | {row['all_history_count']} | {row['complete_count']} | {row['candidate_coverage_failure_count']} | "
            f"{row['activated_count']} | {row['design_truth_reference_covered_count']}/{row['design_truth_reference_denominator']} | "
            f"{counts} | {row['predicted_vs_observed_gain_correlation']['n']} |")
    corr = all_rows["predicted_vs_observed_gain_correlation"]
    fmt = lambda value: "NA" if value is None else f"{value:.6g}"
    lines += ["", f"full/PQ 的全部计划数组完全相同 {all_rows['full_pq_all_arrays_exact_match_count']}/"
        f"{all_rows['full_pq_plan_comparison_denominator']}。full 接受步数 {all_rows['accepted_steps']}，"
        f"其中增益超过 1e-6 的步数 {all_rows['material_accepted_steps']}；最终超出最佳初始方案的历史 "
        f"{all_rows['material_gradient_gain_over_best_seed_count']} 条。", "",
        f"预测增益与实际 design−constant 有符号裕量增益：n={corr['n']}，"
        f"Pearson={fmt(corr['pearson'])}，Spearman={fmt(corr['spearman'])}。"
        +(f" NA 原因：{corr['unavailable_reason']}。" if corr['unavailable_reason'] else ""), "",
        "实际裕量使用完整 SSE，保留正负值；预测分数使用半 SSE。相关是配对描述。",
        "", "| 耗时/调用 | 有效 n | 平均 | 中位 |", "|---|---:|---:|---:|"]
    for name, stats in all_rows["timing"].items():
        lines.append(f"| {name} | {stats['n']} | {fmt(stats['mean'])} | {fmt(stats['median'])} |")
    lines += ["", "| 方法 | 完整/全分母 | 真唯一/未决/误拒 | 有符号裕量均值 | 联合拟合秒均值 |",
        "|---|---:|---:|---:|---:|"]
    for method, row in all_rows["methods"].items():
        counts = "/".join("NA" if row[k] is None else str(row[k])
                          for k in ("only_truth_retained", "unresolved", "false_rejections"))
        lines.append(f"| {method} | {row['complete_count']}/{row['all_history_count']} | {counts} | "
            f"{fmt(row['signed_actual_margin_full_SSE']['mean'])} | {fmt(row['joint_fit_seconds']['mean'])} |")
    lines += ["", "| 场景/设计真拓扑拟合边类 | 历史/边数 | R绝对RMSE pu | R相对RMSE | X绝对RMSE pu | X相对RMSE |",
        "|---|---:|---:|---:|---:|---:|"]
    for case, rows in analysis["by_case"].items():
        errors = rows["methods"]["optimized_designed"]["impedance"]["by_edge_class"]
        for kind, row in errors.items():
            lines.append(f"| {case}/{kind} | {row['history_count']}/{row['edge_count']} | "
                f"{fmt(row['r']['absolute_rmse_pu'])} | {fmt(row['r']['relative_rmse'])} | "
                f"{fmt(row['x']['absolute_rmse_pu'])} | {fmt(row['x']['relative_rmse'])} |")
    lines += ["", "相对误差逐边除以真实参数；弱内部边参数很小时，比值会放大。失败拟合未填零。",
        "完整逐方法、逐场景参数误差和缺失字段理由见 analysis.json。", "",
        f"原校准阈值：{fmt(analysis['calibration_threshold'])}；范围：{analysis['calibration_scope'] or '本阶段无校准门限'}。",
        "本分析只读保存结果，没有更改策略、计量、门限或原 summary。"]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="raw or compact batch base")
    parser.add_argument("--phase", choices=("pilot", "calibration", "evaluation"), required=True)
    args = parser.parse_args(argv)
    trials, provenance, calibration, freeze = load_completed_phase(args.output, args.phase)
    evidence = analyze_trials(trials, phase=args.phase, calibration=calibration, freeze=freeze)
    evidence["input_provenance"] = provenance
    directory = args.output/args.phase
    (directory/"analysis.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (directory/"analysis.md").write_text(analysis_markdown(evidence), encoding="utf-8")
    print(json.dumps(dict(phase=args.phase, histories=evidence["all"]["all_history_count"],
        output=str((directory/"analysis.json").resolve())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
