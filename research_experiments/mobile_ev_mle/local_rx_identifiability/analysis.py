"""Read-only arithmetic summary of the completed frozen local R/X experiment.

No AC solver, fitter, or generator is imported or called.  Truth is read only
for evaluation of already sealed inference.  Missing numerical results retain
their original denominators.  Main completion and bootstrap completion are
reported separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from math import isfinite, isinf, sqrt
from pathlib import Path

import numpy as np
from scipy.stats import norm


CANDIDATES = ("star", "pair01", "pair02", "pair12")
ALPHAS = (.05, .01)
TIE_ABSOLUTE_SSE = 1e-6


def _key(spec):
    return f"{spec['network']}_{spec['regime']}_{spec['seed']}_N{spec['N']}_{spec['boundary']}"


def _read(path, manifest):
    path = Path(path)
    body = path.read_bytes()
    manifest[str(path.resolve())] = hashlib.sha256(body).hexdigest()
    if path.suffix == ".gz":
        body = gzip.decompress(body)
    return json.loads(body)


def _finite(value):
    try:
        value = float(value)
        return value if isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _stats(values):
    values = [float(x) for x in values if x is not None and not np.isnan(float(x))]
    finite = [x for x in values if isfinite(x)]
    infinite = sum(isinf(x) for x in values)
    return dict(available=len(values), finite=len(finite), unbounded=infinite,
                mean=float(np.mean(finite)) if finite and not infinite else None,
                mean_of_finite=float(np.mean(finite)) if finite else None,
                minimum=min(finite) if finite else None,
                maximum=max(finite) if finite and not infinite else None)


def _mean_rows(rows, field):
    return _stats(row.get(field) for row in rows)


def _physical(fitted):
    if fitted.get("qualified"):
        return True
    loss = _finite(fitted.get("loss"))
    return bool(loss is not None and any(
        attempt.get("physical") and _finite(attempt.get("loss")) is not None
        and abs(float(attempt["loss"]) - loss) <= 1e-8 * max(1., abs(loss))
        for attempt in fitted.get("attempts", [])))


def _relative_error(estimated, reference):
    estimated, reference = np.asarray(estimated, float), np.asarray(reference, float)
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(estimated - reference) / denominator) if denominator > 0 else None


def _relative_se(information, r, x):
    """sqrt(sum marginal variances)/norm(true vector), including nulls."""
    m = len(r)
    se = np.asarray(information.get("physical_standard_errors", []), float)
    if se.shape != (2*m,):
        return {}
    fields = {}
    for name, reference, errors in (("r", r, se[:m]), ("x", x, se[m:])):
        for suffix, ids in (("total", slice(None)), ("leaf", slice(1, None))):
            denominator = np.linalg.norm(reference[ids])
            fields[f"{name}_{suffix}"] = float(np.linalg.norm(errors[ids]) / denominator) if denominator > 0 else None
    return fields


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _effective_drop(observed, fitted, rt, xt):
    """Arithmetic on the fixed three-terminal star; no flow recomputation."""
    p, q = np.asarray(observed["p"], float), np.asarray(observed["q"], float)
    # star.M columns are the common branch, followed by the three leaf branches.
    branch_p = np.c_[p.sum(axis=1), p]
    branch_q = np.c_[q.sum(axis=1), q]
    denominator = np.sum(branch_p**2, axis=0)
    kappa = np.divide(np.sum(branch_p*branch_q, axis=0), denominator,
                      out=np.zeros(4), where=denominator > 0)
    r, x = np.asarray(fitted["r"], float), np.asarray(fitted["x"], float)
    true_beta, estimated_beta = rt+kappa*xt, r+kappa*x
    orthogonal_q = branch_q-branch_p*kappa
    drop_error = branch_p*(r-rt)+branch_q*(x-xt)
    true_drop = branch_p*rt+branch_q*xt
    result = dict(
        kappa_observed=kappa.tolist(),
        branch_pq_correlation=[_corr(branch_p[:, j], branch_q[:, j]) for j in range(4)],
        terminal_p_correlation=[_corr(p[:, i], p[:, j]) for i, j in ((0, 1), (0, 2), (1, 2))],
        q_orthogonal_fraction=[float(np.linalg.norm(orthogonal_q[:, j]) / np.linalg.norm(branch_q[:, j]))
                               if np.linalg.norm(branch_q[:, j]) > 0 else None for j in range(4)],
        beta_total_relative_l2=_relative_error(estimated_beta, true_beta),
        beta_leaf_relative_l2=_relative_error(estimated_beta[1:], true_beta[1:]),
        observed_linear_drop_total_relative=float(np.linalg.norm(drop_error)/np.linalg.norm(true_drop)),
        observed_linear_drop_leaf_relative=float(np.linalg.norm(drop_error[:, 1:])/np.linalg.norm(true_drop[:, 1:])),
    )
    # Delta r*P + Delta x*Q = P*Delta beta + Delta x*(Q-kappa*P).
    reconstructed = branch_p*(estimated_beta-true_beta)+orthogonal_q*(x-xt)
    result["drop_decomposition_max_error"] = float(np.max(abs(drop_error-reconstructed)))
    return result


def _reason(fitted):
    attempts = fitted.get("attempts", [])
    loss = _finite(fitted.get("loss"))
    selected = [a for a in attempts if loss is not None and _finite(a.get("loss")) is not None
                and abs(float(a["loss"])-loss) <= 1e-8*max(1., abs(loss))]
    if selected:
        return selected[0].get("reason")
    return "; ".join(str(a.get("error", a.get("reason", "unknown"))) for a in attempts) or fitted.get("status", "missing")


def _certificate_fields(row, wrapper):
    tests = {float(item["alpha"]): item for item in wrapper.get("alpha_results", [])}
    row["certificate_status"] = wrapper.get("status", "missing")
    for alpha in ALPHAS:
        label = f"a{round(alpha*100):02d}"
        item = tests.get(alpha, {})
        computed = item.get("status") == "completed"
        for method in ("raw", "residual"):
            prefix = f"{method}_{label}"
            test = item.get(method, {})
            row[prefix+"_computed"] = computed
            row[prefix+"_reject"] = bool(computed and test.get("reject"))
            row[prefix+"_threshold"] = _finite(test.get("threshold"))
            row[prefix+"_lower"] = _finite(test.get("sse_lower" if method == "raw" else "qp_lower"))
        certificate = item.get("certificate") or {}
        row["bound_delta_"+label] = _finite(certificate.get("remainder_norm"))
        support = _finite(certificate.get("qp_support_lower"))
        row["bound_sqrt_L_"+label] = sqrt(max(0., support)) if support is not None else None
        row["bound_K_"+label] = _finite(certificate.get("lipschitz_k"))


def _audited_point(check, audit_current):
    """Use an existing independently replayed legal point, without refitting."""
    if not audit_current or check.get("replayed") is not True:
        return None
    domain = check.get("domain_checks", {})
    if not domain or not all(value is True for value in domain.values()):
        return None
    for field in ("independent_voltage_max_error", "ac_equation_residual", "power_balance_residual"):
        value = _finite(check.get(field))
        if value is None or value > 2e-10:
            return None
    sse = _finite(check.get("raw_sse"))
    discrepancy = _finite(check.get("raw_sse_absolute_error"))
    if sse is None or sse < 0 or discrepancy is None or discrepancy > 1e-7*max(1., sse):
        return None
    return sse


def gaussian_point_diagnostics(sse):
    """Conditional simple-null NP bound; no nearest-point claim is required."""
    value = _finite(sse)
    if value is None or value < 0:
        raise ValueError("an audited legal point needs a finite nonnegative SSE")
    distance = sqrt(value)
    return dict(distance=distance,
        power_upper_a05=float(norm.cdf(distance-norm.isf(.05))),
        power_upper_a01=float(norm.cdf(distance-norm.isf(.01))),
        forced_two_known_mean_error=float(norm.cdf(-distance/2)))


def _candidate_rows(spec, inference, result, truth, independent=None, audit_current=False):
    diagnostics = result.get("diagnostics", {})
    true_name = truth.get("candidate_name", "star")
    rt, xt = np.asarray(truth.get("r", []), float), np.asarray(truth.get("x", []), float)
    reference_info = diagnostics.get("reference_information_at_truth", {})
    reference_se = _relative_se(reference_info, rt, xt) if len(rt) else {}
    rows = []
    for candidate in CANDIDATES:
        fitted = inference.get("fits", {}).get(candidate, {})
        audit = diagnostics.get("candidates", {}).get(candidate, {})
        information = fitted.get("information", {})
        physical = _physical(fitted)
        row = dict(spec, task_key=_key(spec), candidate=candidate,
                   true_candidate=candidate == true_name,
                   task_status=result.get("status", "missing"),
                   fit_available="theta" in fitted, fit_physical=physical,
                   fit_qualified=bool(fitted.get("qualified")),
                   fit_sse=_finite(fitted.get("loss")),
                   fit_projected_gradient=_finite(fitted.get("projected_gradient")),
                   fit_reason=_reason(fitted), fit_seconds=_finite(fitted.get("seconds")),
                   fit_ac_calls=fitted.get("ac_calls"),
                   information_rank=information.get("rank"),
                   information_width=information.get("width"),
                   active_line_coordinates=information.get("active_line_coordinates", []),
                   active_line_coordinate_count=information.get("active_line_coordinate_count"),
                   interior_CRLB_interpretation_applicable=information.get("interior_CRLB_interpretation_applicable"),
                   response_RMSE_sigma_train=_finite(audit.get("true_PQ_voltage_RMSE_sigma_train")),
                   response_RMSE_sigma_holdout=_finite(audit.get("true_PQ_voltage_RMSE_sigma_holdout")),
                   fixed_two_known_mean_reference=_finite(audit.get("fixed_two_mean_error_reference_holdout")),
                   response_condition="true_PQ_and_true_root_counterfactual" if spec["boundary"] == "U" else "true_PQ_and_known_root",
                   clean_distance_order_pass=audit.get("clean_local_distance_order_pass"),
                   clean_PQ_correction_distance_reduction=_finite(audit.get("PQ_correction_distance_reduction")),
                   zero_extra_edge_max_voltage_error=_finite(diagnostics.get("exact_zero_new_edge_sensitivity", {}).get("voltage_max_errors", {}).get(candidate)),
                   )
        if row["true_candidate"] and len(rt) and "r" in fitted:
            r, x = np.asarray(fitted["r"], float), np.asarray(fitted["x"], float)
            row.update(r_total_relative_l2=_relative_error(r, rt), x_total_relative_l2=_relative_error(x, xt),
                       r_leaf_relative_l2=_relative_error(r[1:], rt[1:]), x_leaf_relative_l2=_relative_error(x[1:], xt[1:]),
                       reference_information_rank=reference_info.get("rank"))
            row.update({"reference_se_"+k: v for k, v in reference_se.items()})
            estimated_se = _relative_se(information, rt, xt)
            row.update({"estimated_se_"+k: v for k, v in estimated_se.items()})
            row.update(_effective_drop(inference["observed"], fitted, rt, xt))
        for name, field in (("fixed_PQ", "clean_fixed_PQ_fit"), ("EIV", "clean_EIV_fit")):
            clean = audit.get(field, {})
            row[f"clean_{name}_qualified"] = bool(clean.get("qualified"))
            row[f"clean_{name}_physical"] = _physical(clean)
            row[f"clean_{name}_sse_upper"] = _finite(clean.get("loss")) if _physical(clean) else None
            row[f"clean_{name}_projected_gradient"] = _finite(clean.get("projected_gradient"))
            row[f"clean_{name}_reason"] = _reason(clean)
            checked = (independent or {}).get("diagnostics", {}).get(candidate, {}).get(field, {})
            point_sse = _audited_point(checked, audit_current)
            row[f"clean_{name}_independently_audited"] = point_sse is not None
            row[f"clean_{name}_audited_point_sse"] = point_sse
            if point_sse is not None:
                row.update({f"clean_{name}_"+key: value for key, value in gaussian_point_diagnostics(point_sse).items()})
            if row["true_candidate"] and spec["boundary"] == "K" and name == "EIV" and point_sse is not None and clean.get("qualified"):
                row["clean_true_r_relative_l2"] = _relative_error(clean["r"], rt)
                row["clean_true_x_relative_l2"] = _relative_error(clean["x"], xt)
        decomposition = audit.get("decomposition_holdout", {})
        for name in ("parameter", "fitted_model_difference", "cross", "total", "identity_error"):
            row["decomposition_"+name] = _finite(decomposition.get(name))
        _certificate_fields(row, inference.get("certificates", {}).get(candidate, {}))
        rows.append(row)
    legal = [row for row in rows if row["fit_physical"] and row["fit_sse"] is not None]
    best = min((row["fit_sse"] for row in legal), default=None)
    complete_qualified = all(row["fit_qualified"] and row["fit_sse"] is not None for row in rows)
    for row in rows:
        row["all_candidates_qualified"] = complete_qualified
        row["sse_minus_best_legal_attained"] = row["fit_sse"]-best if row["fit_sse"] is not None and best is not None else None
        row["first_or_tied_complete_ranking"] = bool(complete_qualified and row["fit_sse"] <= best+TIE_ABSOLUTE_SSE)
        row["strict_rank_complete_ranking"] = (1+sum(other["fit_sse"] < row["fit_sse"] for other in rows)) if complete_qualified else None
    return rows


def _power_summary(rows):
    """At least one difficult wrong candidate per fixed underlying history."""
    by_history = defaultdict(list)
    for row in rows:
        by_history[row["task_key"]].append(row)
    result = {}
    for method in ("EIV", "fixed_PQ"):
        histories = []
        for key, candidates in sorted(by_history.items()):
            usable = [row for row in candidates if not row["true_candidate"]
                      and row.get(f"clean_{method}_independently_audited")]
            entry = dict(task_key=key, audited_wrong_candidates=len(usable), total_wrong_candidates=3,
                         computed=bool(usable))
            if usable:
                selected = min(usable, key=lambda row: row[f"clean_{method}_distance"])
                entry["candidate"] = selected["candidate"]
                for field in ("distance", "power_upper_a05", "power_upper_a01", "forced_two_known_mean_error"):
                    entry[field] = selected[f"clean_{method}_"+field]
            histories.append(entry)
        result[method] = dict(total_conditions=len(histories), computed_conditions=sum(row["computed"] for row in histories),
            selected_candidate_counts=dict(Counter(row["candidate"] for row in histories if row["computed"])),
            histories=histories,
            **{field:_stats(row.get(field) for row in histories)
               for field in ("distance", "power_upper_a05", "power_upper_a01", "forced_two_known_mean_error")})
    return result


def _strict(rows):
    result = {}
    for alpha in ALPHAS:
        label = f"a{round(alpha*100):02d}"
        result[str(alpha)] = {}
        for method in ("raw", "residual"):
            prefix = f"{method}_{label}"
            counts = {}
            for kind, selected in (("true", [r for r in rows if r["true_candidate"]]),
                                   ("wrong", [r for r in rows if not r["true_candidate"]])):
                counts[kind] = dict(total=len(selected), computed=sum(r[prefix+"_computed"] for r in selected),
                    rejected=sum(r[prefix+"_reject"] for r in selected),
                    calculation_failures=sum(not r[prefix+"_computed"] for r in selected),
                    lower=_mean_rows(selected, prefix+"_lower"), threshold=_mean_rows(selected, prefix+"_threshold"))
            result[str(alpha)][method] = counts
    return result


def _group(rows):
    true = [r for r in rows if r["true_candidate"]]
    qualified = [r for r in true if r["fit_qualified"]]
    all_fields = ("r_total_relative_l2", "x_total_relative_l2", "r_leaf_relative_l2", "x_leaf_relative_l2",
                  "beta_total_relative_l2", "beta_leaf_relative_l2", "observed_linear_drop_total_relative",
                  "observed_linear_drop_leaf_relative", "reference_se_r_total", "reference_se_x_total",
                  "reference_se_r_leaf", "reference_se_x_leaf", "estimated_se_r_total", "estimated_se_x_total",
                  "estimated_se_r_leaf", "estimated_se_x_leaf")
    result = dict(conditions=len(true), true_fit_qualified=len(qualified),
                  all_candidate_slots=len(rows), qualified_candidate_slots=sum(r["fit_qualified"] for r in rows),
                  parameter_metrics_qualified={field:_mean_rows(qualified, field) for field in all_fields},
                  parameter_metrics_all_attained={field:_mean_rows(true, field) for field in all_fields},
                  reference_se_all_histories={field:_mean_rows(true, field) for field in all_fields
                                             if field.startswith("reference_se_")},
                  fitted_rank_counts=dict(Counter(str(r.get("information_rank")) for r in true)),
                  reference_rank_counts=dict(Counter(str(r.get("reference_information_rank")) for r in true)),
                  true_active_line_coordinates=_mean_rows(qualified, "active_line_coordinate_count"),
                  hardest_competitor_power_upper=_power_summary(rows),
                  strict=_strict(rows), candidates={})
    clean_true = [row for row in true if row["boundary"] == "K"]
    result["clean_true_parameter_recovery_K"] = dict(
        total=len(clean_true), audited_qualified=sum(row.get("clean_true_r_relative_l2") is not None for row in clean_true),
        r_relative_l2=_mean_rows(clean_true, "clean_true_r_relative_l2"),
        x_relative_l2=_mean_rows(clean_true, "clean_true_x_relative_l2"),
        sse=_mean_rows(clean_true, "clean_EIV_audited_point_sse"))
    complete = [r for r in true if r["all_candidates_qualified"]]
    result["training_local_sse_ranking"] = dict(eligible_conditions=len(complete), total_conditions=len(true),
        true_first_or_tied=sum(r["first_or_tied_complete_ranking"] for r in complete),
        true_unique_first=sum(r["first_or_tied_complete_ranking"] and sum(
            x["first_or_tied_complete_ranking"] for x in rows if x["task_key"] == r["task_key"]) == 1 for r in complete),
        tie_absolute_sse=TIE_ABSOLUTE_SSE,
        true_sse_minus_best=_mean_rows(complete, "sse_minus_best_legal_attained"))
    for field in ("kappa_observed", "branch_pq_correlation", "q_orthogonal_fraction", "terminal_p_correlation"):
        width = 3 if field == "terminal_p_correlation" else 4
        result[field] = [_stats(r.get(field, [None]*width)[j] for r in true) for j in range(width)]
    for candidate in CANDIDATES:
        selected = [r for r in rows if r["candidate"] == candidate]
        result["candidates"][candidate] = dict(
            total=len(selected), qualified=sum(r["fit_qualified"] for r in selected),
            sse=_mean_rows(selected, "fit_sse"),
            response_RMSE_sigma_train=_mean_rows(selected, "response_RMSE_sigma_train"),
            response_RMSE_sigma_holdout=_mean_rows(selected, "response_RMSE_sigma_holdout"),
            clean_fixed_PQ_sse_upper=_mean_rows(selected, "clean_fixed_PQ_sse_upper"),
            clean_EIV_sse_upper=_mean_rows(selected, "clean_EIV_sse_upper"),
            clean_EIV_audited_distance=_mean_rows(selected, "clean_EIV_distance"),
            clean_EIV_power_upper_a05=_mean_rows(selected, "clean_EIV_power_upper_a05"),
            clean_EIV_power_upper_a01=_mean_rows(selected, "clean_EIV_power_upper_a01"),
            clean_fixed_PQ_qualified=sum(r["clean_fixed_PQ_qualified"] for r in selected),
            clean_EIV_qualified=sum(r["clean_EIV_qualified"] for r in selected),
            clean_order_checks=sum(r["clean_distance_order_pass"] is not None for r in selected),
            clean_order_failures=sum(r["clean_distance_order_pass"] is False for r in selected),
            decomposition={name:_mean_rows(selected, "decomposition_"+name)
                           for name in ("parameter", "fitted_model_difference", "cross", "total", "identity_error")})
    result["bound_diagnostics"] = {field:_mean_rows(rows, field) for field in
        ("bound_delta_a05", "bound_sqrt_L_a05", "bound_K_a05", "zero_extra_edge_max_voltage_error", "drop_decomposition_max_error")}
    return result


def _bootstrap(directory, tasks, manifest):
    expected = [(spec, candidate) for spec in tasks if spec["N"] == 96 and spec["boundary"] == "K"
                and spec["seed"] == 31001 for candidate in CANDIDATES]
    rows = []
    for spec, candidate in expected:
        path = directory/"bootstrap"/f"{_key(spec)}_{candidate}.json"
        row = dict(spec, candidate=candidate, true_candidate=candidate == "star", available=path.exists())
        if path.exists():
            value = _read(path, manifest)
            decision = value.get("decision", {})
            row.update(status=value.get("status"), pvalue=_finite(decision.get("pvalue")),
                       original_fit_unqualified=value.get("status") == "original_fit_unqualified",
                       replicates=decision.get("replicates", 0),
                       failed_replicates=decision.get("failed_or_nonstationary_replicates", 0),
                       valid_replicates=decision.get("valid_replicates", 0),
                       reject=bool(value.get("reject", False)), seconds=_finite(value.get("seconds")))
            records = value.get("records", [])
            count = 0
            observed = _finite(value.get("observed_sse"))
            if observed is not None and records:
                for record in records:
                    loss = _finite(record.get("sse"))
                    failed = record.get("success") is not True or record.get("stationary") is not True or loss is None or loss < 0
                    count += int(failed or loss >= observed)
                row["recomputed_pvalue"] = (1+count)/(1+len(records))
                row["pvalue_arithmetic_error"] = abs(row["recomputed_pvalue"]-row["pvalue"]) if row["pvalue"] is not None else None
        else:
            row.update(status="missing", pvalue=None, reject=False, replicates=0,
                       failed_replicates=0, valid_replicates=0, original_fit_unqualified=False)
        rows.append(row)
    complete = all(r["available"] for r in rows)
    return dict(status="complete" if complete else ("partial" if any(r["available"] for r in rows) else "not_started"),
        expected=len(rows), available=sum(r["available"] for r in rows),
        scenario_trajectories=len({(s["network"], s["regime"], s["seed"]) for s, _ in expected}),
        independent_random_families=len({(s["network"], s["seed"]) for s, _ in expected}),
        pairing="regimes share the network/seed random family and meter noise",
        coverage="approximate_candidate_plugin_bootstrap_not_topology_probability",
        rows=rows, true=dict(total=sum(r["true_candidate"] for r in rows),
            computed=sum(r["true_candidate"] and r["status"] == "completed" for r in rows),
            rejected=sum(r["true_candidate"] and r["reject"] for r in rows)),
        wrong=dict(total=sum(not r["true_candidate"] for r in rows),
            computed=sum(not r["true_candidate"] and r["status"] == "completed" for r in rows),
            rejected=sum(not r["true_candidate"] and r["reject"] for r in rows)),
        replicates=sum(r["replicates"] for r in rows), failed_replicates=sum(r["failed_replicates"] for r in rows),
        pvalue_arithmetic_error=_stats(r.get("pvalue_arithmetic_error") for r in rows))


def _format(stat, *, percent=False, digits=2):
    if not isinstance(stat, dict):
        return "—"
    if stat.get("unbounded"):
        return "∞"
    if stat.get("mean") is None:
        return "—"
    return f"{stat['mean']*(100 if percent else 1):.{digits}f}"


def _markdown(summary):
    overall = summary["overall"]
    lines = ["# 小末端R/X与候选解释：冻结批次结果", "",
        f"主矩阵已结束：{summary['completion']['finished_tasks']}/{summary['completion']['expected_tasks']}条件；"
        f"任务级成功{summary['completion']['successful_tasks']}、失败{summary['completion']['failed_tasks']}。"
        f"真候选合格拟合{overall['true_fit_qualified']}/{overall['conditions']}，"
        f"全部候选合格{overall['qualified_candidate_slots']}/{overall['all_candidate_slots']}。", "",
        "12条场景轨迹来自6个独立network/seed随机族；普通/相关负荷共享随机族与计量噪声，regime、96/288和K/U均配对。"
        "每个网络/负荷组内仅3个独立种子，不把48条件或12条场景轨迹当48或12份独立历史。"
        "paper15与paper11是项目合成论文风格网络的原始倍率局部，不是作者实测线路。", "",
        "统计口径更正：冻结configuration中的seed_unit及原审计的基础轨迹数量沿用12条场景轨迹，"
        "其中“独立”一词不准确。原始元数据保留；本分析按cases._rng不含regime的实际生成方式，"
        "将独立随机族计为6。计量、拟合、阈值和逐组数值没有改变。", "",
        "K给定额外准确根电压；U只观测末端P/Q/V。下表误差只平均合格真候选，失败保留在分母；取得值另见summary.json和candidate_details.csv。", "",
        "|网络/负荷|N|边界|真候选合格|R总误差%|X总误差%|R末端%|X末端%|β总误差%|β末端%|真值SE：R末端%|真值SE：X末端%|拟合信息秩|活动坐标均数|",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for group in summary["groups"]:
        values = group["parameter_metrics_qualified"]
        fields = ("r_total_relative_l2", "x_total_relative_l2", "r_leaf_relative_l2", "x_leaf_relative_l2",
                  "beta_total_relative_l2", "beta_leaf_relative_l2", "reference_se_r_leaf", "reference_se_x_leaf")
        numbers = [_format((group["reference_se_all_histories"] if field.startswith("reference_se_") else values)[field], percent=True)
                   for field in fields]
        lines.append("|"+"|".join([f"{group['network']}/{group['regime']}", str(group['N']), group['boundary'],
            f"{group['true_fit_qualified']}/{group['conditions']}", *numbers,
            json.dumps(group["fitted_rank_counts"], ensure_ascii=False),
            _format(group["true_active_line_coordinates"])])+"|")
    lines += ["", "R/X误差定义为向量相对二范数。SE列是已存真值处局部标准误差向量范数除以真参数范数，"
        "不是已验证覆盖或可达精度。真值SE覆盖全部基础条件，不因拟合未合格而删除；误差均值仅用合格拟合。"
        "U共同主干有精确零信息方向，总体SE无穷；末端支路另列。"
        "活动边界、相关激励和估计偏差会影响局部精度解释。", "",
        "有效压降组合β=r+κx，κ=(P·Q)/(P·P)，由观测P/Q按star.M汇总后逐边计算。"
        "P/Q是末端观测的无损功率汇总，用于这项代数诊断；实际拟合仍使用完整AC。"
        "有ΔrP+ΔxQ=PΔβ+Δx(Q−κP)。β误差及Q的正交比例用于检查R/X补偿；"
        "β较准不等于单独R/X较准，也不自动保证不同拓扑可分。", "",
        "|网络/负荷|N|末端P/Q相关范围|末端Q正交比例范围|观测线性压降误差%（合格K）|",
        "|---|---:|---|---|---:|"]
    for group in summary["groups"]:
        if group["boundary"] != "K":
            continue
        def limits(values):
            means = [value["mean"] for value in values if value.get("mean") is not None]
            return f"{min(means):.3f}—{max(means):.3f}" if means else "—"
        lines.append(f"|{group['network']}/{group['regime']}|{group['N']}|"
                     f"{limits(group['branch_pq_correlation'][1:])}|{limits(group['q_orthogonal_fraction'][1:])}|"
                     f"{_format(group['parameter_metrics_qualified']['observed_linear_drop_leaf_relative'], percent=True)}|")
    lines += ["", "范围是三个末端各自跨种子平均值的最小至最大；P/Q相关使用中心化Pearson相关，"
        "Q正交比例为‖Q−κP‖/‖Q‖。K/U观测相同，此表不重复U。", "",
        "## 无噪声恢复检查", "",
        "下面使用真候选star、真实干净P/Q/V和准确根电压，只统计已有clean EIV中独立审计合法且合格的取得值，没有新拟合。", "",
        "|负荷|合格审计/总数|最大R相对误差%|最大X相对误差%|最大SSE|",
        "|---|---:|---:|---:|---:|"]
    for regime in ("ordinary", "correlated"):
        selected = [group["clean_true_parameter_recovery_K"] for group in summary["groups"]
                    if group["boundary"] == "K" and group["regime"] == regime]
        def maximum(field):
            values = [group[field]["maximum"] for group in selected if group[field]["maximum"] is not None]
            return max(values) if values else None
        rr, xx, loss = maximum("r_relative_l2"), maximum("x_relative_l2"), maximum("sse")
        lines.append(f"|{regime}|{sum(group['audited_qualified'] for group in selected)}/{sum(group['total'] for group in selected)}|"
                     f"{100*rr:.6g}|{100*xx:.6g}|{loss:.6g}|" if rr is not None and xx is not None and loss is not None
                     else f"|{regime}|未取得完整参考|—|—|—|")
    lines += ["", "这表明该固定小网、准确根与无噪条件下，现有数值流程可以恢复线路到很小误差；"
        "含噪R/X大误差主要需要从弱信息、计量扰动和活动先验边界解释，不能简单归于AC潮流求解精度。"
        "U的共同主干仍无唯一恢复，不作同样结论。", "",
        "## 训练损失排序与严格拒绝", "",
        "排序只在四候选均合格的完整条件上统计；SSE绝对容差1e-6保留并列。"
        "排序采用局部取得值，不是全局最大似然证明。", "",
        "|网络/负荷|N|边界|可排序/总数|真候选首名或并列|真候选唯一首名|",
        "|---|---:|---|---:|---:|---:|"]
    for group in summary["groups"]:
        rank = group["training_local_sse_ranking"]
        lines.append(f"|{group['network']}/{group['regime']}|{group['N']}|{group['boundary']}|"
                     f"{rank['eligible_conditions']}/{rank['total_conditions']}|{rank['true_first_or_tied']}|{rank['true_unique_first']}|")
    lines += ["", "|α|指标|真候选误拒/全分母（有效计算）|错误候选拒绝/全分母（有效计算）|",
              "|---:|---|---|---|"]
    for alpha in ALPHAS:
        for method in ("raw", "residual"):
            counts = overall["strict"][str(alpha)][method]
            true, wrong = counts["true"], counts["wrong"]
            lines.append(f"|{alpha}|{method}|{true['rejected']}/{true['total']}（{true['computed']}）|"
                         f"{wrong['rejected']}/{wrong['total']}（{wrong['computed']}）|")
    lines += ["", "两指标分别检验，不自动OR合并；可靠性以已声明高斯计量模型、公共物理域覆盖和浮点保护成立为前提。"
        "局部大损失不能触发严格拒绝。逐组/逐候选的下界、阈值、损耗包络与失败计数保存在JSON/CSV。", "",
        "## 事后信息上限：可靠检验最多能排错多少", "",
        "对已有clean合法竞争解释，独立AC审计给出其均值到真实干净均值的完整P/Q/V白化距离d=√SSE。"
        "若一个检验在该错误候选全部合法解释上误拒率≤α，则在这条真实轨迹下，拒绝该候选的功效≤Φ(d−z₁₋α)。"
        "这是简单高斯null的Neyman–Pearson上限；只需一个合法解释，无须证明它是最近点。"
        "以下每条历史选取得距离最小的合法错误解释，再汇总；它指出至少一个难以可靠拒绝的竞争者。", "",
        "|网络/负荷|N|边界|完整P/Q校正：d均值[范围]|α=.05功效上限%均值[范围]|α=.01上限%均值[范围]|固定PQ：α=.05上限%均值[范围]|",
        "|---|---:|---|---|---|---|---|"]
    for group in summary["groups"]:
        power = group["hardest_competitor_power_upper"]
        def interval(stat, percent=False):
            if stat.get("mean") is None:
                return "—"
            factor = 100 if percent else 1
            return f"{factor*stat['mean']:.3f}[{factor*stat['minimum']:.3f},{factor*stat['maximum']:.3f}]"
        lines.append(f"|{group['network']}/{group['regime']}|{group['N']}|{group['boundary']}|"
                     f"{interval(power['EIV']['distance'])}|{interval(power['EIV']['power_upper_a05'], True)}|"
                     f"{interval(power['EIV']['power_upper_a01'], True)}|{interval(power['fixed_PQ']['power_upper_a05'], True)}|")
    lines += ["", "范围是不同固定轨迹的最小至最大，不是置信区间。该上限要求检验全域误拒受控，"
        "单中心bootstrap本身没有这个统一保证。它使用真实干净均值作事后信息诊断，部署不读取真值。"
        "若上限很低，再紧的优化界也不能在同数据、同错误率下获得很高排错功效；"
        "若上限较高而当前检验无功效，仍可能是检验或界过于保守。"
        "它不是拓扑后验概率，也不强迫允许未决的程序作错误二选一。"
        "同两固定均值强制分类的最优错误率Φ(−d/2)另存JSON，含义与功效上限不同。", "",
        "## 参数与响应补偿", "",
        "以下均以真实P/Q和真实根电压回代，只在推断封存后执行。U列是额外边界信息下的反事实诊断，"
        "不是当时可用数据下的在线预测误差；共同主干等价类本身就会影响这项诊断。", "",
        "|网络/负荷|N|边界|候选|合格/总数|训练电压RMSE/σ|留出RMSE/σ|clean固定PQ取得SSE|clean EIV取得SSE|",
        "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for group in summary["groups"]:
        for name, values in group["candidates"].items():
            lines.append(f"|{group['network']}/{group['regime']}|{group['N']}|{group['boundary']}|{name}|"
                f"{values['qualified']}/{values['total']}|{_format(values['response_RMSE_sigma_train'])}|"
                f"{_format(values['response_RMSE_sigma_holdout'])}|{_format(values['clean_fixed_PQ_sse_upper'], digits=4)}|"
                f"{_format(values['clean_EIV_sse_upper'], digits=4)}|")
    lines += ["", "clean列是无噪均值上的局部取得距离上界，未称全局距离；合格数、合法性与顺序检查另存JSON。"
        "响应恒等式保留参数项、拟合模型差项和交叉项。负交叉项可掩盖结构差，不能删去。", "",
        f"恒等式最大误差：{summary['audits']['decomposition_identity_max_error']}; "
        f"新增共享边置零的最大电压误差：{summary['audits']['zero_extra_edge_max_voltage_error']}pu。"
        "置零解释在主实验正下界域外，仅展示允许零边时的精确结构重合。", "",
        "## 数值失败", ""]
    if not summary["unqualified_fits"]:
        lines.append("本批所有候选拟合均达到预定验收。")
    else:
        lines += ["以下保留未合格候选；数值未收敛不直接归因于信息不足。", "",
                  "|条件|候选|PG|活动坐标|停止原因|", "|---|---|---:|---|---|"]
        for row in summary["unqualified_fits"]:
            lines.append(f"|{row['task_key']}|{row['candidate']}|{row.get('fit_projected_gradient')}|"
                         f"{row.get('active_line_coordinates')}|{row.get('fit_reason')}|")
    boot = summary["bootstrap"]
    lines += ["", "## 候选自身bootstrap", "",
        f"状态：{boot['status']}，已取得{boot['available']}/{boot['expected']}个候选条件；"
        f"模拟记录{boot['replicates']}，失败或非驻点{boot['failed_replicates']}。"
        f"对应{boot['scenario_trajectories']}条场景轨迹、{boot['independent_random_families']}个独立network/seed随机族；"
        "普通/相关场景配对，四个候选也不是四份新历史。"
        "p值是拟合参数代入后的近似诊断，失败保守计入上尾，不是拓扑后验概率。", "",
        "|网络/负荷|候选|状态|p|有效/重复|失败重复|拒绝|", "|---|---|---|---:|---:|---:|---|"]
    for row in boot["rows"]:
        lines.append(f"|{row['network']}/{row['regime']}|{row['candidate']}|{row['status']}|{row['pvalue']}|"
                     f"{row['valid_replicates']}/{row['replicates']}|{row['failed_replicates']}|{row['reject']}|")
    lines += ["", f"事后按真值核对：真候选误拒{boot['true']['rejected']}/{boot['true']['total']}"
        f"（有效{boot['true']['computed']}）；错误候选拒绝{boot['wrong']['rejected']}/{boot['wrong']['total']}"
        f"（有效{boot['wrong']['computed']}）。缺失记录保留分母，不能作为已完成检验。", "",
        "## 证据范围", "",
        "本文件由analysis.py只读生成，没有重新拟合、改阈值或生成科学数据。完整输入哈希和分组数值在summary.json，"
        "单候选取得值与失败在candidate_details.csv。预测指标、bootstrap p和局部SE不称为拓扑正确概率。", ""]
    return "\n".join(lines)


def summarize(directory):
    directory = Path(directory).resolve()
    manifest = {}
    configuration = _read(directory/"configuration.json", manifest)
    tasks = configuration["tasks"]
    completion_path = directory/"completed.json"
    if not completion_path.exists():
        raise RuntimeError("main batch has no completed.json; no final summary written")
    completed = _read(completion_path, manifest)
    expected = {_key(spec) for spec in tasks}
    finished = completed.get("results", [])
    if len(tasks) != 48 or len(expected) != 48 or completed.get("total") != 48:
        raise RuntimeError("expected the frozen 48-condition main batch")
    if len(finished) != 48 or {_key(value["spec"]) for value in finished} != expected:
        raise RuntimeError("main completion marker does not contain all 48 tasks")
    rows, independent_errors, artifact_issues, task_failures = [], [], [], []
    independent_path = directory/"audit.json"
    independent_audit = _read(independent_path, manifest) if independent_path.exists() else {}
    independent_slots = {row["key"]: row for row in independent_audit.get("slots", [])}
    source_manifest = directory/"source_manifest.json"
    if source_manifest.exists():
        _read(source_manifest, manifest)
        expected_hash = configuration.get("source_manifest_sha256")
        if expected_hash and manifest[str(source_manifest.resolve())] != expected_hash:
            artifact_issues.append(dict(issue="scientific_source_manifest_hash_mismatch"))
    for spec in tasks:
        key = _key(spec)
        result_path = directory/"results"/f"{key}.json.gz"
        result = _read(result_path, manifest) if result_path.exists() else dict(status="missing")
        inference_path = directory/"inference"/f"{key}.json.gz"
        truth_path = directory/"truth"/f"{key}.json.gz"
        inference = _read(inference_path, manifest) if inference_path.exists() else {}
        truth = _read(truth_path, manifest) if truth_path.exists() else {}
        if result.get("status") != "completed":
            task_failures.append(dict(spec=spec, status=result.get("status"), error=result.get("error")))
        if not inference or not truth:
            artifact_issues.append(dict(task_key=key, issue="missing_inference_or_truth"))
        if inference and (inference.get("truth_used") is not False or "truth" in inference):
            artifact_issues.append(dict(task_key=key, issue="sealed_truth_field_check_failed"))
        if result.get("inference_sha256") and manifest.get(str(inference_path.resolve())) != result["inference_sha256"]:
            artifact_issues.append(dict(task_key=key, issue="inference_hash_mismatch"))
        diagnostics = result.get("diagnostics", {})
        error = _finite(diagnostics.get("independent_truth_voltage_max_error"))
        if error is not None:
            independent_errors.append(error)
        independent = independent_slots.get(key, {})
        audit_current = bool(independent_audit.get("audit_pass") and independent)
        for name, path in (("result", result_path), ("inference", inference_path), ("truth", truth_path)):
            if independent.get(name+"_sha256") != manifest.get(str(path.resolve())):
                audit_current = False
        rows.extend(_candidate_rows(spec, inference, result, truth, independent, audit_current))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["network"], row["regime"], row["N"], row["boundary"])].append(row)
    groups = []
    for (network, regime, count, boundary), selected in sorted(grouped.items()):
        groups.append(dict(network=network, regime=regime, N=count, boundary=boundary, **_group(selected)))
    bootstrap = _bootstrap(directory, tasks, manifest)
    identity = [r["decomposition_identity_error"] for r in rows if r["decomposition_identity_error"] is not None]
    zero = [r["zero_extra_edge_max_voltage_error"] for r in rows if r["zero_extra_edge_max_voltage_error"] is not None]
    analysis_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    audited_clean_count = sum(row.get(f"clean_{method}_independently_audited", False)
                              for row in rows for method in ("EIV", "fixed_PQ"))
    information_complete = audited_clean_count == 2*len(rows)
    summary = dict(schema="local_rx_analysis_v1", analysis_sha256=analysis_hash,
        completion=dict(main_complete=True, expected_tasks=48, finished_tasks=48,
            successful_tasks=48-len(task_failures), failed_tasks=len(task_failures),
            bootstrap_complete=bootstrap["status"] == "complete",
            information_diagnostics_complete=information_complete,
            all_planned_statistics_complete=bootstrap["status"] == "complete" and information_complete),
        statistical_unit="12 scenario trajectories from 6 independent network/seed random families; regime, N and K/U paired; 3 independent seeds within each network/regime group",
        sampling_structure=dict(
            scenario_trajectories=len({(s['network'],s['regime'],s['seed']) for s in tasks}),
            independent_random_families=len({(s['network'],s['seed']) for s in tasks}),
            paired_factors=["regime", "N", "boundary"],
            reason="cases._rng SeedSequence contains seed, network and channel, but not regime",
            original_metadata_preserved=True,
            original_configuration_seed_unit=configuration.get("seed_unit"),
            correction="original 12 independent base trajectories label counts scenarios, not independent random families"),
        parameter_statistic="relative L2 per history then arithmetic mean; primary qualified fits; all attained separately",
        probability_scope="two separate declared-model Gaussian-event tests; bootstrap approximate; no topology posterior",
        information_power_bound_scope="posthoc true mean versus independently audited legal Gaussian null point; uniform composite-null size required; not a posterior",
        overall=_group(rows), groups=groups, bootstrap=bootstrap,
        task_failures=task_failures,
        unqualified_fits=[{key: row.get(key) for key in
            ("task_key", "candidate", "true_candidate", "task_status", "fit_available", "fit_physical",
             "fit_sse", "fit_projected_gradient", "fit_reason", "active_line_coordinates",
             "active_line_coordinate_count", "information_rank", "information_width")}
            for row in rows if not row["fit_qualified"]],
        audits=dict(artifact_issues=artifact_issues,
            independent_truth_voltage_max_error=max(independent_errors, default=None),
            independent_clean_audit_available=bool(independent_audit),
            independently_audited_clean_points=audited_clean_count,
            decomposition_identity_max_error=max(identity, default=None),
            zero_extra_edge_max_voltage_error=max(zero, default=None),
            zero_extra_edge_inside_primary_domain=False,
            clean_distance_order_checks=sum(r["clean_distance_order_pass"] is not None for r in rows),
            clean_distance_order_failures=sum(r["clean_distance_order_pass"] is False for r in rows)),
        source_configuration=configuration, input_sha256=manifest,
        detailed_csv="candidate_details.csv")
    columns = sorted({field for row in rows for field in row})
    csv_path = directory/"candidate_details.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    (directory/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (directory/"results_generated.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    try:
        summary = summarize(args.directory)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"No final analysis written: {error}\n")
    print(json.dumps(dict(main=summary["completion"], true_fit_qualified=summary["overall"]["true_fit_qualified"],
        candidate_fit_qualified=summary["overall"]["qualified_candidate_slots"],
        bootstrap=summary["bootstrap"]["status"], audits=summary["audits"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
