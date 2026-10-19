"""Read-only localization of a frozen response study; no AC, fit or solver imports.

Each independent history has total weight one. Regimes and meter-noise repeats
are conditional observations, not additional independent histories.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp
from scipy.stats import spearmanr

METHODS = ("joint_bc__mobile_bidir", "square__mobile_bidir", "constant", "none")
REGIMES = ("regular", "shift")
FIELDS = ("error", "confidence", "predicted_risk", "true_probability",
          "true_rejected", "voltage_interval90_coverage")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def finite(x):
    try:
        value = float(x)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def avg(items):
    values = [float(x) for x in items if x is not None and np.isfinite(x)]
    return float(np.mean(values)) if values else None


def count_summary(rows, expected=None):
    known = [r for r in rows if r.get("error") is not None]
    n = len(known)
    errors = [r["error"] for r in known]
    expected = len(rows) if expected is None else expected
    total = sum(errors)
    descending = sorted(known, key=lambda r: (-r["error"], r["index"]))
    return {
        "expected_histories": expected, "available_histories": n,
        "missing_histories": expected - n,
        "zero_error_histories": sum(x == 0.0 for x in errors),
        "all_error_histories": sum(x == 1.0 for x in errors),
        "mixed_error_histories": sum(0.0 < x < 1.0 for x in errors),
        **{f"mean_{field}": avg(r.get(field) for r in known) for field in FIELDS},
        "top20_error_share": sum(r["error"] for r in descending[:20]) / total if total else None,
        "top25_error_share": sum(r["error"] for r in descending[:25]) / total if total else None,
        "all_error_mean_confidence": avg(r["confidence"] for r in known if r["error"] == 1.0),
        "all_error_mean_predicted_risk": avg(r["predicted_risk"] for r in known if r["error"] == 1.0),
        "all_error_indices": [r["index"] for r in known if r["error"] == 1.0],
        "worst25_indices": [r["index"] for r in descending[:25]],
    }


def correlation(rows, xname, yname="error"):
    pairs = [(r.get(xname), r.get(yname)) for r in rows]
    pairs = [(float(x), float(y)) for x, y in pairs if finite(x) is not None and finite(y) is not None]
    out = {"x": xname, "y": yname, "available_histories": len(pairs),
           "scope": "descriptive history-level Spearman association, not causal attribution"}
    if len(pairs) < 3 or len({x for x, _ in pairs}) < 2 or len({y for _, y in pairs}) < 2:
        return {**out, "rho": None, "p_value": None, "reason": "insufficient or constant values"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = spearmanr([x for x, _ in pairs], [y for _, y in pairs])
    return {**out, "rho": float(result.statistic), "p_value": float(result.pvalue)}


def history_analysis(summary, csv_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    table = {}
    duplicate_keys = []
    for r in csv_rows:
        key = (int(r["index"]), r["method"], r["model"], r["regime"])
        if key in table:
            duplicate_keys.append(key)
        table[key] = r
    if duplicate_keys:
        raise ValueError(f"duplicate history-metric keys: {duplicate_keys[:3]}")
    all_rows, checks = {}, []
    for method in METHODS:
        records = []
        for h in summary["histories"]:
            index = int(h["index"])
            regimes, missing = [], []
            for regime in REGIMES:
                raw = table.get((index, method, "joint", regime))
                if raw is None or raw["available"].lower() != "true" or finite(raw.get("error")) is None:
                    missing.append({"regime": regime, "reason": raw.get("missing_reason") if raw else "no CSV row"})
                else:
                    regimes.append({"regime": regime, **{f: finite(raw.get(f)) for f in FIELDS},
                                    "n_conditional_noise": int(raw["n_conditional_noise"])})
            record = {"index": index, "seed": h["seed"], "status": h["status"],
                      **h["spec"], "true_name": h.get("true_name"),
                      "active_boundary": h.get("active_training_line_boundary"),
                      "posterior_R_relative_error": finite(h.get("posterior_center_R_relative_error")),
                      "posterior_X_relative_error": finite(h.get("posterior_center_X_relative_error")),
                      "historical_truth_log_weight": finite(h.get("joint_truth_log_mass")),
                      "regimes": regimes, "missing": missing,
                      **{f: avg(r[f] for r in regimes) if len(regimes) == len(REGIMES) else None for f in FIELDS}}
            record["historical_truth_weight"] = (float(np.exp(record["historical_truth_log_weight"]))
                                                  if record["historical_truth_log_weight"] is not None else None)
            records.append(record)
        all_rows[method] = records
        for regime in (*REGIMES, "paired_regime_mean"):
            saved = next((r for r in summary["summary"] if r["group"] == "all" and r["method"] == method
                          and r["model"] == "joint" and r["regime"] == regime), None)
            if saved is None:
                checks.append({"method": method, "regime": regime, "status": "saved_summary_missing"})
                continue
            source = records if regime == "paired_regime_mean" else [
                rr for r in records for rr in r["regimes"] if rr["regime"] == regime]
            for metric, expected in (("error", saved["metrics"]["error"]["mean"]),
                                     ("predicted_risk", saved["declared_Gaussian_integral"]["risk"]["mean"])):
                actual = avg(r[metric] for r in source)
                delta = abs(actual - expected)
                assert delta < 1e-12, (method, regime, metric, actual, expected)
                checks.append({"method": method, "regime": regime, "metric": metric,
                               "recomputed": actual, "saved": expected, "absolute_error": delta, "status": "passed"})
    rows = all_rows[METHODS[0]]
    strata = {}
    for field in ("H", "bank", "excitation", "innovation", "active_boundary", "true_name"):
        bins = defaultdict(list)
        for r in rows:
            bins[str(r.get(field))].append(r)
        strata[field] = {name: count_summary(group) for name, group in sorted(bins.items())}
    comparisons = {}
    for method in METHODS[1:]:
        pairs = [(a, b) for a, b in zip(rows, all_rows[method]) if a["error"] is not None and b["error"] is not None]
        comparisons[method] = {
            "available_paired_histories": len(pairs), "expected_histories": len(rows),
            "mean_BC_minus_comparator_error": avg(a["error"] - b["error"] for a, b in pairs),
            "BC_lower_error_histories": sum(a["error"] < b["error"] for a, b in pairs),
            "same_error_histories": sum(a["error"] == b["error"] for a, b in pairs),
            "BC_higher_error_histories": sum(a["error"] > b["error"] for a, b in pairs),
            "BC_mean_error": avg(a["error"] for a, b in pairs),
            "comparator_mean_error": avg(b["error"] for a, b in pairs),
        }
    return {"summary": count_summary(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
            "method_summaries": {m: count_summary(rr) for m, rr in all_rows.items()},
            "strata": strata, "paired": comparisons,
            "associations": [correlation(rows, f) for f in (
                "posterior_R_relative_error", "posterior_X_relative_error", "historical_truth_log_weight", "historical_truth_weight")],
            "histories": rows, "saved_summary_checks": checks,
            "field_note": "joint_truth_log_mass renamed historical_truth_log_weight: candidate log prior before future data, not truncation log mass"}


def select_margin(pairs, field):
    available = [p for p in pairs if finite(p.get(field)) is not None]
    if not available:
        return {"margin": None, "competitors": [], "reason": "no pair rows"}
    minimum = min(p[field] for p in available)
    return {"margin": minimum, "competitors": [p["competitor"] for p in available if abs(p[field] - minimum) < 1e-10],
            "truth_wins": minimum > 0.0}


def geometry_analysis(diagnostics, histories):
    by_index = {int(r["spec"]["index"]): r for r in diagnostics["records"]}
    result = []
    for h in histories:
        r = by_index.get(h["index"])
        method_map = {m["method"]: m for m in r.get("methods", [])} if r else {}
        rankings = {rk["regime"]: rk for rk in r.get("rankings", [])} if r else {}
        for regime in REGIMES:
            actions = {a["action_key"]: a for a in rankings.get(regime, {}).get("actions", [])}
            for method in METHODS[:3]:
                key = method_map.get(method, {}).get("action_key")
                action = actions.get(key)
                row = {"index": h["index"], "regime": regime, "method": method, "action_key": key,
                       "scope": "posthoc noise-free geometry, not an error probability"}
                if action is None or not action.get("clean_geometry"):
                    result.append({**row, "available": False, "missing_reason": "no saved clean geometry",
                                   "upstream_status": h["status"]})
                    continue
                pairs = []
                for p in action["clean_geometry"].get("pair_rows", []):
                    posterior = float(p["log_posterior_score_gap"])
                    density = float(p["log_density_gap"])
                    residual = 0.5 * float(p["Mahalanobis_square_gap"])
                    pair = {"competitor": p["competitor"], "full": posterior, "uniform_weight": density,
                            "history_weight": posterior - density, "determinant": density - residual,
                            "residual_only": residual, "euclidean": float(p["whitened_square_gap"])}
                    assert abs(pair["full"] - sum(pair[k] for k in ("history_weight", "determinant", "residual_only"))) < 1e-9
                    pairs.append(pair)
                result.append({**row, "available": True, "pair_rows": pairs,
                               "empirical_error": finite(action.get("empirical_error")),
                               "models": {name: select_margin(pairs, name) for name in (
                                   "full", "uniform_weight", "history_weight", "determinant", "residual_only", "euclidean")}})
    aggregates = {}
    for method in METHODS[:3]:
        available = [r for r in result if r["method"] == method and r["available"]]
        aggregates[method] = {
            "expected_history_regime_slots": len(histories) * len(REGIMES),
            "available_history_regime_slots": len(available),
            "independent_available_histories": len({r["index"] for r in available}),
            "missing_history_regime_slots": len(histories) * len(REGIMES) - len(available),
            "negative_full_margin": sum(r["models"]["full"]["margin"] < 0 for r in available),
            "negative_density_margin": sum(r["models"]["uniform_weight"]["margin"] < 0 for r in available),
            "density_positive_weight_flips_negative": sum(
                r["models"]["uniform_weight"]["margin"] > 0 and r["models"]["full"]["margin"] < 0 for r in available),
            "density_negative_weight_rescues": sum(
                r["models"]["uniform_weight"]["margin"] < 0 and r["models"]["full"]["margin"] > 0 for r in available),
            "euclidean_positive_full_negative": sum(
                r["models"]["euclidean"]["margin"] > 0 and r["models"]["full"]["margin"] < 0 for r in available),
            "zero_margin_tie_tolerance": 1e-10,
            "zero_margin_ties": {model: sum(abs(r["models"][model]["margin"]) <= 1e-10 for r in available)
                                 for model in ("full", "uniform_weight", "residual_only", "euclidean")},
            "negative_full_slot_mean_empirical_error": avg(
                r["empirical_error"] for r in available if r["models"]["full"]["margin"] < 0),
            "other_slot_mean_empirical_error": avg(
                r["empirical_error"] for r in available if r["models"]["full"]["margin"] >= 0),
            "negative_full_slot_share_of_all_empirical_errors": (
                sum(r["empirical_error"] for r in available if r["models"]["full"]["margin"] < 0)
                / sum(r["empirical_error"] for r in available)
                if sum(r["empirical_error"] for r in available) else None),
        }
    return {"summary": aggregates, "records": result,
            "scope": "two regimes per independent history; counts are diagnostic slots, not independent sample size; each scoring rule reselects its best rival"}


def gaussian_terms(observations, means, covariances):
    determinant, residual = [], []
    for mean, covariance in zip(means, covariances):
        L = cholesky(covariance, lower=True)
        white = solve_triangular(L, (observations - mean).T, lower=True)
        determinant.append(-float(np.log(np.diag(L)).sum()))
        residual.append(-0.5 * np.sum(white * white, axis=0))
    return np.asarray(determinant), np.asarray(residual).T


def conditional_density(observations, means, covariances, split):
    marginal, conditional = [], []
    for mean, covariance in zip(means, covariances):
        aa, ba = covariance[:split, :split], covariance[split:, :split]
        La = cholesky(aa, lower=True)
        gain_t = solve_triangular(La, ba.T, lower=True)
        gain_t = solve_triangular(La.T, gain_t, lower=False)
        adjusted_mean = mean[split:] + (observations[:, :split] - mean[:split]) @ gain_t
        schur = covariance[split:, split:] - ba @ gain_t
        Lb = cholesky((schur + schur.T) / 2, lower=True)
        ra = solve_triangular(La, (observations[:, :split] - mean[:split]).T, lower=True)
        rb = solve_triangular(Lb, (observations[:, split:] - adjusted_mean).T, lower=True)
        marginal.append(-np.log(np.diag(La)).sum() - 0.5 * np.sum(ra * ra, axis=0))
        conditional.append(-np.log(np.diag(Lb)).sum() - 0.5 * np.sum(rb * rb, axis=0))
    return np.asarray(marginal).T, np.asarray(conditional).T


def raw_analysis(raw_path):
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    if raw["spec"]["index"] != 0 or raw["spec"]["phase"] != "evaluation":
        raise ValueError("raw sensitivity is prespecified only for evaluation history 0")
    output, audit_errors, legacy_rows = [], [], []
    for method in METHODS[:3]:
        key = raw["methods"][method]["action_key"]
        action = raw["actions"][key]
        bank = action["predictions"]["joint"]["forecast"]["bank"]
        means = np.asarray(bank["means"][0], dtype=float)
        cov = np.asarray(bank["covariances"][0], dtype=float)
        lp = np.asarray(bank["log_prior"], dtype=float)
        T, n = np.asarray(action["power_kw"]).shape
        split = 2 * T * n
        for regime in REGIMES:
            saved = action["actual"][regime]
            model = saved["models"]["joint"]
            truth = int(model["true_index"])
            clean = np.asarray(saved["clean_whitened"], dtype=float)
            saved_posterior = np.asarray(model["samples"]["posterior"], dtype=float)
            draws = len(saved_posterior)
            noise = np.random.default_rng(saved["noise_seed"]).normal(size=(draws, len(clean)))
            y = clean + noise
            determinant, residual = gaussian_terms(y, means, cov)
            density = determinant + residual
            scores = lp + density
            posterior = np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
            maximum_error = float(np.max(np.abs(posterior - saved_posterior)))
            stored_choices = np.argmax(saved_posterior, axis=1)
            choices = np.argmax(scores, axis=1)
            assert maximum_error < 1e-9, (method, regime, maximum_error)
            assert np.array_equal(choices, stored_choices)
            assert np.array_equal(choices == truth, np.asarray(model["samples"]["correct"], dtype=bool))
            ma, co = conditional_density(y, means, cov, split)
            schur_error = float(np.max(np.abs(ma + co - density)))
            assert schur_error < 1e-9, schur_error
            clean_det, clean_res = gaussian_terms(clean[None, :], means, cov)
            clean_score = lp + clean_det + clean_res[0]
            rivals = [k for k in range(len(lp)) if k != truth]
            rival = max(rivals, key=lambda k: clean_score[k])
            clean_ma, clean_co = conditional_density(clean[None, :], means, cov, split)
            oracle_means = means.copy()
            oracle_means[truth] = clean
            oracle_det, oracle_res = gaussian_terms(y, oracle_means, cov)
            oracle_scores = lp + oracle_det + oracle_res
            oracle_uniform = oracle_det + oracle_res
            variants = {"stored_full": scores, "uniform_weights": density,
                        "truth_mean_oracle": oracle_scores, "truth_mean_oracle_uniform": oracle_uniform,
                        "conditional_B_given_A_with_history_weights": lp + co}
            output.append({
                "index": 0, "method": method, "action_key": key, "regime": regime,
                "true_index": truth, "true_name": raw["candidate_names"][truth],
                "conditional_noise_draws": draws, "independent_histories": 1,
                "dimensions": {"total": len(clean), "terminal_PQ": split, "voltage_rootPQ": len(clean) - split},
                "posterior_max_absolute_replay_error": maximum_error,
                "argmax_replay_exact": True, "Schur_joint_logdensity_max_error": schur_error,
                "variants": {name: {"error_count": int(np.sum(np.argmax(value, axis=1) != truth)),
                                     "error_rate": float(np.mean(np.argmax(value, axis=1) != truth)),
                                     "first_noise_selected": raw["candidate_names"][int(np.argmax(value[0]))],
                                     "selected_counts": np.bincount(np.argmax(value, axis=1), minlength=len(lp)).tolist()}
                             for name, value in variants.items()},
                "clean_best_rival": {"index": rival, "name": raw["candidate_names"][rival],
                    "history_weight_gap": float(lp[truth] - lp[rival]),
                    "determinant_gap": float(clean_det[truth] - clean_det[rival]),
                    "residual_gap": float(clean_res[0, truth] - clean_res[0, rival]),
                    "full_gap": float(clean_score[truth] - clean_score[rival]),
                    "terminal_PQ_marginal_density_gap": float(clean_ma[0, truth] - clean_ma[0, rival]),
                    "conditional_voltage_rootPQ_density_gap": float(clean_co[0, truth] - clean_co[0, rival])},
            })
            legacy = raw.get("legacy_joint_audit", {}).get("results", {}).get(regime + "__" + key)
            if legacy is not None:
                legacy_rows.append({"index": 0, "method": method, "regime": regime, "action_key": key,
                                    **{field: legacy.get(field) for field in ("selected", "M", "complete", "global_optimality_certified", "losses", "failures", "history_scope")},
                                    "Bayes_first_noise_selected": raw["candidate_names"][int(choices[0])]})
    return {"scope": "one prespecified history only, no re-fitting or new physical simulation",
            "oracle_scope": "truth mean replacement merges line/background/source/single-center errors; not deployable and not a performance upper bound",
            "records": output, "failures": audit_errors,
            "stored_legacy_AC_excerpt": legacy_rows,
            "legacy_scope": "read-only saved joint R/X AC refits; P/Q fixed input and different objective, one noise draw, local fits, small SSE margins; not overall repair or calibrated confidence"}


def markdown(result):
    history = result["history_analysis"]
    stats = history["summary"]
    lines = ["# 既有贝叶斯响应误差定位", "",
             "本检查只重算已存证据；新增 AC、拟合、MILP、观测和计划调用均为 0。", "",
             f"预定历史 {stats['expected_histories']}，完整可评价 {stats['available_histories']}，缺失 {stats['missing_histories']}。两个未来状态先在每份历史内平均，32 次噪声不增加独立样本量。", "",
             f"贝叶斯移动充放电：预计风险 {100*stats['mean_predicted_risk']:.4f}%，实际误判 {100*stats['mean_error']:.4f}%。",
             f"零误判/全部误判/混合历史：{stats['zero_error_histories']}/{stats['all_error_histories']}/{stats['mixed_error_histories']}。",
             f"最差 20/25 份历史占全部错误的 {100*stats['top20_error_share']:.2f}%/{100*stats['top25_error_share']:.2f}%。",
             f"全部误判历史平均置信 {100*stats['all_error_mean_confidence']:.3f}%，预计风险 {100*stats['all_error_mean_predicted_risk']:.4f}%。", "",
             "## 无噪评分定位", "",
             "下面是历史×未来状态的诊断次数，并非独立历史数或平均风险。", "",
              "|方法|可得状态|联合评分错|去历史权重后错|历史权重救回|历史权重翻错|欧氏最近但联合评分错|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method, r in result["geometry_analysis"]["summary"].items():
        lines.append(f"|{method}|{r['available_history_regime_slots']}|{r['negative_full_margin']}|{r['negative_density_margin']}|{r['density_negative_weight_rescues']}|{r['density_positive_weight_flips_negative']}|{r['euclidean_positive_full_negative']}|")
    g = result["geometry_analysis"]["summary"][METHODS[0]]
    lines.extend(["", f"贝叶斯方案无噪评分为负的 {g['negative_full_margin']} 个状态，实际平均误判 {100*g['negative_full_slot_mean_empirical_error']:.4f}%；其余状态为 {100*g['other_slot_mean_empirical_error']:.4f}%。前者承载全部实际错误的 {100*g['negative_full_slot_share_of_all_empirical_errors']:.4f}%。这些是对保存数据的描述性拆分。"])
    lines.extend(["", "欧氏最近而联合评分错反映均值偏差、协方差加权及历史权重的耦合，不能仅归因于协方差大小。", "", "## 描述性关联", ""])
    for r in history["associations"]:
        rho = "缺失" if r["rho"] is None else f"{r['rho']:.4f}"
        lines.append(f"- {r['x']} 与误判的 Spearman：{rho}，历史 n={r['available_histories']}。")
    lines.extend(["", "## 预定历史 0 的独立重放", "",
                  "|方法|状态|原错误/32|统一权重|真均值替换|真均值+统一权重|去P/Q边际|", "|---|---|---:|---:|---:|---:|---:|"])
    for r in result["raw_audit"]["records"]:
        v = r["variants"]
        names = ("stored_full", "uniform_weights", "truth_mean_oracle", "truth_mean_oracle_uniform", "conditional_B_given_A_with_history_weights")
        lines.append(f"|{r['method']}|{r['regime']}|" + "|".join(str(v[n]["error_count"]) for n in names) + "|")
    lines.extend(["", "历史 0 的已存完整 AC 重拟合（只读取，未新增拟合）：", ""])
    for r in result["raw_audit"]["stored_legacy_AC_excerpt"]:
        lines.append(f"- {r['method']} / {r['regime']}：AC 选 {r['selected']}，M={r['M']:.6f}；贝叶斯同首个噪声选 {r['Bayes_first_noise_selected']}；全局证明={r['global_optimality_certified']}。")
    lines.extend(["", "真均值替换使用真值，仅定位综合均值误差；不可部署，也不构成性能上界。一个历史的敏感性不能推广为总体改善。", "",
                  f"与既有汇总逐项核对 {len(history['saved_summary_checks'])} 项；完整 JSON 保存缺失、分层、配对结果、各竞争者三项分数及所有输入哈希。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--raw-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = {"analysis": args.analysis, "history_metrics": args.analysis.with_name("history_metrics.csv"),
             "diagnostics": args.diagnostics, "raw_audit": args.raw_audit,
             "script": Path(__file__), "math": Path(__file__).with_name("ev_bayes_response_error_localization_math.md")}
    result = {"schema": "ev_bayes_response_error_localization_v1", "status": "running",
              "config": {"methods": METHODS, "regimes": REGIMES, "model": "joint", "raw_history": 0,
                         "posterior_tolerance": 1e-9, "summary_tolerance": 1e-12},
              "scope": "read-only saved-evidence postprocessing; no policy tuning, new samples or numerical inference",
              "new_calls": {"AC": 0, "fits": 0, "MILP": 0, "observations": 0, "plans": 0},
              "inputs": {k: {"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha(p)} for k, p in paths.items()},
              "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__},
              "command": sys.argv, "failures": []}
    try:
        summary = load(args.analysis)
        result["policy_id"] = summary["policy_id"]
        result["history_analysis"] = history_analysis(summary, paths["history_metrics"])
        result["geometry_analysis"] = geometry_analysis(load(args.diagnostics), result["history_analysis"]["histories"])
        result["raw_audit"] = raw_analysis(args.raw_audit)
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        result["failures"].append({"type": type(exc).__name__, "message": str(exc)})
    result["seconds"] = time.perf_counter() - started
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    if result["status"] == "complete":
        (args.output / "result.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    else:
        (args.output / "result.md").write_text("# 检查失败\n\n" + json.dumps(result["failures"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "seconds": result["seconds"], "output": str(args.output), "failures": result["failures"]}, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
