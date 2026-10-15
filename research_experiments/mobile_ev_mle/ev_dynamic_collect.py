"""Read saved dynamic-EV trials; summarize, rank-calibrate, or evaluate.

This module imports only the standard library. It never runs a strategy,
simulation, AC fit, or solver. The independent unit is a saved base history.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

FAMILIES = ("star", "clade_01", "clade_02", "clade_12")
DOMAINS = ("charge_fixed", "v2g_fixed", "charge_mobile", "v2g_mobile")
RENAME = {"quality_fixed": "quality_charge_fixed",
          "quality_mobile": "quality_v2g_mobile",
          **{name: "design_"+name for name in DOMAINS}}
CONTRASTS = {
    "timing": ("design_v2g_fixed", "timing_reverse_fixed"),
    "amplitude": ("design_charge_fixed", "amplitude_half_fixed"),
    "discharge": ("design_v2g_fixed", "design_charge_fixed"),
    "movement": ("design_v2g_mobile", "design_v2g_fixed"),
    "charge_movement": ("design_charge_mobile", "design_charge_fixed"),
    "forecast_policy": ("design_v2g_mobile", "design_mean_v2g_mobile"),
}
for _domain in DOMAINS:
    CONTRASTS[_domain+"_vs_constant"] = ("design_"+_domain, "constant")
    CONTRASTS[_domain+"_vs_quality"] = ("design_"+_domain, "quality_"+_domain)
SCHEMA = "ev_dynamic_saved_collection_v1"
CAL_SCHEMA = "ev_dynamic_candidate_rank_thresholds_v1"
SCOPE = ("Method-wise, true-candidate-family rank calibration under exchangeability "
         "of independent histories in the declared synthetic distribution. "
         "Rank retention requires true-candidate coverage; omitted truths are counted "
         "separately and included in the empirical overall false-reject rate. "
         "No field guarantee or simultaneous multi-strategy error control.")


def object_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       allow_nan=False)+"\n").encode("utf-8")
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def collector_manifest():
    return dict(source_sha256={Path(__file__).name:
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                imports="standard library only", new_fit_calls=0,
                new_selector_calls=0, new_simulation_calls=0)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _loss(value):
    return _finite(value) and value >= 0


def _encode(value):
    return float(value) if math.isfinite(value) else "inf"


def _decode(value):
    if value == "inf":
        return math.inf
    if not _loss(value):
        raise ValueError("threshold must be a nonnegative finite number or 'inf'")
    return float(value)


def rank_threshold(scores, alpha=.1):
    """Conservative split-rank quantile; missing/nonfinite scores are +infinity."""
    if not _finite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    values = sorted(float(s) if _loss(s) else math.inf for s in scores)
    n = len(values)
    rank = math.ceil((n+1)*(1-alpha))
    return dict(n=n, rank=rank, threshold=_encode(values[rank-1] if rank <= n else math.inf),
                sorted_scores=[_encode(s) for s in values],
                missing_or_infinite=sum(not math.isfinite(s) for s in values))


def expected_methods(config):
    modes = config.get("select_modes")
    if not isinstance(modes, list) or not modes or len(set(modes)) != len(modes):
        raise ValueError("freeze must declare distinct select_modes")
    names = [RENAME.get(name, name) for name in modes]+["highlow"]
    if config.get("mean_policy_ablation"):
        names.append("design_mean_v2g_mobile")
    if config.get("diagnostics"):
        if "design_charge_fixed" in names:
            names.append("amplitude_half_fixed")
        if "design_v2g_fixed" in names:
            names.append("timing_reverse_fixed")
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("ambiguous declared method names")
    return names


def _identity(value):
    config, sources = value.get("config"), value.get("source_sha256")
    if not isinstance(config, dict) or not isinstance(sources, dict) or not sources:
        raise ValueError("missing frozen config/source manifest")
    policy = object_hash(dict(config=config, source_sha256=sources))
    if policy != value.get("policy_id"):
        raise ValueError("policy_id does not match config/source content")
    return dict(policy_id=policy, config_sha256=object_hash(config),
                source_manifest_sha256=object_hash(sources))


def _family(trial):
    index = trial.get("family_index")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(FAMILIES):
        raise ValueError("missing declared true family_index")
    family = FAMILIES[index]
    if trial.get("true_name") not in (None, family):
        raise ValueError("true_name differs from declared family")
    return family


def _fit_map(block, names):
    if not isinstance(block, dict) or block.get("all_complete") is not True or block.get("failures"):
        return None
    rows = block.get("fits")
    if not isinstance(rows, list):
        return None
    result = {}
    for row in rows:
        name = row.get("candidate", row.get("name"))
        if name not in names or name in result or row.get("name", name) != name:
            return None
        if row.get("success") is not True or not _loss(row.get("loss")):
            return None
        if any(not isinstance(row.get(k), list) or not row[k]
               or not all(_finite(v) and v > 0 for v in row[k]) for k in ("r", "x")):
            return None
        if len(row["r"]) != len(row["x"]):
            return None
        result[name] = row
    return result if set(result) == set(names) else None


def _block_failures(block):
    if not isinstance(block, dict):
        return dict(missing=True, failed_candidates=0, all_complete=False)
    failures = block.get("failures", {})
    return dict(missing=False, failed_candidates=len(failures) if isinstance(failures, (dict, list)) else 1,
                all_complete=block.get("all_complete") is True)


def _selection_record(trial, method):
    if method == "design_mean_v2g_mobile":
        return trial.get("mean_selection", {}).get("methods", {}).get("v2g_mobile", {})
    key = method.removeprefix("design_")
    key = {"quality_charge_fixed": "quality_fixed",
           "quality_v2g_mobile": "quality_mobile"}.get(key, key)
    return trial.get("selection", {}).get("methods", {}).get(key, {})


def _origin(trial, method):
    row = _selection_record(trial, method)
    raw = row.get("selected_origin")
    is_new = row.get("selected_is_new_MILP_proposal") is True
    if is_new:
        category = "new"
    elif raw and raw.startswith("quality"):
        category = "quality"
    elif raw in DOMAINS:
        category = "parent"
    elif raw == "MILP":
        category = "MILP_duplicate"
    elif raw:
        category = "deterministic_baseline"
    elif method.startswith("quality_"):
        category = "quality"
    else:
        category = "not_applicable"
    return dict(category=category, raw=raw, selected_is_new_MILP_proposal=is_new,
                recovered_proposal_failures=row.get("failures", []),
                initial_best_origin=row.get("initial_best_origin"),
                selected_score=row.get("selected_score"))


def make_record(trial, methods):
    """Extract one independent history, keeping failed methods in the denominator."""
    seed = trial.get("seed")
    if seed is None or isinstance(seed, bool):
        raise ValueError("missing independent history seed")
    family = _family(trial)
    bank = trial.get("candidates")
    bank_missing = not isinstance(bank, list) or not bank
    names = list(FAMILIES) if bank_missing else [row["name"] for row in bank]
    if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("candidate names must be distinct")
    omitted = bool(trial.get("candidate_miss") or (not bank_missing and family not in names))
    raw = trial.get("methods", [])
    if not isinstance(raw, list):
        raise ValueError("methods must be a list")
    indexed = {row["name"]: row for row in raw}
    if len(indexed) != len(raw) or set(indexed)-set(methods):
        raise ValueError("duplicate or undeclared method")
    training = trial.get("training_result", {})
    validation = trial.get("validation_result", {})
    tr, vr = _fit_map(training, names), _fit_map(validation, names)
    output = {}
    for method in methods:
        row = indexed.get(method, {})
        reasons = []
        noisy = row.get("noisy", {})
        fr = _fit_map(noisy, names)
        values = row.get("loss_by_candidate", {})
        if bank_missing:
            reasons.append("candidate_metadata_missing")
        if tr is None:
            reasons.append("training_fit_failure")
        if vr is None:
            reasons.append("validation_source_fit_failure")
        if row.get("status") != "complete" or row.get("power_kw") is None:
            reasons.append(row.get("status", "missing_method_failure"))
        if row.get("operation", {}).get("success") is not True:
            reasons.append("operation_failure")
        if fr is None:
            reasons.append("future_source_fit_failure")
        if not isinstance(values, dict) or set(values) != set(names) or not all(_loss(v) for v in values.values()):
            reasons.append("missing_or_invalid_candidate_loss")
        if not reasons:
            if any(not math.isclose(values[n], vr[n]["loss"]+fr[n]["loss"],
                                    rel_tol=1e-12, abs_tol=1e-9) for n in names):
                reasons.append("validation_future_SSE_mismatch")
            if any(vr[n][k] != tr[n][k] or fr[n][k] != tr[n][k] for n in names for k in ("r", "x")):
                reasons.append("training_lines_changed")
            if any(fr[n].get("diagnostics", {}).get("line_parameters_fixed") is False for n in names):
                reasons.append("future_lines_not_fixed")
        complete = not reasons
        loss_values = dict(values) if complete else {}
        margin = (min(v for n, v in loss_values.items() if n != family)-loss_values[family]
                  if complete and not omitted and family in loss_values and len(loss_values) > 1 else None)
        minimum = min(loss_values.values()) if loss_values else None
        nonconformity = (max(0., loss_values[family]-minimum)
                         if complete and not omitted and family in loss_values else "inf")
        operation = row.get("operation", {})
        output[method] = dict(status="complete" if complete else "failure",
            raw_status=row.get("status", "missing_method_failure"),
            failure_reasons=reasons, losses=loss_values, M=margin,
            saved_M=row.get("M"), calibration_score=nonconformity,
            source_failures=dict(validation=_block_failures(validation), future=_block_failures(noisy),
                                 noiseless_diagnostic=_block_failures(row.get("noiseless"))),
            energy={key:operation.get(key) for key in
                ("grid_energy_kwh", "charging_kwh", "discharging_kwh", "throughput_kwh",
                 "driving_kwh", "worker_minutes", "moves", "power_variance")},
            origin=_origin(trial, method), logical_future_fit_cost=noisy.get("cost", {}),
            logical_future_fit_seconds=noisy.get("seconds"))
    return dict(seed=str(seed), family=family, candidate_names=names,
                candidate_miss=omitted, candidate_metadata_missing=bank_missing,
                status=trial.get("status"), methods=output,
                training_parameter_error=trial.get("training_parameter_error", {}),
                training_lines={n:{k:row.get(k) for k in ("r", "x", "loss")}
                                for n,row in (tr or {}).items()},
                training_fit_failure=tr is None,
                training_cost=training.get("cost", {}),
                actual_solver_calls=trial.get("actual_solver_calls", {}),
                seconds=trial.get("seconds"))


def _read_directory(directory):
    directory = Path(directory)
    freeze_path = directory/"freeze.json"
    freeze = read_json(freeze_path)
    identity = _identity(freeze)
    methods = expected_methods(freeze["config"])
    records, manifest, seen = [], [], set()
    for path in sorted(directory.glob("trial*.json.gz")):
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("trial changed during read: "+path.name)
        trial = json.loads(gzip.decompress(data))
        if _identity(trial) != identity:
            raise ValueError("trial policy differs from frozen directory")
        if trial.get("source_unchanged") is not True:
            raise ValueError("source not verified unchanged during trial")
        for selector in ("selection", "mean_selection"):
            if any(trial.get(selector, {}).get(key, False) is not False
                   for key in ("uses_validation", "uses_truth", "uses_future_noise")):
                raise ValueError("selector information boundary violated")
        if trial.get("validation_used_by_selector", False) is not False or trial.get("future_noise_used_by_selector", False) is not False:
            raise ValueError("trial information boundary violated")
        record = make_record(trial, methods)
        if record["seed"] in seen:
            raise ValueError("duplicate independent history seed")
        seen.add(record["seed"])
        records.append(record)
        manifest.append(dict(name=path.name, seed=record["seed"],
                             sha256=hashlib.sha256(data).hexdigest(), compressed_bytes=len(data)))
    if not records:
        raise ValueError("no saved trial*.json.gz records")
    return freeze, identity, methods, records, manifest


def _stats(values, denominator):
    finite = [float(v) for v in values if _finite(v)]
    return dict(total=denominator, valid=len(finite), missing_or_failed=denominator-len(finite),
                mean=statistics.fmean(finite) if finite else None,
                median=statistics.median(finite) if finite else None,
                positive=sum(v > 0 for v in finite), negative=sum(v < 0 for v in finite),
                zero=sum(v == 0 for v in finite), sum=math.fsum(finite) if finite else None)


def _summary(records, methods):
    n = len(records)
    result = dict(independent_histories=n,
        family_counts=dict(Counter(r["family"] for r in records)),
        candidate_omissions=sum(r["candidate_miss"] for r in records),
        candidate_metadata_missing=sum(r["candidate_metadata_missing"] for r in records),
        training_fit_failures=sum(r["training_fit_failure"] for r in records),
        training_parameter_error={k:_stats([r["training_parameter_error"].get(k) for r in records], n)
                                  for k in ("R", "X")},
        runtime_seconds=_stats([r["seconds"] for r in records], n), methods={}, paired_contrasts={},
        cost_scope="Actual solver totals are per history; per-method logical fit costs can share cached curves.")
    cost_keys = sorted({k for r in records for k in r["actual_solver_calls"]})
    result["actual_solver_calls"] = {key:_stats([r["actual_solver_calls"].get(key) for r in records], n)
                                     for key in cost_keys}
    for method in methods:
        rows = [r["methods"][method] for r in records]
        result["methods"][method] = dict(M=_stats([row["M"] for row in rows], n),
            complete=sum(row["status"] == "complete" for row in rows),
            failed=sum(row["status"] != "complete" for row in rows),
            status_counts=dict(Counter(row["raw_status"] for row in rows)),
            failure_reasons=dict(Counter(reason for row in rows for reason in set(row["failure_reasons"]))),
            future_source_fit_failures=sum(not row["source_failures"]["future"]["all_complete"] for row in rows),
            validation_source_fit_failures=sum(not row["source_failures"]["validation"]["all_complete"] for row in rows),
            noiseless_diagnostic_fit_failures=sum(not row["source_failures"]["noiseless_diagnostic"]["all_complete"] for row in rows),
            selected_origin=dict(Counter(row["origin"]["category"] for row in rows)),
            selected_origin_raw=dict(Counter(str(row["origin"]["raw"]) for row in rows)),
            recovered_proposal_failures=sum(bool(row["origin"]["recovered_proposal_failures"]) for row in rows),
            energy={key:_stats([row["energy"].get(key) for row in rows], n)
                    for key in ("grid_energy_kwh", "charging_kwh", "discharging_kwh",
                                "throughput_kwh", "driving_kwh", "worker_minutes", "moves")},
            logical_future_fit_seconds=_stats([row["logical_future_fit_seconds"] for row in rows], n),
            logical_future_fit_cost={key:_stats([row["logical_future_fit_cost"].get(key) for row in rows], n)
                                     for key in ("ac_calls", "fit_calls", "solver_calls")})
    for name, (left,right) in CONTRASTS.items():
        if left not in methods or right not in methods:
            continue
        values = []
        for record in records:
            a,b = record["methods"][left]["M"],record["methods"][right]["M"]
            values.append(a-b if _finite(a) and _finite(b) else None)
        result["paired_contrasts"][name] = dict(left=left, right=right,
                                               delta_M=_stats(values,n))
    return result


def _base_report(freeze, identity, methods, records, manifest):
    return dict(schema=SCHEMA, **identity, source_sha256=freeze["source_sha256"],
                collector=collector_manifest(), methods=methods,
                seeds=[r["seed"] for r in records], trial_files=manifest,
                records=records, summary=_summary(records, methods),
                objective="validation+actual future output SSE at fixed training R/X",
                M_is_topology_probability=False,
                rank_retention_requires_true_candidate_coverage=True,
                candidate_omission_counted_in_overall_false_reject=True,
                rank_retention_requires_within_true_family_exchangeability=True,
                calibration_is_methodwise=True, simultaneous_multi_strategy_error_control=False,
                scope=SCOPE)


def _planned_seed_check(freeze, records, phase):
    planned = freeze["config"].get(phase+"_seeds")
    if planned is not None and any(r["seed"] not in {str(v) for v in planned} for r in records):
        raise ValueError("unplanned "+phase+" seed")


def _calibrate(report, freeze, *, alpha, expected_per_family):
    if not isinstance(expected_per_family, int) or isinstance(expected_per_family, bool) or expected_per_family < 1:
        raise ValueError("expected_per_family must be a positive integer")
    records,methods = report["records"],report["methods"]
    counts = Counter(r["family"] for r in records)
    if counts != {family:expected_per_family for family in FAMILIES}:
        raise ValueError("calibration requires "+str(expected_per_family)+" independent histories per true family")
    _planned_seed_check(freeze, records, "calibration")
    calibration = {}
    for method in methods:
        calibration[method] = {}
        for family in FAMILIES:
            rows = [r for r in records if r["family"] == family]
            calibration[method][family] = rank_threshold(
                [r["methods"][method]["calibration_score"] for r in rows], alpha)
    report.update(schema=CAL_SCHEMA, mode="calibrate", alpha=alpha,
        expected_per_family=expected_per_family, calibration=calibration,
        score="max(0,J_true-min_candidate J); missing/failure/omitted truth -> inf",
        decision="retain k when J_k-min_candidate J <= q(method,k); method failure -> keep all")
    report["thresholds_id"] = object_hash(report)
    return report


def _evaluate(report, freeze, thresholds):
    if thresholds.get("schema") != CAL_SCHEMA:
        raise ValueError("unknown calibration threshold schema")
    content = {k:v for k,v in thresholds.items() if k != "thresholds_id"}
    if object_hash(content) != thresholds.get("thresholds_id"):
        raise ValueError("threshold content/hash mismatch")
    for key in ("policy_id", "config_sha256", "source_manifest_sha256"):
        if thresholds.get(key) != report.get(key):
            raise ValueError("calibration/evaluation frozen policy differs")
    if thresholds.get("collector") != report["collector"]:
        raise ValueError("collector source differs from calibration")
    if thresholds.get("methods") != report["methods"]:
        raise ValueError("calibration/evaluation methods differ")
    for method in report["methods"]:
        if set(thresholds.get("calibration", {}).get(method, {})) != set(FAMILIES):
            raise ValueError("incomplete candidate-family thresholds")
        for family in FAMILIES:
            _decode(thresholds["calibration"][method][family].get("threshold"))
    if set(thresholds.get("seeds", [])) & set(report["seeds"]):
        raise ValueError("calibration and evaluation seeds overlap")
    _planned_seed_check(freeze, report["records"], "evaluation")
    decisions = []
    for record in report["records"]:
        local = dict(seed=record["seed"], family=record["family"],
                     candidate_miss=record["candidate_miss"], methods={})
        for method in report["methods"]:
            row = record["methods"][method]
            failed = row["status"] != "complete"
            if failed:
                retained = list(record["candidate_names"])
            else:
                minimum = min(row["losses"].values())
                retained = []
                for candidate in record["candidate_names"]:
                    item = thresholds["calibration"][method].get(candidate)
                    q = math.inf if item is None else _decode(item["threshold"])
                    if row["losses"][candidate]-minimum <= q:
                        retained.append(candidate)
            rejected = record["candidate_miss"] or record["family"] not in retained
            local["methods"][method] = dict(retained=retained, failed_keep_all=failed,
                failure_reasons=row["failure_reasons"], false_reject=bool(rejected),
                unique_correct=bool(not rejected and len(retained) == 1),
                unique_incorrect=bool(rejected and len(retained) == 1),
                unresolved=len(retained) != 1, retained_count=len(retained))
        decisions.append(local)
    def counts(items, method):
        rows = [r["methods"][method] for r in items]
        n = len(rows)
        return dict(total=n, failed_keep_all=sum(r["failed_keep_all"] for r in rows),
            candidate_omissions=sum(r["candidate_miss"] for r in items),
            false_reject=sum(r["false_reject"] for r in rows),
            false_reject_rate=sum(r["false_reject"] for r in rows)/n if n else None,
            unique_correct=sum(r["unique_correct"] for r in rows),
            unique_incorrect=sum(r["unique_incorrect"] for r in rows),
            unresolved=sum(r["unresolved"] for r in rows),
            valid_computations=sum(not r["failed_keep_all"] for r in rows))
    results = {}
    for method in report["methods"]:
        results[method] = dict(overall=counts(decisions,method),
            by_true_family={family:counts([r for r in decisions if r["family"] == family],method)
                            for family in FAMILIES})
    report.update(mode="evaluate", alpha=thresholds["alpha"],
                  thresholds_id=thresholds["thresholds_id"],
                  calibration_seeds=thresholds["seeds"],
                  calibration_trial_files=thresholds["trial_files"],
                  decisions=decisions, evaluation=results)
    return report


def collect(directory, *, mode="summarize", thresholds=None, alpha=.1, expected_per_family=9):
    """Read and aggregate raw trials; default calibration is 36=4*9 histories."""
    if mode not in ("summarize", "calibrate", "evaluate"):
        raise ValueError("unknown collection mode")
    freeze, identity, methods, records, manifest = _read_directory(directory)
    report = _base_report(freeze, identity, methods, records, manifest)
    if mode == "calibrate":
        return _calibrate(report, freeze, alpha=alpha, expected_per_family=expected_per_family)
    if mode == "evaluate":
        if thresholds is None:
            raise ValueError("evaluation requires frozen thresholds")
        saved = read_json(thresholds) if isinstance(thresholds, (str, Path)) else thresholds
        evaluated = _evaluate(report, freeze, saved)
        if isinstance(thresholds, (str, Path)):
            evaluated["threshold_file_sha256"] = hashlib.sha256(Path(thresholds).read_bytes()).hexdigest()
        return evaluated
    report["mode"] = "summarize"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("summarize", "calibrate", "evaluate"), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--alpha", type=float, default=.1)
    args = parser.parse_args()
    if args.mode == "evaluate" and args.thresholds is None:
        parser.error("--thresholds is required in evaluate mode")
    if args.mode != "evaluate" and args.thresholds is not None:
        parser.error("--thresholds applies only to evaluate mode")
    result = collect(args.directory, mode=args.mode, thresholds=args.thresholds, alpha=args.alpha)
    write_json(args.output, result)
    print(json.dumps(dict(mode=args.mode, policy_id=result["policy_id"],
                         histories=len(result["records"]), output=str(args.output))))


if __name__ == "__main__":
    main()
