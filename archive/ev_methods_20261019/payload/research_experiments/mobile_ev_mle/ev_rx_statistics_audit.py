"""Independent saved-data statistical audit; Python standard library only.

Rebuild J from validation and noisy-future fit losses, not saved J, M, collector
records, or analyzer outputs. Reproduce method-wise family rank thresholds and
evaluation decisions. All planned history/method slots survive input failures.
No scientific module, optimizer, AC predictor, or fit routine is imported.
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

DOMAINS = ("charge_fixed", "v2g_fixed", "charge_mobile", "v2g_mobile")
METHODS = ("none", "constant", *(p+"_"+d for p in ("linear", "square", "mean", "nearest") for d in DOMAINS), "highlow")
FAMILIES = ("star", "clade_01", "clade_02", "clade_12")
SUPPORTS = {
    "star": ((0, 1, 2), (0,), (1,), (2,)),
    "clade_01": ((0, 1, 2), (0, 1), (0,), (1,), (2,)),
    "clade_02": ((0, 1, 2), (0, 2), (0,), (1,), (2,)),
    "clade_12": ((0, 1, 2), (1, 2), (0,), (1,), (2,)),
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value):
    if value in ("inf", "+inf", "Infinity"):
        return math.inf
    if value in ("-inf", "-Infinity"):
        return -math.inf
    return float(value)


def serial(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    path = Path(path)
    content = path.read_bytes()
    if path.suffix == ".gz":
        content = gzip.decompress(content)
    return json.loads(content)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Checks:
    def __init__(self):
        self.counts = Counter()
        self.failures = []
        self.max_absolute_difference = {}

    def check(self, passed, label, *, category="structural", actual=None, expected=None):
        self.counts[category] += 1
        if not passed:
            self.failures.append(dict(label=label, category=category, actual=serial(actual), expected=serial(expected)))
        return bool(passed)

    def equal(self, actual, expected, label, *, category="numeric"):
        try:
            a, b = number(actual), number(expected)
            if math.isfinite(a) and math.isfinite(b):
                difference = abs(a-b)
                self.max_absolute_difference[category] = max(self.max_absolute_difference.get(category, 0.), difference)
                passed = math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
            else:
                passed = a == b
        except (ValueError, TypeError):
            passed = actual == expected
        return self.check(passed, label, category=category, actual=actual, expected=expected)


def _fit_map(result, candidates, checks, label, *, training=None):
    """Validate saved state metadata; does not evaluate AC or optimize a state."""
    result = result if isinstance(result, dict) else {}
    rows = result.get("fits", [])
    named = [r.get("candidate", r.get("name")) for r in rows]
    coverage = len(named) == len(set(named)) and set(named) == set(candidates)
    checks.check(len(named) == len(set(named)) and set(named) <= set(candidates)
        and (result.get("all_complete") is not True or coverage),
        label+"/fit_coverage_consistency", category="fit_state_coverage", actual=named, expected=candidates)
    complete = bool(coverage and result.get("all_complete") is True and not result.get("failures"))
    by_name = {}
    for row in rows:
        name = row.get("candidate", row.get("name"))
        attained = row.get("success") is True
        valid = attained and finite(row.get("loss")) and row["loss"] >= 0
        checks.check(not attained or valid,
            label+"/"+str(name)+"/success_and_loss", category="fit_state_metadata")
        if "half_loss" in row and finite(row.get("loss")):
            checks.equal(row["half_loss"], row["loss"]/2 if finite(row.get("loss")) else None,
                label+"/"+str(name)+"/half_loss", category="fit_half_loss")
        if training is not None and attained:
            parent = training.get(name, {})
            for coord in ("r", "x"):
                valid = checks.check(row.get(coord) == parent.get(coord) and row.get(coord) is not None,
                    label+"/"+str(name)+"/fixed_"+coord, category="fixed_training_lines") and valid
            diagnostic = row.get("diagnostics", {})
            valid = checks.check(diagnostic.get("line_parameters_fixed") is True
                and diagnostic.get("source_node") == 1 and diagnostic.get("P_Q_inputs_fixed") is True,
                label+"/"+str(name)+"/conditional_input_source_flags", category="fit_state_metadata") and valid
        by_name[name] = row
        complete = complete and valid
    return by_name, complete


def family_from_truth(trial):
    supports = tuple(tuple(s) for s in trial.get("truth", {}).get("supports", []))
    return next((name for name, value in SUPPORTS.items() if supports == value), None)


def extract_trial(trial, expected_seed, expected_family, checks):
    seed = str(expected_seed)
    family = family_from_truth(trial)
    checks.check(family == expected_family, seed+"/truth_structure", actual=family, expected=expected_family)
    bank = [r["name"] for r in trial.get("candidates", [])]
    checks.check(bool(bank) and len(bank) == len(set(bank)) and set(bank) <= set(FAMILIES), seed+"/candidate_bank")
    missed = family not in bank
    checks.check(trial.get("candidate_miss") is missed and trial.get("true_name") == (None if missed else family),
        seed+"/true_candidate_metadata")
    training, train_ok = _fit_map(trial.get("training_result", {}), bank, checks, seed+"/training")
    validation, val_ok = _fit_map(trial.get("validation_result", {}), bank, checks, seed+"/validation", training=training)
    rows = trial.get("methods", [])
    names = [r.get("name") for r in rows]
    checks.check(len(names) == len(set(names)) and set(names) <= set(METHODS), seed+"/declared_method_coverage", category="method_coverage")
    by_method = {r.get("name"): r for r in rows}
    output = dict(seed=seed, family=family or expected_family, candidates=bank,
        candidate_miss=missed, methods={})
    for method in METHODS:
        row = by_method.get(method, {})
        future, future_ok = _fit_map(row.get("noisy", {}), bank, checks, seed+"/"+method+"/noisy", training=training)
        failures = []
        for ok, reason in ((train_ok, "training_fit_failure"), (val_ok, "validation_fit_failure"),
                (future_ok, "noisy_future_fit_failure"), (row.get("status") == "complete", "method_status_failure"),
                (row.get("power_kw") is not None, "missing_power"),
                (row.get("operation", {}).get("success") is True, "operation_failure")):
            if not ok:
                failures.append(reason)
        complete = not failures
        J = {name: validation[name]["loss"]+future[name]["loss"] for name in bank} if complete else {}
        if complete and not all(finite(value) for value in J.values()):
            failures.append("combined_loss_nonfinite")
            complete, J = False, {}
        M = min((v for name, v in J.items() if name != family), default=math.inf)-J[family] if family in J and len(J)>1 else None
        score = J[family]-min(J.values()) if family in J else math.inf
        if complete:
            saved = row.get("loss_by_candidate", {})
            checks.check(set(saved) == set(J), seed+"/"+method+"/J_coverage", category="J_coverage")
            for candidate in bank:
                checks.equal(saved.get(candidate), J[candidate], seed+"/"+method+"/J/"+candidate, category="J_saved_vs_rebuilt")
            checks.equal(row.get("M"), M, seed+"/"+method+"/M", category="M_saved_vs_rebuilt")
        output["methods"][method] = dict(complete=complete, failure_reasons=failures,
            J=J, M=M, calibration_score=score)
    return output


def load_phase(directory, phase, checks):
    directory = Path(directory)
    freeze = read_json(directory/"freeze.json")
    cfg = freeze["config"]
    planned = [str(s) for s in cfg[phase+"_seeds"]]
    checks.check(len(planned) == len(set(planned)), phase+"/distinct_planned_seeds")
    methods = list(cfg["select_modes"])+["highlow"]
    checks.check(methods == list(METHODS), phase+"/all19_declared", actual=methods, expected=METHODS)
    source = freeze["source_sha256"]
    identity = dict(policy_id=object_hash(dict(config=cfg, source_sha256=source)),
        config_sha256=object_hash(cfg), source_manifest_sha256=object_hash(source))
    checks.check(identity["policy_id"] == freeze["policy_id"], phase+"/freeze_policy_hash", category="provenance")
    paths = list(sorted(directory.glob("trial_*.json.gz")))
    mapped = {p.name[len("trial_"):-len(".json.gz")]: p for p in paths}
    checks.check(set(mapped) == set(planned), phase+"/planned_raw_coverage", category="provenance", actual=list(mapped), expected=planned)
    records = {}
    files = []
    input_errors = []
    for index, seed in enumerate(planned):
        path = mapped.get(seed)
        expected_family = FAMILIES[index % len(FAMILIES)]
        try:
            if path is None:
                raise ValueError("planned raw file missing")
            data = path.read_bytes()
            files.append(dict(name=path.name, seed=seed, sha256=hashlib.sha256(data).hexdigest(), compressed_bytes=len(data)))
            trial = json.loads(gzip.decompress(data))
            checks.check(str(trial.get("seed")) == seed, phase+"/"+seed+"/seed", category="provenance")
            checks.check(trial.get("config") == cfg and trial.get("policy_id") == freeze["policy_id"]
                and trial.get("source_sha256") == source and trial.get("source_unchanged") is True,
                phase+"/"+seed+"/frozen_policy", category="provenance")
            checks.check(trial.get("validation_used_by_selector") is False and trial.get("future_noise_used_by_selector") is False,
                phase+"/"+seed+"/information_boundary", category="provenance")
            records[seed] = extract_trial(trial, seed, expected_family, checks)
        except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
            input_errors.append(dict(seed=seed, path=None if path is None else str(path), error=type(error).__name__+": "+str(error)))
            checks.check(False, phase+"/"+seed+"/raw_read", category="input_failure", actual=str(error))
            records[seed] = dict(seed=seed, family=expected_family, candidates=list(FAMILIES), candidate_miss=False,
                input_failure=True, methods={name:dict(complete=False, failure_reasons=["input_failure"],
                    J={}, M=None, calibration_score=math.inf) for name in METHODS})
    slots = [r["methods"][name] for r in records.values() for name in METHODS]
    return dict(directory=str(directory.resolve()), freeze=freeze, identity=identity,
        freeze_sha256=file_hash(directory/"freeze.json"), planned=planned, records=records,
        input_files=files, input_errors=input_errors,
        denominators=dict(independent_histories=len(planned),method_slots=19*len(planned),
            complete_method_slots=sum(row["complete"] for row in slots),
            failed_method_slots=sum(not row["complete"] for row in slots),
            candidate_omissions=sum(r["candidate_miss"] for r in records.values()),
            family_counts=dict(Counter(r["family"] for r in records.values()))))


def rank_threshold(scores, alpha):
    values = sorted(scores)
    rank = math.ceil((len(values)+1)*(1-alpha))
    return dict(n=len(values), rank=rank, threshold=values[rank-1] if rank <= len(values) else math.inf,
        sorted_scores=values, missing_or_infinite=sum(not math.isfinite(v) for v in values))


def decide(J, candidates, truth, thresholds, *, complete=True):
    if complete and J:
        minimum = min(J.values())
        retained = [name for name in candidates if J[name]-minimum <= thresholds[name]]
    else:
        retained = list(candidates)
    return dict(retained=retained, retained_count=len(retained), failed_keep_all=not complete,
        false_reject=truth not in retained, unique_correct=len(retained)==1 and truth in retained,
        unique_incorrect=len(retained)==1 and truth not in retained, unresolved=len(retained)!=1)


def descriptive(values, total, tolerance=1e-4):
    values = [v for v in values if finite(v)]
    return dict(total=total, valid=len(values), unavailable=total-len(values),
        mean=statistics.fmean(values) if values else None, median=statistics.median(values) if values else None,
        positive=sum(v>tolerance for v in values), ties=sum(abs(v)<=tolerance for v in values),
        negative=sum(v < -tolerance for v in values), sign_tolerance=tolerance)


def aggregate_decisions(records, decisions, method, family=None):
    selected = [r for r in records.values() if family is None or r["family"]==family]
    rows = [decisions[r["seed"]][method] for r in selected]
    keys = ("failed_keep_all", "false_reject", "unique_correct", "unique_incorrect", "unresolved")
    result = {key:sum(row[key] for row in rows) for key in keys}
    result.update(total=len(rows), valid_computations=sum(not row["failed_keep_all"] for row in rows),
        candidate_omissions=sum(r["candidate_miss"] for r in selected),
        false_reject_rate=result["false_reject"]/len(rows) if rows else None)
    return result


def compare_record_report(phase, report, checks):
    saved = {str(r["seed"]):r for r in report.get("records", [])}
    checks.check(set(saved)==set(phase["planned"]), "report/record_seed_coverage", category="reported_record")
    for seed, record in phase["records"].items():
        for method, own in record["methods"].items():
            stored = saved.get(seed, {}).get("methods", {}).get(method, {})
            checks.check((stored.get("status")=="complete") == own["complete"], seed+"/"+method+"/reported_complete", category="reported_record")
            if own["complete"]:
                for name, value in own["J"].items():
                    checks.equal(stored.get("losses", {}).get(name), value, seed+"/"+method+"/reported_J/"+name,
                        category="reported_J")
                checks.equal(stored.get("M"), own["M"], seed+"/"+method+"/reported_M", category="reported_M")
            checks.equal(stored.get("calibration_score"), own["calibration_score"], seed+"/"+method+"/reported_score",
                category="reported_score")
    actual_files = {r["seed"]:r["sha256"] for r in phase["input_files"]}
    saved_files = {str(r["seed"]):r["sha256"] for r in report.get("trial_files", [])}
    checks.check(actual_files==saved_files, "report/raw_file_hashes", category="provenance")
    for method in METHODS:
        values = [r["methods"][method]["M"] for r in phase["records"].values()
            if finite(r["methods"][method]["M"])]
        own = dict(total=len(phase["planned"]), valid=len(values),
            missing_or_failed=len(phase["planned"])-len(values),
            mean=statistics.fmean(values) if values else None,
            median=statistics.median(values) if values else None,
            sum=math.fsum(values),
            positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),zero=sum(v==0 for v in values))
        stored = report.get("summary", {}).get("methods", {}).get(method, {}).get("M", {})
        for key, value in own.items():
            checks.equal(stored.get(key), value, "report/"+method+"/M_summary/"+key, category="M_summary")


def audit(calibration_directory, evaluation_directory):
    checks = Checks()
    cal = load_phase(calibration_directory, "calibration", checks)
    eva = load_phase(evaluation_directory, "evaluation", checks)
    threshold_path = Path(calibration_directory)/"thresholds.json"
    evaluation_path = Path(evaluation_directory)/"evaluation.json"
    thresholds = read_json(threshold_path)
    evaluated = read_json(evaluation_path)
    checks.check(cal["freeze"]["config"]==eva["freeze"]["config"]
        and cal["freeze"]["source_sha256"]==eva["freeze"]["source_sha256"], "cal_eval/same_frozen_strategy", category="provenance")
    checks.check(not set(cal["planned"]) & set(eva["planned"]), "cal_eval/independent_seed_sets", category="provenance")
    checks.check(len(cal["planned"])==36 and len(eva["planned"])==12, "formal/36cal_12eval")
    for phase, report in ((cal, thresholds), (eva, evaluated)):
        checks.check(report.get("methods")==list(METHODS), "report/all19_methods", category="method_coverage")
        for key, value in phase["identity"].items():
            checks.check(report.get(key)==value, "report/"+key, category="provenance", actual=report.get(key), expected=value)
        checks.check(report.get("source_sha256")==phase["freeze"]["source_sha256"], "report/source_manifest", category="provenance")
        compare_record_report(phase, report, checks)
    checks.check(evaluated.get("threshold_file_sha256")==file_hash(threshold_path), "evaluation/threshold_file_sha256", category="provenance")
    checks.check(evaluated.get("thresholds_id")==thresholds.get("thresholds_id"), "evaluation/threshold_id", category="provenance")
    alpha = cal["freeze"]["config"]["alpha"]
    checks.equal(alpha, .1, "calibration/alpha")
    checks.equal(thresholds.get("alpha"), alpha, "threshold_report/alpha")
    checks.equal(evaluated.get("alpha"), alpha, "evaluation_report/alpha")
    checks.equal(thresholds.get("expected_per_family"), 9, "threshold_report/expected_per_family")
    rebuilt = {}
    for method in METHODS:
        rebuilt[method] = {}
        for family in FAMILIES:
            scores = [r["methods"][method]["calibration_score"] for r in cal["records"].values() if r["family"]==family]
            own = rank_threshold(scores, alpha)
            rebuilt[method][family] = own
            saved = thresholds.get("calibration", {}).get(method, {}).get(family, {})
            checks.check(own["n"]==9 and own["rank"]==9, method+"/"+family+"/nine_calibration_histories", category="threshold_sample")
            for key in ("n", "rank", "missing_or_infinite", "threshold"):
                checks.equal(saved.get(key), own[key], method+"/"+family+"/"+key, category="threshold_rank_value")
            saved_scores = saved.get("sorted_scores", [])
            checks.check(len(saved_scores)==len(own["sorted_scores"]), method+"/"+family+"/sorted_score_count", category="threshold_sorted_score")
            for index, value in enumerate(own["sorted_scores"]):
                checks.equal(saved_scores[index] if index<len(saved_scores) else None, value,
                    method+"/"+family+"/sorted_score/"+str(index), category="threshold_sorted_score")
    decisions = {}
    saved_decisions = {str(r["seed"]):r for r in evaluated.get("decisions", [])}
    checks.check(set(saved_decisions)==set(eva["planned"]), "evaluation/decision_seed_coverage", category="decision")
    for seed, record in eva["records"].items():
        decisions[seed] = {}
        for method, row in record["methods"].items():
            own = decide(row["J"], record["candidates"], record["family"],
                {family:rebuilt[method][family]["threshold"] for family in FAMILIES}, complete=row["complete"])
            decisions[seed][method] = own
            saved = saved_decisions.get(seed, {}).get("methods", {}).get(method, {})
            checks.check(set(saved.get("retained", []))==set(own["retained"]), seed+"/"+method+"/retained", category="retained_set")
            for key in ("retained_count", "failed_keep_all", "false_reject", "unique_correct", "unique_incorrect", "unresolved"):
                checks.check(saved.get(key)==own[key], seed+"/"+method+"/"+key, category="decision_flag", actual=saved.get(key), expected=own[key])
    evaluation = {}
    for method in METHODS:
        own = dict(overall=aggregate_decisions(eva["records"], decisions, method),
            by_true_family={family:aggregate_decisions(eva["records"], decisions, method, family) for family in FAMILIES})
        evaluation[method] = own
        for family, values in [("overall", own["overall"]), *own["by_true_family"].items()]:
            saved = evaluated.get("evaluation", {}).get(method, {})
            saved = saved.get("overall", {}) if family=="overall" else saved.get("by_true_family", {}).get(family, {})
            for key, value in values.items():
                checks.equal(saved.get(key), value, method+"/"+family+"/aggregate/"+key, category="decision_aggregate")
    comparisons = {}
    for domain in DOMAINS:
        for prefix in ("square", "mean", "linear"):
            left, right = "nearest_"+domain, prefix+"_"+domain
            rows = []
            for seed, record in eva["records"].items():
                a, b = record["methods"][left], record["methods"][right]
                good = a["complete"] and b["complete"] and finite(a["M"]) and finite(b["M"])
                rows.append(dict(seed=seed, family=record["family"], complete=good,
                    delta_M=a["M"]-b["M"] if good else None))
            comparisons[left+"__minus__"+right] = dict(left=left, right=right, rows=rows,
                delta_M=descriptive([r["delta_M"] for r in rows],len(eva["planned"])))
    return serial(dict(schema="ev_rx_independent_statistics_audit_v1", success=not checks.failures,
        check_count=sum(checks.counts.values()), check_counts=dict(checks.counts), differences=checks.failures,
        max_absolute_difference=checks.max_absolute_difference, calibration_denominators=cal["denominators"],
        evaluation_denominators=eva["denominators"], thresholds_rebuilt=rebuilt,
        decisions_rebuilt=decisions, evaluation_rebuilt=evaluation, same_domain_paired_differences=comparisons,
        calibration_rebuilt_records=cal["records"], evaluation_rebuilt_records=eva["records"],
        input_errors=cal["input_errors"]+eva["input_errors"],
        sources=dict(calibration_directory=cal["directory"],evaluation_directory=eva["directory"],
            calibration_raw=cal["input_files"],evaluation_raw=eva["input_files"],
            calibration_freeze_sha256=cal["freeze_sha256"],evaluation_freeze_sha256=eva["freeze_sha256"],
            thresholds_sha256=file_hash(threshold_path),evaluation_report_sha256=file_hash(evaluation_path),
            frozen_scientific_identity=cal["identity"],script_sha256=file_hash(__file__)),
        formulas=dict(J="validation fit SSE + noisy-future fit SSE; lines fixed to training",
            M="min incorrect candidate J - true candidate J",
            calibration_score="Jtrue - min all candidate J; failure/omitted truth -> +infinity",
            threshold="rank ceil((n+1)*(1-alpha)); n=9, alpha=.1 -> ninth ordered score",
            retention="Jcandidate-minJ <= that candidate family's threshold; failed method -> keep all candidates"),
        scope="Independent standard-library arithmetic from saved fit states; no AC/global-optimality/field-probability claim.",
        statistical_unit="One independent history; paired methods/families do not multiply its sample size.",
        new_fit_calls=0,new_AC_calls=0,new_solver_calls=0,imports="Python standard library only"))


def self_test():
    """Known rejection, ties, conservative failure and missing-truth fixtures."""
    threshold = rank_threshold([0.]*8+[1.], .1)
    assert threshold["n"]==9 and threshold["rank"]==9 and threshold["threshold"]==1.
    wrong = decide({"a":2.,"b":0.},["a","b"],"a",{"a":1.,"b":0.})
    assert wrong["retained"]==["b"] and wrong["false_reject"] and wrong["unique_incorrect"]
    tie = decide({"a":1.,"b":0.},["a","b"],"a",{"a":1.,"b":0.})
    assert tie["retained"]==["a","b"] and not tie["false_reject"] and tie["unresolved"]
    failed = decide({},["a","b"],"a",{"a":0.,"b":0.},complete=False)
    assert failed["failed_keep_all"] and not failed["false_reject"] and failed["unresolved"]
    omitted = decide({},["a","b"],"c",{"a":0.,"b":0.},complete=False)
    assert omitted["false_reject"] and omitted["retained_count"]==2
    assert math.isinf(rank_threshold([0.]*8+[math.inf],.1)["threshold"])
    return dict(success=True,fixtures=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-directory",type=Path)
    parser.add_argument("--evaluation-directory",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        if args.calibration_directory is None:
            return
    if any(v is None for v in (args.calibration_directory,args.evaluation_directory,args.output)):
        parser.error("calibration-directory, evaluation-directory and output are required")
    report = audit(args.calibration_directory,args.evaluation_directory)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(success=report["success"],checks=report["check_count"],differences=len(report["differences"]),
        calibration_slots=report["calibration_denominators"],evaluation_slots=report["evaluation_denominators"],
        output=str(args.output.resolve()))))
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
