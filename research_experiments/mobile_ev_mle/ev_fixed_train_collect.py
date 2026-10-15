"""Saved-record rank calibration for separate weak8 and rich32 policies.

Pure standard-library adapter: no fitter, selector, simulator, NumPy or solver
is imported. The score uses validation+future SSE at the saved training lines.
The two training conditions share base histories and are calibrated separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

from . import ev_policy_calibration as cal

METHODS = ("none", "constant", "quality_fixed", "quality_mobile", "random_mobile",
    "charge_fixed", "v2g_fixed", "charge_mobile", "v2g_mobile", "refined_v2g_mobile", "capped_v2g_mobile")
PRIMARY = ("refined_v2g_mobile", "capped_v2g_mobile")
CANDIDATES = ("clade_01", "clade_02", "clade_12", "star")
CONDITIONS = ("weak8", "rich32")
SCHEMA = "ev_fixed_train_standard_records_v1"


def read_json(path):
    path = Path(path);data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def write_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def collector_manifest():
    return dict(source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(cal.__file__))},
        imports="standard library and ev_policy_calibration only", new_fit_calls=0,
        new_selector_calls=0, old_threshold_reused=False)


def _finite_loss(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _fit_rows(rows, names):
    if not isinstance(rows, list):return None
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):return None
        name = row.get("candidate", row.get("name"))
        if name in indexed or name not in names:return None
        if row.get("name", name) != name:return None
        if row.get("success") is not True or not _finite_loss(row.get("loss")):return None
        if any(not isinstance(row.get(k), list) or not row[k] or not all(_finite_loss(v) for v in row[k])
               for k in ("r", "x")) or len(row["r"]) != len(row["x"]):return None
        indexed[name] = row
    return indexed if set(indexed) == set(names) else None


def _condition(trial):
    if trial.get("condition") is not None:return trial["condition"]
    config = trial.get("config", {})
    return "weak8" if config.get("training_points") == 8 and not config.get("training_support") else None


def make_standard_record(trial, *, policy_id, reference_id, config_sha256,
                         source_manifest_sha256, stratum="declared"):
    """Convert one saved trial; failed computations remain keep-all records.

    Complete losses must equal each candidate's validation SSE plus actual
    future SSE, with both sets of R/X exactly equal to its training arrays.
    Candidate coverage failures remain separate from numerical/operation ones.
    """
    expected = dict(policy_id=policy_id, config_sha256=config_sha256,
        source_manifest_sha256=source_manifest_sha256)
    if cal.freeze_identity(trial) != expected:
        raise ValueError("trial config/source differs from frozen policy")
    if trial.get("source_unchanged") is not True or trial.get("source_changed_during_trial"):
        raise ValueError("trial source was changed or not verified unchanged")
    for key in ("validation_used_by_selector", "truth_used_by_selector", "future_noise_used_by_selector"):
        if trial.get(key, trial.get("config", {}).get(key, False)) is not False:
            raise ValueError("selection information boundary violated: "+key)
    bank = trial.get("candidates", [])
    names = [row["name"] for row in bank]
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("saved candidate bank must contain distinct names")
    raw_true = trial.get("true_name")
    true_name = raw_true if isinstance(raw_true, str) and raw_true else "__uncovered_truth__"
    rows = trial.get("methods", [])
    if not isinstance(rows, list):raise ValueError("saved methods must be a list")
    indexed = {r.get("name"):r for r in rows}
    if len(indexed) != len(rows) or any(name not in METHODS for name in indexed):
        raise ValueError("duplicate or unexpected saved method")
    training = trial.get("training_result", {})
    validation = trial.get("validation_result", {})
    tr = _fit_rows(training.get("fits"), names)
    vr = _fit_rows(validation.get("fits"), names)
    historical_ok = training.get("all_complete") is True and not training.get("failures") and tr is not None
    validation_ok = validation.get("all_complete") is True and not validation.get("failures") and vr is not None
    bank_matches = set(names) == set(CANDIDATES)
    methods = {}
    for name in METHODS:
        row = indexed.get(name, {})
        raw_status = row.get("status", "missing_method_failure")
        status = raw_status
        losses = row.get("combined_loss", {})
        noisy = row.get("noisy", {})
        fr = _fit_rows(noisy.get("fits"), names)
        if not historical_ok:
            status = "historical_fit_failure"
        elif not validation_ok:
            status = "validation_fit_failure"
        elif not bank_matches:
            status = "candidate_bank_fit_failure"
        elif raw_status in ("selection_failure", "operation_failure", "design_failure") or row.get("power_kw") is None:
            status = "operation_design_failure" if name in indexed else "missing_method_fit_failure"
        elif raw_status == "complete":
            if noisy.get("all_complete") is not True or noisy.get("failures") or fr is None:
                status = "future_fit_failure"
            elif not isinstance(losses, dict) or set(losses) != set(names) or not all(_finite_loss(v) for v in losses.values()):
                status = "invalid_fixed_fit_loss_failure"
            elif any(losses[n] != vr[n]["loss"]+fr[n]["loss"] for n in names):
                status = "combined_fixed_fit_loss_mismatch_failure"
            elif any(vr[n].get(k) != tr[n].get(k) or fr[n].get(k) != tr[n].get(k)
                     for n in names for k in ("r", "x")):
                status = "changed_training_lines_fit_failure"
        methods[name] = dict(status=status, raw_status=raw_status,
            losses=dict(losses) if status == "complete" else {},
            objective="validation+actual_future output SSE at fixed training R/X",
            training_loss_included=False)
    history = trial.get("history_id", trial.get("seed"))
    unit = trial.get("independent_unit_id", trial.get("seed"))
    if history is None or unit is None or isinstance(history, bool) or isinstance(unit, bool):
        raise ValueError("missing independent base history identifier")
    return dict(**expected, reference_id=reference_id, history_id=str(history), independent_unit_id=str(unit),
        stratum=stratum, true_name=true_name, raw_true_name=raw_true, candidate_names=names,
        candidate_miss=bool(trial.get("candidate_miss") or true_name not in names), methods=methods,
        training_condition=_condition(trial), base_history_is_paired_across_conditions=True,
        source_interface=1, P_Q_fixed_inputs=True, adapter="fixed_train_saved_validation_plus_future_no_refitting")


def _planned(freeze, key, supplied):
    values = list(supplied if supplied is not None else freeze["config"][key+"_seeds"])
    values = [str(v) for v in values]
    if not values or len(set(values)) != len(values):raise ValueError("duplicate/empty planned history identifiers")
    return values


def _read_phase(directory, freeze, phase, planned, condition, spec):
    records, manifest, seen = [], [], set()
    if directory is None:return records, manifest
    for path in sorted(Path(directory).glob("trial_*.json.gz")):
        before = path.stat();data = path.read_bytes();after = path.stat()
        if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
            raise ValueError("raw changed during read: "+path.name)
        trial = json.loads(gzip.decompress(data))
        seed = str(trial.get("seed"))
        if seed in seen or seed not in planned:raise ValueError("duplicate or unplanned base history")
        seen.add(seed)
        if trial.get("phase", phase) != phase:raise ValueError("saved trial phase differs")
        if _condition(trial) != condition:raise ValueError("training conditions cannot be pooled")
        record = make_standard_record(trial, **cal.freeze_identity(freeze), reference_id=spec.reference_id)
        records.append(record)
        manifest.append(dict(name=path.name, seed=seed, phase=phase,
            sha256=hashlib.sha256(data).hexdigest(), compressed_bytes=len(data), mtime_ns=after.st_mtime_ns))
        del trial, data
    return records, manifest


def _balance(records, planned_count, phase):
    counts = dict(Counter(r["true_name"] for r in records))
    target = 9 if phase == "calibration" else 3
    expected = 36 if phase == "calibration" else 12
    complete = len(records) == planned_count
    coverage_ok = all(not r["candidate_miss"] for r in records)
    balanced = complete and planned_count == expected and counts == {name:target for name in CANDIDATES}
    if complete and coverage_ok and not balanced:
        raise ValueError("complete formal "+phase+" must have "+str(target)+" histories per true candidate")
    return dict(actual=len(records), planned=planned_count, complete=complete, true_candidate_counts=counts,
        prespecified_balanced_design=balanced, candidate_coverage_complete=coverage_ok)


def risk_report_from_standard_records(bundle):
    """Portable statistical replay; performs no raw file read or optimization."""
    if bundle.get("schema") != SCHEMA:raise ValueError("unknown fixed-training standard-record schema")
    if cal.object_hash({k:v for k,v in bundle.items() if k != "bundle_id"}) != bundle.get("bundle_id"):
        raise ValueError("standard-record content/hash mismatch")
    spec = cal.CalibrationSpec(**bundle["spec"])
    if spec.methods != METHODS or spec.familywise or spec.candidates != CANDIDATES:
        raise ValueError("fixed-training report requires the declared individual methods/candidates")
    if bundle["condition"] not in CONDITIONS:raise ValueError("unknown training condition")
    cr, er = bundle["calibration"], bundle["evaluation"]
    balances = {}
    for phase, records in (("calibration", cr), ("evaluation", er)):
        planned = bundle["planned_"+phase+"_ids"]
        if any(r["history_id"] not in planned for r in records):raise ValueError("unplanned standard record")
        if any(r["training_condition"] != bundle["condition"] for r in records):raise ValueError("mixed training conditions")
        balances[phase] = _balance(records, len(planned), phase)
    answer = dict(condition=bundle["condition"], alpha=spec.alpha, standard_records=bundle,
        planned_calibration=len(bundle["planned_calibration_ids"]), planned_evaluation=len(bundle["planned_evaluation_ids"]),
        actual_calibration=len(cr), actual_evaluation=len(er), phase_balance=balances,
        base_independent_history_units=len({r["independent_unit_id"] for r in cr+er}),
        paired_across_conditions=True, cross_condition_any_error_control=False,
        topology_probability=False, old_threshold_applied=False,
        scope="separate condition-wise rank calibration in its declared synthetic reference; same base seeds are paired across conditions")
    for label, methods, familywise in (("individual_policies", METHODS, False), ("primary_family", PRIMARY, True)):
        local = replace(spec, methods=methods, familywise=familywise)
        cdata = cr if methods == METHODS else [dict(r, methods={m:r["methods"][m] for m in methods}) for r in cr]
        edata = er if methods == METHODS else [dict(r, methods={m:r["methods"][m] for m in methods}) for r in er]
        calibration = cal.calibrate(cdata, local, expected_history_ids=[r["history_id"] for r in cr],
            planned_history_ids=bundle["planned_calibration_ids"])
        evaluation = cal.evaluate(edata, calibration, expected_history_ids=[r["history_id"] for r in er])
        answer[label] = dict(calibration=calibration, evaluation=evaluation)
    return answer


def collect_condition(calibration_dir, evaluation_dir=None, *, condition, planned_calibration_ids=None,
                      planned_evaluation_ids=None, alpha=.1, reference_id=None, stage_before=None):
    """Collect completed records, or save calibration first with evaluation=None.

    Pass the earlier cal-only report (object or path) as stage_before in the
    final call: both calibration objects and identities must remain exact.
    Execution-order evidence is owned by the external runner; an empty future
    directory alone is not used to assert that no evaluation process started.
    """
    if condition not in CONDITIONS:raise ValueError("condition must be weak8 or rich32")
    cd = Path(calibration_dir)
    freeze = read_json(cd/"freeze.json")
    if tuple(freeze.get("config", {}).get("methods", ())) != METHODS:
        raise ValueError("frozen policy must declare the eleven methods in their fixed order")
    identity = cal.freeze_identity(freeze)
    if evaluation_dir is not None:
        ef = read_json(Path(evaluation_dir)/"freeze.json")
        if cal.freeze_identity(ef) != identity:raise ValueError("calibration/evaluation policy versions differ")
    spec = cal.spec_from_freeze(freeze, reference_id=reference_id or "fixed_training_"+condition+"_synthetic_v1",
        candidates=CANDIDATES, methods=METHODS, strata=("declared",), alpha=alpha, familywise=False)
    planned_c = _planned(freeze, "calibration", planned_calibration_ids)
    planned_e = _planned(freeze, "evaluation", planned_evaluation_ids)
    if len(planned_c) != 36 or len(planned_e) != 12:
        raise ValueError("formal design requires 36 calibration and 12 evaluation base histories per condition")
    if set(planned_c) & set(planned_e):raise ValueError("calibration/evaluation planned histories overlap")
    cr, cm = _read_phase(cd, freeze, "calibration", planned_c, condition, spec)
    er, em = _read_phase(evaluation_dir, freeze, "evaluation", planned_e, condition, spec)
    bundle = dict(schema=SCHEMA, condition=condition, spec=spec.summary(), calibration=cr, evaluation=er,
        planned_calibration_ids=planned_c, planned_evaluation_ids=planned_e,
        raw_manifest=dict(calibration=cm, evaluation=em),
        collector_manifest=collector_manifest(), input_freeze_identity=identity,
        statistical_unit="one base asset/history per condition; paired across training conditions")
    bundle["bundle_id"] = cal.object_hash(bundle)
    answer = risk_report_from_standard_records(bundle)
    answer.update(collected_at_UTC=datetime.now(timezone.utc).isoformat(),
        collector_manifest=bundle["collector_manifest"], calibration_saved_without_evaluation=len(er)==0,
        calibration_complete=len(cr)==36, execution_order_must_be_verified_by_runner=True)
    if stage_before is not None:
        old = read_json(stage_before) if isinstance(stage_before, (str, Path)) else stage_before
        if old["condition"] != condition or old["actual_calibration"] != 36 or old["actual_evaluation"] != 0:
            raise ValueError("stage_before must be a complete calibration-only report for this condition")
        if old.get("collector_manifest") != answer["collector_manifest"]:
            raise ValueError("collector/calibrator source changed after the stage-before lock")
        old_replay = risk_report_from_standard_records(old["standard_records"])
        for group in ("individual_policies", "primary_family"):
            if old_replay[group]["calibration"] != old[group]["calibration"]:
                raise ValueError("stage-before calibration does not match its saved standard records")
            if answer[group]["calibration"] != old[group]["calibration"]:
                raise ValueError("calibration changed after the stage-before lock")
        answer["stage_before_calibration_exactly_preserved"] = True
        answer["stage_before_calibration_ids"] = {g:old[g]["calibration"]["calibration_id"]
            for g in ("individual_policies", "primary_family")}
    return answer


def paired_condition_units(reports):
    """Count shared units once; make no simultaneous cross-condition guarantee."""
    labels = [r["condition"] for r in reports]
    if len(set(labels)) != len(labels):raise ValueError("duplicate condition report")
    units = [{r["independent_unit_id"] for phase in ("calibration", "evaluation")
              for r in report["standard_records"][phase]} for report in reports]
    return dict(conditions=labels, condition_record_count=sum(len(u) for u in units),
        distinct_base_history_units=len(set().union(*units)) if units else 0,
        paired_base_units=len(set.intersection(*units)) if units else 0,
        condition_records_are_not_independent_histories=True, cross_condition_any_error_control=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--stage-before", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    answer = collect_condition(args.calibration, args.evaluation, condition=args.condition, stage_before=args.stage_before)
    write_json(args.output, answer)
    print("condition="+args.condition+" calibration="+str(answer["actual_calibration"])+" evaluation="+str(answer["actual_evaluation"]))


if __name__ == "__main__":main()
