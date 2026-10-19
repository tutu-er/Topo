"""Independent saved-model replay for the structural-information AC study.

No optimizer, study generator or study summarizer is called. Every H/N/joint
meter model comes from the saved record. Numerical failures remain in all
history and calibration denominators. Source checking is read-only.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np
from .likelihood import Observations, MeasurementModel, RadialCandidate, ac_predict, concat_observations
from .single_site_milp import SingleSiteConstraints, SingleSitePlan, check_single_site_plan
from .gradient_price_evidence import replay_fit, _bounds_audit, _close

HERE = Path(__file__).resolve().parent
CHANNELS = ("p", "q", "v", "master_p", "master_q", "source_p", "source_q")
METHODS = ("none", "constant_fixed", "quality_global", "quality_local", "information_seed",
           "information_fixed", "information_random", "information_designed", "nominal_information", "profile_designed")


def read_json(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n").encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def _obs(values):
    return Observations(**{key: None if value is None else np.asarray(value, float) for key, value in values.items()})


def _meter(values):
    if values.get("fit_bias") or values.get("calibration") is not None:
        raise ValueError("this frozen study requires known independently calibrated biases")
    return MeasurementModel(**{key: np.asarray(value, float) if isinstance(value, list) else value
                               for key, value in values.items()})


def _family(supports, n):
    return tuple(sorted({tuple(sorted(map(int, support))) for support in supports if 1 < len(support) < n}))


def _digest_policy(trial):
    data = json.dumps(dict(config=trial["config"], source_sha256=trial["source_sha256"]),
                      sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def audit_source(source_hashes, *, source_root=HERE, snapshot=None, check_current_bytes=True):
    """Check frozen bytes, path containment and optional snapshot without writes."""
    root = Path(source_root).resolve()
    allowed = root.parent
    errors, rows = [], []
    saved = None if snapshot is None else read_json(snapshot)
    if saved is not None and saved.get("source_sha256") != source_hashes:
        errors.append("source_snapshot_hash_map")
    for name, expected in source_hashes.items():
        target = (root/name).resolve()
        try:
            target.relative_to(allowed)
        except ValueError:
            errors.append("source_path_outside_research_experiments:"+name)
            continue
        current = target.read_bytes() if target.is_file() else None
        actual = None if current is None else hashlib.sha256(current).hexdigest()
        exact = actual == expected
        status = "exact" if exact else "missing" if current is None else "content_or_line_endings_changed"
        if saved is not None:
            try:
                original = base64.b64decode(saved["files"][name], validate=True)
                if hashlib.sha256(original).hexdigest() != expected:
                    errors.append("source_snapshot_bytes:"+name)
                elif current is not None and not exact and current.replace(b"\r\n", b"\n") == original.replace(b"\r\n", b"\n"):
                    status = "line_endings_only"
            except (KeyError, TypeError, ValueError):
                errors.append("source_snapshot_invalid:"+name)
        if check_current_bytes and not exact:
            errors.append("frozen_source_bytes:"+name)
        rows.append(dict(name=name, status=status, expected_sha256=expected, actual_sha256=actual))
    return dict(success=not errors, errors=errors, files=rows, checked_count=len(rows),
                exact_count=sum(row["status"] == "exact" for row in rows), restoration_attempted=False,
                current_source_bytes_required=check_current_bytes, snapshot_verified=saved is not None)


def _quality_audit(trial, history, historical, future, joint, errors):
    quality = trial["quality_record"]
    kind = trial["quality"]
    matrix = {"uniform": np.full((4, 3), 5),
              "temporal": np.tile(np.array([1, 9, 9, 1])[:, None], (1, 3)),
              "spatial": np.array([[9, 1, 1], [1, 9, 1], [1, 1, 9], [9, 9, 9]])}[kind]
    permutation = np.asarray(quality["row_permutation"])
    if permutation.shape != (4,) or set(permutation.tolist()) != set(range(4)):
        raise ValueError("invalid public quality permutation")
    matrix = matrix[permutation.astype(int)]
    _close(matrix, quality["v_repeat_counts"], "quality_V_counts", errors, atol=0., rtol=0.)
    if quality["v_total_raw_reads"] != 60 or int(matrix.sum()) != 60:
        errors.append("quality_total_V_raw_reads")
    if quality.get("kind") != kind or quality.get("public_plan_uses_truth") or quality.get("public_plan_uses_future_noise"):
        errors.append("public_quality_information")
    for name in CHANNELS:
        value = getattr(history, name)
        if value is None:
            if any(getattr(meter, name+"_sigma") is not None for meter in (historical, future, joint)):
                errors.append("sigma_for_absent_channel:"+name)
            continue
        tail = (history.n,) if name in ("p", "q", "v") else ()
        old = np.broadcast_to(np.asarray(getattr(historical, name+"_sigma"), float), (history.t, *tail))
        new = np.broadcast_to(np.asarray(getattr(future, name+"_sigma"), float), (4, *tail))
        combined = np.broadcast_to(np.asarray(getattr(joint, name+"_sigma"), float), (history.t+4, *tail))
        count = matrix if name == "v" else np.full((4, *tail), 5)
        if np.any(old <= 0) or np.any(new <= 0) or not np.isfinite(combined).all():
            errors.append("positive_finite_sigma:"+name)
        _close(old, np.broadcast_to(old[0], old.shape), "time_constant_raw_sigma:"+name, errors, atol=0., rtol=0.)
        _close(count, quality["per_channel_repeat_counts"].get(name), "quality_counts:"+name, errors, atol=0., rtol=0.)
        _close(new, old[0]/np.sqrt(count), "averaged_sigma:"+name, errors, atol=1e-15, rtol=1e-12)
        _close(combined, np.concatenate((old, new)), "joint_sigma:"+name, errors, atol=1e-15, rtol=1e-12)
    for channel in ("p", "q", "v"):
        for meter in (future, joint):
            _close(getattr(meter, "bias_"+channel), getattr(historical, "bias_"+channel),
                   "bias_not_averaged:"+channel, errors, atol=1e-15, rtol=0.)
    for key, model in (("history", historical), ("future", future), ("joint", joint)):
        for field, saved in quality[key+"_model"].items():
            actual = getattr(model, field)
            if actual is None or isinstance(actual, bool):
                if actual != saved:
                    errors.append("duplicate_model_record:"+key+":"+field)
            else:
                _close(actual, saved, "duplicate_model_record:"+key+":"+field, errors, atol=0., rtol=0.)
    return dict(kind=kind, V_raw_read_count=60, fixed_bias_verified=True,
                actual_H_N_joint_sigmas_checked=True, row_permutation=permutation.tolist())


def _seed_audit(trial, errors, require_drawn_quality):
    streams = np.random.SeedSequence(int(trial["seed"])).spawn(5)
    seeds = {name: int(stream.generate_state(1)[0]) for name, stream in zip(
        ("generator", "policy", "evaluation_noise", "public_quality", "design_integration"), streams)}
    if seeds != trial["seeds"]:
        errors.append("five_independent_seed_streams")
    rng = np.random.default_rng(seeds["evaluation_noise"])
    white = {name: rng.standard_normal((4, 3) if name in ("p", "q", "v") else (4,))
             for name in ("p", "q", "v", "master_p", "master_q")}
    for name, value in white.items():
        _close(value, trial["white_evaluation_noise"][name], "independent_future_noise:"+name, errors, atol=0., rtol=0.)
    public = np.random.default_rng(seeds["public_quality"])
    drawn = str(public.choice(("uniform", "temporal", "spatial"), p=trial["config"]["population"]["quality_mass"]))
    permutation = public.permutation(4).tolist()
    if permutation != trial["quality_record"]["row_permutation"]:
        errors.append("public_quality_permutation_seed")
    if require_drawn_quality and drawn != trial["quality"]:
        errors.append("formal_random_quality_choice")


def _truth_future(trial, action, model, white):
    truth, h = trial["truth"], len(trial["history_observations"]["p"])
    family = _family(truth["supports"], 3)
    candidate = RadialCandidate.from_supports("replay_truth", 3, family)
    physical = ac_predict(candidate, np.asarray(truth["r_pu"]), np.asarray(truth["x_pu"]),
        np.asarray(truth["p"])[h:]+np.asarray(action["p"]),
        np.asarray(truth["q"])[h:]+np.asarray(action["q"]),
        np.asarray(truth["source_p"])[h:], np.asarray(truth["source_q"])[h:], root_v=1.)
    means = {}
    for name in ("p", "q", "v", "master_p", "master_q"):
        value = physical[name]
        if name in ("p", "q", "v"):
            value = value+getattr(model, "bias_"+name)
        means[name] = value+np.broadcast_to(getattr(model, name+"_sigma"), value.shape)*np.asarray(white[name])
    return physical, means


def audit_trial(trial, *, calibration=None, expected=None, require_drawn_quality=False):
    """Audit one independent H or one explicit paired pilot quality condition."""
    began = perf_counter()
    errors, methods, history_replays = [], [], []
    try:
        if _digest_policy(trial) != trial["policy_id"]:
            errors.append("policy_id_digest")
        if expected is not None:
            for name in ("config", "source_sha256", "policy_id"):
                if trial[name] != expected[name]:
                    errors.append("frozen_"+name)
        if trial.get("selection_uses_truth") or trial.get("selection_uses_future_noise") or trial["selection"].get("selector_uses_truth") or trial["selection"].get("selector_uses_future_noise"):
            errors.append("selector_information_isolation")
        history = _obs(trial["history_observations"])
        models = {name: _meter(value) for name, value in trial["measurement_models"].items()}
        quality = _quality_audit(trial, history, models["history"], models["future"], models["joint"], errors)
        _seed_audit(trial, errors, require_drawn_quality)
        history_replays = [replay_fit(row, history, models["history"]) for row in trial["historical_fits"]]
        for index, replay in enumerate(history_replays):
            errors.extend("H["+str(index)+"]:"+error for error in replay["errors"])
        definitions = {row["name"]: row["supports"] for row in trial["historical_fits"]}
        for row in trial.get("proposal", {}).get("bank", []):
            definitions.setdefault(row["name"], row["supports"])
        names = trial["candidate_names"]
        if len(names) != len(set(names)):
            errors.append("duplicate_candidate_names")
        family = _family(trial["truth"]["supports"], history.n)
        matching = [name for name in names if name in definitions and _family(definitions[name], history.n) == family]
        covered = bool(matching)
        if covered != trial["true_covered"] or covered and trial["true_name"] not in matching:
            errors.append("truth_coverage_name")
        constraints = SingleSiteConstraints(**trial["device_constraints"])
        threshold = None if calibration is None else math.inf if calibration["relative"]["vacuous"] else float(calibration["relative"]["threshold"])
        seen, actual_joint_fit_count, actual_joint_ac_count, unique_joint_replay_count = {}, 0, 0, 0
        expected_methods = [row["method"] for row in trial["records"]]
        declared_methods = trial["config"].get("methods", trial["config"].get("method_names", list(trial["selection"]["methods"])))
        if len(expected_methods) != len(set(expected_methods)) or set(expected_methods) != set(declared_methods):
            errors.append("method_record_denominator")
        all_complete, true_gaps = True, []
        for row in trial["records"]:
            method, local, replays, losses = row["method"], [], [], {}
            action = row.get("action")
            physical = None
            if action is not None:
                plan = SingleSitePlan(action["site"], np.asarray(action["p"]), np.asarray(action["q"]),
                    np.asarray(action["sites"]), np.asarray(action["soc"]), action.get("solver", {}))
                if method == "none":
                    operation = dict(feasible=bool(np.all(plan.p == 0) and np.all(plan.q == 0)), no_EV=True)
                else:
                    operation = check_single_site_plan(plan, constraints)
                if row["status"] != "operational_failure" and not operation["feasible"]:
                    local.append("decoded_operation")
                if len(plan.p) != trial["config"]["future_points"]:
                    local.append("action_horizon")
                try:
                    physical, expected_values = _truth_future(trial, action, models["future"], trial["white_evaluation_noise"])
                    vmin, vmax = float(abs(physical["voltage"]).min()), float(abs(physical["voltage"]).max())
                    operational_voltage = .9 <= vmin and vmax <= 1.1
                    if row["status"] == "complete" and not operational_voltage:
                        local.append("complete_actual_voltage_violation")
                    if "operational_voltage_ok" in row and row["operational_voltage_ok"] != operational_voltage:
                        local.append("actual_voltage_flag")
                    for key, value in (("voltage_min", vmin), ("voltage_max", vmax)):
                        if key in row:
                            _close(value, row[key], key, local, atol=2e-10, rtol=1e-10)
                    if "observations" in row:
                        for name, value in expected_values.items():
                            _close(value, row["observations"][name], "generated_future:"+name, local, atol=2e-10, rtol=1e-10)
                except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                    if row["status"] != "physical_failure":
                        local.append("actual_truth_AC:"+str(error))
            else:
                operation = dict(feasible=False, recorded_failure=True)
                if row["status"] != "operational_failure":
                    local.append("missing_action_outside_operational_failure")
            if "observations" in row:
                future = _obs(row["observations"])
                joint = concat_observations(history, future)
                replays = [replay_fit(fit, joint, models["joint"]) for fit in row["fits"]]
                for fit, replay in zip(row["fits"], replays):
                    local.extend(fit["name"]+":"+error for error in replay["errors"])
                    if fit["success"] and replay.get("replayed"):
                        losses[fit["name"]] = replay["full_loss"]
                key = (np.asarray(action["p"], dtype="<f8").tobytes(), np.asarray(action["q"], dtype="<f8").tobytes())
                cached = key in seen
                if row.get("equivalent_action_fit_cache_hit") != cached:
                    local.append("joint_fit_cache_hit")
                if cached:
                    first = seen[key]
                    if row["fits"] != first["fits"]:
                        local.append("cached_saved_fits_differ")
                    if row.get("fit_calls") != 0 or row.get("ac_calls") != 0:
                        local.append("cache_charged_execution")
                else:
                    seen[key] = row
                    if row.get("fit_calls") != len(names):
                        local.append("actual_joint_fit_count")
                    actual_joint_fit_count += row.get("fit_calls", 0)
                    unique_joint_replay_count += sum(replay.get("replayed", False) for replay in replays)
                    actual_joint_ac_count += row.get("ac_calls", 0)
            elif row.get("fits"):
                local.append("fit_without_joint_measurements")
            complete = bool(row["status"] == "complete" and set(losses) == set(names))
            if row["status"] == "complete" and not complete:
                local.append("incomplete_saved_joint_fit")
            all_complete &= complete
            for name, value in losses.items():
                _close(value, row["losses"].get(name), "loss_table:"+name, local)
            gaps = {name: value-min(losses.values()) for name, value in losses.items()} if complete else {}
            margin = None
            if complete:
                winner = min(losses, key=lambda name: (losses[name], name))
                if row.get("winner") != winner:
                    local.append("winner")
                for name, value in gaps.items():
                    _close(value, row["gaps"].get(name), "candidate_gap:"+name, local)
                if covered:
                    true_name = trial["true_name"]
                    gap = gaps[true_name]
                    true_gaps.append(gap)
                    margin = min(value for name, value in losses.items() if name != true_name)-losses[true_name]
                    _close(gap, row.get("true_gap"), "true_gap", local)
                    _close(margin, row.get("signed_margin_full_SSE"), "signed_actual_margin", local)
            decisions = {}
            if threshold is not None:
                retained = [name for name in names if gaps[name] <= threshold] if complete else list(names)
                decisions = dict(retained_candidates=retained, false_rejection=bool(covered and trial["true_name"] not in retained),
                    only_truth_retained=bool(covered and retained == [trial["true_name"]]), unresolved=len(retained) > 1)
                for name, value in decisions.items():
                    if row.get(name) != value:
                        local.append("decision_"+name)
            errors.extend(method+":"+error for error in local)
            methods.append(dict(method=method, success=not local, errors=local, recorded_status=row["status"],
                operation=operation, fits=replays, losses=losses, joint_complete=complete,
                signed_margin_full_SSE=margin, decisions=decisions,
                selection_failure=trial["selection"]["methods"][method]["status"] == "selection_failure_fallback"))
        hcomplete = not trial["historical_failures"] and len(history_replays) == len(names) and all(
            row["success"] and row.get("loss") is not None for row in trial["historical_fits"])
        complete = bool(hcomplete and all_complete and covered)
        if complete != trial["complete"]:
            errors.append("trial_complete")
        score = max(true_gaps) if complete else math.inf
        if complete:
            _close(score, trial["relative_score"], "relative_score", errors)
        elif trial.get("relative_score") is not None and not math.isinf(float(trial["relative_score"])):
            errors.append("failed_calibration_score_not_infinite")
        if trial["calibration_score_is_infinite"] != (not complete):
            errors.append("calibration_infinity_flag")
        if actual_joint_fit_count != trial["actual_joint_fit_calls"] or len(seen) != trial["joint_unique_action_count"]:
            errors.append("cached_actual_fit_denominator")
        expected_total = trial["historical_cost"]["ac_calls"]+trial["selection"]["selector_ac_calls"]+actual_joint_ac_count
        if expected_total != trial["total_ac_calls"]:
            errors.append("actual_total_AC_denominator")
        bound_errors, bound_count, bound_conflicts = _bounds_audit(trial["selection"])
        errors.extend(bound_errors)
        if calibration is not None:
            if trial["history_id"] in calibration["calibration_history_ids"]:
                errors.append("calibration_evaluation_overlap")
            if calibration["policy_id"] != trial["policy_id"]:
                errors.append("calibration_policy")
            any_false = any(method["decisions"]["false_rejection"] for method in methods)
            if trial.get("any_method_false_rejection") != any_false:
                errors.append("any_method_false_rejection")
        all_fits = history_replays+[fit for method in methods for fit in method["fits"]]
        return dict(success=not errors, errors=errors, history_id=trial["history_id"], run_id=trial["run_id"],
            quality=trial["quality"], case=trial["case"], policy_id=trial["policy_id"], quality_audit=quality,
            candidate_coverage=covered, recomputed_complete=complete,
            recomputed_relative_score=None if math.isinf(score) else score, calibration_score_is_infinite=not complete,
            recorded_fit_replay_count=sum(fit["replayed"] for fit in all_fits),
            actual_unique_fit_replay_count=sum(fit["replayed"] for fit in history_replays)+unique_joint_replay_count,
            actual_unique_fit_attempt_record_count=len(history_replays)+actual_joint_fit_count,
            recorded_failure_count=sum(not row["success"] for row in trial["historical_fits"])+sum(method["recorded_status"] != "complete" for method in methods),
            actual_joint_fit_calls=actual_joint_fit_count, joint_unique_action_count=len(seen),
            milp_bound_checks=bound_count, recorded_bound_conflicts=bound_conflicts,
            history_fits=history_replays, methods=methods, seconds=perf_counter()-began)
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        errors.append("trial_audit_exception:"+type(error).__name__+":"+str(error))
        return dict(success=False, errors=errors, history_id=trial.get("history_id"), run_id=trial.get("run_id"),
                    methods=methods, history_fits=history_replays, recomputed_complete=False, candidate_coverage=False,
                    calibration_score_is_infinite=True, recomputed_relative_score=None, seconds=perf_counter()-began)


def audit_calibration(audits, saved, alpha):
    """Reconstruct exchange rank independently; every failed H keeps +infinity."""
    n = len(audits)
    rank = math.ceil((n+1)*(1-float(alpha)))
    scores = sorted(math.inf if audit["calibration_score_is_infinite"] else audit["recomputed_relative_score"] for audit in audits)
    threshold = scores[rank-1] if rank <= n else math.inf
    errors = []
    ids = [audit["history_id"] for audit in audits]
    if len(set(ids)) != n or set(ids) != set(saved["calibration_history_ids"]):
        errors.append("calibration_independent_history_ids")
    gate = saved["relative"]
    if gate["rank"] != rank or gate["sample_count"] != n:
        errors.append("calibration_rank_count")
    if gate["vacuous"] != math.isinf(threshold):
        errors.append("calibration_vacuity")
    if math.isfinite(threshold):
        _close(threshold, gate["threshold"], "calibration_threshold", errors)
    if sum(audit["calibration_score_is_infinite"] for audit in audits) != saved["failed_history_count"] or saved["all_history_count"] != n:
        errors.append("calibration_failure_denominator")
    if "rank_tail_bound" in gate:
        _close((n+1-rank)/(n+1), gate["rank_tail_bound"], "calibration_rank_tail_bound", errors, atol=1e-12, rtol=0.)
    return dict(success=not errors, errors=errors, rank=rank, independent_history_count=n,
        threshold=None if math.isinf(threshold) else threshold, vacuous=math.isinf(threshold),
        failed_history_count=sum(audit["calibration_score_is_infinite"] for audit in audits))


def _summary_audit(trials, audits, saved, decisions):
    """Reconstruct counts without the study summarizer."""
    errors = []
    if any(not audit["success"] for audit in audits):
        return dict(success=False, errors=["summary_requires_successful_independent_trial_audits"], reconstructed={}, methods={})
    expected = dict(run_count=len(audits), independent_history_count=len({audit["history_id"] for audit in audits}),
        complete_count=sum(audit["recomputed_complete"] for audit in audits),
        failure_count=sum(not audit["recomputed_complete"] for audit in audits),
        candidate_coverage_failures=sum(not audit["candidate_coverage"] for audit in audits))
    for name, value in expected.items():
        if saved.get(name) != value:
            errors.append("summary_"+name)
    methods = {}
    method_names = [row["method"] for row in trials[0]["records"]] if trials else []
    if set(saved["method_summary"]) != set(method_names):
        errors.append("summary_method_set")
    for name in method_names:
        rows = [next(method for method in audit["methods"] if method["method"] == name) for audit in audits]
        result = dict(all_history_count=len(rows), complete_count=sum(row["joint_complete"] for row in rows),
            failed_count=sum(not row["joint_complete"] for row in rows), selection_failures=sum(row["selection_failure"] for row in rows))
        if decisions:
            result.update(only_truth_retained=sum(row["decisions"]["only_truth_retained"] for row in rows),
                unresolved=sum(row["decisions"]["unresolved"] for row in rows),
                false_rejections=sum(row["decisions"]["false_rejection"] for row in rows),
                false_rejection_denominator=sum(audit["candidate_coverage"] for audit in audits))
        for key, value in result.items():
            if saved["method_summary"][name].get(key) != value:
                errors.append("method_summary:"+name+":"+key)
        margins = [row["signed_margin_full_SSE"] for row in rows if row["joint_complete"]]
        if margins:
            _close(np.mean(margins), saved["method_summary"][name]["signed_margin_mean"], "method_mean_margin:"+name, errors)
            _close(np.median(margins), saved["method_summary"][name]["signed_margin_median"], "method_median_margin:"+name, errors)
        methods[name] = result
    if decisions:
        actual_false = sum(any(row["decisions"]["false_rejection"] for row in audit["methods"]) for audit in audits)
        if saved["any_method_false_rejections"] != actual_false:
            errors.append("summary_any_method_false_rejections")
        expected["any_method_false_rejections"] = actual_false
    pair_audits = []
    for pair in saved.get("paired_effects", []):
        clusters = {}
        valid_conditions = 0
        for audit in audits:
            row = {method["method"]: method for method in audit["methods"]}
            a, b = row[pair["method"]], row[pair["baseline"]]
            if a["joint_complete"] and b["joint_complete"] and a["signed_margin_full_SSE"] is not None and b["signed_margin_full_SSE"] is not None:
                clusters.setdefault(audit["history_id"], []).append(a["signed_margin_full_SSE"]-b["signed_margin_full_SSE"])
                valid_conditions += 1
        differences = [float(np.mean(values)) for _, values in sorted(clusters.items())]
        n = len(differences)
        if pair.get("paired_count") != n or pair.get("paired_condition_count") != valid_conditions:
            errors.append("paired_independent_denominator:"+pair["baseline"])
        result = pair.get("signed_margin_effect")
        if n and result is not None:
            if result["history_count"] != n or result["unit"] != "independent_paired_history":
                errors.append("paired_history_unit:"+pair["baseline"])
            _close(np.mean(differences), result["effect"], "paired_effect:"+pair["baseline"], errors)
            if n > 1:
                _close(np.std(differences, ddof=1)/np.sqrt(n), result["paired_standard_error"], "paired_SE:"+pair["baseline"], errors)
        pair_audits.append(dict(method=pair["method"], baseline=pair["baseline"], independent_history_count=n,
            paired_condition_count=valid_conditions, history_mean_difference=None if not n else float(np.mean(differences))))
    return dict(success=not errors, errors=errors, reconstructed=expected, methods=methods, paired_effect_audits=pair_audits)


def audit_phase(output, phase, *, report_directory=None, check_current_source=True):
    """Read only a complete saved phase; write separate audit evidence if asked."""
    output = Path(output)
    directory = output/phase
    began = perf_counter()
    config = read_json(directory/"config.json")
    paths = sorted(directory.glob("trial_*.json.gz"))
    if len(paths) != config["run_count"]:
        raise ValueError("phase incomplete: saved count differs from planned run_count")
    expected = {name: config[name] for name in ("config", "source_sha256", "policy_id")}
    freeze = read_json(output/"freeze.json") if (output/"freeze.json").exists() else None
    errors = []
    if phase != "pilot" and freeze != expected:
        errors.append("phase_freeze_differs")
    source = audit_source(expected["source_sha256"], snapshot=output/"science_source_snapshot.json.gz" if (output/"science_source_snapshot.json.gz").exists() else None, check_current_bytes=check_current_source)
    if source is not None:
        errors.extend(source["errors"])
    calibration = read_json(output/"calibration/thresholds.json") if phase == "evaluation" else None
    trials = [read_json(path) for path in paths]
    audits = [audit_trial(trial, calibration=calibration, expected=expected, require_drawn_quality=phase != "pilot") for trial in trials]
    errors.extend(str(audit["run_id"])+":"+error for audit in audits for error in audit["errors"])
    ids = [trial["history_id"] for trial in trials]
    expected_ids = set(map(str, config["seeds"]))
    if set(ids) != expected_ids or phase != "pilot" and len(set(ids)) != len(ids):
        errors.append("phase_independent_history_ids")
    if phase == "pilot" and Counter(ids) != Counter({str(seed): 3 for seed in config["seeds"]}):
        errors.append("pilot_three_quality_conditions_per_history")
    calibration_audit = None
    calibration_dependencies = []
    if phase == "calibration":
        calibration_audit = audit_calibration(audits, read_json(directory/"thresholds.json"), config["config"]["alpha"])
        errors.extend(calibration_audit["errors"])
    elif phase == "evaluation":
        if set(ids) & set(calibration["calibration_history_ids"]):
            errors.append("formal_calibration_evaluation_overlap")
        if config["policy_id"] != calibration["policy_id"]:
            errors.append("formal_calibration_policy")
        cal_config = read_json(output/"calibration/config.json")
        cal_paths = sorted((output/"calibration").glob("trial_*.json.gz"))
        if len(cal_paths) != cal_config["run_count"] or cal_config["policy_id"] != config["policy_id"]:
            errors.append("calibration_dependency_count_or_policy")
        calibration_dependencies = [audit_trial(read_json(path), expected=expected, require_drawn_quality=True) for path in cal_paths]
        errors.extend("calibration_dependency:"+str(audit["run_id"])+":"+error for audit in calibration_dependencies for error in audit["errors"])
        calibration_audit = audit_calibration(calibration_dependencies, calibration, config["config"]["alpha"])
        errors.extend(calibration_audit["errors"])
    summary = _summary_audit(trials, audits, read_json(directory/"summary.json"), phase == "evaluation")
    errors.extend(summary["errors"])
    report = dict(success=not errors, errors=errors, phase=phase, source_audit=source,
        planned_run_count=config["run_count"], audited_run_count=len(audits),
        independent_history_count=len(set(ids)), passed_run_count=sum(audit["success"] for audit in audits),
        recorded_fit_replay_count=sum(audit.get("recorded_fit_replay_count", 0) for audit in audits),
        actual_unique_fit_replay_count=sum(audit.get("actual_unique_fit_replay_count", 0) for audit in audits),
        calibration_audit=calibration_audit, summary_audit=summary, audits=audits,
        calibration_dependency_run_count=len(calibration_dependencies),
        calibration_dependency_recorded_fit_replay_count=sum(audit.get("recorded_fit_replay_count", 0) for audit in calibration_dependencies),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seconds=perf_counter()-began, optimizer_or_generator_called=False, actual_saved_models_used=True)
    if report_directory is not None:
        write_json(Path(report_directory)/(phase+"_replay.json.gz"), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("pilot", "calibration", "evaluation"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-current-source", action="store_true")
    args = parser.parse_args()
    if args.trial:
        result = audit_trial(read_json(args.trial))
        if args.report:
            write_json(args.report, result)
    elif args.output and args.phase:
        result = audit_phase(args.output, args.phase, report_directory=args.report,
                             check_current_source=not args.skip_current_source)
    else:
        parser.error("provide --trial or --output and --phase")
    print(json.dumps({key: value for key, value in result.items() if key not in ("audits", "methods", "history_fits", "source_audit", "summary_audit")}, ensure_ascii=False))
    if not result["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()