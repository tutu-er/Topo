"""Small full-AC random-price integration audit; does not change a selector.

Predeclared two historical datasets, fixed physical forecasts, and independent
antithetic *measurement-noise* draws. Each draw gets a full H+N joint refit.
The four antithetic pairs, not eight individual draws, are integration units.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import gzip
import json
import platform
import sys
import traceback

import numpy as np
import scipy

from .gradient_price import GradientPriceEvaluator, CHANNEL_ORDER
from .likelihood import RadialCandidate, fit_candidate
from .simulation import generate, observe, model as measurement_model

CONFIG = {
    "audit": "full_AC_random_directional_profile_price",
    "cases": [{"case": "strong", "history_seed": 830001},
              {"case": "weak", "history_seed": 830002}],
    "n_terminals": 3, "history_count": 8, "future_count": 4,
    "history_generation_future_count": 0,
    "generation_truth_family": [[0, 1]],
    "candidates": ["star", "cherry01"],
    "site": 0, "power_kw": [0.0, 6.0, 3.0, 3.0],
    "direction": [0.0, 0.0, 1.0, -1.0],
    "step_kw": 0.01, "base_kva": 20.0, "q_over_p": 0.0,
    "design_rng_seed": 832001, "independent_base_noise_draws": 4,
    "antithetic_total_noise_draws": 8,
    "noise_scope": "additive_meter_noise_only; physical_background_and_source_fixed",
    "noise_pair_indices": [[0, 4], [1, 5], [2, 6], [3, 7]],
    "noise_bank_shared_across_two_histories": True,
    "fit_options": {"multistart": 2, "max_nfev": 250,
                    "linear_solver": "exact", "tol": 1e-10},
    "historical_fit_seed_offset": 833000,
    "joint_fit_seed": 834001,
    "absolute_directional_error_tolerance": 0.02,
    "relative_directional_error_tolerance": 0.002,
    "statistical_unit": "independent antithetic pair conditional on one H",
    "interpretation": "integration diagnostic; not EIG, topology probability or a population calibration",
    "selector_changed": False, "formal_thresholds_changed": False,
}

SOURCE_NAMES = ("gradient_price_random_audit.py", "gradient_price.py",
                "likelihood.py", "simulation.py", "candidates.py", "design.py")


def _safe(value):
    """Keep arrays/states and explicit nonfinite nulls in portable strict JSON."""
    if is_dataclass(value):
        return _safe(asdict(value))
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, complex):
        return {"real": _safe(value.real), "imag": _safe(value.imag)}
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _source_hashes():
    root = Path(__file__).resolve().parent
    return {name: sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def _save_evidence(record, output):
    """Checkpoint after each completed call; original failures remain in record."""
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_safe(record), ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    temporary = output.with_name(output.name+".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(text)
    temporary.replace(output)


def _pair_index(evaluation):
    return {row["pair_id"]: row for row in evaluation["pairs"]}


def _save_evaluation(evaluation):
    return {key: (_safe(value) if key != "fit_results" else
                  [{"reference": key_tuple[0], "sample": key_tuple[1],
                    "candidate": key_tuple[2], "fit": _safe(fitted)}
                   for key_tuple, fitted in value.items()])
            for key, value in evaluation.items()}


def _paired_stats(values):
    """Unbiased variance over the four independent pair averages."""
    values = np.asarray(values, float)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("all eight values required; failed samples cannot be dropped")
    paired = .5*(values[:4]+values[4:])
    return {
        "sample_mean": float(values.mean()), "n_samples": 8,
        "n_independent_antithetic_pairs": 4,
        "antithetic_pair_means": paired.tolist(),
        "standard_error": float(paired.std(ddof=1)/np.sqrt(len(paired))),
        "statistical_unit": "antithetic_pair_conditional_on_H",
    }


def _active_counts(evaluation):
    return [{"reference": row["reference"], "sample": row["sample"],
             "topology": row["topology"], "success": row["success"],
             "active_bound_count": row.get("diagnostics", {}).get("active_bound_count"),
             "optimality": row.get("optimality"),
             "ac_equation_residual": row.get("diagnostics", {}).get("ac_equation_residual"),
             "power_balance_residual": row.get("diagnostics", {}).get("power_balance_residual")}
            for row in evaluation["fit_records"]]


def _pair_checks(mean, center, plus, minus):
    mean_pairs, center_pairs = _pair_index(mean), _pair_index(center)
    plus_pairs, minus_pairs = _pair_index(plus), _pair_index(minus)
    direction = np.asarray(CONFIG["direction"], float)
    step = CONFIG["step_kw"]
    checks = []
    for pair_id, central in center_pairs.items():
        views = {"mean": mean_pairs.get(pair_id), "saa_center": central,
                 "saa_plus": plus_pairs.get(pair_id), "saa_minus": minus_pairs.get(pair_id)}
        successful = all(row is not None and row["success"] for row in views.values())
        check = {"pair_id": pair_id, "success": successful,
                 "required_design_sample_denominator": 8, "required_evaluations": 4}
        if not successful:
            check.update(passed=False, failure_kind="necessary_fit_or_prediction_failed",
                         evaluation_success={name: None if row is None else row["success"]
                                             for name, row in views.items()})
            checks.append(check)
            continue
        nominal, p, m = views["mean"], views["saa_plus"], views["saa_minus"]
        samples, forward, central_units, sample_prices = [], [], [], []
        for index in range(8):
            cs, ps, ms = central["samples"][index], p["samples"][index], m["samples"][index]
            if any(row["sample"] != index for row in (cs, ps, ms)):
                raise AssertionError("sample alignment changed")
            forward_value = (ps["score"]-cs["score"])/step
            central_value = (ps["score"]-ms["score"])/(2*step)
            price = float(np.asarray(cs["gradient"])@direction)
            forward.append(forward_value)
            central_units.append(central_value)
            sample_prices.append(price)
            samples.append({
                "sample": index, "antithetic_unit": index % 4,
                "noise_sign": 1 if index < 4 else -1,
                "score_center": cs["score"], "score_plus": ps["score"],
                "score_minus": ms["score"],
                "gradient_kw": cs["gradient"], "channel_gradients_kw": cs["channel_gradients"],
                "pq_only_gradient_kw": cs["pq_only_gradient"],
                "directional_price_per_kw": price,
                "random_unit_increment_forward_per_kw": forward_value,
                "random_unit_increment_central_per_kw": central_value,
                "price_minus_central_increment_per_kw": price-central_value,
            })
        price = float(np.asarray(central["gradient"])@direction)
        fd = float((p["score"]-m["score"])/(2*step))
        error = abs(price-fd)
        tolerance = (CONFIG["absolute_directional_error_tolerance"]+
                     CONFIG["relative_directional_error_tolerance"]*max(abs(price), abs(fd)))
        sample_average_identity = (
            abs(np.mean(central_units)-fd) <= 1e-8*max(1., abs(fd))
            and abs(np.mean(sample_prices)-price) <= 1e-10*max(1., abs(price)))
        check.update(
            mean_score=nominal["score"], saa_score=central["score"],
            mean_gradient_kw=nominal["gradient"], saa_gradient_kw=central["gradient"],
            mean_channel_gradients_kw=nominal["channel_gradients"],
            saa_channel_gradients_kw=central["channel_gradients"],
            mean_directional_price_per_kw=float(np.asarray(nominal["gradient"])@direction),
            saa_directional_price_per_kw=price, saa_central_difference_per_kw=fd,
            absolute_error=error, accepted_error_limit=tolerance,
            derivative_matches_saa_difference=error <= tolerance,
            sample_average_identity=sample_average_identity,
            saa_minus_mean_score=central["score"]-nominal["score"],
            saa_minus_mean_directional_price_per_kw=
                price-float(np.asarray(nominal["gradient"])@direction),
            random_unit_increment_forward=_paired_stats(forward),
            random_unit_increment_central=_paired_stats(central_units),
            sample_directional_price=_paired_stats(sample_prices),
            score_integration=_paired_stats([row["score"] for row in central["samples"]]),
            samples=samples,
            passed=bool(error <= tolerance and sample_average_identity),
        )
        checks.append(check)
    return checks


def _brief(record):
    cases = []
    for case in record["cases"]:
        evaluations = case.get("evaluations", {})
        pairs = case.get("pair_checks", [])
        cases.append({
            "case": case["case"], "history_seed": case["history_seed"],
            "status": case["status"], "error": case.get("error"),
            "history_fit_successes": sum(row["success"] for row in case.get("history_fits", [])),
            "history_fit_denominator": 2,
            "evaluation_successes": sum(row["success"] for row in evaluations.values()),
            "evaluation_denominator": 4,
            "pair_passes": sum(row.get("passed", False) for row in pairs),
            "pair_denominator": 2,
            "pair_diagnostics": [
                {key: row.get(key) for key in (
                    "pair_id", "success", "passed", "mean_score", "saa_score",
                    "mean_directional_price_per_kw", "saa_directional_price_per_kw",
                    "saa_central_difference_per_kw", "absolute_error", "accepted_error_limit",
                    "saa_minus_mean_score", "saa_minus_mean_directional_price_per_kw",
                    "random_unit_increment_forward", "random_unit_increment_central",
                    "sample_directional_price", "score_integration")}
                for row in pairs],
            "fit_calls": case.get("joint_fit_calls", 0),
            "joint_ac_calls": case.get("joint_ac_calls", 0),
            "historical_ac_calls": case.get("historical_ac_calls", 0),
            "seconds": case.get("seconds"),
            "failure_denominators": {
                label: {"failed_fits": sum(not row["success"] for row in value["fit_records"]),
                        "attempted_fits": len(value["fit_records"]),
                        "planned_fits": 4 if label == "mean_center" else 32,
                        "failures": value["failures"]}
                for label, value in evaluations.items()},
            "active_bound_counts": case.get("active_bound_counts", {}),
        })
    return {
        "config": record["config"], "status": record["status"],
        "case_denominator": 2, "pair_denominator": 4,
        "passed_pairs": sum(row["pair_passes"] for row in cases),
        "source_hashes_start": record["source_hashes_start"],
        "source_hashes_end": record.get("source_hashes_end"),
        "source_unchanged_during_run": record.get("source_unchanged_during_run"),
        "seconds": record.get("seconds"), "cases": cases,
        "interpretation": CONFIG["interpretation"],
        "mean_equals_expected_assumed": False,
    }


def main():
    global CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("symmetric_v1", "asymmetric_v2"),
                        default="symmetric_v1")
    args = parser.parse_args()
    CONFIG = dict(CONFIG)
    CONFIG["audit_variant"] = args.variant
    if args.variant == "asymmetric_v2":
        CONFIG.update(power_kw=[0.0, 5.0, 4.0, 3.0],
                      direction=[0.0, -1.0, 1.0, 0.0],
                      direction_purpose="nonzero physical path price diagnostic; no exact-expectation equality assumption")
    else:
        CONFIG["direction_purpose"] = "symmetric-slot finite-SAA sampling-error control"
    root = Path(__file__).resolve().parent
    stem = "random_price_audit_v2" if args.variant == "asymmetric_v2" else "random_price_audit"
    output = root/"evidence"/"gradient_price"/(stem+".json.gz")
    summary = output.with_name("summary_v2.json" if args.variant == "asymmetric_v2"
                               else "random_price_audit_summary.json")
    if output.exists() or summary.exists():
        raise FileExistsError("Preserve existing audit: choose a new explicitly approved run filename.")
    began = perf_counter()
    record = {
        "config": CONFIG, "status": "running", "cases": [],
        "source_hashes_start": _source_hashes(),
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform()},
        "nonfinite_serialization": "nonfinite scalar fields become null; fit success and failures retained",
    }
    rng = np.random.default_rng(CONFIG["design_rng_seed"])
    positive = rng.normal(size=(4, CONFIG["future_count"], 11))
    bank = np.concatenate((positive, -positive), axis=0)
    record["design_noise_samples"] = bank
    record["design_noise_hash"] = sha256(bank.astype("<f8").tobytes()).hexdigest()
    _save_evidence(record, output)
    for case_config in CONFIG["cases"]:
        started = perf_counter()
        case = {**case_config, "status": "running", "history_fits": [],
                "evaluations": {}, "pair_checks": [], "active_bound_counts": {}}
        record["cases"].append(case)
        try:
            path = generate(case_config["history_seed"], case_config["case"],
                n=CONFIG["n_terminals"], history_count=CONFIG["history_count"],
                future_count=0, truth_family=((0, 1),))
            if len(path.p) != CONFIG["history_count"]:
                raise AssertionError("generation unexpectedly included true future states")
            history, _unused_physical_history = observe(path, np.arange(CONFIG["history_count"]))
            del path, _unused_physical_history
            case["selector_input_boundary"] = {
                "only_observed_H_passed": True, "true_future_generated": False,
                "actual_future_noise_passed": False, "truth_parameters_passed": False}
            candidates = (
                RadialCandidate.from_supports("star", 3, []),
                RadialCandidate.from_supports("cherry01", 3, [(0, 1)]))
            meter = measurement_model(3)
            fitted = {}
            for index, candidate in enumerate(candidates):
                fit = fit_candidate(candidate, history, meter,
                    seed=CONFIG["historical_fit_seed_offset"]+case_config["history_seed"]+index,
                    **CONFIG["fit_options"])
                fitted[candidate.name] = fit
                case["history_fits"].append(_safe(fit))
                case["historical_ac_calls"] = sum(
                    row["diagnostics"].get("ac_evaluations", 0) for row in case["history_fits"])
                _save_evidence(record, output)
            if not all(fit.success for fit in fitted.values()):
                case["status"] = "history_fit_failed"
                continue
            common = dict(horizon=CONFIG["future_count"], base_kva=CONFIG["base_kva"],
                q_over_p=CONFIG["q_over_p"], fit_options=CONFIG["fit_options"],
                seed=CONFIG["joint_fit_seed"])
            mean_evaluator = GradientPriceEvaluator(history, candidates, fitted, meter, **common)
            saa_evaluator = GradientPriceEvaluator(history, candidates, fitted, meter,
                                                   design_noise_samples=bank, **common)
            case["history_hash"] = mean_evaluator.history_hash
            if saa_evaluator.history_hash != mean_evaluator.history_hash:
                raise AssertionError("mean/SAA histories differ")
            case["meter_model"] = _safe(meter)
            case["history_observations"] = _safe(history)
            case["fixed_forecasts"] = _safe(mean_evaluator.forecasts)
            u = np.asarray(CONFIG["power_kw"])
            direction = np.asarray(CONFIG["direction"])
            h = CONFIG["step_kw"]
            calls = [
                ("mean_center", mean_evaluator, u),
                ("saa_center", saa_evaluator, u),
                ("saa_plus", saa_evaluator, u+h*direction),
                ("saa_minus", saa_evaluator, u-h*direction),
            ]
            raw = {}
            for label, evaluator, power in calls:
                evaluated = evaluator.evaluate(CONFIG["site"], power)
                raw[label] = evaluated
                case["evaluations"][label] = _save_evaluation(evaluated)
                case["active_bound_counts"][label] = _active_counts(evaluated)
                case["joint_fit_calls"] = sum(value["fit_calls"] for value in raw.values())
                case["joint_ac_calls"] = sum(value["total_ac_calls"] for value in raw.values())
                _save_evidence(record, output)
                print(json.dumps({"case": case["case"], "evaluation": label,
                                  "success": evaluated["success"],
                                  "score": evaluated["score"], "fit_calls": evaluated["fit_calls"],
                                  "ac_calls": evaluated["total_ac_calls"],
                                  "seconds": evaluated["seconds"]}), flush=True)
            case["pair_checks"] = _pair_checks(raw["mean_center"], raw["saa_center"],
                                               raw["saa_plus"], raw["saa_minus"])
            case["status"] = ("passed" if all(row["passed"] for row in case["pair_checks"])
                              and len(case["pair_checks"]) == 2 else "diagnostic_failed")
        except Exception as error:
            case["status"] = "exception"
            case["error"] = {"type": type(error).__name__, "message": str(error),
                             "traceback": traceback.format_exc()}
            print(json.dumps({"case": case["case"], "exception": str(error)}), flush=True)
        finally:
            case["seconds"] = perf_counter()-started
            _save_evidence(record, output)
    record["source_hashes_end"] = _source_hashes()
    record["source_unchanged_during_run"] = record["source_hashes_start"] == record["source_hashes_end"]
    record["status"] = ("passed" if all(case["status"] == "passed" for case in record["cases"])
                        and len(record["cases"]) == 2 and record["source_unchanged_during_run"]
                        else "diagnostic_failed")
    record["seconds"] = perf_counter()-began
    _save_evidence(record, output)
    brief = _brief(record)
    summary.write_text(json.dumps(_safe(brief), ensure_ascii=False, allow_nan=False,
                                  indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "passed_pairs": brief["passed_pairs"],
                      "pair_denominator": brief["pair_denominator"],
                      "seconds": record["seconds"], "evidence": str(output),
                      "summary": str(summary)}), flush=True)
    return 0 if record["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
