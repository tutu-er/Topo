"""Offline line/load/source forecast diagnosis for retained v4 actions.

Synthetic truth is regenerated solely for post-selection diagnosis. No action
is selected, no candidate is fitted, and no model or policy is changed. The
three-factor allocation always concerns the SAME voltage-increment MSE; it
does not allocate likelihood loss, decision power, or topology probability.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import permutations, product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import ev_mobile_v4_cases as cases
from . import ev_mobile_v4_domain as nested
from . import ev_mobile_v2_review as io
from . import likelihood as lk
from . import gradient_price_study as serial_io


FACTORS = ("lines", "load", "source")
MASKS = tuple("".join(map(str, bits)) for bits in product((0, 1), repeat=3))
HERE = Path(__file__).resolve().parent


def error_summary(predicted, target, *, units):
    predicted, target = np.asarray(predicted, float), np.asarray(target, float)
    if predicted.shape != target.shape or not predicted.size or not np.isfinite([predicted, target]).all():
        raise ValueError("matching nonempty finite prediction and target arrays required")
    error = predicted - target
    denominator = float(np.linalg.norm(target))
    return dict(units=units, count=int(error.size), RMSE=float(np.sqrt(np.mean(error ** 2))),
        bias=float(np.mean(error)), max_absolute_error=float(np.max(abs(error))),
        relative_L2=None if denominator <= 1e-15 else float(np.linalg.norm(error) / denominator),
        target_RMS=float(np.sqrt(np.mean(target ** 2))))


def shapley_error_reduction(errors):
    """Allocate E(000)-E(111) by averaging all six correction orders.

    E(S) is the voltage-increment MSE with exactly factors S replaced by truth.
    Contributions can be negative: a correction can expose a cancelling error.
    The six-order average removes the choice of one preferred replacement order.
    """
    if set(errors) != set(MASKS):
        raise ValueError("all eight line/load/source MSE values required")
    values = {mask: float(value) for mask, value in errors.items()}
    if not np.isfinite(list(values.values())).all() or min(values.values()) < 0.:
        raise ValueError("nonnegative finite MSE values required")
    contributions = np.zeros(3)
    for order in permutations(range(3)):
        bits = [0, 0, 0]
        before = "000"
        for index in order:
            bits[index] = 1
            after = "".join(map(str, bits))
            contributions[index] += (values[before] - values[after]) / 6.
            before = after
    target = values["000"] - values["111"]
    return dict(units="pu_voltage_squared", target="same voltage increment MSE reduction",
        contributions=dict(zip(FACTORS, map(float, contributions))),
        total_reduction=target, sum_contributions=float(contributions.sum()),
        identity_error=float(abs(contributions.sum() - target)),
        fractions_of_nominal_MSE=None if values["000"] <= 1e-30 else
            dict(zip(FACTORS, map(float, contributions / values["000"]))),
        negative_contribution_meaning="truth correction exposes another cancelling prediction error",
        is_likelihood_or_decision_power_attribution=False)


def _line_comparison(candidate, fitted, truth_candidate, truth_r, truth_x):
    r, x = np.asarray(fitted["r"], float), np.asarray(fitted["x"], float)
    if r.shape != (candidate.m,) or x.shape != r.shape or not np.isfinite([r, x]).all():
        raise ValueError("finite fitted line arrays in declared candidate edge order required")
    true_index = {tuple(support): index for index, support in enumerate(truth_candidate.supports)}
    fitted_index = {tuple(support): index for index, support in enumerate(candidate.supports)}
    common = [support for support in candidate.supports if tuple(support) in true_index]
    source = [fitted_index[tuple(s)] for s in common]
    target = [true_index[tuple(s)] for s in common]
    rows = []
    for support, i, j in zip(common, source, target):
        rows.append(dict(support=list(support), fitted_R_pu=float(r[i]), true_R_pu=float(truth_r[j]),
            fitted_X_pu=float(x[i]), true_X_pu=float(truth_x[j]),
            relative_R_error=float((r[i] - truth_r[j]) / truth_r[j]),
            relative_X_error=float((x[i] - truth_x[j]) / truth_x[j])))
    return dict(common_edge_count=len(common), fitted_edge_count=candidate.m,
        true_edge_count=truth_candidate.m, common_edges=rows,
        common_R=error_summary(r[source], truth_r[target], units="pu_impedance"),
        common_X=error_summary(x[source], truth_x[target], units="pu_impedance"),
        extra_fitted_supports=[list(s) for s in candidate.supports if tuple(s) not in true_index],
        missed_true_supports=[list(s) for s in truth_candidate.supports if tuple(s) not in fitted_index],
        unmatched_edges_have_no_assigned_parameter_error=True,
        comparison_scope="equal downstream support only; false topology edges are not treated as matching true lines")


def _response_factorial(candidate, trained, truth, forecast, true_inputs, power, predictor, derivative, baseline_cache):
    """Full AC increments at eight fixed input/parameter combinations; no fit."""
    base_kw = true_inputs["base_kw"]
    r_fit, x_fit = np.asarray(trained["r"]), np.asarray(trained["x"])
    sources = forecast["source_by_candidate"][candidate.name]
    settings, calls, physical_failures = {}, 0, []
    for mask in MASKS:
        r, x = (truth["r"], truth["x"]) if mask[0] == "1" else (r_fit, x_fit)
        p, q = (true_inputs["p"], true_inputs["q"]) if mask[1] == "1" else (forecast["p"], forecast["q"])
        sp, sq = (true_inputs["source_p"], true_inputs["source_q"]) if mask[2] == "1" else (sources["source_p"], sources["source_q"])
        try:
            if mask not in baseline_cache:
                calls += 1
                baseline_cache[mask] = predictor(candidate, r, x, p, q, sp, sq, true_inputs["root_v"])
            calls += 1
            physical = predictor(candidate, r, x, np.asarray(p) + power / base_kw,
                                 q, sp, sq, true_inputs["root_v"])
            delta = np.asarray(physical["v"]) - np.asarray(baseline_cache[mask]["v"])
            settings[mask] = dict(delta=delta, physical=physical)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            physical_failures.append(dict(mask=mask, failure=type(exc).__name__ + ": " + str(exc)))
    if set(settings) != set(MASKS):
        return dict(success=False, failures=physical_failures, completed_masks=list(settings),
                    production_AC_calls=calls, candidate_refits=0)
    target = settings["111"]["delta"]
    comparison, errors = {}, {}
    for mask in MASKS:
        row = error_summary(settings[mask]["delta"], target, units="pu_voltage")
        errors[mask] = row["RMSE"] ** 2
        comparison[mask] = row
    # Derivatives at the same action, for the sole trained true-topology reference
    # versus true lines/load/source. Their differences do not include future noise.
    predicted_J = np.asarray(derivative(candidate, settings["000"]["physical"], True)["v"])
    true_J = np.asarray(derivative(candidate, settings["111"]["physical"], True)["v"])
    m, n = candidate.m, candidate.n
    p_columns, q_columns = slice(2 * m, 2 * m + n), slice(2 * m + n, 2 * m + 2 * n)
    derivatives = dict(
        V_per_P=error_summary(predicted_J[:, :, p_columns] / base_kw,
                             true_J[:, :, p_columns] / base_kw, units="pu_voltage_per_kW"),
        V_per_Q=error_summary(predicted_J[:, :, q_columns] / base_kw,
                             true_J[:, :, q_columns] / base_kw, units="pu_voltage_per_kvar"))
    # A local R/X cancellation check for the increment, using the true-input
    # derivative difference between u and zero. It is not a refit compensation.
    true_J_zero = np.asarray(derivative(candidate, baseline_cache["111"], True)["v"])
    J_delta = true_J - true_J_zero
    dr, dx = r_fit - np.asarray(truth["r"]), x_fit - np.asarray(truth["x"])
    r_component = np.einsum("tne,e->tn", J_delta[:, :, :m], dr)
    x_component = np.einsum("tne,e->tn", J_delta[:, :, m:2 * m], dx)
    denominator = float(np.linalg.norm(r_component) + np.linalg.norm(x_component))
    line_cancellation = dict(R_component_RMS_pu=float(np.sqrt(np.mean(r_component ** 2))),
        X_component_RMS_pu=float(np.sqrt(np.mean(x_component ** 2))),
        combined_RMS_pu=float(np.sqrt(np.mean((r_component + x_component) ** 2))),
        cancellation_ratio=None if denominator <= 1e-30 else
            float(np.linalg.norm(r_component + x_component) / denominator),
        opposite_R_X_error_count=int(np.count_nonzero(dr * dx < 0.)), edge_count=m,
        scope="first-order trained-line increment error at true parameters and true inputs; no parameter refitting")
    return dict(success=True, voltage_increment_errors=comparison,
        MSE_allocation=shapley_error_reduction(errors), derivative_prediction_error=derivatives,
        R_X_increment_cancellation=line_cancellation,
        nominal_delta_V_pu=settings["000"]["delta"], true_delta_V_pu=target,
        true_increment_RMS_pu=float(np.sqrt(np.mean(target ** 2))),
        mask_meaning="lines/load/source; 0=trained or forecast, 1=synthetic truth correction",
        target="V(u)-V(0), same physical action, latent inputs, no future measurement noise",
        production_AC_calls=calls, analytic_derivative_calls=3, candidate_refits=0)


def diagnose_record(record, *, case_factory=cases.make_case, predictor=lk.ac_predict,
                    derivative=lk._response_derivatives):
    began = perf_counter()
    missing, failures = [], []
    spec = record.get("spec", {})
    required = ("observed", "training_result", "forecast", "methods", "config", "true_name")
    for key in required:
        if key not in record:
            missing.append(dict(field=key, reason=record.get("failure", "saved field unavailable")))
    if missing:
        return dict(success=False, spec=spec, missing=missing, failures=failures,
                    posthoc_truth_only=True, candidate_refits=0, action_selection_calls=0)
    regular = case_factory(spec["bank"], spec["seed"], spec["family_index"], "regular", H=96)
    historical_count = len(record["observed"]["p"])
    for channel in ("p", "q", "v", "root_v", "master_p", "master_q"):
        regenerated = np.concatenate((np.asarray(getattr(regular["train"], channel)),
                                      np.asarray(getattr(regular["validation"], channel))))
        io.close(regenerated, record["observed"][channel], "regenerated observed D/" + channel, atol=0., rtol=0.)
    truth = regular["truth"]
    bank = {candidate.name: candidate for candidate in regular["bank"]}
    true_name = record["true_name"]
    io.require(true_name in bank and bank[true_name].supports == truth["candidate"].supports,
               "saved true label differs from regenerated synthetic truth")
    trained = {row.get("candidate", row.get("name")): row for row in record["training_result"].get("fits", [])}
    line_rows = {}
    for name, candidate in bank.items():
        if name not in trained or not trained[name].get("success"):
            missing.append(dict(field="trained_lines/" + name, reason="successful observed-D fit unavailable"))
            continue
        line_rows[name] = _line_comparison(candidate, trained[name], truth["candidate"],
                                            np.asarray(truth["r"]), np.asarray(truth["x"]))
    if true_name not in line_rows:
        return dict(success=False, spec=spec, missing=missing, failures=failures,
                    line_parameter_errors=line_rows, posthoc_truth_only=True,
                    candidate_refits=0, action_selection_calls=0)
    power, method_names = {}, {}
    for name, method in record["methods"].items():
        key = nested.action_key(method["power_kw"])
        io.require(key == method["action_key"], "retained action identity differs")
        power.setdefault(key, np.asarray(method["power_kw"], float))
        method_names.setdefault(key, []).append(name)
    rows, total_calls = {}, 0
    forecast, base_kw = record["forecast"], float(record["config"]["base_kw"])
    for regime in spec["regimes"]:
        case = regular if regime == "regular" else case_factory(spec["bank"], spec["seed"], spec["family_index"], regime, H=96)
        start = case["future_start"]
        T = int(spec["T"])
        future = {key: np.asarray(case[key])[start:start + T] for key in ("p", "q", "source_p", "source_q", "root_v")}
        future["base_kw"] = base_kw
        load = {}
        for channel, units in (("p", "kW"), ("q", "kvar")):
            pred, target = np.asarray(forecast[channel]) * base_kw, future[channel] * base_kw
            load[channel] = dict(error_summary(pred, target, units=units),
                per_slot_RMSE=np.sqrt(np.mean((pred - target) ** 2, axis=1)),
                per_terminal_RMSE=np.sqrt(np.mean((pred - target) ** 2, axis=0)))
        sources = {}
        true_reference_source = forecast["source_by_candidate"][true_name]
        for name in bank:
            source = forecast["source_by_candidate"].get(name)
            if source is None:
                missing.append(dict(field="forecast_source/" + name, reason="candidate source forecast unavailable"))
                continue
            sources[name] = {}
            for channel, units in (("source_p", "kW"), ("source_q", "kvar")):
                predicted = np.asarray(source[channel]) * base_kw
                target = future[channel] * base_kw
                comparison = error_summary(predicted, target, units=units)
                comparison["per_slot_error"] = predicted - target
                comparison["difference_from_true_candidate_forecast"] = error_summary(
                    predicted, np.asarray(true_reference_source[channel]) * base_kw, units=units)
                sources[name][channel] = comparison
        action_rows, baseline_cache = {}, {}
        for key, action in power.items():
            try:
                result = _response_factorial(bank[true_name], trained[true_name], truth,
                    forecast, future, action, predictor, derivative, baseline_cache)
                result.update(methods=method_names[key], power_kw=action)
                action_rows[key] = result
                total_calls += result["production_AC_calls"]
                if not result["success"]:
                    failures.extend(dict(regime=regime, action_key=key, **failure) for failure in result["failures"])
            except (ValueError, RuntimeError, KeyError, FloatingPointError, np.linalg.LinAlgError) as exc:
                action_rows[key] = dict(success=False, methods=method_names[key],
                    failure=type(exc).__name__ + ": " + str(exc))
                failures.append(dict(regime=regime, action_key=key, failure=action_rows[key]["failure"]))
        rows[regime] = dict(load_forecast=load, source_forecast_by_candidate=sources,
            true_candidate=true_name, actions=action_rows,
            target="latent ordinary load and one physical source; EV and future measurement noise excluded",
            unique_actions=len(power), method_slots=len(record["methods"]))
    return dict(success=not missing and not failures, spec=spec, observed_count=historical_count,
        true_name=true_name, line_parameter_errors=line_rows, regimes=rows,
        missing=missing, failures=failures, seconds=perf_counter() - began,
        posthoc_truth_only=True, selector_modified=False, action_selection_calls=0,
        candidate_refits=0, production_AC_calls=total_calls,
        production_AC_call_scope="direct factorial response calls only; case regeneration calls reported separately",
        regenerated_case_calls=1 + sum(regime != "regular" for regime in spec["regimes"]),
        expected_candidate_count=len(bank), parameter_error_candidate_count=len(line_rows),
        mask_order=list(FACTORS), scope="saved actions, full AC response and physical derivatives; no action reoptimization or likelihood refitting")


def run(directory, output, *, phase="development"):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    frozen = io.read(directory / "freeze.json")
    for name in ("ev_mobile_v4_cases.py", "ev_mobile_v4_domain.py"):
        io.require(sha256((HERE / name).read_bytes()).hexdigest() == frozen["source_sha256"][name],
                   "frozen generator/domain source changed: " + name)
    rows = []
    for file in sorted((directory / phase).glob("*.json.gz")):
        record = io.read(file)
        io.require(record.get("policy_id") == frozen["policy_id"], "record/freeze policy mismatch")
        try:
            result = diagnose_record(record)
        except (ValueError, RuntimeError, KeyError, TypeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            result = dict(success=False, spec=record.get("spec"), missing=[],
                failures=[dict(failure=type(exc).__name__ + ": " + str(exc))],
                posthoc_truth_only=True, candidate_refits=0, action_selection_calls=0)
        result.update(input_file=str(file.resolve()), input_sha256=sha256(file.read_bytes()).hexdigest())
        rows.append(result)
        print("FORECAST_AUDIT", file.name, result["success"], flush=True)
    expected = frozen["specifications"][phase]
    key = lambda value: serial_io.identity(serial_io.serial(value), {})
    ids = [key(row.get("spec")) for row in rows]
    coverage = len(ids) == len(set(ids)) and set(ids) == {key(spec) for spec in expected}
    result = dict(success=coverage and all(row["success"] for row in rows), exact_case_coverage=coverage,
        phase=phase, policy_id=frozen["policy_id"], expected_case_slots=len(expected), saved_case_slots=len(rows),
        case_results=rows, posthoc_truth_only=True, candidate_refits=0, action_selection_calls=0,
        production_AC_calls=sum(row.get("production_AC_calls", 0) for row in rows),
        input_freeze_sha256=sha256((directory / "freeze.json").read_bytes()).hexdigest(),
        audit_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        statistical_unit="independent history; methods and regimes remain paired within a history")
    io.write(output, serial_io.serial(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("development", "evaluation"), default="development")
    args = parser.parse_args()
    result = run(args.directory, args.output, phase=args.phase)
    print(json.dumps({key: result[key] for key in
        ("success", "expected_case_slots", "saved_case_slots", "production_AC_calls", "candidate_refits")}))


if __name__ == "__main__":
    main()
