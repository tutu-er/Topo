"""Offline eight-corner response/ranking diagnosis of already executed EV plans.

No selector is run or changed. Truth enters only the declared diagnostic
reference network; every competing line vector remains its single training fit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations, product
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import scipy

from . import gradient_price_study as io
from .ev_fixed_response import FixedResponse
from .ev_fixed_train_fit import fit_fixed_inputs
from .information_context import meter_from_record
from .likelihood import Observations, RadialCandidate

HERE = Path(__file__).resolve().parent
METHODS = ("constant", "square_v2g_fixed", "gradac_v2g_fixed", "nearest_v2g_fixed", "nearest_v2g_mobile")
SCENARIOS = tuple("".join(map(str, bits)) for bits in product((0, 1), repeat=3))
CHANNELS = ("v", "master_p", "master_q")
BASE_KW = 20.
ORDER_TOL = 1e-6
ENDPOINT_ATOL, ENDPOINT_RTOL = 1e-5, 1e-8


def control_id(power):
    # Same eight-decimal equivalence used by the saved policy's actual-curve cache.
    u = np.asarray(power, float)
    return hashlib.sha256(np.round(u, 8).astype("<f8").tobytes()+repr(u.shape).encode()).hexdigest()[:16]


def factorial(values):
    """Exact anchored main/interaction contrasts and average replacement effects."""
    if set(values) != set(SCENARIOS) or not all(np.isfinite(v) for v in values.values()):
        return dict(success=False, failure="all eight finite corners required")
    effects = {}
    labels = ("theta", "load", "source")
    for mask in SCENARIOS[1:]:
        active = [i for i, bit in enumerate(mask) if bit == "1"]
        value = 0.
        for bits in product((0, 1), repeat=len(active)):
            corner = ["0"]*3
            for index, bit in zip(active, bits): corner[index] = str(bit)
            value += (-1)**(len(active)-sum(bits))*values["".join(corner)]
        effects["_".join(labels[i] for i in active)] = float(value)
    average = {}
    for index, label in enumerate(labels):
        total = 0.
        for mask in SCENARIOS[1:]:
            if mask[index] == "1":
                key = "_".join(labels[i] for i, bit in enumerate(mask) if bit == "1")
                total += effects[key]/mask.count("1")
        average[label] = float(total)
    delta = values["111"]-values["000"]
    return dict(success=True, all_forecast=float(values["000"]), all_true=float(values["111"]),
        total_change=float(delta), effects=effects, average_replacement=average,
        identity_error=float(sum(effects.values())-delta),
        average_identity_error=float(sum(average.values())-delta))


def _ranks(values, tolerance):
    order = sorted(range(len(values)), key=lambda i:values[i])
    ranks = np.empty(len(values), float)
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and abs(values[order[end]]-values[order[start]]) <= tolerance:
            end += 1
        ranks[order[start:end]] = (start+end-1)/2.
        start = end
    return ranks


def rank_agreement(predicted, actual, *, tolerance=ORDER_TOL):
    """Strict pair accounting; predictive ties are reported, never hidden."""
    ids = sorted(set(predicted) & set(actual))
    ids = [i for i in ids if np.isfinite(predicted[i]) and np.isfinite(actual[i])]
    x, y = [predicted[i] for i in ids], [actual[i] for i in ids]
    a, b = _ranks(x, tolerance), _ranks(y, tolerance)
    rho = float(np.corrcoef(a, b)[0, 1]) if len(ids) >= 2 and np.ptp(a) and np.ptp(b) else None
    counts = Counter(); rows = []
    for i, j in combinations(ids, 2):
        dp, da = predicted[i]-predicted[j], actual[i]-actual[j]
        sp, sa = (0 if abs(v) <= tolerance else (1 if v > 0 else -1) for v in (dp, da))
        label = "actual_tie" if sa == 0 else "predicted_tie" if sp == 0 else "concordant" if sp == sa else "discordant"
        counts[label] += 1
        rows.append(dict(left=i, right=j, predicted_delta=float(dp), actual_delta=float(da), classification=label))
    strict = counts["concordant"]+counts["discordant"]+counts["predicted_tie"]
    return dict(curves=len(ids), spearman=rho, pairs=len(rows), actual_strict_pairs=strict,
        strict_order_accuracy=counts["concordant"]/strict if strict else None,
        counts={k:counts[k] for k in ("concordant", "discordant", "predicted_tie", "actual_tie")},
        rows=rows, tolerance=tolerance)


def response_errors(values, baseline, true_values, true_baseline, sigma, n):
    errors = dict(absolute=np.asarray(values)-true_values,
        increment=(np.asarray(values)-baseline)-(true_values-true_baseline))
    result = {}
    for label, error in errors.items():
        channels = {}
        for channel, section in (("v", slice(0, n)), ("master_p", slice(n, n+1)), ("master_q", slice(n+1, n+2))):
            raw, white = error[:, section], error[:, section]/sigma[:, section]
            channels[channel] = dict(RMSE_pu=float(np.sqrt(np.mean(raw**2))),
                whitened_SSE=float(np.sum(white**2)))
            if channel != "v": channels[channel]["RMSE_kW_or_kvar"] = BASE_KW*channels[channel]["RMSE_pu"]
        result[label] = dict(channels=channels, whitened_SSE=sum(v["whitened_SSE"] for v in channels.values()))
    return result


def select_saved_actions(trial):
    indexed = {m["name"]:m for m in trial.get("methods", [])}
    actions, unavailable = {}, []
    for name in METHODS:
        row = indexed.get(name, {})
        if row.get("status") != "complete" or row.get("power_kw") is None:
            unavailable.append(dict(method=name, status=row.get("status", "missing")))
            continue
        key = control_id(row["power_kw"])
        if key not in actions:
            actions[key] = dict(id=key, aliases=[], power_kw=np.asarray(row["power_kw"], float),
                saved_clean_M=float(row["clean_future_M"]), saved_M=float(row["M"]),
                operation=row.get("operation", {}), saved_noiseless=row.get("noiseless", {}))
        actions[key]["aliases"].append(name)
        if abs(float(row["clean_future_M"])-actions[key]["saved_clean_M"]) > ENDPOINT_ATOL:
            raise ValueError("equivalent saved controls have different clean losses")
    return actions, unavailable


def _vector(physical):
    return np.column_stack((physical["v"], physical["master_p"], physical["master_q"]))


def _source_fit(candidate, fit, p, q, root, target, meter, config, seed, counter):
    # The original actual_curve fitter uses these same direct P/Q inputs and
    # constant training-source-mean initial point. Biases cancel exactly.
    obs = Observations(p=p+np.asarray(meter.bias_p), q=q+np.asarray(meter.bias_q),
        v=target[:, :candidate.n]+np.asarray(meter.bias_v),
        master_p=target[:, candidate.n], master_q=target[:, candidate.n+1], root_v=root)
    initial = SimpleNamespace(r=fit.r, x=fit.x,
        source_p=np.full(len(p), np.mean(fit.source_p)), source_q=np.full(len(p), np.mean(fit.source_q)))
    result = fit_fixed_inputs(candidate, obs, meter, fixed_r=fit.r, fixed_x=fit.x,
        initial=initial, seed=seed, call_counter=counter, **config["profile"], **config["source"])
    return result


def original_score_map(trial):
    values = {}
    for name, log in trial.get("selection", {}).get("methods", {}).items():
        if not name.startswith(("gradac_", "nearest_")):
            continue
        selected = log.get("selected_profile", log.get("selected_prediction", {}))
        if selected.get("power_kw") is not None and log.get("selected_score") is not None:
            values[control_id(selected["power_kw"])] = log["selected_score"]
        for initial in log.get("initial_candidates", []):
            profile = initial.get("profile", initial.get("prediction", {}))
            score = initial.get("score", initial.get("AC_score"))
            if profile.get("power_kw") is not None and score is not None:
                values.setdefault(control_id(profile["power_kw"]), score)
    return values


def diagnose_trial(trial):
    began = perf_counter()
    frozen_checks = {}
    for name in ("ev_fixed_response.py", "ev_fixed_train_fit.py", "likelihood.py", "information_context.py"):
        expected = trial.get("source_sha256", {}).get(name)
        if expected is not None:
            frozen_checks[name] = hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected
    if frozen_checks and not all(frozen_checks.values()):
        raise ValueError("numerical backend differs from the saved policy's frozen bytes")
    H, V, T = (trial["config"][k] for k in ("training_points", "validation_points", "horizon"))
    n = len(trial["training"]["p"][0]); start = H+V
    bank = [RadialCandidate.from_supports(row["name"], n,
        [tuple(s) for s in row["supports"] if 1 < len(s) < n]) for row in trial["candidates"]]
    candidates = {c.name:c for c in bank}; true = trial.get("true_name")
    if true not in candidates:
        return dict(seed=trial["seed"], success=False, failure="true candidate absent", actions={})
    fits = {r["candidate"]:SimpleNamespace(**{k:np.asarray(r[k], float) for k in
        ("r", "x", "source_p", "source_q")}) for r in trial["training_result"]["fits"]}
    meter = meter_from_record(trial["future_meter"])
    sigma = np.column_stack((np.broadcast_to(meter.v_sigma, (T, n)),
        np.broadcast_to(meter.master_p_sigma, (T,)), np.broadcast_to(meter.master_q_sigma, (T,))))
    forecast, background = trial["forecast"], trial["background"]
    inputs = [(np.asarray(forecast["p"]), np.asarray(forecast["q"])),
        (np.asarray(background["p_kw"])[start:]/BASE_KW, np.asarray(background["q_kvar"])[start:]/BASE_KW)]
    src = forecast["source_by_candidate"][true]
    sources = [np.column_stack((src["source_p"], src["source_q"])),
        np.column_stack((background["source_p_kw"], background["source_q_kvar"]))[start:]/BASE_KW]
    root = np.asarray(background["root_v"])[start:]
    if not np.array_equal(root, np.asarray(forecast["root"])):
        raise ValueError("root trajectory differs; a fourth factor would be needed")
    references = [FixedResponse(candidates[true], fits[true].r, fits[true].x, method="ac"),
        FixedResponse(candidates[true], trial["truth"]["r"], trial["truth"]["x"], method="ac")]
    baseline = {}
    zero = np.zeros((T, n))
    for mask in SCENARIOS:
        i, j, k = map(int, mask); p, q = inputs[j]; s = sources[k]
        baseline[mask] = references[i].evaluate(p, q, s[:, 0], s[:, 1], root, derivatives=False)["values"]
    actions, unavailable = select_saved_actions(trial)
    saved_scores = original_score_map(trial)
    val = {r["candidate"]:float(r["loss"]) for r in trial["validation_result"]["fits"]}
    counters = {}; completed_profiles = failed_profiles = 0; output = {}
    for key, action in actions.items():
        u = action["power_kw"]
        ptrue, qtrue = inputs[1]; strue = sources[1]
        true_y = references[1].evaluate(ptrue+u/BASE_KW, qtrue, strue[:, 0], strue[:, 1], root, derivatives=False)["values"]
        result = {k:v for k,v in action.items() if k != "saved_noiseless"}
        result.update(original_all_reference_score=saved_scores.get(key), scenarios={})
        for mask in SCENARIOS:
            corner_began = perf_counter(); i, j, k = map(int, mask)
            p, q = inputs[j]; source = sources[k]; total = p+u/BASE_KW
            y = references[i].evaluate(total, q, source[:, 0], source[:, 1], root, derivatives=False)["values"]
            errors = response_errors(y, baseline[mask], true_y, baseline["111"], sigma, n)
            losses, profiles = {}, {}
            for index, candidate in enumerate(bank):
                name = candidate.name
                if name == true and i == 0:
                    losses[name] = 0.
                    profiles[name] = dict(success=True, SSE=0., source=source,
                        reason="same fixed lines and feasible reference source; exact zero witness", solver_calls=0)
                    continue
                try:
                    fitted = _source_fit(candidate, fits[name], total, q, root, y, meter,
                        trial["config"], int(trial["seed"])+600+index, counters)
                    row = dict(success=bool(fitted.success), SSE=float(fitted.loss),
                        source=np.column_stack((fitted.source_p, fitted.source_q)),
                        message=fitted.message, nfev=fitted.nfev,
                        total_nfev=fitted.diagnostics.get("total_nfev"),
                        line_parameters_fixed=True)
                    if fitted.success:
                        losses[name] = float(fitted.loss); completed_profiles += 1
                        residual = (_vector(fitted.predicted)-y)/sigma
                        row["channel_SSE"] = dict(v=float(np.sum(residual[:, :n]**2)),
                            master_p=float(np.sum(residual[:, n]**2)), master_q=float(np.sum(residual[:, n+1]**2)))
                    else: failed_profiles += 1
                except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    row = dict(success=False, failure=type(exc).__name__+":"+str(exc)); failed_profiles += 1
                profiles[name] = row
            success = len(losses) == len(bank)
            corner = dict(success=success, response_error=errors, profiles=profiles,
                seconds=perf_counter()-corner_began)
            if success:
                nearest = min(v for name,v in losses.items() if name != true)
                joints = {name:val[name]+value for name,value in losses.items()}
                nearest_joint = min(v for name,v in joints.items() if name != true)
                corner.update(half_nearest_distance=nearest/2, M=nearest_joint-joints[true],
                    own_future_SSE=losses[true], future_loss_by_candidate=losses,
                    nearest_future=[name for name,value in losses.items() if name != true and abs(value-nearest)<=ORDER_TOL],
                    nearest_joint=[name for name,value in joints.items() if name != true and abs(value-nearest_joint)<=ORDER_TOL])
            result["scenarios"][mask] = corner
        full = result["scenarios"]["111"]
        delta = full.get("M", np.nan)-action["saved_clean_M"]
        result["endpoint_audit"] = dict(success=bool(full["success"] and np.isclose(full["M"], action["saved_clean_M"],
            atol=ENDPOINT_ATOL, rtol=ENDPOINT_RTOL)), recomputed_M=full.get("M"),
            saved_clean_M=action["saved_clean_M"], difference=float(delta),
            atol=ENDPOINT_ATOL, rtol=ENDPOINT_RTOL,
            future_PQ_noise_removed=True, noisy_validation_retained=True,
            candidate_loss_differences={name:full.get("future_loss_by_candidate", {}).get(name, np.nan)-float(saved["loss"])
                for saved in action["saved_noiseless"].get("fits", []) for name in [saved["candidate"]]})
        scalar = {"M":lambda row:row.get("M", np.nan),
            "half_nearest_distance":lambda row:row.get("half_nearest_distance", np.nan),
            "absolute_response_error":lambda row:row["response_error"]["absolute"]["whitened_SSE"],
            "increment_response_error":lambda row:row["response_error"]["increment"]["whitened_SSE"]}
        result["decomposition"] = {name:factorial({mask:fn(row) for mask,row in result["scenarios"].items()})
            for name,fn in scalar.items()}
        output[key] = result
    # Fixed-V2G curves are the main same-domain comparison; other actions are
    # explicitly marked cross-domain and never pooled into that claim.
    fixed_ids = [key for key,row in output.items() if any(name in row["aliases"] for name in
        ("square_v2g_fixed", "gradac_v2g_fixed", "nearest_v2g_fixed"))]
    blocks = {"fixed_v2g":dict(ids=fixed_ids, cross_domain=False),
        "all_diagnostic_actions":dict(ids=list(output), cross_domain=True)}
    rankings = {}
    for block, info in blocks.items():
        actual = {key:output[key]["saved_clean_M"] for key in info["ids"]}
        compared = {}
        for metric in ("half_nearest_distance", "M"):
            compared[metric] = {}
            for mask in SCENARIOS:
                predicted = {key:output[key]["scenarios"][mask][metric] for key in info["ids"]
                    if output[key]["scenarios"][mask]["success"]}
                compared[metric][mask] = rank_agreement(predicted, actual)
        compared["original_all_reference_score"] = rank_agreement(
            {key:output[key]["original_all_reference_score"] for key in info["ids"]
                if output[key]["original_all_reference_score"] is not None}, actual)
        rankings[block] = dict(info, comparisons=compared)
    differences = []
    for left, right in combinations(fixed_ids, 2):
        values = {mask:output[left]["scenarios"][mask].get("M", np.nan)
            -output[right]["scenarios"][mask].get("M", np.nan) for mask in SCENARIOS}
        differences.append(dict(left=left, right=right, left_aliases=output[left]["aliases"],
            right_aliases=output[right]["aliases"], actual_clean_delta=output[left]["saved_clean_M"]-output[right]["saved_clean_M"],
            margin_differences=values, decomposition=factorial(values)))
    return dict(seed=trial["seed"], true_candidate=true, policy_id=trial["policy_id"],
        success=not unavailable and not failed_profiles and all(r["endpoint_audit"]["success"] for r in output.values()),
        requested_methods=list(METHODS), unavailable_methods=unavailable, unique_actions=len(output), actions=output,
        rankings=rankings, fixed_domain_margin_differences=differences, frozen_numerical_source_checks=frozen_checks,
        source_profiles_completed=completed_profiles, source_profiles_failed=failed_profiles,
        source_fit_cost=counters, response_counters=[r.counters for r in references], seconds=perf_counter()-began,
        truth_scope="offline reference-network replacement only; trained competitor lines always fixed",
        source_initialization="same training-source mean and noisy-free future replay seed as original actual_curve")


def _summary(records):
    completed = [r for r in records if r.get("actions")]
    ranking = {}
    for metric in ("half_nearest_distance", "M"):
        ranking[metric] = {}
        for mask in SCENARIOS:
            rows = [r["rankings"]["fixed_v2g"]["comparisons"][metric][mask] for r in completed]
            rho = [row["spearman"] for row in rows if row["spearman"] is not None]
            count = Counter({k:sum(row["counts"][k] for row in rows) for k in
                ("concordant", "discordant", "predicted_tie", "actual_tie")})
            ranking[metric][mask] = dict(independent_histories=len(records), histories_with_defined_spearman=len(rho),
                mean_spearman=float(np.mean(rho)) if rho else None, pair_counts=dict(count))
    attribution = []
    for history in completed:
        groups = dict(all_unique=list(history["actions"]), fixed_v2g=history["rankings"]["fixed_v2g"]["ids"])
        group_results = {}
        for name, ids in groups.items():
            if not ids: continue
            actions = [history["actions"][key] for key in ids]
            mean_error = {mask:float(np.mean([a["scenarios"][mask]["response_error"]["increment"]["whitened_SSE"]
                for a in actions])) for mask in SCENARIOS}
            baseline_error = mean_error["000"]
            group_results[name] = dict(unique_actions=len(actions), mean_increment_whitened_SSE=mean_error,
                remaining_fraction_after_single_replacement={factor:mean_error[mask]/baseline_error if baseline_error else None
                    for factor, mask in (("theta", "100"), ("load", "010"), ("source", "001"))},
                average_replacement_fraction={factor:-float(np.mean([a["decomposition"]["increment_response_error"]
                    ["average_replacement"][factor] for a in actions]))/baseline_error if baseline_error else None
                    for factor in ("theta", "load", "source")},
                forecast_channel_RMSE_pu={channel:float(np.sqrt(np.mean([a["scenarios"]["000"]["response_error"]
                    ["increment"]["channels"][channel]["RMSE_pu"]**2 for a in actions]))) for channel in CHANNELS})
        attribution.append(dict(seed=history["seed"], groups=group_results))
    return dict(independent_histories=len(records), complete_histories=sum(r["success"] for r in records),
        unique_actions=sum(r.get("unique_actions", 0) for r in records),
        source_profiles_completed=sum(r.get("source_profiles_completed", 0) for r in records),
        source_profiles_failed=sum(r.get("source_profiles_failed", 0) for r in records),
        endpoint_failures=[dict(seed=r["seed"], action=key, **action["endpoint_audit"])
            for r in completed for key,action in r["actions"].items() if not action["endpoint_audit"]["success"]],
        response_attribution_by_history=attribution,
        margin_difference_effects_by_history=[dict(seed=r["seed"], action_pairs=len(r["fixed_domain_margin_differences"]),
            mean_absolute_effect={key:float(np.mean([abs(pair["decomposition"]["effects"][key])
                for pair in r["fixed_domain_margin_differences"]]))
                for key in ("theta", "load", "source", "theta_load", "theta_source", "load_source", "theta_load_source")})
            for r in completed if r["fixed_domain_margin_differences"]
                and all(pair["decomposition"]["success"] for pair in r["fixed_domain_margin_differences"])],
        nearest_joint_changes=[dict(seed=r["seed"], action=key, aliases=a["aliases"],
            forecast=a["scenarios"]["000"].get("nearest_joint"), actual=a["scenarios"]["111"].get("nearest_joint"))
            for r in completed for key,a in r["actions"].items()
            if a["scenarios"]["000"].get("nearest_joint") != a["scenarios"]["111"].get("nearest_joint")],
        endpoint_max_absolute_error=max((abs(a["endpoint_audit"]["difference"]) for r in completed
            for a in r["actions"].values()), default=None), fixed_domain_rankings=ranking,
        seconds=sum(r["seconds"] for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/gradient_flow_development_v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/response_ranking_diagnostics_v1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(993101, 993105)))
    parser.add_argument("--history-role", choices=("original_development", "former_evaluation_now_development"),
        default="original_development")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    config = dict(version="offline_response_eight_corners_v1", seeds=args.seeds, methods=list(METHODS), history_role=args.history_role,
        source_profile="original fixed-input AC least-squares; same source initialization/seeds",
        factors=["reference theta", "background P/Q", "reference same-interface source"],
        selector_calls=0, endpoint_tolerance=dict(atol=ENDPOINT_ATOL, rtol=ENDPOINT_RTOL),
        independent_unit="one old development history; action/model pairs do not increase n")
    scientific = ("ev_response_ranking_diagnose.py", "ev_response_ranking_diagnostics.md",
        "ev_fixed_response.py", "ev_fixed_train_fit.py", "likelihood.py", "information_context.py")
    hashes = {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in scientific}
    io.write_json(args.output/"config.json", dict(config=config, source_sha256=hashes,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)))
    records, manifest = [], []
    for seed in args.seeds:
        path = args.input/f"trial_{seed}.json.gz"
        raw = path.read_bytes(); trial = io.read_json(path)
        began = perf_counter()
        try:
            row = diagnose_trial(trial)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            row = dict(seed=seed, success=False, failure=type(exc).__name__+":"+str(exc),
                actions={}, seconds=perf_counter()-began)
        records.append(row)
        manifest.append(dict(seed=seed, path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
            policy_id=trial["policy_id"]))
        io.write_json(args.output/f"diagnostic_{seed}.json.gz", io.serial(row), compressed=True)
        io.write_json(args.output/"checkpoint.json", dict(completed=len(records), planned=len(args.seeds), last_seed=seed))
        print(json.dumps(dict(seed=seed, success=row["success"], actions=row.get("unique_actions"),
            profiles=row.get("source_profiles_completed"), failed=row.get("source_profiles_failed"), seconds=row["seconds"])), flush=True)
        if any(not action["endpoint_audit"]["success"] for action in row.get("actions", {}).values()):
            io.write_json(args.output/"endpoint_failure.json", io.serial(row))
            raise RuntimeError("full-truth endpoint differs from saved clean_M; inspect before continuing")
    report = dict(config=config, source_sha256=hashes, inputs=manifest, summary=_summary(records))
    report["source_unchanged"] = all(hashlib.sha256((HERE/name).read_bytes()).hexdigest() == value for name,value in hashes.items())
    if not report["source_unchanged"]:
        raise RuntimeError("diagnostic source changed during run; preserve checkpoint and rerun a new version")
    io.write_json(args.output/"summary.json", io.serial(report))


if __name__ == "__main__":
    main()
