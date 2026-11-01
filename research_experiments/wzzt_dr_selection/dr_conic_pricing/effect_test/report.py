"""Aggregate-only export of ended conic-price effect tests.

This is post-run descriptive evaluation, not part of the frozen price rule.
No observation trajectories or fitted network parameters are exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(array):
    return float(np.sqrt(np.mean(np.square(array))))


def power_packet(row):
    folder = Path(row["physical_output"])
    with (folder/"physical_truth.pkl").open("rb") as stream:
        events = pickle.load(stream)["events"]
    with (folder/"packet.pkl").open("rb") as stream:
        observed = pickle.load(stream)
    return events, observed["training"]+observed["validation"]


def export(root, audit_path, output):
    root, output = Path(root), Path(output)
    audit = read(audit_path)
    if audit["status"] != "complete":
        raise RuntimeError("completed independent audit required")
    freeze = read(root/"price_freeze.json")
    rows, cases, lp_total = [], {}, 0
    background_signatures, response_signatures = {}, {}
    for key, path in freeze["case_roots"].items():
        folder = Path(path)
        config = read(folder/"protocol.json")
        scored = read(folder/"test/scored.json")
        summary = read(folder/"summary.json")
        if not (folder/"test/fits_frozen.json").is_file():
            raise RuntimeError("all fits must be frozen before evaluation")
        lp_total += scored["actual_LP_calls"]
        cases[key] = dict(summary=summary, competition=config["competition"],
            original_voltage_sigma=config["voltage_sigma"], pq_sigma=config["pq_sigma"],
            original_history_rows=config["original_history_rows"], final_rows=config["future_rows"],
            budget=config["hard_four_event_budget"], policy_details=config["policy_details"],
            optimization_status={mode:read(folder/(mode+"_optimization.json"))["status"] for mode in ("socp", "misocp")},
            planning_model_validation={mode:{**{k:read(folder/(mode+"_moment_validation.json"))[k]
                for k in ("samples", "successes", "failures", "budget_failures", "device_failures")},
                "paired_no_dr_successes":read(folder/(mode+"_moment_validation.json"))["paired_no_dr"]["successes"],
                "threshold":read(folder/(mode+"_moment_validation.json"))["threshold"]}
                for mode in ("socp", "misocp")})
        for old in scored["rows"]:
            row = dict(case=key, path=old["path"], policy=old["policy"], status=old["status"],
                       actual_payment=old.get("actual_payment"), worst_payment=old.get("worst_payment"),
                       excitation_physical=old.get("excitation_physical"),
                       excitation_observed=old.get("excitation_observed"),
                       excitation_threshold=old.get("excitation_threshold"),
                       excitation_physical_joint_pass=old.get("excitation_physical_joint_pass"),
                       excitation_observed_joint_pass=old.get("excitation_observed_joint_pass"))
            if old["status"] == "complete":
                row["models"] = old["models"]
                trace = old["trace"]
                candidate_points = trace["selection_candidates"]
                row["mechanism"] = dict(
                    accepted_updates=trace["accepted_updates"], accepted_cross_updates=trace["accepted_cross_updates"],
                    removed_blocks=trace["removed_blocks"], wall_limited=trace["wall_limited"],
                    search_stops=trace["search_stops"], final_source=trace["final_source"],
                    correct_candidate_available=any(p["full_errors"] == 0 for p in candidate_points),
                    correct_candidate_validation_eligible=any(p["full_errors"] == 0 and p["eligible"] for p in candidate_points),
                    W_reached_correct_from_wrong_start=any(points[0]["full_errors"] > 0
                        and any(p["full_errors"] == 0 for p in points[1:]) for points in trace["path_points"].values()),
                    pre_contraction_errors=trace["pre_contraction_errors"], final_errors=trace["final_errors"],
                    correct_lost_at_contraction=trace["pre_contraction_errors"] == 0 and trace["final_errors"] > 0)
                zero = next(r for r in scored["rows"] if r["policy"] == "zero" and r["path"] == old["path"])
                if zero.get("excitation_physical") is not None:
                    row["physical_excitation_change_vs_zero"] = (np.asarray(old["excitation_physical"])
                                                                 -np.asarray(zero["excitation_physical"])).tolist()
                events, observed = power_packet(old)
                base_events, _ = power_packet(zero)
                if old["policy"] == "zero":
                    digest = hashlib.sha256()
                    for event_row in events:
                        for channel in ("natural_P_pu", "natural_Q_pu", "root_pu"):
                            digest.update(np.asarray(event_row[channel]).tobytes())
                    background_signatures[(key, old["path"])] = digest.hexdigest()
                    response_signatures[(key, old["path"])] = hashlib.sha256(
                        json.dumps(old["factors_by_event"], sort_keys=True).encode()).hexdigest()
                signals, errors = [], []
                for event in (0, 2):
                    actual, base = events[event], base_events[event]
                    target = np.asarray(actual["root_pu"])[:, None]**2-np.asarray(actual["V_pu"])**2
                    target_zero = np.asarray(base["root_pu"])[:, None]**2-np.asarray(base["V_pu"])**2
                    signals.append(target-target_zero)
                    measured = observed[event]["V_root_observed"].to_numpy()[:, None]**2-observed[event]["V_terminal"].to_numpy()**2
                    errors.append(measured-target)
                signal, error = np.concatenate(signals), np.concatenate(errors)
                row["posthoc_voltage_diagnostic"] = dict(
                    scope="two paid days; paired noiseless squared-voltage-drop change versus zero",
                    DR_signal_rms_pu2=rms(signal), metering_error_rms_pu2=rms(error),
                    signal_to_metering_error_rms=rms(signal)/rms(error) if rms(error) > 0 else None,
                    not_a_causal_noise_ablation=True)
            rows.append(row)
    value = dict(status="complete", planned_records=16, completed_records=sum(r["status"] == "complete" for r in rows),
        case_path_strata=4, distinct_natural_background_paths=len(set(background_signatures.values())),
        distinct_response_factor_streams=len(set(response_signatures.values())),
        background_reuse_records=[dict(case=k[0], path=k[1], background_sha256=v)
                                  for k,v in background_signatures.items()],
        independence_scope="Two natural four-day backgrounds shared across the historical/noise conditions; four separately seeded response streams. Four strata are not four iid natural paths.",
        rows=rows, cases=cases, actual_LP_calls=lp_total,
        physical_events=64, ac_time_points=64*96,
        independent_audit_totals=audit["totals"], independent_audit_sha256=sha(audit_path),
        price_freeze_sha256=sha(root/"price_freeze.json"), source_snapshot_sha256=sha(root/"source_snapshot.zip"),
        export_source_sha256=sha(__file__), actual_recovery_probability_guarantee=False,
        no_household_or_PQV_or_AC_trajectories_exported=True,
        assumptions=dict(response="synthetic continuous relative response, not behavior calibration",
                         cross_node_response_correlation=.8,
                         joint_excitation_requirement="90pct under planning moments; not topology recovery",
                         background="original producer retained; finite historical factor model is an approximation",
                         PV="not added", EV="no EV field calibration or actual EV declared in this pilot"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:value[k] for k in ("status", "planned_records", "completed_records", "actual_LP_calls")}))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.audit, args.output)


if __name__ == "__main__":
    main()
