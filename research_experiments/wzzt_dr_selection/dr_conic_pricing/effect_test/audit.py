"""Independent read-only audit of a frozen continuous-DR actual-effect trial.

No project model, selector, identifier, physical producer or solver is imported.
Private trusted packets are read only after every case has terminal worker and
fit-freeze receipts. Output contains aggregate/scalar metrics, never trajectories
or continuous fitted network parameters. This checks the hash-bound controller
and its declared ordering; it is not an operating-system file-access audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import traceback

import numpy as np
import pandas as pd


POLICIES = ("zero", "rotation", "socp", "misocp")
PATHS = (1100, 1101)
MODELS = ("multistart_original_rule", "rnj_bank", "baseline25")
TOL = 1e-7


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(item):
    if not isinstance(item, dict) or not {"path", "sha256"} <= set(item):
        raise ValueError("explicit path/hash reference required")
    path = Path(item["path"])
    if sha(path) != item["sha256"]:
        raise AssertionError("bound input hash changed: "+str(path))
    return path


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def vector(value, count=96):
    return np.broadcast_to(np.asarray(value, float), (count,)).copy()


def read_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def own_input_hash(scenarios):
    frames = [pd.concat([s[key] for s in scenarios]) for key in
              ("P_terminal", "Q_terminal", "V_terminal", "V_root_observed")]
    assert all(f.index.equals(frames[0].index) for f in frames)
    assert frames[0].index.is_unique and frames[0].index.is_monotonic_increasing
    days = [str(v) for v in frames[0].index.normalize()]
    digest = hashlib.sha256()
    for value in frames:
        digest.update(pd.util.hash_pandas_object(value, index=True).values.tobytes())
    digest.update(json.dumps(list(frames[0].columns), default=str).encode())
    digest.update(json.dumps(days).encode())
    return digest.hexdigest(), frames


def supports(point, labels):
    if "support_labels" in point:
        values = point["support_labels"]
    else:
        values = [[labels[int(i)] for i in s] for s in
                  point.get("support_indices", point.get("supports", []))]
    family = {tuple(sorted(int(v) for v in s)) for s in values}
    assert all(s and set(s) <= set(labels) for s in family)
    return family


def error(family, truth):
    missing, extra = truth-family, family-truth
    return dict(full_errors=len(missing)+len(extra), full_recovery=not missing and not extra,
                missing_count=len(missing), extra_count=len(extra),
                predicted_blocks=len(family), truth_blocks=len(truth))


def numeric_supports(point):
    return point.get("supports", point.get("support_indices"))


def own_loss(point, frames):
    p, q, v, root = frames
    n = p.shape[1]
    r, x = np.zeros((n, n)), np.zeros((n, n))
    for support, rw, xw in zip(numeric_supports(point), point["r"], point["x"], strict=True):
        r[np.ix_(support, support)] += rw
        x[np.ix_(support, support)] += xw
    y = root.to_numpy()[:, None]**2-v.to_numpy()**2
    return float(np.mean(np.abs(p.to_numpy()@r.T+q.to_numpy()@x.T-y)))


def independently_select(points, calibration, frames):
    vals = np.asarray([own_loss(point, frames) for point in points])
    for point, val in zip(points, vals, strict=True):
        if "validation_mae" in point:
            assert abs(val-point["validation_mae"]) <= 1e-11
    best = vals.min()
    eligible = [i for i, point in enumerate(points) if vals[i] <= 1.03*best+1e-12]
    selected = min(eligible, key=lambda i: (len(numeric_supports(points[i])), vals[i], i))
    point = points[selected]
    threshold = .16*calibration["median_root_depth"]
    keep = [i for i, s in enumerate(numeric_supports(point)) if len(s) == 1 or
            calibration["a"]*point["r"][i]+calibration["b"]*point["x"][i]+1e-10 >= threshold]
    family = {tuple(s) for i, s in enumerate(numeric_supports(point)) if i in keep}
    return point, family, float(best), len(eligible)


def audit_selection(fit, truth_supports, valid_frames):
    labels = fit["terminal_labels"]
    truth = {tuple(sorted(s)) for s in truth_supports}
    assert len(fit["bank"]) == 15
    calibration = fit["calibrations"]["raw"]
    points_by_model = {"rnj_bank": fit["bank"],
                       "baseline25": fit["bank"]+fit["searches"]["primary"]["path"],
                       "multistart_original_rule": fit["selection_candidates"]}
    records = {}
    for name in MODELS:
        model = fit["models"][name]
        raw, contracted, minimum, eligible = independently_select(points_by_model[name], calibration, valid_frames)
        stored_raw = model["selected_uncontracted"]
        assert numeric_supports(raw) == numeric_supports(stored_raw)
        assert np.array_equal(raw["r"], stored_raw["r"]) and np.array_equal(raw["x"], stored_raw["x"])
        predicted = {tuple(sorted(labels[i] for i in s)) for s in contracted}
        assert predicted == supports(model, labels)
        record = error(predicted, truth)
        record.update(raw_full_errors=error(supports(raw, labels), truth)["full_errors"],
                      validation_minimum=minimum, eligible_candidates=eligible,
                      selected_origin=model.get("selected_origin"))
        assert abs(own_loss(model, valid_frames)-model["validation_mae"]) <= 1e-11
        records[name] = record
    accepted = sum(max(0, len(s["path"])-1) for s in fit["searches"].values())
    for search in fit["searches"].values():
        for before, after in zip(search["path"], search["path"][1:]):
            assert before["objective"]-after["objective"] > 1e-10+.001*before["objective"]
    for solved in fit["LP_solves"]:
        assert solved["status"] == 2 and solved["accepted"]
    return records, dict(accepted_updates=accepted,
                         wall_limited=any(s["reason"] == "pool_wall_limit" for s in fit["searches"].values()),
                         path_stops={k: s["reason"] for k, s in fit["searches"].items()},
                         lp_calls=len(fit["LP_solves"]))


def resource_checks(resource, delta):
    base = vector(resource["baseline_kw"])
    lower, upper = vector(resource.get("power_lower_kw", 0.)), vector(resource["power_upper_kw"])
    prefix = .25*np.cumsum(delta)
    assert abs(float(prefix[-1])) <= TOL
    assert np.max(lower-base-delta) <= TOL and np.max(base+delta-upper) <= TOL
    assert np.max(vector(resource["prefix_delta_lower_kwh"])-prefix) <= TOL
    assert np.max(prefix-vector(resource["prefix_delta_upper_kwh"])) <= TOL
    available = np.broadcast_to(np.asarray(resource.get("available", True), bool), (96,))
    assert np.max(np.abs(delta[~available]), initial=0.) <= TOL
    battery = resource.get("battery")
    if battery:
        energy = battery["initial_kwh"]+battery.get("eta_charge", 1.)*.25*np.cumsum(base+delta)
        assert energy.min() >= battery.get("soc_min", 0.)*battery["capacity_kwh"]-TOL
        assert energy.max() <= battery.get("soc_max", 1.)*battery["capacity_kwh"]+TOL
        assert abs(energy[-1]-battery["terminal_kwh"]) <= TOL
        if resource.get("service_deadline_slot") is not None:
            assert abs(energy[resource["service_deadline_slot"]-1]-battery["terminal_kwh"]) <= TOL


def audit_physics(plan, request, receipt, private, packet, labels, problem):
    n = len(labels)
    positions = {label: i for i, label in enumerate(labels)}
    whole_pq, observed_pq, fees, expected_fees, hard_fees = [], [], [], [], []
    for event, row in enumerate(private["events"]):
        assert row["event"] == event and row["status"] == "complete"
        x = np.asarray(plan["expected_amplitudes_by_event"][event], float)
        prices = np.asarray(plan["prices_by_event"][event], float)
        factors = np.asarray(request["factors_by_event"][event], float)
        d = np.asarray(plan["response_factors_by_event"][event], float).reshape(len(x), 2)
        actual = x*(1+d@factors)
        assert np.max(np.abs(factors), initial=0.) <= 1
        assert np.allclose(row["actual_amplitudes"], actual, rtol=0, atol=1e-12)
        assert np.array_equal(row["expected_amplitudes"], x)
        assert np.array_equal(row["prices"], prices) and np.array_equal(row["factors"], factors)
        assert actual.min(initial=0.) >= -TOL
        dp, dq = np.zeros((96, n)), np.zeros((96, n))
        grouped = {r["resource_id"]: np.zeros(96) for r in plan["resources_by_event"][event]}
        resources = {r["resource_id"]: r for r in plan["resources_by_event"][event]}
        total_fee = expected_fee = hard_fee = 0.
        active = {key: 0 for key in resources}
        for j, module in enumerate(plan["modules_by_event"][event]):
            assert len(module["profiles"]) == 1
            key, shape = next(iter(module["profiles"].items()))
            shape = np.asarray(shape, float)
            assert shape.shape == (96,) and abs(shape.sum()) <= TOL
            assert actual[j] <= module["amplitude_upper_kw"]+TOL
            if actual[j] > TOL:
                active[key] += 1
            grouped[key] += shape*actual[j]
            reduction_hours = .25*np.maximum(-shape, 0.).sum()
            assert abs(reduction_hours-plan["reduction_hours_by_event"][event][j]) <= TOL
            total_fee += prices[j]*reduction_hours*actual[j]
            expected_fee += prices[j]*reduction_hours*x[j]
            hard_fee += prices[j]*reduction_hours*x[j]*(1+np.abs(d[j]).sum())
        assert all(count <= 1 for count in active.values())
        for key, resource in resources.items():
            delta = grouped[key]
            resource_checks(resource, delta)
            pos = positions[resource["node_label"]]
            dp[:, pos] += delta
            dq[:, pos] += vector(resource.get("reactive_ratio", .35))*delta
            stored = row["resource_audit"]["resources"][key]
            assert np.allclose(delta, stored["executed_delta_kw"], atol=1e-12, rtol=0)
        assert np.allclose(dp, row["delta_kw"], rtol=0, atol=1e-12)
        assert np.allclose(dq, row["delta_kvar"], rtol=0, atol=1e-12)
        assert np.allclose((row["P_pu"]-row["natural_P_pu"])*receipt["base_kw"], dp, atol=1e-10, rtol=0)
        assert np.allclose((row["Q_pu"]-row["natural_Q_pu"])*receipt["base_kw"], dq, atol=1e-10, rtol=0)
        assert abs(total_fee-receipt["event_records"][event]["actual_payment"]) <= TOL
        checks = receipt["event_records"][event]["checks"]
        assert checks["ac_all_96_converged"] and checks["electrical_equations_pass"] and checks["voltage_limits_pass"]
        assert checks["branch_ampacity_checked"] is False
        whole_pq.extend((np.asarray(row["P_pu"]).ravel()*receipt["base_kw"],
                         np.asarray(row["Q_pu"]).ravel()*receipt["base_kw"]))
        scene = packet["training"][event] if event < 2 else packet["validation"][event-2]
        observed_pq.extend((scene["P_terminal"].to_numpy().ravel()*receipt["base_kw"],
                            scene["Q_terminal"].to_numpy().ravel()*receipt["base_kw"]))
        fees.append(float(total_fee)); expected_fees.append(float(expected_fee)); hard_fees.append(float(hard_fee))
    assert len(private["events"]) == 4
    assert abs(sum(fees)-receipt["actual_payment"]) <= TOL
    assert abs(sum(expected_fees)-receipt["expected_payment"]) <= TOL
    assert abs(sum(hard_fees)-receipt["worst_payment"]) <= TOL
    assert sum(fees) <= receipt["incentive_budget"]+TOL and sum(hard_fees) <= receipt["incentive_budget"]+TOL
    g, offset = np.asarray(problem["benefit_directions"]), np.asarray(problem["benefit_offsets"])
    physical_benefit = offset+g@np.concatenate(whole_pq)
    observed_benefit = offset+g@np.concatenate(observed_pq)
    assert np.allclose(physical_benefit, receipt["excitation_physical"], atol=1e-10, rtol=0)
    assert np.allclose(observed_benefit, receipt["excitation_observed"], atol=1e-10, rtol=0)
    return dict(actual_payment=sum(fees), expected_payment=sum(expected_fees), hard_payment=sum(hard_fees),
                physical_excitation_pass=bool(np.all(physical_benefit >= plan["excitation_threshold"])),
                observed_excitation_pass=bool(np.all(observed_benefit >= plan["excitation_threshold"])))


def audit(root):
    root = Path(root).resolve()
    master = read(root/"price_freeze.json")
    assert master["status"] == "frozen" and master["prices_frozen_before_any_future"]
    assert set(master["policies"]) == set(POLICIES) and tuple(master["test_paths"]) == PATHS
    for path, digest in master["source_sha256"].items():
        assert sha(path) == digest
    cases = master["case_roots"]
    assert len(cases) == 2
    # All terminal/fit-freeze gates are checked before any truth/PQV packet read.
    case_packets = {}
    for key, path in cases.items():
        case_root = Path(path)
        c = read(reference(master["case_protocols"][key]))
        assert reference(master["case_protocols"][key]) == case_root/"protocol.json"
        assert set(c["plans"]) == set(POLICIES)
        for source, digest in c["source_sha256"].items():
            assert sha(source) == digest
        for plan in c["plans"].values():
            reference(plan)
        reference(c["problem"]); reference(c["history_packet"])
        led = read(case_root/"test/observations.json")
        frozen = read(case_root/"test/fits_frozen.json")
        scored = read(case_root/"test/scored.json")
        assert frozen["status"] == "frozen" and frozen["truth_has_not_yet_been_opened_by_controller"]
        assert frozen["observation_ledger_sha256"] == sha(case_root/"test/observations.json")
        assert scored["fits_frozen_sha256"] == sha(case_root/"test/fits_frozen.json")
        assert led["planned"] == scored["planned"] == 8
        expected = {(policy, path) for policy in POLICIES for path in PATHS}
        assert {(row["policy"], row["path"]) for row in led["rows"]} == expected
        assert len(led["rows"]) == len(scored["rows"]) == 8
        for worker in range(c["workers"]):
            assert (case_root/f"test/worker_{worker}_ended.json").is_file()
        case_packets[key] = (case_root, c, led, frozen, scored)
    rows, crn, counted_jobs, totals = [], {}, set(), dict(
        physical_complete=0, joint_complete=0, unique_joint_complete=0, failed_or_unresolved=0,
        wall_limited_complete=0, unique_wall_limited_complete=0, lp_calls=0)
    for key, (case_root, c, led, frozen, scored) in case_packets.items():
        history = read_pickle(reference(c["history_packet"]))["history"]
        assert sum(len(s["P_terminal"]) for s in history["training"]) == 288
        assert sum(len(s["P_terminal"]) for s in history["validation"]) == 288
        problem = read(reference(c["problem"]))
        jobs = {j["key"]: j for j in frozen["jobs"]}
        for scored_row in scored["rows"]:
            row = dict(case=key, policy=scored_row["policy"], path=scored_row["path"],
                       status=scored_row["status"], actual_payment=scored_row.get("actual_payment"),
                       models=None)
            if scored_row["status"] != "complete":
                totals["failed_or_unresolved"] += 1
                rows.append(row)
                continue
            physical = Path(scored_row["physical_output"])
            receipt = read(physical/"events_receipt.json")
            assert stamp(master["created_utc"]) <= stamp(receipt["started_utc"])
            request = read(physical.parent/"request.json")
            assert sha(physical.parent/"request.json") == scored_row["request_sha256"]
            plan = read(reference(c["plans"][row["policy"]]))
            assert reference(request["plan_packet"]) == reference(c["plans"][row["policy"]])
            assert request["protocol"]["sha256"] == sha(case_root/"protocol.json")
            for filename in ("packet.pkl", "physical_truth.pkl", "truth_evaluation.json"):
                assert sha(physical/filename) == receipt["private_files"][filename]["sha256"]
            packet = read_pickle(physical/"packet.pkl")
            private = read_pickle(physical/"physical_truth.pkl")
            phy = audit_physics(plan, request, receipt, private, packet, c["labels"], problem)
            row.update(phy)
            totals["physical_complete"] += 1
            for event, event_row in enumerate(private["events"]):
                cell = (key, row["path"], event)
                values = tuple(np.asarray(event_row[k]) for k in ("natural_P_pu", "natural_Q_pu", "root_pu", "factors"))
                if cell not in crn:
                    crn[cell] = values
                else:
                    assert all(np.array_equal(a, b) for a, b in zip(values, crn[cell], strict=True))
            fit = read(reference(scored_row["result"]))
            job = jobs[scored_row["fit_key"]]
            assert job["status"] == "complete" and job["result"]["sha256"] == scored_row["result"]["sha256"]
            assert fit["status"] == "complete" and fit["training_rows"] == fit["validation_rows"] == 480
            for split in ("training", "validation"):
                full = history[split]+packet[split]
                assert sum(len(s["P_terminal"]) for s in full) == 480
                digest, frames = own_input_hash(full)
                assert digest == fit["input_sha256"][split]
                if split == "validation":
                    valid_frames = frames
            truth = read(physical/"truth_evaluation.json")
            assert truth["zero_impedance_merged"] and truth["arbitrarily_weak_positive_lines_retained"]
            own_models, diagnostics = audit_selection(fit, truth["true_supports"], valid_frames)
            for name in MODELS:
                for metric in ("full_errors", "full_recovery"):
                    assert own_models[name][metric] == scored_row["models"][name][metric]
            row.update(models=own_models, diagnostics=diagnostics)
            totals["joint_complete"] += 1
            totals["wall_limited_complete"] += int(diagnostics["wall_limited"])
            unique_job = (key, scored_row["fit_key"])
            if unique_job not in counted_jobs:
                counted_jobs.add(unique_job)
                totals["unique_joint_complete"] += 1
                totals["unique_wall_limited_complete"] += int(diagnostics["wall_limited"])
                totals["lp_calls"] += diagnostics["lp_calls"]
            rows.append(row)
    paired = []
    for key in cases:
        for path in PATHS:
            cells = {row["policy"]: row for row in rows if row["case"] == key and row["path"] == path}
            for policy in POLICIES:
                a, b = cells[policy], cells["zero"]
                delta = None if not a["models"] or not b["models"] else (
                    a["models"]["multistart_original_rule"]["full_errors"]
                    -b["models"]["multistart_original_rule"]["full_errors"])
                paired.append(dict(case=key, path=path, policy=policy, full_error_change_vs_zero=delta,
                                   actual_payment=a["actual_payment"]))
    return dict(status="complete", planned_records=16, case_path_denominator=4,
                independent_cases=2, policies_per_case_path=4, rows=rows, paired=paired, totals=totals,
                price_freeze_sha256=sha(root/"price_freeze.json"), audit_source_sha256=sha(__file__),
                scope="independent scalar arithmetic, final selection and source/input/ordering audit",
                boundaries=["No new AC, QP, LP, SOCP or topology search performed by this audit.",
                            "Hashes/source and receipts document controller ordering; no operating-system access log is claimed.",
                            "Physical worker necessarily knows its generating graph before fit; truth is excluded from planner/identifier inputs.",
                            "Exact-zero physical lines omitted; all positive weak lines retained by evaluator. Numerical fit pruning remains the original rule.",
                            "Branch ampacity is not checked by the existing physical checker.",
                            "Two paths per case diagnose mechanism and do not certify population 90% excitation or recovery."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("audit outputs are never overwritten")
    try:
        result = audit(args.root)
    except Exception as exc:
        result = dict(status="audit_failed", error_type=type(exc).__name__, message=str(exc),
                      traceback=traceback.format_exc(), audit_source_sha256=sha(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result.get(key) for key in ("status", "planned_records", "totals", "message")}, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
