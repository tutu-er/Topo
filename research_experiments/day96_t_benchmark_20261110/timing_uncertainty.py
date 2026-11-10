"""Bound infrastructure-lost wall time after all native evaluations return.

The immutable recovery debit is an estimate, not a termination-time bound.
This postprocessor leaves that estimate and every worker result untouched,
then uses the host-confirmed upper bound to check each original 4-hour cap.
No optimizer, refit, candidate selection, truth or observation arrays are used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def bound_attempt(result, attempts, *, original_cap):
    """Return one evaluation's operational interval; never add proxy twice."""
    successful = float(result.get("successful_total_seconds", result["total_seconds"]))
    if not math.isfinite(successful) or successful < 0 or abs(successful-float(result["total_seconds"])) > 1e-7:
        raise AssertionError("Successful attempt timing does not match its unchanged native total")
    lower = sum(float(row["timing_lower_known_seconds"]) for row in attempts)
    proxy = sum(float(row["timing_proxy_debit_seconds"]) for row in attempts)
    upper = sum(float(row["timing_upper_receipt_seconds"]) for row in attempts)
    upper_reported = sum(float(row["timing_upper_reported_host_seconds"]) for row in attempts)
    if not all(math.isfinite(value) and value >= 0 for value in (lower, proxy, upper, upper_reported)):
        raise AssertionError("Infrastructure timing bounds must be finite and nonnegative")
    if lower > upper + 1e-7 or proxy > upper + 1e-7 or upper_reported > upper + 1e-7:
        raise AssertionError("Reported infrastructure bounds are inconsistent")
    charge = float(result.get("interrupted_attempt_seconds", 0.))
    if abs(charge-proxy) > 1e-7:
        raise AssertionError("Final result's immutable v1 debit differs from the recovery audit")
    cap = float(result.get("original_total_cap", original_cap))
    if cap != float(original_cap) or any(float(row["original_total_cap"]) != cap for row in attempts):
        raise AssertionError("The original per-evaluation budget changed")
    if not math.isfinite(cap) or cap <= 0:
        raise AssertionError("A positive original budget is required")
    logical = float(result.get("logical_total_seconds", successful+proxy))
    if abs(logical-(successful+proxy)) > 1e-7:
        raise AssertionError("Proxy-charged logical total is not successful + proxy")
    operational_lower, operational_proxy, operational_upper = successful+lower, successful+proxy, successful+upper
    # The upper is added to successful time, NOT to the already charged logical total.
    if abs(operational_upper-(logical-proxy+upper)) > 1e-7:
        raise AssertionError("Upper-bound accounting double-counted the proxy debit")
    certified = operational_upper <= cap
    exceeded = operational_lower > cap
    unresolved = not certified and not exceeded
    return dict(successful_algorithm_seconds=successful, interrupted_attempts=len(attempts),
                infrastructure_lower_known_seconds=lower, infrastructure_proxy_debit_seconds=proxy,
                infrastructure_upper_reported_host_seconds=upper_reported,
                infrastructure_upper_receipt_seconds=upper,
                operational_lower_seconds=operational_lower, operational_total_seconds=operational_proxy,
                operational_upper_seconds=operational_upper, original_total_cap_seconds=cap,
                conservative_budget_margin_seconds=cap-operational_upper,
                budget_conservative_certified=certified, budget_violation_proven=exceeded, budget_unresolved=unresolved,
                budget_status="certified_within" if certified else "certified_exceeded" if exceeded else "unresolved_upper_exceeds_cap",
                exact_interrupted_termination_time_known=not bool(attempts),
                total_seconds_semantics="final successful native attempt; operational_total_seconds is the unchanged proxy estimate")


def check(output, recovery_audit):
    started = perf_counter()
    output, recovery_audit = Path(output).resolve(), Path(recovery_audit).resolve()
    protocol_file = output / "protocol.json"
    protocol = read(protocol_file)
    methods = [row["id"] for row in protocol["methods"]]
    pairs = {(job, method) for job in protocol["jobs"] for method in methods}
    if len(pairs) != protocol["planned"] or len(pairs) != 144:
        raise ValueError("This final timing check expects all 144 native evaluations")
    missing = [pair for pair in pairs if not (output/"jobs"/pair[0]/pair[1]/"result.json").is_file()]
    if missing:
        raise RuntimeError(f"Timing certification waits for all native returns; {len(missing)} missing")
    native_audit_file = output / "analysis/verification.json"
    native = read(native_audit_file)
    if not (native.get("passed") is True and native.get("complete") is True and native.get("verified") == len(pairs)):
        raise RuntimeError("The full native numerical audit must pass before conservative timing certification")
    audit = read(recovery_audit)
    if audit.get("passed") is not True or audit["protocol_sha256"] != sha(protocol_file):
        raise AssertionError("Recovery audit is invalid or refers to a different frozen protocol")
    charges_file = output / "recovery/charges.json"
    host_file = output / "recovery/host_process_liveness.json"
    charges_sha, host_sha = sha(charges_file), sha(host_file)
    if charges_sha != audit["charges_sha256"] or host_sha != audit["host_liveness_receipt_sha256"]:
        raise AssertionError("Immutable recovery charges or host liveness receipt changed")
    charges, host = read(charges_file), read(host_file)
    if any(row["alive"] is not False for row in host["interrupted_worker_pids"]):
        raise AssertionError("The host receipt does not confirm all old workers terminated")
    grouped = defaultdict(list)
    for row in audit["interrupted_attempts"]:
        pair = (row["job"], row["method"])
        if pair not in pairs:
            raise AssertionError("Unknown interrupted job/method in the recovery audit")
        key = pair[0]+"/"+pair[1]
        if key not in charges["charges"]:
            raise AssertionError("Recovery audit contains an interruption absent from immutable v1 charges")
        if abs(float(charges["charges"][key]["interrupted_attempt_seconds"])-float(row["timing_proxy_debit_seconds"])) > 1e-7:
            raise AssertionError("Recovery audit proxy differs from the immutable v1 charge")
        grouped[pair].append(row)
    if sum(map(len, grouped.values())) != audit["interrupted_infrastructure_attempts"]:
        raise AssertionError("Interrupted attempt denominator differs from the audited recovery receipt")
    rows, result_hashes = [], {}
    catalog = read(output/"catalog.json")
    for job in protocol["jobs"]:
        for method in methods:
            file = output/"jobs"/job/method/"result.json"
            result = read(file)
            if (result["job"], result["method"]) != (job, method):
                raise AssertionError("Native result identity mismatch")
            if result["protocol_sha256"] != sha(protocol_file):
                raise AssertionError("Native result protocol changed")
            bounds = bound_attempt(result, grouped.get((job, method), []), original_cap=protocol["limits"]["total"])
            rows.append(dict(job=job, method=method, case=catalog[job]["case"], bundle=catalog[job]["bundle"],
                             voltage_sigma=catalog[job]["voltage_sigma"], **bounds))
            result_hashes[job+"/"+method] = sha(file)
    frame = pd.DataFrame(rows)
    summaries = []
    for method, group in frame.groupby("method", sort=False):
        summaries.append(dict(method=method, planned=len(protocol["jobs"]), returned=len(group),
                              interrupted_attempts=int(group.interrupted_attempts.sum()),
                              affected_evaluations=int((group.interrupted_attempts > 0).sum()),
                              successful_algorithm_seconds=float(group.successful_algorithm_seconds.sum()),
                              operational_lower_seconds=float(group.operational_lower_seconds.sum()),
                              operational_total_seconds=float(group.operational_total_seconds.sum()),
                              operational_upper_seconds=float(group.operational_upper_seconds.sum()),
                              conservative_budget_certified=int(group.budget_conservative_certified.sum()),
                              unresolved_budgets=int(group.budget_unresolved.sum()), proven_exceeded_budgets=int(group.budget_violation_proven.sum()),
                              minimum_conservative_margin_seconds=float(group.conservative_budget_margin_seconds.min())))
    summary = pd.DataFrame(summaries)
    target = output/"timing_uncertainty"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target/"per_model.csv", index=False, lineterminator="\n")
    summary.to_csv(target/"summary.csv", index=False, lineterminator="\n")
    certified = int(frame.budget_conservative_certified.sum())
    unresolved, exceeded = int(frame.budget_unresolved.sum()), int(frame.budget_violation_proven.sum())
    receipt = dict(passed=certified == len(pairs), complete=True, planned=len(pairs), returned=len(rows),
                   all_original_budget_caps_conservatively_certified=certified == len(pairs),
                   budget_conservative_certified=certified, budget_unresolved=unresolved, budget_violation_proven=exceeded,
                   interrupted_infrastructure_attempts=int(frame.interrupted_attempts.sum()),
                   successful_algorithm_seconds=float(frame.successful_algorithm_seconds.sum()),
                   operational_lower_seconds=float(frame.operational_lower_seconds.sum()),
                   operational_total_seconds=float(frame.operational_total_seconds.sum()),
                   operational_upper_seconds=float(frame.operational_upper_seconds.sum()),
                   operational_total_is_proxy_estimate=True, exact_old_worker_termination_time_known=False,
                   formula="successful_total_seconds + host-receipt upper lost-attempt seconds <= original_total_cap",
                   no_proxy_double_counting=True, unknown_upper_exceedance_reported_as_unresolved=True,
                   immutable_v1_charges_sha256=charges_sha, host_liveness_receipt_sha256=host_sha,
                   recovery_audit_sha256=sha(recovery_audit), native_analysis_receipt_sha256=sha(native_audit_file),
                   protocol_sha256=sha(protocol_file), script_sha256=sha(__file__), native_result_sha256=result_hashes,
                   worker_results_modified=False, charges_modified=False, optimizer_source_modified=False,
                   new_solver_calls=0, new_fit_calls=0, workers_stopped=0, workers_relaunched=0,
                   scope="per-evaluation algorithm wall bounds with shared frontend once; not solver CPU time or manager wall time",
                   seconds=perf_counter()-started)
    # Prove this read-only computation left the two immutable source receipts unchanged.
    if sha(charges_file) != charges_sha or sha(host_file) != host_sha:
        raise AssertionError("Recovery source receipts changed during the timing audit")
    write(target/"verification.json", receipt)
    print(summary.to_string(index=False))
    print(json.dumps({key: value for key, value in receipt.items() if key != "native_result_sha256"}, ensure_ascii=False))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=ROOT/"outputs/day96_t_benchmark_20261110")
    parser.add_argument("--recovery-audit", type=Path, default=ROOT/"docs/day96_t_benchmark_20261110/recovery_audit.json")
    args = parser.parse_args()
    check(args.output, args.recovery_audit)
