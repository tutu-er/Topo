"""Small descriptive diagnostics after the complete MIQP/L1 audit passes.

No observations, truth files, fitting code, or native solver are imported.
The input is the 192 SHA-verified saved models and their existing audit rows.
Run only after all 96 new and 96 paired parent models have passed analysis::

    python -m research_experiments.miqp_loss_benchmark_20261111.diagnostics OUTPUT

Writes OUTPUT/analysis/diagnostics/{summary.json,solver_calls.csv,
selector_changes.csv,t_effects.csv,interactions.csv,seed_changes.csv,coverage.csv}.
summary.json defines every contrast and records input/output hashes.  The CSVs
contain individual saved calls/models; summary arrays contain descriptive
counts and means.  A changed saved-point index is kept separate from a changed
parameter fit, support family, or positive proper-support topology.

When the reboot ledger exists, recovery_audit.json/recovery_charges.csv verify
the 78 preserved results, eight interrupted attempts and reduced retry caps.
loss_sensitivity.csv retains all48 as primary and additionally reports
new_uninterrupted (ledger-selected 44) and both_uninterrupted (zero recorded
interruption debit on both sides of each matched L1/MSE contrast).  No score
is used to define a sensitivity cohort.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
LOSSES = ("mae", "scaled_mse")
METHODS = ("noT", "T003")
VIEWS = ("primary_mae", "secondary_mse")
TIE = 1e-10


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                          encoding="utf-8", newline="\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("An expected complete diagnostics table is empty: "+str(path))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)


def canonical(supports):
    return tuple(sorted({tuple(map(int, C)) for C in supports}, key=lambda C: (len(C), C)))


def topology(point, n):
    return frozenset(tuple(C) for C, r, x in zip(point["supports"], point["r"], point["x"])
                     if 1 < len(C) < n and max(r, x) > 1e-9)


def fingerprint(point):
    return tuple(sorted((tuple(C), float(r), float(x)) for C, r, x in
                        zip(point["supports"], point["r"], point["x"])))


def finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def gaps(upper, lower):
    """Native minimization gap, with an explicit zero-incumbent convention."""
    upper, lower = finite(upper), finite(lower)
    if upper is None or lower is None:
        return dict(incumbent=upper, bound=lower, absolute_gap=None, relative_gap=None,
                    relative_gap_kind="unavailable")
    if lower > upper+2e-8:
        raise ValueError("A saved solver lower bound exceeds its incumbent.")
    absolute = max(0., upper-lower)
    relative = absolute/abs(upper) if upper != 0. else (0. if absolute == 0. else None)
    return dict(incumbent=upper, bound=lower, absolute_gap=absolute, relative_gap=relative,
                relative_gap_kind="finite" if relative is not None else "infinite_zero_incumbent")


def calls(result):
    for event_index, event in enumerate(result["trajectory"].get("attempts", [])):
        for trial_index, call in enumerate(event.get("trials", [event])):
            if call.get("status") is not None:
                yield event.get("kind", "unknown"), event_index, trial_index, call
    event = result.get("exit_event", {})
    if event.get("status") is not None:
        yield "final_exit", None, None, event


def average(values):
    values = [float(v) for v in values if v is not None]
    return sum(values)/len(values) if values else None


def close(a, b, message, tolerance=1e-6):
    a, b = finite(a), finite(b)
    if a is None or b is None or abs(a-b) > tolerance:
        raise ValueError(f"{message}: {a!r} versus {b!r}")


def utc(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("A recovery timestamp lacks its UTC offset.")
    return value.astimezone(timezone.utc)


def interrupted_seconds(result):
    value = float(result.get("interrupted_attempt_seconds", 0.))
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid saved infrastructure-interruption debit.")
    return value


def recovery_receipts(output, protocol, directory, ledger, models, before_keys):
    """Bind all 18 published recovery results to their exact helper receipts."""
    expected = {(job, method) for job in protocol["jobs"] for method in METHODS}-before_keys
    if len(expected) != 18:
        raise ValueError("The preserved78 complement must contain 18 dispatched recovery models.")
    receipts = {}
    ledger_file = directory/"charges.json"; ledger_sha = sha(ledger_file)
    source = (output/"source").resolve()
    for job, method in sorted(expected):
        key = job+"/"+method; result = models[job, "scaled_mse", method]
        entry = ledger["charges"].get(key); debit = float(entry["interrupted_attempt_seconds"]) if entry else 0.
        record = result["recovery"]
        if (record["job"] != job or record["method"] != method or record["protocol_sha256"] != ledger["protocol_sha256"]
                or result["recovery_charges_sha256"] != ledger_sha or record["recovery_charges_sha256"] != ledger_sha
                or Path(record["recovery_charges_file"]).resolve() != ledger_file.resolve()):
            raise ValueError("Published recovery result lost its ledger/protocol identity.")
        wrapper, native = Path(record["wrapper_file"]).resolve(), Path(record["native_worker_file"]).resolve()
        if sha(wrapper) != record["wrapper_sha256"] or not native.is_relative_to(source):
            raise ValueError("Recovery helper source changed or native worker escaped the frozen source.")
        relative = native.relative_to(source).as_posix()
        if sha(native) != record["native_worker_sha256"] or protocol["source_hashes"].get(relative) != record["native_worker_sha256"]:
            raise ValueError("Published recovery native worker differs from its frozen source.")
        payload_file = (directory/"new_attempts"/job/method/"payload.json").resolve()
        if Path(record["payload_file"]).resolve() != payload_file or sha(payload_file) != record["payload_sha256"]:
            raise ValueError("Published recovery payload reference changed.")
        payload = read(payload_file); control = payload["recovery_control"]
        if (payload["job"] != job or payload["method"] != method or payload["source_hashes"] != protocol["source_hashes"]
                or payload["protocol_sha256"] != ledger["protocol_sha256"] or control["charges_sha256"] != ledger_sha
                or Path(control["charges_file"]).resolve() != ledger_file.resolve()):
            raise ValueError("Recovery payload source/ledger identity changed.")
        method_flags = next(row for row in protocol["methods"] if row["id"] == method)
        if (payload["flags"] != method_flags
                or payload["input_hashes"] != {s: protocol["input_hashes"][job][s] for s in ("train", "validation")}
                or payload["common_sha256"] != sha(payload["common"])):
            raise ValueError("Recovery changed method flags, observed inputs, or common preprocessing.")
        close(control["interrupted_attempt_seconds"], debit, "Recovery payload debit")
        close(record["interrupted_attempt_seconds"], debit, "Recovery metadata debit")
        close(record["original_total_cap"], protocol["limits"]["total"], "Recovery metadata original cap")
        remaining = float(protocol["limits"]["total"])-debit
        close(record["remaining_total_cap"], remaining, "Recovery metadata remaining cap")
        expected_limits = dict(protocol["limits"], total=remaining)
        if result["limits"] != expected_limits:
            raise ValueError("Recovery changed a numerical setting besides the remaining total allowance.")
        if (record["common_charged_once"] is not True or record["optimizer_source_modified"] is not False
                or bool(record["new_never_started_worker"]) != (entry is None)
                or bool(record["charge_is_conservative_upper_bound"]) != bool(entry)
                or int(record["interrupted_attempt_count"]) != int(bool(entry))):
            raise ValueError("Recovery metadata changed its source/timing boundary.")
        if entry:
            old_file = (directory/"jobs"/job/method/"payload.json").resolve()
            if (Path(record["original_payload_file"]).resolve() != old_file
                    or Path(control["original_payload_file"]).resolve() != old_file
                    or record["original_payload_sha256"] != sha(old_file)
                    or record["original_payload_sha256"] != entry["preserved_files"]["payload.json"]
                    or {k: v for k, v in payload.items() if k != "recovery_control"} != read(old_file)):
                raise ValueError("Recovery modified an interrupted original optimizer payload.")
        elif (record["original_payload_file"] is not None or record["original_payload_sha256"] is not None
                or control.get("original_payload_file") is not None):
            raise ValueError("A never-started model unexpectedly references an interrupted payload.")
        folder = output/"jobs"/job/method
        canonical_file = folder/"recovery.json"; published = read(canonical_file)
        expected_receipt = dict(record, stage="published", result_sha256=sha(folder/"result.json"),
            logical_total_seconds=result["logical_total_seconds"], successful_total_seconds=result["successful_total_seconds"],
            logical_total_cap_exceeded=result["logical_total_cap_exceeded"])
        if published != expected_receipt:
            raise ValueError("Canonical recovery receipt and embedded published metadata differ.")
        pending = folder/"result.pending.json"
        if sha(pending) != record["native_pending_sha256"]:
            raise ValueError("The preserved native result before accounting changed.")
        for field, value in read(pending).items():
            if field != "operational_total_seconds" and result.get(field) != value:
                raise ValueError("Recovery changed a native result field besides operational accounting: "+field)
        close(result["logical_total_seconds"], float(result["total_seconds"])+debit, "Published logical timing")
        close(result["operational_total_seconds"], result["logical_total_seconds"], "Published operational timing")
        close(interrupted_seconds(result), debit, "Published interrupted debit")
        close(result["original_total_cap"], protocol["limits"]["total"], "Published original cap")
        if (result["total_seconds"] > remaining+1. or result["logical_total_seconds"] > protocol["limits"]["total"]+1.
                or result["logical_total_cap_exceeded"]):
            raise ValueError("Published recovery exceeds a remaining or logical original allowance.")
        if not utc(ledger["verified_system_boot_utc"]) <= utc(record["started_utc"]) <= utc(record["completed_utc"]):
            raise ValueError("Recovery execution timestamps precede the externally verified host reboot.")
        receipts[key] = dict(recovery_receipt_sha256=sha(canonical_file), wrapper_sha256=record["wrapper_sha256"],
            native_worker_sha256=record["native_worker_sha256"], payload_sha256=record["payload_sha256"],
            native_pending_sha256=record["native_pending_sha256"], never_started=entry is None,
            wrapper_wall_seconds=record["wrapper_wall_seconds"], interrupted_attempt_seconds=debit,
            remaining_total_cap=remaining, logical_upper_seconds=result["logical_total_seconds"])
    if any("recovery" in models[job, "scaled_mse", method] for job, method in before_keys):
        raise ValueError("An originally completed preserved result was republished through recovery.")
    return receipts


def audit_recovery(output, protocol, models):
    """Check immutable recovery evidence; never query/stop/relaunch a process."""
    directory = output/"recovery/host_restart_20261111"
    file = directory/"charges.json"
    if not file.exists():
        if any(interrupted_seconds(r) > 0 for (job, loss, method), r in models.items() if loss == "scaled_mse"):
            raise ValueError("New models record interruption debits but the authoritative ledger is absent.")
        return dict(applicable=False, affected_inputs=[], affected_input_count=0,
                    primary_inputs=48, new_uninterrupted_inputs=48), []
    ledger = read(file)
    if ledger["protocol_sha256"] != sha(output/"protocol.json") or ledger["catalog_sha256"] != sha(output/"catalog.json"):
        raise ValueError("Recovery ledger refers to a different frozen experiment.")
    if (ledger["returned_preserved"] != 78 or ledger["interrupted_attempts"] != 8
            or ledger["never_started"] != 10 or ledger["logical_models_planned"] != 96
            or ledger["attempts_after_recovery_expected"] != 104):
        raise ValueError("The recorded 78+8+10 recovery denominator changed.")
    if (ledger["model_or_data_configuration_changed"] is not False
            or ledger["optimizer_processes_stopped_by_capture"] is not False
            or ledger["frozen_sources_verified"] is not True):
        raise ValueError("The immutable restart ledger violates its recovery scope.")
    capture_source = Path(__file__).with_name("capture_recovery.py")
    if sha(capture_source) != ledger["capture_script_sha256"]:
        raise ValueError("The saved capture-helper SHA does not match its source.")
    previous = ledger["result_hashes_before_recovery"]
    if len(previous) != 78:
        raise ValueError("The ledger does not preserve exactly 78 returned result hashes.")
    before_keys = set()
    for relative, digest in previous.items():
        path = (output/relative).resolve()
        if not path.is_relative_to((output/"jobs").resolve()) or sha(path) != digest:
            raise ValueError("A completed pre-restart result was changed: "+relative)
        parts = Path(relative).parts
        if len(parts) != 4 or parts[0] != "jobs" or parts[3] != "result.json":
            raise ValueError("Malformed preserved result path.")
        before_keys.add((parts[1], parts[2]))
    charges = ledger["charges"]
    if len(charges) != 8:
        raise ValueError("The ledger must contain eight interrupted logical methods.")
    receipts = recovery_receipts(output, protocol, directory, ledger, models, before_keys)
    boot = utc(ledger["verified_system_boot_utc"])
    state_file = directory/"all_worker_state.json"; state = read(state_file)
    heartbeat = datetime.fromtimestamp(state_file.stat().st_mtime, timezone.utc)
    if (state["returned"] != 78 or state["failed"] != 0 or len(state["active"]) != 8 or state["queued"] != 10):
        raise ValueError("Preserved controller state disagrees with the recovery denominator.")
    active = {str(Path(row["result"]).resolve()): int(row["pid"]) for row in state["active"]}
    rows, affected = [], set()
    for key, charge in charges.items():
        job, method = charge["job"], charge["method"]
        if key != job+"/"+method or job not in protocol["jobs"] or method not in METHODS or (job, method) in before_keys:
            raise ValueError("An interrupted key is malformed or overwrites a preserved result.")
        affected.add(job); saved = directory/"jobs"/job/method
        for name, digest in charge["preserved_files"].items():
            path = (saved/name).resolve()
            if path.parent != saved.resolve() or sha(path) != digest:
                raise ValueError("A preserved interrupted-attempt file changed: "+key+"/"+name)
        payload_file = saved/"payload.json"; payload, attempt = read(payload_file), read(saved/"attempt.json")
        if attempt["state"] != "running" or attempt["payload_sha256"] != sha(payload_file):
            raise ValueError("Interrupted original attempt and payload do not match.")
        original_result = str((output/"jobs"/job/method/"result.json").resolve())
        if (str(Path(payload["result"]).resolve()) != original_result or str(Path(attempt["result"]).resolve()) != original_result
                or active.get(original_result) != int(attempt["pid"]) or int(charge["original_pid"]) != int(attempt["pid"])):
            raise ValueError("Interrupted process/payload identity disagrees with the preserved controller state.")
        if payload["protocol_sha256"] != ledger["protocol_sha256"] or payload["source_hashes"] != protocol["source_hashes"]:
            raise ValueError("Interrupted original worker used a different frozen source/protocol.")
        if payload["input_hashes"] != {s: protocol["input_hashes"][job][s] for s in ("train", "validation")}:
            raise ValueError("Interrupted original observations differ from the frozen protocol.")
        if payload["common_sha256"] != sha(payload["common"]):
            raise ValueError("Interrupted worker common preprocessing changed.")
        common = read(payload["common"])
        loaded = read(saved/"loaded_sources_before.json")
        if loaded["passed"] is not True:
            raise ValueError("Original worker did not pass its frozen-import check.")
        for module in loaded["loaded_modules"].values():
            if protocol["source_hashes"].get(module["relative_path"]) != module["sha256"]:
                raise ValueError("Original worker's loaded-source SHA differs from the frozen protocol.")
        start = utc(charge["payload_created_utc"]); last = utc(charge["last_controller_heartbeat_utc"])
        finish = utc(charge["termination_upper_bound_utc"])
        close(start.timestamp(), payload_file.stat().st_mtime, "Original payload timestamp", tolerance=1e-5)
        close(last.timestamp(), heartbeat.timestamp(), "Original controller heartbeat timestamp", tolerance=1e-5)
        if not start <= utc(attempt["started_utc"]) <= last < finish or finish != boot:
            raise ValueError("Recovery timestamps do not provide the recorded pre-spawn-to-boot upper interval.")
        debit = float(charge["interrupted_attempt_seconds"])
        close(debit, (finish-start).total_seconds(), "Conservative interrupted-attempt debit")
        close(charge["last_seen_wall_proxy_seconds"], (last-start).total_seconds(), "Last heartbeat proxy")
        close(charge["original_total_cap"], protocol["limits"]["total"], "Original logical cap")
        remaining = float(charge["remaining_total_cap"])
        close(remaining, float(charge["original_total_cap"])-debit, "Reduced retry cap")
        if not debit > 0 or not remaining > float(common["seconds"])+float(protocol["limits"]["post_reserve"]):
            raise ValueError("The retry allowance does not retain the required common/post-analysis reserve.")
        if charge["charge_is_conservative_upper_bound"] is not True or charge["solver_failure"] is not False:
            raise ValueError("Infrastructure interruption was relabeled or lost its upper-bound convention.")
        result = models[job, "scaled_mse", method]
        close(interrupted_seconds(result), debit, "Native result interruption debit")
        close(result["original_total_cap"], charge["original_total_cap"], "Native original cap")
        close(result["limits"]["total"], remaining, "Native remaining cap")
        close(result["successful_total_seconds"], result["total_seconds"], "Successful native time")
        close(result["logical_total_seconds"], float(result["total_seconds"])+debit, "Logical upper timing")
        close(result["operational_total_seconds"], result["logical_total_seconds"], "Operational upper timing")
        if int(result["interrupted_attempt_count"]) != 1 or result["logical_total_seconds"] > float(charge["original_total_cap"])+1.:
            raise ValueError("Final logical time exceeds the original allowance or records an incorrect attempt count.")
        rows.append(dict(job=job, method=method, debit_upper_seconds=debit, remaining_total_cap_seconds=remaining,
            successful_total_seconds=float(result["total_seconds"]), logical_upper_seconds=float(result["logical_total_seconds"]),
            original_total_cap_seconds=float(charge["original_total_cap"]),
            budget_upper_certified=float(result["total_seconds"])+debit <= float(charge["original_total_cap"])+1.,
            preserved_payload_sha256=sha(payload_file),
            recovery_receipt_sha256=receipts[key]["recovery_receipt_sha256"],
            recovery_wrapper_sha256=receipts[key]["wrapper_sha256"],
            source_math="verified-system-boot timestamp minus preserved pre-spawn payload mtime"))
    total = sum(r["debit_upper_seconds"] for r in rows)
    close(total, ledger["total_interrupted_wall_upper_seconds"], "Total upper debit")
    new_charged = {(job, method) for (job, loss, method), result in models.items()
                   if loss == "scaled_mse" and interrupted_seconds(result) > 0}
    if new_charged != {(c["job"], c["method"]) for c in charges.values()} or len(affected) != 4:
        raise ValueError("Final charged methods or ledger-selected affected input set changed.")
    return dict(applicable=True, passed=True, ledger_sha256=sha(file), ledger_path=str(file),
        capture_source_sha256=sha(capture_source), preserved_controller_state_sha256=sha(state_file),
        preserved_completed_results=78, verified_completed_hashes=78, interrupted_attempts=8, never_started=10,
        logical_models=96, known_native_attempts=104, total_interrupted_upper_seconds=total,
        published_recovery_receipts=len(receipts), recovery_receipts=receipts,
        affected_inputs=sorted(affected), affected_input_count=4, primary_inputs=48, new_uninterrupted_inputs=44,
        budget_upper_certified=all(r["budget_upper_certified"] for r in rows),
        budget_tolerance_seconds=1.,
        bound_provenance="arithmetic/source identities independently checked; host boot and no-live-process facts are the externally verified assertions retained in the immutable ledger",
        termination_upper_bound_utc=ledger["verified_system_boot_utc"],
        host_liveness_assertion=ledger["liveness_audit"], new_native_optimizer_calls=0), rows


def loss_sensitivity(jobs, models, views, recovery):
    """Matched cohorts are defined solely by infrastructure records."""
    new_affected = set(recovery["affected_inputs"])
    summaries, cohorts = [], []
    for method in METHODS:
        both_affected = {job for job in jobs if any(interrupted_seconds(models[job, loss, method]) > 0 for loss in LOSSES)}
        definitions = (("all48", set()), ("new_uninterrupted", new_affected),
                       ("both_uninterrupted", both_affected))
        for name, excluded in definitions:
            included = [job for job in jobs if job not in excluded]
            if not included:
                raise ValueError("An infrastructure-defined sensitivity cohort is empty.")
            cohorts.append(dict(cohort=name, method=method, planned=len(included), excluded_inputs=sorted(excluded),
                included_inputs=included, selection_uses_F1=False,
                exclusion_source="none" if name == "all48" else ("new immutable reboot ledger, either new method" if name == "new_uninterrupted"
                    else "positive interrupted_attempt_seconds in either model of this method's L1/MSE pair")))
            for view in VIEWS:
                paired = [(views[job, "mae", method, view], views[job, "scaled_mse", method, view]) for job in included]
                deltas = [float(b["clade_f1"])-float(a["clade_f1"]) for a, b in paired]
                summaries.append(dict(cohort=name, method=method, view=view, planned=len(included), paired=len(included),
                    excluded_inputs=len(excluded), L1_F1=average(a["clade_f1"] for a, b in paired),
                    MSE_F1=average(b["clade_f1"] for a, b in paired), delta_F1=average(deltas),
                    L1_exact=sum(bool(a["exact"]) for a, b in paired), MSE_exact=sum(bool(b["exact"]) for a, b in paired),
                    delta_exact=sum(int(b["exact"])-int(a["exact"]) for a, b in paired),
                    new_exact=sum(b["exact"] and not a["exact"] for a, b in paired),
                    lost_exact=sum(a["exact"] and not b["exact"] for a, b in paired),
                    F1_wins=sum(d > TIE for d in deltas), F1_ties=sum(abs(d) <= TIE for d in deltas), F1_losses=sum(d < -TIE for d in deltas),
                    L1_successful_seconds=sum(float(a["total_seconds"]) for a, b in paired),
                    MSE_successful_seconds=sum(float(b["total_seconds"]) for a, b in paired),
                    L1_recorded_operational_seconds=sum(float(a["operational_total_seconds"]) for a, b in paired),
                    MSE_recorded_operational_upper_seconds=sum(float(b["operational_total_seconds"]) for a, b in paired),
                    MSE_interrupted_evaluations=sum(interrupted_seconds(models[job, "scaled_mse", method]) > 0 for job in included),
                    L1_interrupted_evaluations=sum(interrupted_seconds(models[job, "mae", method]) > 0 for job in included)))
    return summaries, cohorts


def contrast_summary(rows, keys, field):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, group in groups.items():
        item = dict(zip(keys, key)); values = [float(r[field]) for r in group]
        item.update(paired=len(group), mean=average(values), total=sum(values),
                    wins=sum(v > TIE for v in values), ties=sum(abs(v) <= TIE for v in values),
                    losses=sum(v < -TIE for v in values))
        output.append(item)
    return output


def guard(output):
    protocol, verification = read(output/"protocol.json"), read(output/"analysis/verification.json")
    if (protocol["planned"] != 96 or len(protocol["jobs"]) != 48
            or tuple(m["id"] for m in protocol["methods"]) != METHODS):
        raise ValueError("The fixed 48-input / 96-new-model protocol is required.")
    if (not verification["passed"] or not verification["complete"]
            or verification["new_native_verified"] != 96 or verification["reused_parent_verified"] != 96):
        raise RuntimeError("Diagnostics require all 96 new and 96 paired parent models to pass independent analysis.")
    if any(verification[k] != 0 for k in ("new_native_failed", "new_native_invalid", "new_native_pending")):
        raise ValueError("A nonzero unresolved new-model denominator remains.")
    if verification["protocol_sha256"] != sha(output/"protocol.json") or verification["catalog_sha256"] != sha(output/"catalog.json"):
        raise ValueError("The verified new protocol or catalog changed.")
    parent = Path(protocol["parent_output"])
    parent_verification = read(parent/"analysis/verification.json")
    if (not parent_verification["passed"] or not parent_verification["complete"]
            or parent_verification["planned"] != 144 or parent_verification["verified"] != 144):
        raise ValueError("Complete passed parent L1 verification is required.")
    if verification["parent_analysis_verification_sha256"] != sha(parent/"analysis/verification.json"):
        raise ValueError("The reused parent analysis changed.")
    if protocol["parent_protocol_sha256"] != sha(parent/"protocol.json") or protocol["parent_catalog_sha256"] != sha(parent/"catalog.json"):
        raise ValueError("The reused parent protocol or catalog changed.")
    for base, receipt in ((output, verification), (parent, parent_verification)):
        if sha(base/"analysis_source/analyze.py") != receipt["analyzer_sha256"]:
            raise ValueError("An authoritative frozen post-analysis source changed.")
    for base, hashes in ((output, protocol["source_hashes"]), (parent, protocol["parent_source_hashes"])):
        for name, digest in hashes.items():
            if sha(base/"source"/name) != digest:
                raise ValueError("A frozen native source changed: "+name)
    return protocol, parent, verification


def diagnostics(output):
    output = Path(output).resolve(); started = perf_counter()
    # This guard runs before any saved metrics are read or output is created.
    protocol, parent, verification = guard(output)
    catalog = read(output/"catalog.json")
    models, view_rows, model_rows, source_hashes = {}, {}, {}, {}
    verification_hashes = {}
    for job in protocol["jobs"]:
        audit_file = output/"analysis/jobs"/job/"verified.json"
        audit = read(audit_file)
        if audit["analyzer_sha256"] != verification["analyzer_sha256"] or audit["protocol_sha256"] != verification["protocol_sha256"]:
            raise ValueError("A per-input audit refers to a different analysis or protocol.")
        verification_hashes[job] = sha(audit_file)
        for row in audit["rows"]:
            key = (job, row["fit_loss"], row["method"])
            if row["status"] != "verified" or key in model_rows:
                raise ValueError("Missing, repeated, or unverified model row.")
            model_rows[key] = row
        for row in audit["views"]:
            key = (job, row["fit_loss"], row["method"], row["view"])
            if row["status"] != "verified" or key in view_rows:
                raise ValueError("Missing, repeated, or unverified selector-view row.")
            view_rows[key] = row
        for loss_name, base in (("mae", parent), ("scaled_mse", output)):
            for method in METHODS:
                file = base/"jobs"/job/method/"result.json"; measured = sha(file)
                if audit["raw_hashes"][loss_name+"/"+method] != measured:
                    raise ValueError("A saved model differs from the current independent audit.")
                if loss_name == "mae":
                    provenance = audit["parent_analysis"][method]
                    if (provenance["result_sha256"] != measured
                            or provenance["job_verification_sha256"] != sha(parent/"analysis/jobs"/job/"verified.json")
                            or provenance["analysis_verification_sha256"] != sha(parent/"analysis/verification.json")):
                        raise ValueError("A reused parent per-input audit changed.")
                source_hashes[job+"/"+loss_name+"/"+method] = measured
                models[job, loss_name, method] = read(file)
    expected_models = {(j, loss, method) for j in protocol["jobs"] for loss in LOSSES for method in METHODS}
    expected_views = {(*key, view) for key in expected_models for view in VIEWS}
    if set(models) != set(model_rows) or set(models) != expected_models or set(view_rows) != expected_views:
        raise ValueError("The verified 192-model / 384-view denominator changed.")
    recovery, recovery_rows = audit_recovery(output, protocol, models)
    sensitivity_rows, sensitivity_cohorts = loss_sensitivity(protocol["jobs"], models, view_rows, recovery)

    solver_rows, selectors, seed_rows, coverage_rows = [], [], [], []
    for (job, loss_name, method), result in models.items():
        meta = catalog[job]
        header = dict(job=job, case=meta["case"], seed_index=meta["seed_index"],
                      voltage_sigma=meta["voltage_sigma"], fit_loss=loss_name, method=method)
        for index, (stage, event_index, trial_index, call) in enumerate(calls(result)):
            upper = call.get("objective", call.get("final_objective"))
            lower = call.get("dual_bound", call.get("bound", call.get("final_bound")))
            gap = gaps(upper, lower)
            constant = float(call.get("protected_count_cost", call.get("fixed_count_cost", 0.)))
            regularized = gaps(None if gap["incumbent"] is None else gap["incumbent"]+constant,
                               None if gap["bound"] is None else gap["bound"]+constant)
            solver_rows.append(dict(**header, call_index=index, stage=stage, event_index=event_index,
                trial_index=trial_index, status=int(call["status"]), seconds=finite(call.get("seconds")),
                saved_relative_gap=finite(call.get("relative_gap", call.get("gap"))),
                requested_relative_gap=finite(call.get("requested_relative_gap")), **gap,
                protected_count_constant=constant, regularized_absolute_gap=regularized["absolute_gap"],
                regularized_relative_gap=regularized["relative_gap"],
                gap_scope="native model objective; protected count constant separately added when recorded"))
        by_index = {int(row["index"]): row for row in result["output_candidates"]}
        primary = view_rows[job, loss_name, method, "primary_mae"]
        secondary = view_rows[job, loss_name, method, "secondary_mse"]
        a, b = by_index[int(primary["selected_index"])], by_index[int(secondary["selected_index"])]
        if int(result["selected"]["index"]) != int(a["index"]) or fingerprint(result["selected"]["point"]) != fingerprint(a["point"]):
            raise ValueError("The primary view does not anchor the actual selected full model.")
        if "selected_mse_view" in result and (int(result["selected_mse_view"]["index"]) != int(b["index"])
                or fingerprint(result["selected_mse_view"]["point"]) != fingerprint(b["point"])):
            raise ValueError("The secondary view does not anchor its saved full model.")
        n = len(meta["labels"])
        selectors.append(dict(**header, output_library_size=len(by_index), primary_index=a["index"], secondary_index=b["index"],
            saved_point_changed=a["index"] != b["index"], parameter_fit_changed=fingerprint(a["point"]) != fingerprint(b["point"]),
            support_family_changed=canonical(a["point"]["supports"]) != canonical(b["point"]["supports"]),
            topology_changed=topology(a["point"], n) != topology(b["point"], n),
            delta_F1=secondary["clade_f1"]-primary["clade_f1"], delta_exact=int(secondary["exact"])-int(primary["exact"]),
            primary_F1=primary["clade_f1"], secondary_F1=secondary["clade_f1"],
            primary_exact=bool(primary["exact"]), secondary_exact=bool(secondary["exact"]),
            primary_validation_mae=primary["validation_mae"], secondary_validation_mae=secondary["validation_mae"],
            primary_validation_mse=primary["validation_mse"], secondary_validation_mse=secondary["validation_mse"]))
        row = model_rows[job, loss_name, method]
        raw, mapped = bool(row["raw_pool_has_true_tree"]), bool(row["output_pool_has_true_tree"])
        united = bool(row["own_pool_union_has_true_connections"])
        coverage_rows.append(dict(**header, raw_pool_has_true_tree=raw, output_pool_has_true_tree=mapped,
            own_pool_union_has_true_connections=united, own_pool_union_true_recall=row["own_pool_union_true_recall"],
            truth_lost_by_output_mapping=raw and not mapped, truth_created_by_output_mapping=not raw and mapped,
            primary_misses_available_true_tree=mapped and not primary["exact"],
            secondary_misses_available_true_tree=mapped and not secondary["exact"],
            union_connections_without_raw_true_tree=united and not raw,
            primary_exact=bool(primary["exact"]), secondary_exact=bool(secondary["exact"])))

    for job in protocol["jobs"]:
        for method in METHODS:
            old, new = (models[job, loss_name, method]["trajectory"]["seeding"] for loss_name in LOSSES)
            a, b = canonical(old["seeds"]), canonical(new["seeds"])
            seed_rows.append(dict(job=job, case=catalog[job]["case"], seed_index=catalog[job]["seed_index"],
                voltage_sigma=catalog[job]["voltage_sigma"], method=method, seed_family_changed=a != b,
                L1_seed_count=len(a), MSE_seed_count=len(b), removed_seed_count=len(set(a)-set(b)),
                added_seed_count=len(set(b)-set(a)), L1_ranking_complete=bool(old["ranking_complete"]),
                MSE_ranking_complete=bool(new["ranking_complete"]),
                L1_seeds=json.dumps(a), MSE_seeds=json.dumps(b)))

    t_rows, interaction_rows = [], []
    within_loss_seed_changes = []
    for job in protocol["jobs"]:
        for loss_name in LOSSES:
            seeds = [canonical(models[job, loss_name, method]["trajectory"]["seeding"]["seeds"]) for method in METHODS]
            if seeds[0] != seeds[1]:
                within_loss_seed_changes.append(dict(job=job, fit_loss=loss_name, noT_seeds=seeds[0], T003_seeds=seeds[1]))
            for view in VIEWS:
                a, b = (view_rows[job, loss_name, method, view] for method in METHODS)
                t_rows.append(dict(job=job, case=catalog[job]["case"], seed_index=catalog[job]["seed_index"],
                    voltage_sigma=catalog[job]["voltage_sigma"], fit_loss=loss_name, view=view,
                    delta_F1=b["clade_f1"]-a["clade_f1"], delta_exact=int(b["exact"])-int(a["exact"]),
                    delta_seconds=b["total_seconds"]-a["total_seconds"],
                    new_exact=bool(b["exact"] and not a["exact"]), lost_exact=bool(a["exact"] and not b["exact"]),
                    same_initial_seed_family=seeds[0] == seeds[1]))
        for view in VIEWS:
            cells = {(loss_name, method): view_rows[job, loss_name, method, view]
                     for loss_name in LOSSES for method in METHODS}
            def interaction(field):
                return ((float(cells["scaled_mse", "T003"][field])-float(cells["scaled_mse", "noT"][field]))
                       -(float(cells["mae", "T003"][field])-float(cells["mae", "noT"][field])))
            interaction_rows.append(dict(job=job, case=catalog[job]["case"], seed_index=catalog[job]["seed_index"],
                voltage_sigma=catalog[job]["voltage_sigma"], view=view, F1_interaction=interaction("clade_f1"),
                exact_interaction=interaction("exact"), seconds_interaction=interaction("total_seconds")))

    solver_summary = []
    for loss_name in LOSSES:
        for method in METHODS:
            group = [r for r in solver_rows if r["fit_loss"] == loss_name and r["method"] == method]
            for stage in ["all", *sorted({r["stage"] for r in group})]:
                subset = group if stage == "all" else [r for r in group if r["stage"] == stage]
                gap_values = [r["relative_gap"] for r in subset if r["relative_gap"] is not None]
                solver_summary.append(dict(fit_loss=loss_name, method=method, stage=stage, calls=len(subset),
                    status2=sum(r["status"] == 2 for r in subset), status9=sum(r["status"] == 9 for r in subset),
                    other_statuses={str(s): sum(r["status"] == s for r in subset) for s in sorted({r["status"] for r in subset}-{2, 9})},
                    computable_gaps=len(gap_values), max_relative_gap=max(gap_values) if gap_values else None,
                    mean_relative_gap=average(gap_values), solver_seconds=sum(r["seconds"] or 0. for r in subset)))
    selector_summary, seed_summary, coverage_summary = [], [], []
    for loss_name in LOSSES:
        for method in METHODS:
            subset = [r for r in selectors if r["fit_loss"] == loss_name and r["method"] == method]
            selector_summary.append(dict(fit_loss=loss_name, method=method, paired=len(subset),
                **{name: sum(bool(r[name]) for r in subset) for name in
                   ("saved_point_changed", "parameter_fit_changed", "support_family_changed", "topology_changed")},
                delta_F1=average(r["delta_F1"] for r in subset), delta_exact=sum(r["delta_exact"] for r in subset),
                new_exact=sum(r["secondary_exact"] and not r["primary_exact"] for r in subset),
                lost_exact=sum(r["primary_exact"] and not r["secondary_exact"] for r in subset)))
            subset = [r for r in coverage_rows if r["fit_loss"] == loss_name and r["method"] == method]
            coverage_summary.append(dict(fit_loss=loss_name, method=method, models=len(subset),
                **{name: sum(bool(r[name]) for r in subset) for name in
                   ("raw_pool_has_true_tree", "output_pool_has_true_tree", "own_pool_union_has_true_connections",
                    "truth_lost_by_output_mapping", "truth_created_by_output_mapping",
                    "primary_misses_available_true_tree", "secondary_misses_available_true_tree",
                    "union_connections_without_raw_true_tree", "primary_exact", "secondary_exact")}))
    for method in METHODS:
        subset = [r for r in seed_rows if r["method"] == method]
        seed_summary.append(dict(method=method, paired=len(subset), changed=sum(r["seed_family_changed"] for r in subset),
            L1_incomplete_rankings=sum(not r["L1_ranking_complete"] for r in subset),
            MSE_incomplete_rankings=sum(not r["MSE_ranking_complete"] for r in subset)))

    destination = output/"analysis/diagnostics"; destination.mkdir(parents=True, exist_ok=True)
    tables = dict(solver_calls=solver_rows, selector_changes=selectors, t_effects=t_rows,
                  interactions=interaction_rows, seed_changes=seed_rows, coverage=coverage_rows,
                  loss_sensitivity=sensitivity_rows)
    if recovery_rows:
        tables["recovery_charges"] = recovery_rows
    for name, rows in tables.items():
        write_csv(destination/(name+".csv"), rows)
    write(destination/"recovery_audit.json", recovery)
    summary = dict(passed=True, complete=True, new_native_verified=96, reused_parent_verified=96,
        logical_models=192, selector_views=384, new_fit_calls=0, new_native_optimizer_calls=0,
        observations_opened=False, truth_files_opened=False, script_sha256=sha(__file__),
        protocol_sha256=sha(output/"protocol.json"), catalog_sha256=sha(output/"catalog.json"),
        native_analysis_verification_sha256=sha(output/"analysis/verification.json"),
        native_analyzer_sha256=verification["analyzer_sha256"],
        parent_analysis_verification_sha256=sha(parent/"analysis/verification.json"),
        per_input_verification_sha256=verification_hashes, native_result_sha256=source_hashes,
        output_sha256={name+".csv": sha(destination/(name+".csv")) for name in tables},
        recovery_audit_sha256=sha(destination/"recovery_audit.json"),
        recovery=recovery, loss_sensitivity_cohorts=sensitivity_cohorts,
        loss_sensitivity=sensitivity_rows,
        solver=solver_summary, selectors=selector_summary, across_loss_seeds=seed_summary,
        within_loss_T_seed_changes=within_loss_seed_changes, coverage=coverage_summary,
        T_F1=contrast_summary(t_rows, ("fit_loss", "view"), "delta_F1"),
        T_exact=contrast_summary(t_rows, ("fit_loss", "view"), "delta_exact"),
        T_seconds=contrast_summary(t_rows, ("fit_loss", "view"), "delta_seconds"),
        F1_interaction=contrast_summary(interaction_rows, ("view",), "F1_interaction"),
        exact_interaction=contrast_summary(interaction_rows, ("view",), "exact_interaction"),
        field_definitions=dict(
            passed="all descriptive inputs remain SHA-anchored to complete native verification; structural status9 is reported rather than hidden",
            solver_gap="max(0,incumbent-bound)/abs(incumbent); native objective units; E included; zero-incumbent nonzero gap flagged separately",
            saved_point_changed="the selected output-library index differs; topology may remain identical",
            parameter_fit_changed="exact saved support/r/x tuple differs; no refit",
            topology_changed="positive (>1e-9) proper internal support family differs, matching audited F1",
            T_delta="T003 minus noT within the same training loss and validation view",
            interaction="(MSE:T003-noT) minus (L1:T003-noT), computed per fixed input",
            seed_change="RNJ selected small-seed family differs across training losses; rankings are data-only but loss-dependent",
            raw_pool="saved own raw trajectory models only, not the full feasible search domain",
            output_pool="same own path after .24 contraction and training refit",
            union_coverage="all true connections occur somewhere across saved raw/output models; this does not assert a single true full model",
            output_mapping="both contraction and subsequent refitting; not attributed to contraction alone",
            misses_available_truth="an exact model is in the fitted output pool but that validation view selects a different topology",
            timing="successful algorithm wall plus per-method charged common costs; parent infrastructure charges remain in native analysis; solver_seconds is a subset"),
        interpretation="descriptive paired diagnostics on four fixed networks and four globally shared seed groups; sensitivity cohorts exclude infrastructure debit only, never scores; all48 remains primary; no tuning or independent causal identification",
        elapsed_seconds=perf_counter()-started)
    write(destination/"summary.json", summary)
    print(json.dumps(dict(passed=True, complete=True, destination=str(destination),
                         solver_calls=len(solver_rows), logical_models=192, new_native_optimizer_calls=0)))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    diagnostics(parser.parse_args().output)
