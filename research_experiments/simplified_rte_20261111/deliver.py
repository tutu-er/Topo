"""Deliver only a fully verified four-input simplified-RTE pilot.

Postprocessing stays outside the immutable native source. The archive contains
this pilot's package and records; its frozen parent archive is referenced by
SHA-256, without repeating parent source, observations or license files.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from time import sleep
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research_experiments/simplified_rte_20261111/"
OWN_SOURCE = ("__init__.py", "engine.py", "run.py", "analyze.py", "verify.py", "README.md")
EXCLUDED_SUFFIXES = {".npz", ".npy", ".pyc", ".pyo", ".pem", ".key", ".lic", ".pfx", ".p12"}


def read(path):
    for attempt in range(100):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 99:
                raise
            sleep(.05)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    for attempt in range(100):
        try:
            temporary.write_text(serialized, encoding="utf-8", newline="\n")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            sleep(.05)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def family(point):
    return {tuple(s) for s in point["supports"]}


def fingerprint(point):
    rows = sorted(zip(point["supports"], point["r"], point["x"], strict=True),
                  key=lambda row: (len(row[0]), tuple(row[0])))
    value = dict(supports=[list(s) for s, _, _ in rows],
                 r=[float(r) for _, r, _ in rows], x=[float(x) for _, _, x in rows])
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def finite_nonnegative(value, label):
    value = float(value)
    if not math.isfinite(value) or value < 0.:
        raise ValueError(f"Invalid {label}")
    return value


def validate(output):
    output = Path(output).resolve()
    protocol, catalog = read(output/"protocol.json"), read(output/"catalog.json")
    verification = read(output/"analysis/verification.json")
    completion = read(output/"completion.json")
    if not (verification["complete"] and verification["passed"]):
        raise ValueError("Independent verification is incomplete or failed; delivery is not permitted")
    if not (protocol["planned"] == verification["planned_new"] == verification["verified_new"] == 4
            and verification["baseline_planned"] == verification["baseline_verified"] == 4):
        raise ValueError("Delivery requires all four new and four historical baseline inputs")
    if verification["errors"] or not verification["all_denominators_preserved"]:
        raise ValueError("Verification reports an error or a lost denominator")
    if not (completion["planned"] == completion["returned"] == 4 and completion["failed"] == completion["pending"] == 0):
        raise ValueError("The finite native manager has not completed all four inputs")
    if sha(output/"protocol.json") != verification["protocol_sha256"]:
        raise ValueError("The verified protocol changed")
    if sha(output/"catalog.json") != protocol["catalog_sha256"] or set(catalog) != set(protocol["jobs"]):
        raise ValueError("The fixed pilot catalog changed")
    if float(protocol["nominal_method_hours"]) != 16. or float(protocol["limits"]["total"]) != 14400.:
        raise ValueError("The declared four-input budget changed")
    for name, expected in {**protocol["source_hashes"], **protocol["source_artifact_hashes"]}.items():
        if sha(output/"source"/name) != expected:
            raise ValueError(f"A frozen native source changed: {name}")
    for name, expected in verification["result_hashes"].items():
        if sha(output/"analysis"/name) != expected:
            raise ValueError(f"An independently verified analysis table changed: {name}")
    parent = Path(protocol["parent_output"]).resolve()
    if sha(parent/"protocol.json") != protocol["parent_protocol_sha256"] or sha(parent/"analysis/verification.json") != protocol["parent_verification_sha256"]:
        raise ValueError("The verified parent provenance changed")
    results = {}
    for job in protocol["jobs"]:
        file = output/"jobs"/job/"result.json"
        if sha(file) != verification["native_result_sha256"][job]:
            raise ValueError(f"A verified native result changed: {job}")
        result = read(file)
        if result["protocol_sha256"] != verification["protocol_sha256"]:
            raise ValueError(f"A native result references another protocol: {job}")
        if sha(file.with_name("candidates.json")) != result["candidate_export_sha256"]:
            raise ValueError(f"A native model export changed: {job}")
        attempt = read(file.with_name("attempt.json"))
        if attempt["state"] != "finished" or attempt["exit_code"] != 0:
            raise ValueError(f"A native attempt has not finished successfully: {job}")
        if result["total_cap_exceeded"]:
            raise ValueError(f"A native method exceeded its declared budget: {job}")
        if sha(parent/"common"/job/"common.json") != protocol["parent_common_sha256"][job]:
            raise ValueError(f"Shared preprocessing changed: {job}")
        if sha(parent/"jobs"/job/"T003/result.json") != protocol["parent_T003_result_sha256"][job]:
            raise ValueError(f"A paired historical baseline changed: {job}")
        results[job] = result
    return output, protocol, catalog, verification, completion, parent, results


def stage_rows(catalog, results):
    """Record proposals separately from actual accepted path updates."""
    rows = []
    for job, result in results.items():
        trajectory = result["trajectory"]
        path = trajectory["path"]
        current, next_index = path[0], 1
        n = max(i for support in current["supports"] for i in support)+1
        bins = {stage: dict(job=job, case=catalog[job]["case"], stage=stage, calls=0,
            skipped_calls=0,
            accepted_proposals=0, accepted_path_updates=0, repeated_W_not_appended=0,
            native_solve_calls=0, MIP_timeouts=0, pool_count_max=None if stage == "W" else 0,
            optional_candidates_max=None if stage == "W" else 0,
            RNJ_guidance_proper_count=sum(1 < len(s) < n for s in trajectory["rnj_pool"]),
            temporary_old_blocks_unlocked=0, accepted_old_blocks_removed=0,
            proposed_zero_release_events=0, proposed_zero_blocks_removed=0,
            applied_zero_release_events=0, applied_zero_blocks_removed=0,
            seconds=0., pool_scope="implicit unrestricted W domain; RNJ count is guidance only" if stage == "W" else "explicit finite T/E candidate pool")
                for stage in ("W", "T", "E")}
        for event in trajectory["events"]:
            stage = event["stage"]
            row = bins[stage]; row["calls"] += 1
            row["skipped_calls"] += bool(event.get("skipped"))
            proposed_accepted = bool(event.get("accepted"))
            proposed = event.get("point")
            applied = bool(proposed_accepted and proposed is not None and next_index < len(path)
                           and fingerprint(path[next_index]) == fingerprint(proposed))
            if proposed_accepted and not applied and stage != "W":
                raise ValueError(f"An accepted {stage} update is absent from the native path: {job}")
            row["accepted_proposals"] += proposed_accepted
            row["accepted_path_updates"] += applied
            row["repeated_W_not_appended"] += proposed_accepted and not applied
            solver = event.get("solver", {})
            row["native_solve_calls"] += bool(solver.get("solve_started"))
            row["MIP_timeouts"] += solver.get("status") == 9
            if "pool" in event:
                row["pool_count_max"] = max(row["pool_count_max"] or 0, len(event["pool"]))
            if "candidates" in event:
                row["optional_candidates_max"] = max(row["optional_candidates_max"] or 0, len(event["candidates"]))
            row["seconds"] += finite_nonnegative(event.get("seconds", 0.), "stage seconds")
            before_family = family(current)
            if stage in {"T", "E"}:
                keep = {tuple(s) for s in event.get("keep", current["supports"])}
                row["temporary_old_blocks_unlocked"] += sum(1 < len(s) < n and s not in keep for s in before_family)
            zeros = [z for z in event.get("zero_events", []) if z.get("accepted")]
            zero_count = sum(len(z["removed"]) for z in zeros)
            row["proposed_zero_release_events"] += len(zeros)
            row["proposed_zero_blocks_removed"] += zero_count
            if applied:
                row["accepted_old_blocks_removed"] += sum(1 < len(s) < n for s in before_family-family(proposed))
                row["applied_zero_release_events"] += len(zeros)
                row["applied_zero_blocks_removed"] += zero_count
                current = path[next_index]; next_index += 1
        if next_index != len(path):
            raise ValueError(f"Native event/path accounting mismatch: {job}")
        rows.extend(bins.values())
    for stage in ("W", "T", "E"):
        values = [row for row in rows if row["stage"] == stage]
        total = dict(values[0]); total.update(job="ALL", case="ALL")
        for name in total:
            if name in {"job", "case", "stage", "pool_scope"}:
                continue
            if name.endswith("_max"):
                available = [row[name] for row in values if row[name] is not None]
                total[name] = max(available) if available else None
            else:
                total[name] = sum(row[name] for row in values)
        rows.append(total)
    return rows


def allowed(file):
    return (file.is_file() and file.suffix.lower() not in EXCLUDED_SUFFIXES
            and not file.name.endswith((".tmp", ".lock")) and file.name.lower() not in {".env", "credentials", "secrets.json"})


def deliver(output):
    output, protocol, catalog, verification, completion, parent, results = validate(output)
    docs = ROOT/"docs"/output.name
    docs.mkdir(parents=True, exist_ok=True)
    for name in ("MODEL.md", "PLAN.md", "PARAMETERS.json", "toy_verification.json"):
        if not (docs/name).is_file():
            raise FileNotFoundError(f"Required reviewed research artifact is missing: {name}")
    toy = read(docs/"toy_verification.json")
    if not toy["passed"] or toy["case_count"] != 8:
        raise ValueError("The complete eight-toy verification is required")
    if toy["engine_sha256"] != protocol["source_hashes"][PACKAGE+"engine.py"] or toy["script_sha256"] != protocol["source_hashes"][PACKAGE+"verify.py"]:
        raise ValueError("The toy receipt does not cover the frozen native implementation")
    source = output/"source"
    actual_analyzer = next((file for file in (source/PACKAGE/"analyze.py", Path(__file__).with_name("analyze.py"))
                            if file.is_file() and sha(file) == verification["analyzer_sha256"]), None)
    if actual_analyzer is None:
        raise ValueError("The actual analysis source cannot be recovered by its recorded SHA")
    post_source = output/"analysis_source"
    post_source.mkdir(exist_ok=True)
    shutil.copy2(actual_analyzer, post_source/"analyze.py")
    shutil.copy2(Path(__file__).resolve(), post_source/"deliver.py")
    stages = stage_rows(catalog, results)
    write_csv(output/"analysis/stages.csv", stages)
    for file in (output/"analysis").rglob("*"):
        if file.is_file() and file.suffix.lower() in {".csv", ".json"}:
            destination = docs/file.relative_to(output/"analysis")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
    for name in ("protocol.json", "catalog.json", "completion.json"):
        shutil.copy2(output/name, docs/name)
    timing_rows = []
    for job, result in results.items():
        fresh = finite_nonnegative(result["fresh_seconds"], "fresh method seconds")
        common = finite_nonnegative(result["common_seconds"], "charged shared common seconds")
        total = finite_nonnegative(result["total_seconds"], "total method seconds")
        if abs(total-fresh-common) > 1e-6:
            raise ValueError(f"Fresh/common timing does not reconcile: {job}")
        timing_rows.append(dict(job=job, case=catalog[job]["case"], fresh_seconds=fresh,
            reused_common_charged_seconds=common, total_method_seconds=total,
            structural_seconds=result["structural_seconds"], output_seconds=result["output_seconds"],
            total_cap_seconds=protocol["limits"]["total"], total_cap_exceeded=result["total_cap_exceeded"]))
    write_csv(docs/"timing_components.csv", timing_rows)
    metadata = dict(
        native_inputs=4, nominal_method_hours=16., method_cap_seconds=14400.,
        workers=protocol["workers"], solver_threads=1, W_per_MIQP_cap_seconds=3600., fixed_QP_cap_seconds=300.,
        T_E_cap_scope="their allocated structural-stage remaining wall time, as recorded per native solve",
        total_method_seconds=sum(row["total_method_seconds"] for row in timing_rows),
        fresh_native_seconds=sum(row["fresh_seconds"] for row in timing_rows),
        reused_common_charged_seconds=sum(row["reused_common_charged_seconds"] for row in timing_rows),
        native_manager_calendar_seconds=finite_nonnegative(completion["wall_seconds"], "manager calendar seconds"),
        calendar_scope="elapsed native pilot manager clock; excludes this postprocessing and prior shared preprocessing",
        common_scope="historically computed RNJ/QP common cost charged once to each new input; not new work in this pilot",
        comparison_scope=protocol["comparison_scope"], independent_fair_runtime_comparison=False,
        MIP_timeouts=verification["MIP_timeouts"], failed=0, invalid=0, pending=0,
        protocol_sha256=sha(output/"protocol.json"), verification_sha256=sha(output/"analysis/verification.json"),
        stages_csv_sha256=sha(output/"analysis/stages.csv"),
        postprocessing_source_sha256={name: sha(post_source/name) for name in ("analyze.py", "deliver.py")},
        no_plots_or_PDF_generated=True, recorded_utc=datetime.now(timezone.utc).isoformat())
    write(docs/"delivery_metadata.json", metadata)
    parent_docs = ROOT/"docs"/parent.name
    parent_archive = parent_docs/"records.zip"
    old_receipt = read(parent_docs/"archive_receipt.json")
    if sha(parent_archive) != old_receipt["sha256"]:
        raise ValueError("The referenced complete parent archive changed")
    parent_reference = dict(
        parent_output=str(parent), parent_protocol_sha256=protocol["parent_protocol_sha256"],
        parent_verification_sha256=protocol["parent_verification_sha256"],
        records_zip=str(parent_archive), records_zip_sha256=sha(parent_archive),
        records_zip_bytes=parent_archive.stat().st_size, parent_zip_copied=False,
        parent_source_copied=False, raw_NPZ_copied=False,
        common_sha256=protocol["parent_common_sha256"], baseline_T003_result_sha256=protocol["parent_T003_result_sha256"])
    write(docs/"parent_reference.json", parent_reference)
    files = {}
    for name in OWN_SOURCE:
        file = source/PACKAGE/name
        files["native_source/"+PACKAGE+name] = file
    for top in ("jobs", "analysis", "analysis_source"):
        for file in (output/top).rglob("*"):
            if allowed(file):
                files["run/"+file.relative_to(output).as_posix()] = file
    for file in output.iterdir():
        if allowed(file) and file.suffix.lower() in {".json", ".log"}:
            files["run/"+file.name] = file
    for job in protocol["jobs"]:
        common_file = parent/"common"/job/"common.json"
        files[f"shared_parent/common/{job}/common.json"] = common_file
        baseline_file = parent/"jobs"/job/"T003/result.json"
        files[f"paired_parent/{job}/T003/result.json"] = baseline_file
    for file in docs.rglob("*"):
        if allowed(file) and file.name not in {"records.zip", "manifest.json", "archive_receipt.json"}:
            files["report/"+file.relative_to(docs).as_posix()] = file
    hashes = {name: sha(file) for name, file in sorted(files.items())}
    archive = docs/"records.zip"
    temporary = archive.with_name(archive.name+f".{os.getpid()}.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, file in sorted(files.items()):
            bundle.write(file, name)
        bundle.writestr("manifest.json", json.dumps(hashes, sort_keys=True, indent=2)+"\n")
    with zipfile.ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
        if set(bundle.namelist()) != set(hashes) | {"manifest.json"}:
            raise ValueError("Archive membership does not match its manifest")
        if json.loads(bundle.read("manifest.json")) != hashes:
            raise ValueError("The bundled manifest differs from the verified member inventory")
        if any(hashlib.sha256(bundle.read(name)).hexdigest() != expected for name, expected in hashes.items()):
            raise ValueError("An archive member failed SHA-256 verification")
        if any(Path(name).suffix.lower() in EXCLUDED_SUFFIXES or name.endswith("records.zip") for name in bundle.namelist()):
            raise ValueError("An excluded raw array, license/key or nested archive was bundled")
        copied_native = [name for name in bundle.namelist() if name.startswith("native_source/")]
        if len(copied_native) != len(OWN_SOURCE) or any(not name.startswith("native_source/"+PACKAGE) for name in copied_native):
            raise ValueError("The archive must contain only the new package's native source")
    for attempt in range(100):
        try:
            temporary.replace(archive)
            break
        except PermissionError:
            if attempt == 99:
                raise
            sleep(.05)
    receipt = dict(
        complete=True, passed=True, verified_new=4, baseline_verified=4, failed=0, invalid=0, pending=0,
        archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive), files=len(hashes),
        zip_crc_passed=True, all_manifest_hashes_passed=True, own_native_files=len(OWN_SOURCE),
        parent_zip_copied=False, parent_source_copied=False, raw_observation_arrays_bundled=False,
        native_timeouts_preserved=verification["MIP_timeouts"], budget_exceeded=0,
        total_method_seconds=metadata["total_method_seconds"], fresh_native_seconds=metadata["fresh_native_seconds"],
        reused_common_charged_seconds=metadata["reused_common_charged_seconds"],
        native_manager_calendar_seconds=metadata["native_manager_calendar_seconds"],
        protocol_sha256=metadata["protocol_sha256"], verification_sha256=metadata["verification_sha256"],
        delivery_source_sha256=sha(__file__), recorded_utc=datetime.now(timezone.utc).isoformat())
    write(docs/"archive_receipt.json", receipt)
    write(docs/"manifest.json", {file.relative_to(docs).as_posix(): sha(file) for file in docs.rglob("*")
                                if file.is_file() and file != docs/"manifest.json"})
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=ROOT/"outputs/simplified_rte_20261111")
    args = parser.parse_args()
    deliver(args.output)
