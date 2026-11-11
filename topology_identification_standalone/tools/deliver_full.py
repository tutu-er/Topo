"""Archive a finished, independently analyzed 48 x 10 run, with local data.

This postprocessor never starts an optimizer or changes the original output.
It permits terminal failures and unresolved solver gaps, retains every planned
denominator, and separates archive integrity from numerical-analysis success.
Run only after the actual plots and the final human report are ready.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
METHODS = {"rnj", "nj", "000", "100", "010", "001", "110", "101", "011", "111"}
TERMINAL = {"fitted", "complete", "failed", "common_failed", "interrupted", "budget_exceeded"}
ANALYZED = {"verified", "failed", "common_failed", "interrupted", "budget_exceeded", "invalid"}
FORBIDDEN_SUFFIXES = {".zip", ".pyc", ".pyo", ".lic", ".key", ".pem", ".pfx", ".p12"}
PRIVATE_NAMES = {"credentials.json", "secrets.json", "tokens.json", "gurobi.lic"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def local(base, relative):
    """Portable relative-path handling, including Windows JSON separators."""
    relative = Path(str(relative).replace("\\", "/"))
    if relative.is_absolute() or any(":" in part for part in relative.parts):
        raise ValueError("Expected a relative archived path: "+str(relative))
    path = (Path(base)/relative).resolve()
    if not path.is_relative_to(Path(base).resolve()):
        raise ValueError("An archive path escapes its source folder")
    return path


def allowed(path):
    path = Path(path)
    return (path.is_file() and not path.is_symlink()
            and path.suffix.lower() not in FORBIDDEN_SUFFIXES
            and path.name.lower() not in PRIVATE_NAMES
            and not any(part in {".git", "__pycache__", ".venv", "venv"} or part.startswith(".env")
                        for part in path.parts))


def verified_file(path, expected, label):
    if not Path(path).is_file() or sha(path) != expected:
        raise ValueError(label+" does not match its recorded SHA-256: "+str(path))


def source_with_hash(expected, candidates, label):
    for path in candidates:
        if Path(path).is_file() and sha(path) == expected:
            return Path(path)
    raise ValueError("The exact "+label+" source is unavailable; archive it before editing the live tool")


def validate(output, report):
    protocol = read(output/"protocol.json")
    if (len(protocol["jobs"]) != 48 or len(set(protocol["jobs"])) != 48
            or set(protocol["methods"]) != METHODS or len(protocol["methods"]) != 10
            or protocol.get("planned_methods") != 480 or protocol.get("planned_common") != 48):
        raise ValueError("Expected the full 48-input, ten-method frozen protocol")
    if (protocol.get("primary_selection") != "validation_mse"
            or protocol.get("methods_fresh") is not True or protocol.get("common_fresh") is not True
            or protocol.get("reused_historical_fits") is not False):
        raise ValueError("This delivery is restricted to the fresh MSE-primary experiment")
    if (output/"run.lock").exists():
        raise ValueError("The native manager lock still exists; the run is not closed")
    completion = read(output/"completion.json")
    if completion.get("completion") is not True:
        raise ValueError("The original manager has not declared completion")
    for kind, planned in (("common", 48), ("methods", 480)):
        counts = completion[kind]
        if counts.get("planned") != planned or counts.get("pending") or counts.get("attempts_without_result"):
            raise ValueError("Pending native tasks cannot be archived as a complete run")
        if sum(int(counts.get(state, 0)) for state in TERMINAL) != planned:
            raise ValueError("Native terminal counts do not retain their planned denominator")
    verified_file(output/"catalog.json", protocol["catalog_hash"], "Frozen catalog")
    runtime = output/"runtime"
    for name, expected in protocol["runtime_file_hashes"].items():
        file = local(runtime, name)
        if not allowed(file):
            raise ValueError("The declared runtime contains a prohibited file: "+name)
        verified_file(file, expected, "Frozen runtime/data")
    actual_runtime = {file.relative_to(runtime).as_posix() for file in runtime.rglob("*") if allowed(file)}
    if actual_runtime != set(protocol["runtime_file_hashes"]):
        raise ValueError("The frozen runtime contains undeclared or missing files")
    inventory = read(runtime/"data/manifest.json")["files"]
    if len(inventory) != 208:
        raise ValueError("The declared self-contained benchmark data inventory changed")
    for name, expected in inventory.items():
        verified_file(local(runtime, name), expected, "Bundled cases/data")

    analysis = output/"analysis"
    verification = read(analysis/"verification.json")
    if (verification.get("complete") is not True or verification.get("native_queue_finished") is not True
            or verification.get("planned") != 480 or verification.get("pending") != 0
            or verification.get("all_planned_denominators_preserved") is not True):
        raise ValueError("A complete independent analysis with all 480 denominators is required")
    if ("mse" not in str(verification.get("selection_metric", "")).lower()
            or verification.get("MAE_used_for_final_selection") is not False
            or verification.get("native_optimizer_imported") is not False):
        raise ValueError("The analysis has an incompatible selection/information policy")
    if verification["protocol_sha256"] != sha(output/"protocol.json"):
        raise ValueError("Independent analysis belongs to another protocol")
    if verification["runtime_file_hashes"] != protocol["runtime_file_hashes"]:
        raise ValueError("Independent analysis used a different runtime inventory")
    for name, expected in verification["artifacts"].items():
        verified_file(local(analysis, name), expected, "Independent analysis artifact")
    with (analysis/"per_model.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_keys = {(job, method) for job in protocol["jobs"] for method in METHODS}
    if len(rows) != 480 or {(row["job"], row["method"]) for row in rows} != expected_keys:
        raise ValueError("The independent model table omits or repeats planned cells")
    if any(row["status"] not in ANALYZED for row in rows):
        raise ValueError("The independent model table still contains pending cells")
    counts = {status: sum(row["status"] == status for row in rows) for status in ANALYZED}
    checked_counts = dict(verified=counts["verified"], failed=counts["failed"]+counts["common_failed"],
                          interrupted=counts["interrupted"], budget_exceeded=counts["budget_exceeded"], invalid=counts["invalid"])
    if any(verification[key] != value for key, value in checked_counts.items()):
        raise ValueError("Numerical status counts disagree with the model table")
    by_key = {(row["job"], row["method"]): row for row in rows}
    policy_violations = []
    for job in protocol["jobs"]:
        common_file = output/"common"/job/"common.json"
        verified_file(common_file, verification["common_sha256"][job], "Fresh common record")
        if read(common_file).get("status") not in TERMINAL:
            raise ValueError("A common record is not terminal")
        for method in METHODS:
            file = output/"runs"/job/method/"result.json"
            verified_file(file, verification["native_result_sha256"][job+"/"+method], "Fresh method record")
            wrapper = read(file)
            if wrapper.get("status") not in TERMINAL or wrapper.get("protocol_sha256") != verification["protocol_sha256"]:
                raise ValueError("A raw method record is not a terminal result for this protocol")
            index = file.with_name("terminal_status.json")
            if index.exists():
                terminal = read(index)
                if terminal.get("fullresult_sha256") != sha(file):
                    raise ValueError("A terminal-status index differs from its complete result")
            native = wrapper.get("result")
            if isinstance(native, dict) and (native.get("selection_metric") != "validation_mse"
                    or native.get("selected") != native.get("selected_mse_view")):
                policy_violations.append(dict(job=job, method=method, status=by_key[job, method]["status"]))
                if by_key[job, method]["status"] == "verified":
                    raise ValueError("A verified method did not actually return primary MSE selection")

    figures = output/"figures"
    plot = read(figures/"plot_manifest.json")
    if plot.get("protocol_sha256") != verification["protocol_sha256"]:
        raise ValueError("The figures belong to another protocol or a synthetic fixture")
    if len(plot["outputs"]) != 24:
        raise ValueError("Expected eight actual figures in PNG/SVG/PDF, not mock/test plots")
    stems = {}
    for name, expected in plot["outputs"].items():
        verified_file(local(figures, name), expected, "Actual figure")
        stems.setdefault(Path(name).stem, set()).add(Path(name).suffix)
    if len(stems) != 8 or any(formats != {".png", ".svg", ".pdf"} for formats in stems.values()):
        raise ValueError("The actual figure set has missing or mixed formats")
    for name, expected in plot["analysis_input_sha256"].items():
        verified_file(local(analysis, name), expected, "Figure analysis input")
    analyzer = local(output, verification["analyzer_source_archive"])
    verified_file(analyzer, verification["analyzer_sha256"], "Actual independent analyzer")
    plotter = source_with_hash(plot["plot_source_sha256"],
        [ROOT/"tools/plot_full.py", output/"post_source/plot_full.py", report/"post_source/plot_full.py"], "plotter")
    if not (report/"CONCLUSIONS.md").is_file() or not (report/"CONCLUSIONS.md").stat().st_size:
        raise ValueError("Write the final root-reviewed CONCLUSIONS.md before archiving")
    return dict(protocol=protocol, completion=completion, verification=verification, counts=checked_counts,
                timing=read(analysis/"timing.json"), plot=plot, analyzer=analyzer, plotter=plotter,
                policy_violations=policy_violations)


def collect_run(output):
    return {file.relative_to(output).as_posix(): file for file in sorted(output.rglob("*")) if allowed(file)}


def reproduce_text(name):
    return f"""# Recompute this self-contained archive

Extract the entire `{name}/` folder. All observed arrays and physical cases are
inside `runtime/data/` and `runtime/cases/`; no original project is required.
Install Python >=3.10 and the dependencies in `runtime/requirements.txt`.
MIQP needs an independently available Gurobi license, which is not bundled.

From inside this extracted folder, replay the numerical audit and plots:

```text
python -I post_source/analyze_full.py . --bootstrap 2000
python -I post_source/plot_full.py .
```

The audit uses NumPy and the standard library, without a native optimizer.
Plotting additionally needs Matplotlib. Existing archived report and figures
remain the original evidence; use a copy if you want to preserve them unchanged.

For a small algorithm demonstration:

```text
python -I runtime/run.py demo --methods rnj nj pure_wzzt rnj_wzzt rte
```

To rerun all 480 methods freshly, choose a NEW output folder:

```text
python -I runtime/tools/full_benchmark.py freeze ../recomputed_full48 --workers 8
python -I runtime/tools/full_benchmark.py run ../recomputed_full48 --workers 8
python -I post_source/analyze_full.py ../recomputed_full48 --bootstrap 2000
python -I post_source/plot_full.py ../recomputed_full48
```

This full recomputation has a nominal 1920 method-hour allowance, not a promised
elapsed duration. Failed, interrupted and budget-exceeded archived results are
retained; archive integrity does not mean all numerical models succeeded.
Historical absolute paths in payloads describe provenance only. Do not invoke
an archived worker payload directly after relocation; create a fresh run using
the frozen local runner as shown above.
"""


def deliver(output, *, report=None, archive=None, estimate=False):
    output = Path(output).resolve()
    report = Path(report).resolve() if report else ROOT/"experiments/full48_mse"
    archive = Path(archive).resolve() if archive else report/"records.zip"
    if report.is_relative_to(output) or archive.is_relative_to(output):
        raise ValueError("Publication files must stay outside the immutable original run")
    evidence = validate(output, report)
    run_files = collect_run(output)
    copy_sources = [p for p in (output/"analysis").iterdir() if p.is_file() and p.suffix in {".csv", ".json"}]
    copy_sources += [output/"figures"/name for name in [*evidence["plot"]["outputs"], "plot_manifest.json"]]
    copy_sources += [evidence["analyzer"], evidence["plotter"], Path(__file__).resolve()]
    preflight = dict(raw_run_files=len(run_files), raw_run_bytes=sum(p.stat().st_size for p in run_files.values()),
                     publication_copy_bytes=sum(p.stat().st_size for p in copy_sources),
                     runtime_npz_files=sum(name.startswith("runtime/") and name.endswith(".npz") for name in run_files),
                     runtime_npz_bytes=sum(p.stat().st_size for name, p in run_files.items() if name.startswith("runtime/") and name.endswith(".npz")),
                     archive_destination=str(archive), counts=evidence["counts"],
                     note="Uncompressed preflight; generated metadata/report bytes are additional. Git inclusion is a separate root decision.")
    if estimate:
        print(json.dumps(preflight, indent=2)); return preflight
    if archive.exists():
        raise FileExistsError("The archive already exists; keep it and choose another --archive path instead of overwriting evidence")
    report.mkdir(parents=True, exist_ok=True); archive.parent.mkdir(parents=True, exist_ok=True)
    if (report/"protocol.json").exists() and sha(report/"protocol.json") != sha(output/"protocol.json"):
        raise ValueError("The report folder already belongs to a different protocol")
    published = {}

    def copy(source, relative):
        target = local(report, relative); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(source) != sha(target):
            raise RuntimeError("A publication copy differs from its raw source")
        published[str(relative).replace("\\", "/")] = target

    for source in (output/"analysis").iterdir():
        if source.is_file() and source.suffix in {".csv", ".json"}:
            copy(source, source.name)
    for name in ("protocol.json", "catalog.json", "state.json", "completion.json"):
        copy(output/name, name)
    for name in [*evidence["plot"]["outputs"], "plot_manifest.json"]:
        copy(output/"figures"/name, "figures/"+name)
    for name, source in (("analyze_full.py", evidence["analyzer"]), ("plot_full.py", evidence["plotter"]),
                         ("deliver_full.py", Path(__file__).resolve())):
        copy(source, "post_source/"+name)
    for name in ("PLAN.md", "CONCLUSIONS.md", "RUN_STATUS.json", "README.md", "MODEL.md", "PARAMETERS.json"):
        if (report/name).is_file():
            published[name] = report/name
    (report/"REPRODUCE.md").write_text(reproduce_text(output.name), encoding="utf-8", newline="\n")
    published["REPRODUCE.md"] = report/"REPRODUCE.md"
    metadata = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(output/"protocol.json"), verification_sha256=sha(output/"analysis/verification.json"),
        planned_methods=480, planned_inputs=48, planned_common=48, counts=evidence["counts"], pending=0,
        numerical_analysis_passed=evidence["verification"]["passed"],
        all_models_verified=evidence["verification"]["all_models_verified"],
        primary_selection="validation_mse", MAE_used_for_final_selection=False,
        native_timeouts=evidence["verification"]["native_timeouts"],
        all_native_gap_met=evidence["verification"]["all_native_gap_met"],
        timing=evidence["timing"], nominal_method_hours=evidence["protocol"]["nominal_method_hours"],
        manager_log_available=(output/"manager.log").is_file(), manager_stdout_complete_capture=False,
        manager_log_note="The original manager had no complete stdout redirection; no complete manager log is claimed. All available logs, manager attempts and terminal records are preserved.",
        actual_post_sources={name: sha(published["post_source/"+name]) for name in ("analyze_full.py", "plot_full.py", "deliver_full.py")},
        policy_violations_in_nonverified_records=evidence["policy_violations"], preflight=preflight,
        source_data_included=True, original_run_modified=False, mock_scores_included=False)
    write(report/"delivery_metadata.json", metadata); published["delivery_metadata.json"] = report/"delivery_metadata.json"
    files = {output.name+"/"+name: source for name, source in run_files.items()}
    files.update({output.name+"/report/"+name: source for name, source in published.items()})
    for name in ("analyze_full.py", "plot_full.py", "deliver_full.py"):
        files[output.name+"/post_source/"+name] = published["post_source/"+name]
    files[output.name+"/REPRODUCE.md"] = published["REPRODUCE.md"]
    hashes = {name: sha(source) for name, source in sorted(files.items())}
    print(json.dumps(dict(preflight=preflight, payload_files=len(files), payload_bytes=sum(p.stat().st_size for p in files.values())), indent=2), flush=True)
    temporary = archive.with_name(archive.name+".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as bundle:
        for name, source in sorted(files.items()):
            bundle.write(source, name)
        bundle.writestr("manifest.json", json.dumps(hashes, indent=2, sort_keys=True)+"\n")
    with zipfile.ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed")
        if set(bundle.namelist()) != set(hashes) | {"manifest.json"} or read_manifest(bundle) != hashes:
            raise RuntimeError("Archive membership differs from its manifest")
        if any(hashlib.sha256(bundle.read(name)).hexdigest() != expected for name, expected in hashes.items()):
            raise RuntimeError("An archive member failed SHA-256 verification")
        if any(Path(name).suffix.lower() in FORBIDDEN_SUFFIXES for name in bundle.namelist()):
            raise RuntimeError("The archive includes a nested zip, cache or credential")
    if any(sha(source) != hashes[name] for name, source in files.items()):
        raise RuntimeError("A source record/report changed during archiving; temporary evidence is retained")
    temporary.replace(archive)
    receipt = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), archive=str(archive),
        bytes=archive.stat().st_size, sha256=sha(archive), payload_files=len(files),
        zip_crc_passed=True, all_member_hashes_passed=True, archive_integrity_passed=True,
        numerical_analysis_passed=evidence["verification"]["passed"],
        all_models_verified=evidence["verification"]["all_models_verified"],
        native_queue_complete=True, planned_methods=480, planned_inputs=48, planned_common=48,
        counts=evidence["counts"], pending=0, native_timeouts=metadata["native_timeouts"],
        timing=evidence["timing"], manager_log_available=metadata["manager_log_available"],
        manager_stdout_complete_capture=False, runtime_npz_files=preflight["runtime_npz_files"],
        external_project_or_data_required=False, Gurobi_license_bundled=False,
        original_run_modified=False, delivery_source_sha256=sha(__file__),
        protocol_sha256=metadata["protocol_sha256"], verification_sha256=metadata["verification_sha256"])
    write(report/"archive_receipt.json", receipt)
    outer = {name: sha(path) for name, path in published.items()}
    outer["archive_receipt.json"] = sha(report/"archive_receipt.json")
    if archive.is_relative_to(report):
        outer[archive.relative_to(report).as_posix()] = receipt["sha256"]
    write(report/"manifest.json", outer)
    print(json.dumps(receipt, indent=2)); return receipt


def read_manifest(bundle):
    return json.loads(bundle.read("manifest.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--estimate", action="store_true", help="read-only size preflight after all evidence and final report are ready")
    args = parser.parse_args()
    deliver(args.output, report=args.report, archive=args.archive, estimate=args.estimate)
