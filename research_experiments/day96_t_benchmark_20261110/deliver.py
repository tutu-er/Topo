"""Archive verified day96 evidence and sources without copying measurement arrays."""
from pathlib import Path
import hashlib
import json
import platform
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/day96_t_benchmark_20261110"
DOCS = ROOT / "docs/day96_t_benchmark_20261110"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import scipy
    import gurobipy as gp
    native_receipt = OUTPUT / "analysis/verification.json"
    protocol_file = OUTPUT / "protocol.json"
    verification = read(native_receipt)
    assert verification["passed"] and verification["verified"] == 144
    assert verification["protocol_sha256"] == sha(protocol_file)
    bridge_file = OUTPUT / "bridge_candidates/summary.json"
    bridge = read(bridge_file)
    assert bridge["exports_verified"] == bridge["requested_exports"] == 144 and bridge["native_results_complete"]
    compatibility = read(OUTPUT / "compatibility_analysis/verification.json")
    assert compatibility["passed"] and compatibility["native_evaluations"] == 144
    assert compatibility["native_protocol_sha256"] == sha(protocol_file)
    assert compatibility["native_analysis_receipt_sha256"] == sha(native_receipt)
    assert compatibility["bridge_summary_sha256"] == sha(bridge_file)
    timing = read(OUTPUT / "timing_uncertainty/verification.json")
    assert timing["complete"] and timing["returned"] == timing["planned"] == 144
    assert timing["protocol_sha256"] == sha(protocol_file)
    assert timing["native_analysis_receipt_sha256"] == sha(native_receipt)
    stages = read(OUTPUT / "stage_effects/verification.json")
    assert stages["passed"] and stages["planned_native_results"] == 144
    assert stages["native_verification_sha256"] == sha(native_receipt)
    assert stages["native_protocol_sha256"] == sha(protocol_file)
    candidates = read(OUTPUT / "candidate_diagnostics/verification.json")
    assert candidates["passed"] and candidates["native_evaluations"] == 144
    assert candidates["native_analysis_receipt_sha256"] == sha(native_receipt)
    assert candidates["bridge_summary_sha256"] == sha(bridge_file)
    gates = read(OUTPUT / "tail_gate_diagnostics/verification.json")
    assert gates["passed"] and gates["native_verified"] == 144
    assert gates["native_verification_sha256"] == sha(native_receipt)
    assert gates["stage_verification_sha256"] == sha(OUTPUT / "stage_effects/verification.json")
    # Unresolved or exceeded conservative caps remain reportable evidence;
    # they are not silently converted to a successful budget certificate.
    figures = read(DOCS / "figure_receipt.json")
    assert figures["verification_sha256"] == sha(native_receipt)
    assert read(DOCS / "visual_review.json")["passed"]
    for file in (OUTPUT / "analysis").iterdir():
        if file.is_file():
            shutil.copy2(file, DOCS / file.name)
    for file in (OUTPUT / "compatibility_analysis").iterdir():
        if file.is_file():
            shutil.copy2(file, DOCS / ("compatibility_"+file.name))
    for file in (OUTPUT / "timing_uncertainty").iterdir():
        if file.is_file():
            shutil.copy2(file, DOCS / ("timing_"+file.name))
    for file in (OUTPUT / "stage_effects").iterdir():
        if file.is_file():
            shutil.copy2(file, DOCS / ("stages_"+file.name))
    for directory, prefix in (("candidate_diagnostics", "candidate_diagnostic_"),
                              ("tail_gate_diagnostics", "tail_gate_")):
        for file in (OUTPUT / directory).iterdir():
            if file.is_file():
                shutil.copy2(file, DOCS / (prefix+file.name))
    (DOCS / "bridge_export_summary.json").write_text(
        json.dumps({k:v for k,v in bridge.items() if k != "receipts"}, indent=2)+"\n", encoding="utf-8")
    environment = dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                       scipy=scipy.__version__, gurobi=gp.gurobi.version(), platform=platform.platform(),
                       scope="recorded after the finite batch; numerical libraries not changed during it")
    (DOCS / "environment.json").write_text(json.dumps(environment, indent=2)+"\n", encoding="utf-8")
    files = {}
    for top in ("source", "analysis_source", "analysis", "bridge_candidates", "compatibility_analysis", "timing_uncertainty", "stage_effects", "candidate_diagnostics", "tail_gate_diagnostics", "recovery", "common", "networks"):
        for file in (OUTPUT / top).rglob("*"):
            if file.is_file() and file.suffix not in {".pyc", ".npz"}:
                files["run/"+file.relative_to(OUTPUT).as_posix()] = file
    for file in (OUTPUT / "jobs").rglob("*.json"):
        if file.name in {"result.json", "result.pending.json", "native_recovery_payload.json", "recovery.json", "candidates.json", "error.json", "payload.json"}:
            files["run/"+file.relative_to(OUTPUT).as_posix()] = file
    for name in ("configuration.json", "protocol.json", "catalog.json", "physical_seeds.json", "data_errors.json",
                 "pilot_completion.json", "all_completion.json", "pilot_state.json", "all_state.json", "postprocess.json"):
        file = OUTPUT / name
        if file.exists():
            files["run/"+name] = file
    for file in (OUTPUT / "data").rglob("diagnostics.json"):
        files["run/"+file.relative_to(OUTPUT).as_posix()] = file
    for file in Path(__file__).parent.iterdir():
        if file.is_file():
            files["driver/"+file.name] = file
    for file in DOCS.iterdir():
        if file.is_file() and file.name not in {"records.zip", "manifest.json", "archive_receipt.json"}:
            files["report/"+file.name] = file
    hashes = {name:sha(file) for name, file in sorted(files.items())}
    archive = DOCS / "records.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, file in sorted(files.items()):
            bundle.write(file, name)
        bundle.writestr("manifest.json", json.dumps(hashes, indent=2)+"\n")
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert all(hashlib.sha256(bundle.read(name)).hexdigest() == digest for name, digest in hashes.items())
        assert not any(name.lower().endswith((".npz", ".pem", ".key")) for name in bundle.namelist())
    receipt = dict(bytes=archive.stat().st_size, sha256=sha(archive), files=len(hashes),
                   zip_crc_passed=True, all_manifest_hashes_passed=True,
                   raw_measurement_arrays_bundled=False, verified_method_runs=144,
                   timing_evaluations=timing["returned"],
                   all_original_budget_caps_conservatively_certified=timing["all_original_budget_caps_conservatively_certified"],
                   unresolved_budgets=timing["budget_unresolved"], proven_exceeded_budgets=timing["budget_violation_proven"])
    (DOCS / "archive_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    hashes = {file.name:sha(file) for file in DOCS.iterdir() if file.is_file() and file.name != "manifest.json"}
    (DOCS / "manifest.json").write_text(json.dumps(hashes, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
