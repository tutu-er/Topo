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
    verification = read(OUTPUT / "analysis/verification.json")
    assert verification["passed"] and verification["verified"] == 144
    assert read(OUTPUT / "bridge_candidates/summary.json")["exports_verified"] == 144
    assert read(OUTPUT / "compatibility_analysis/verification.json")["passed"]
    assert read(DOCS / "visual_review.json")["passed"]
    for file in (OUTPUT / "analysis").iterdir():
        if file.is_file():
            shutil.copy2(file, DOCS / file.name)
    for file in (OUTPUT / "compatibility_analysis").iterdir():
        if file.is_file():
            shutil.copy2(file, DOCS / ("compatibility_"+file.name))
    bridge = read(OUTPUT / "bridge_candidates/summary.json")
    (DOCS / "bridge_export_summary.json").write_text(
        json.dumps({k:v for k,v in bridge.items() if k != "receipts"}, indent=2)+"\n", encoding="utf-8")
    environment = dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                       scipy=scipy.__version__, gurobi=gp.gurobi.version(), platform=platform.platform(),
                       scope="recorded after the finite batch; numerical libraries not changed during it")
    (DOCS / "environment.json").write_text(json.dumps(environment, indent=2)+"\n", encoding="utf-8")
    files = {}
    for top in ("source", "analysis_source", "analysis", "bridge_candidates", "compatibility_analysis", "timing_uncertainty", "recovery", "common", "networks"):
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
                   raw_measurement_arrays_bundled=False, verified_method_runs=144)
    (DOCS / "archive_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    hashes = {file.name:sha(file) for file in DOCS.iterdir() if file.is_file() and file.name != "manifest.json"}
    (DOCS / "manifest.json").write_text(json.dumps(hashes, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
