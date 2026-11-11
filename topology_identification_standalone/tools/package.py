"""Build a portable zip, excluding local outputs, caches and credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(destination):
    destination = Path(destination).resolve(); destination.parent.mkdir(parents=True, exist_ok=True)
    folders = {"cases", "data", "tests", "experiments", "reference_results", "validation", "tools"}
    forbidden = {".key", ".lic", ".pem", ".pfx", ".p12", ".pyc", ".pyo", ".zip"}
    files = []
    for file in ROOT.rglob("*"):
        if not file.is_file() or file == destination or file.name == "bundle_manifest.json":
            continue
        relative = file.relative_to(ROOT)
        if "__pycache__" in relative.parts or any(part.startswith(".env") for part in relative.parts):
            continue
        if len(relative.parts) > 1 and relative.parts[0] not in folders:
            continue
        if file.suffix.lower() in forbidden:
            continue
        if len(relative.parts) == 1 and file.suffix not in {".py", ".md", ".txt"} and file.name not in {".gitattributes", ".gitignore"}:
            continue
        files.append(file)
    hashes = {file.relative_to(ROOT).as_posix(): sha(file) for file in sorted(files)}
    manifest = dict(files=hashes, count=len(hashes), runtime_data_included=True,
                    development_outputs_included=False, parent_project_required=False)
    manifest_file = ROOT/"bundle_manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in files:
            bundle.write(file, "topology_identification_standalone/"+file.relative_to(ROOT).as_posix())
        bundle.write(manifest_file, "topology_identification_standalone/bundle_manifest.json")
    with zipfile.ZipFile(destination) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Zip CRC failed")
        for name, expected in hashes.items():
            if hashlib.sha256(bundle.read("topology_identification_standalone/"+name)).hexdigest() != expected:
                raise RuntimeError("Zip SHA failed: "+name)
    receipt = dict(archive=str(destination), bytes=destination.stat().st_size, sha256=sha(destination),
                   payload_files=len(files), crc_passed=True, all_member_hashes_passed=True,
                   source_manifest_sha256=sha(manifest_file), data_arrays_included=True,
                   external_project_or_dataset_required=False,
                   dependencies="Python, NumPy, SciPy; Gurobi and its license for MIQP")
    destination.with_suffix(".receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="path of the zip file to create")
    args = parser.parse_args()
    package(args.destination)
