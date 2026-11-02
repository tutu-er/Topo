"""Archive this finite diagnostic and frozen sources; no numerical execution."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/t_regularization_audit_20261102"
DOCS = ROOT / "docs/t_regularization_audit_20261102"
FROZEN = ROOT / "outputs/full_ablation_20261102/source"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for path in (OUTPUT / "analysis").iterdir():
        if path.is_file():
            shutil.copy2(path, DOCS / path.name)
    visual = dict(passed=True, reviewed_files=["lambda_intervals.png"],
                  method="rendered PNG inspected in view_image; legend clear of intervals",
                  numerical_fitting=False)
    (DOCS / "visual_review.json").write_text(json.dumps(visual, indent=2) + "\n", encoding="utf-8")
    archive = DOCS / "records.zip"
    files = {}
    files.update({"diagnostic/" + p.relative_to(OUTPUT).as_posix(): p
                  for p in OUTPUT.rglob("*") if p.is_file() and (p.name == "profile.json" or "analysis" in p.parts)})
    files.update({"driver/" + p.name: p for p in Path(__file__).parent.iterdir() if p.is_file()})
    files.update({"frozen_source/" + p.relative_to(FROZEN).as_posix(): p for p in FROZEN.rglob("*.py")})
    files.update({"report/" + p.name: p for p in DOCS.iterdir()
                  if p.is_file() and p.name not in {"records.zip", "manifest.json", "archive_receipt.json"}})
    original = ROOT / "outputs/full_ablation_20261102"
    for name in ("protocol.json", "catalog.json"):
        files["parent/" + name] = original / name
    checks = {name: sha(path) for name, path in files.items()}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, path in sorted(files.items()):
            bundle.write(path, name)
        bundle.writestr("manifest.json", json.dumps(checks, indent=2) + "\n")
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert all(hashlib.sha256(bundle.read(name)).hexdigest() == digest for name, digest in checks.items())
        assert not any(name.lower().endswith((".npz", ".pem", ".key")) for name in bundle.namelist())
    receipt = dict(archive="records.zip", bytes=archive.stat().st_size, sha256=sha(archive),
                   verified_files=len(checks), zip_crc_passed=True, all_manifest_hashes_passed=True,
                   raw_measurement_arrays_bundled=False)
    (DOCS / "archive_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    checks = {p.name: sha(p) for p in DOCS.iterdir() if p.is_file() and p.name != "manifest.json"}
    (DOCS / "manifest.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
