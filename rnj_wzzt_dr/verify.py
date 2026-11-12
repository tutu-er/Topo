"""Check the packaged sources; original locations are optional read-only checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify(*, originals=False):
    root = Path(__file__).resolve().parent
    doc = json.loads((root/"PROVENANCE.json").read_text(encoding="utf-8"))
    issues, missing_originals = [], []
    for name, expected in doc.get("package_files", {}).items():
        p = root/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            issues.append(dict(path=name, reason="package_file_changed"))
    for row in doc["sources"]:
        p = root/row["destination"]
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row["packaged_sha256"]:
            issues.append(dict(path=row["destination"], reason="packaged_source_changed"))
        if originals:
            p = Path(doc["original_roots"][row["source_root"]])/row["source"]
            if not p.exists():
                missing_originals.append(str(p))
            elif hashlib.sha256(p.read_bytes()).hexdigest() != row["source_sha256"]:
                issues.append(dict(path=str(p), reason="original_source_changed"))
    return dict(passed=not issues, checked=len(doc["sources"]),
                package_files_checked=len(doc.get("package_files", {})), issues=issues,
                missing_originals=missing_originals, active_core_requires_original_sources=False,
                compatibility_AC_requires_frozen_assets=True)
