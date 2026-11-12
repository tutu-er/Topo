"""Read-only SHA-256 verification of the archived originals; no project imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-local", action="store_true")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    repository = base.parent.parent
    doc = json.loads((base/"manifest.json").read_text(encoding="utf-8"))
    if args.include_local:
        rows = [json.loads(line) for line in (base/"LOCAL_MANIFEST.jsonl").read_text(encoding="utf-8").splitlines()]
    else:
        rows = doc["tracked_files"]
    issues = []
    for row in rows:
        path = (repository/row["archived"]).resolve()
        if not path.is_relative_to(base/"legacy"):
            raise ValueError("Archive reference escapes its directory")
        if not path.is_file():
            issues.append(dict(original=row["original"], reason="missing"))
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            issues.append(dict(original=row["original"], reason="changed"))
    print(json.dumps(dict(status="passed" if not issues else "failed", checked=len(rows),
                         include_local=args.include_local, issues=issues), ensure_ascii=False, indent=2))
    return bool(issues)


if __name__ == "__main__":
    raise SystemExit(main())
