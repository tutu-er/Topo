"""Check handoff references without importing the project or running experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--include-private", action="store_true",
                        help="Only test existence of private pointers; do not read trajectories")
    parser.add_argument("--workspace", type=Path, help="Override workspace root after moving checkout")
    parser.add_argument("--main", type=Path, help="Override read-only algorithm/data-source root")
    parser.add_argument("--private", type=Path, help="Override local private recovery root")
    args = parser.parse_args()
    doc = json.loads(args.manifest.read_text(encoding="utf-8"))
    roots = dict(doc["roots"])
    for name in roots:
        if getattr(args, name) is not None:
            roots[name] = str(getattr(args, name))
    counts = dict(checked=0, matched=0, private_exists=0, skipped=0,
                  missing=0, changed=0, unreadable=0)
    issues = []
    for row in doc["entries"]:
        private = row["visibility"] == "private"
        if private and not args.include_private:
            counts["skipped"] += 1
            continue
        counts["checked"] += 1
        path = Path(roots[row["root"]])/row["path"]
        try:
            if not path.exists():
                counts["missing"] += 1
                issues.append(dict(id=row["id"], status="missing", path=str(path)))
            elif private:
                counts["private_exists"] += 1
            elif row["kind"] == "directory":
                counts["matched"] += 1
            elif hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]:
                counts["matched"] += 1
            else:
                counts["changed"] += 1
                issues.append(dict(id=row["id"], status="changed_since_snapshot", path=str(path)))
        except OSError as exc:
            counts["unreadable"] += 1
            issues.append(dict(id=row["id"], status="unreadable", error=str(exc)))
    result = dict(status="passed" if not issues else "review_needed", counts=counts,
                  issues=issues, actions="read-only paths/hashes; no fitting, optimizer, random draw or AC")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
