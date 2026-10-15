"""Read-only audit, or explicit newline-only restoration, of 78/79 sources.

New snapshots store source_sha256/source_base64. A temporary data-only adapter
feeds the existing whole-batch path/link/content preflight. Neither snapshot
nor freeze is rewritten, and snapshot contents are never executed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from . import ev_original_source_restore as safe

HERE = Path(__file__).resolve().parent
DEFAULT_DIRECTORY = HERE/"evidence/ev_fixed_train/development_v2"


def _digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _directory(value):
    if value is None or str(value) in ("weak8", "development_v2"):return DEFAULT_DIRECTORY
    if str(value) in ("rich8", "rich32", "rich96"):
        return HERE/"evidence/ev_fixed_train/training_support_dev_v1"/str(value)
    return Path(value)


def audit_source_bytes(directory=None, *, source_root=HERE, repository_root=None,
                       restore_line_endings=False, expected_count=None, temporary_root=None):
    """Adapt new evidence to the old safe audit without executing source data.

    By default only the 78-source weak or 79-source rich freeze is accepted.
    expected_count is exposed for small controlled test fixtures. All source
    target resolution and newline writes are delegated to the unchanged safe
    API. Temporary adapter files live in an explicitly bounded repo/outputs.
    """
    directory = _directory(directory).resolve()
    source_root = Path(source_root).resolve()
    repository_root = Path(repository_root).resolve() if repository_root is not None else source_root.parents[1]
    if source_root != repository_root/"research_experiments/mobile_ev_mle":
        raise ValueError("source_root must be repository/research_experiments/mobile_ev_mle")
    if not repository_root.is_dir():raise ValueError("repository root does not exist")
    snapshot, freeze = directory/"source_snapshot.json.gz", directory/"freeze.json"
    input_hashes = (_digest(snapshot), _digest(freeze))
    saved, frozen = safe.load(snapshot), safe.load(freeze)
    if not isinstance(saved, dict) or set(saved) != {"source_sha256", "source_base64"}:
        raise ValueError("expected new source_sha256/source_base64 snapshot schema")
    hashes, encoded = saved["source_sha256"], saved["source_base64"]
    if (not isinstance(hashes, dict) or not isinstance(encoded, dict) or not hashes
            or hashes != frozen.get("source_sha256") or set(encoded) != set(hashes)):
        raise ValueError("new snapshot and external freeze source manifests differ")
    count = len(hashes)
    if expected_count is None and count not in (78, 79):
        raise ValueError("fixed-training freeze must contain 78 or 79 sources")
    if expected_count is not None and (isinstance(expected_count, bool) or count != expected_count):
        raise ValueError("unexpected fixed-training source count")
    if frozen.get("source_count", count) != count:raise ValueError("freeze source_count differs from manifest")
    old_schema = {}
    for name, sha in hashes.items():
        safe.source_repository_path(name)
        if not isinstance(sha, str) or re.fullmatch("[0-9a-f]{64}", sha) is None:
            raise ValueError("invalid frozen SHA: "+name)
        try:data = base64.b64decode(encoded[name], validate=True)
        except (ValueError, TypeError) as error:raise ValueError("invalid encoded source bytes: "+name) from error
        if hashlib.sha256(data).hexdigest() != sha:raise ValueError("snapshot decoded bytes/hash mismatch: "+name)
        old_schema[name] = dict(sha256=sha, bytes_base64=encoded[name])
    # Validate the resolved absolute cleanup target before creating a temporary
    # directory that the standard library will later remove recursively.
    area = Path(temporary_root).resolve() if temporary_root is not None else repository_root/"outputs/ev_source_restore_temporary"
    area = area.resolve()
    if not area.is_relative_to(repository_root) or area == repository_root:
        raise ValueError("temporary adapter area must be inside the intended repository")
    area.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="fixed_source_adapter_", dir=area) as temporary:
        temporary = Path(temporary).resolve()
        if not temporary.is_relative_to(area) or temporary == area:
            raise RuntimeError("temporary cleanup target escapes the intended adapter area")
        adapted = temporary/"legacy_snapshot.json"
        adapted.write_bytes(json.dumps(old_schema, sort_keys=True, allow_nan=False).encode("utf-8"))
        options = dict(freeze=freeze, source_root=source_root, repository_root=repository_root, expected_count=count)
        # Always perform the whole-batch read-only preflight first. A bad or
        # missing source blocks every requested newline restoration.
        result = safe.audit_source_bytes(adapted, **options)
        if input_hashes != (_digest(snapshot), _digest(freeze)):
            raise RuntimeError("original snapshot/freeze changed during adapter preflight")
        if restore_line_endings and result["restoration_safe"]:
            # The safe API repeats source/path/object checks before its first
            # write and verifies the external freeze throughout restoration.
            result = safe.audit_source_bytes(adapted, restore_line_endings=True, **options)
        elif restore_line_endings:
            result["restore_requested"] = True
        adapter_sha = _digest(adapted)
    if input_hashes != (_digest(snapshot), _digest(freeze)):
        raise RuntimeError("original snapshot/freeze changed during source audit")
    result.update(snapshot=str(snapshot), freeze=str(freeze), snapshot_sha256=input_hashes[0], freeze_sha256=input_hashes[1],
        evidence_inputs_unchanged=True, source_writes=result["restored_count"],
        new_snapshot_schema="source_sha256/source_base64", snapshot_manifest_equals_external_freeze=True,
        adapter_snapshot_sha256=adapter_sha, temporary_adapter_retained=False,
        adapter="data-only conversion followed by existing canonical-path whole-batch safe audit",
        safe_helper_sha256=_digest(safe.__file__), wrapper_sha256=_digest(__file__))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=str(DEFAULT_DIRECTORY),
        help="evidence directory, or development_v2/weak8/rich8/rich32/rich96")
    parser.add_argument("--source-root", type=Path, default=HERE)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--expected-source-count", type=int, choices=(78,79))
    parser.add_argument("--restore-line-endings", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = audit_source_bytes(args.directory, source_root=args.source_root, repository_root=args.repository_root,
            temporary_root=args.temporary_root, restore_line_endings=args.restore_line_endings,
            expected_count=args.expected_source_count)
    except (ValueError, RuntimeError, OSError) as error:
        result = dict(success=False, restore_requested=args.restore_line_endings,
            source_writes=None if args.restore_line_endings else 0,
            source_write_count_scope="unknown after restore-path exception" if args.restore_line_endings else "read-only path",
            errors=[str(error)])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!="files"}, ensure_ascii=False, allow_nan=False))
    if not result["success"]:raise SystemExit(2)


if __name__ == "__main__":main()
