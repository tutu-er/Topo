"""Check 72 frozen scientific sources; optionally restore newline-only drift.

Reuses the canonical-path, whole-batch preflight and exact-byte checks of the
original-stage utility. Snapshot source bytes are data and are never executed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import ev_original_source_restore as original
from .ev_policy_calibration import freeze_identity

HERE = Path(__file__).resolve().parent
DEFAULT_DIRECTORY = HERE/'evidence/ev_compact/delivery_development_v2/development_ensemble_v2'


def audit_directory(directory=DEFAULT_DIRECTORY, *, source_root=HERE,
                    repository_root=None, restore_line_endings=False):
    directory=Path(directory)
    freeze=directory/'freeze.json'
    before=original.digest(freeze.read_bytes())
    identity=freeze_identity(original.load(freeze))
    result=original.audit_source_bytes(directory/'source_snapshot.json.gz',
        freeze=freeze,source_root=source_root,repository_root=repository_root,
        restore_line_endings=restore_line_endings,expected_count=72)
    if original.digest(freeze.read_bytes())!=before:
        raise RuntimeError('freeze changed during compact source audit')
    return dict(result,policy_identity=identity,
        tool_sha256=original.digest(Path(__file__).read_bytes()),
        preflight_tool_sha256=original.digest(Path(original.__file__).read_bytes()),
        policy_identity_tool_sha256=original.digest(Path(__import__(
            freeze_identity.__module__,fromlist=['__file__']).__file__).read_bytes()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=DEFAULT_DIRECTORY)
    parser.add_argument('--source-root',type=Path,default=HERE)
    parser.add_argument('--repository-root',type=Path)
    parser.add_argument('--restore-line-endings',action='store_true')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    try:
        result=audit_directory(args.directory,source_root=args.source_root,
            repository_root=args.repository_root,restore_line_endings=args.restore_line_endings)
    except (ValueError,RuntimeError,OSError) as error:
        result=dict(success=False,errors=[str(error)],restore_requested=args.restore_line_endings)
    if args.report:
        with args.report.open('x',encoding='utf-8') as handle:
            json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,allow_nan=False))
    if not result['success']:raise SystemExit(2)


if __name__=='__main__':main()
