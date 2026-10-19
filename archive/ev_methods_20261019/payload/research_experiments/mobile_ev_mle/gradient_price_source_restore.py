"""Audit exact frozen source bytes; optionally restore CRLF/LF differences.

The byte snapshot must match the external freeze SHA-256 map. Every target is
resolved beneath research_experiments. A real content change blocks the whole
restore preflight; the tool never extracts an archive or rewrites code logic.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path,PureWindowsPath
import re

HERE=Path(__file__).resolve().parent
DEFAULT_SNAPSHOT=HERE/'evidence'/'gradient_price'/'science_source_snapshot.json.gz'


def default_freeze():
    base=HERE/'evidence'/'gradient_price'/'v2_supported'
    raw=base/'freeze.json'
    return raw if raw.exists() else base/'compact'/'freeze.json'


def digest(data):return hashlib.sha256(data).hexdigest()


def _load(path):
    path=Path(path);data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def audit_source_bytes(snapshot=DEFAULT_SNAPSHOT,*,source_root=HERE,freeze=None,
                       restore_line_endings=False):
    """Default read-only audit. Restore only validated CRLF/LF changes.

    No write starts until every snapshot byte hash, destination and current
    content passes preflight. Missing files and genuine code edits are blocked.
    """
    source_root=Path(source_root).resolve()
    allowed=source_root.parent.resolve()
    if allowed.name!='research_experiments':
        raise ValueError('source_root must be a direct child of research_experiments')
    snapshot=Path(snapshot);saved=_load(snapshot)
    frozen=_load(default_freeze() if freeze is None else freeze)
    if saved.get('schema')!='gradient_price_source_bytes_v1':
        raise ValueError('unknown source snapshot schema')
    hashes=saved.get('source_sha256');files=saved.get('files')
    if not isinstance(hashes,dict) or not isinstance(files,dict) or not hashes:
        raise ValueError('nonempty source_sha256/files maps required')
    if hashes!=frozen.get('source_sha256'):
        raise ValueError('snapshot source hashes differ from external freeze')
    if set(files)!=set(hashes):raise ValueError('snapshot file keys differ from frozen hashes')
    targets=set();work=[];rows=[]
    for name,expected in hashes.items():
        if not isinstance(name,str) or not name:
            raise ValueError('source name must be a nonempty relative path')
        windows=PureWindowsPath(name)
        if Path(name).is_absolute() or windows.drive or windows.root:
            raise ValueError('absolute source path rejected: '+name)
        target=(source_root/name).resolve()
        try:target.relative_to(allowed)
        except ValueError as error:raise ValueError('source path escapes research_experiments: '+name) from error
        if target==allowed or target in targets:
            raise ValueError('directory or duplicate resolved target rejected: '+name)
        targets.add(target)
        if not isinstance(expected,str) or re.fullmatch('[0-9a-f]{64}',expected) is None:
            raise ValueError('invalid frozen SHA-256: '+name)
        try:data=base64.b64decode(files[name],validate=True)
        except (ValueError,TypeError) as error:raise ValueError('invalid base64 source bytes: '+name) from error
        if digest(data)!=expected:raise ValueError('snapshot bytes do not match frozen SHA-256: '+name)
        row=dict(name=name,path=str(target),expected_sha256=expected)
        if not target.is_file():
            row.update(status='missing',current_sha256=None);current=None
        else:
            current=target.read_bytes();actual=digest(current)
            if actual==expected:status='exact'
            elif current.replace(b'\r\n',b'\n')==data.replace(b'\r\n',b'\n'):status='line_endings_only'
            else:status='content_changed'
            row.update(status=status,current_sha256=actual)
        work.append((target,data,current,row));rows.append(row)
    exact=sum(r['status']=='exact' for r in rows)
    endings=sum(r['status']=='line_endings_only' for r in rows)
    changed=sum(r['status']=='content_changed' for r in rows)
    missing=sum(r['status']=='missing' for r in rows)
    safe=not changed and not missing;restored=0
    errors=[]
    if changed:errors.append('genuine source content changes; restore refused for every file')
    if missing:errors.append('missing source files; restore refused for every file')
    if restore_line_endings and safe:
        # A re-read also guards against edits made after the whole-batch audit.
        for target,data,current,row in work:
            if target.read_bytes()!=current:
                raise RuntimeError('source changed after preflight; restore stopped: '+row['name'])
        for target,data,current,row in work:
            if row['status']=='line_endings_only':
                if target.read_bytes()!=current:
                    raise RuntimeError('source changed before write; restore stopped: '+row['name'])
                target.write_bytes(data)
                if digest(target.read_bytes())!=row['expected_sha256']:
                    raise RuntimeError('post-restore SHA-256 mismatch: '+row['name'])
                row['status']='restored_line_endings';restored+=1
    for target,data,current,row in work:
        row['after_sha256']=digest(target.read_bytes()) if target.is_file() else None
        row['exact_after']=row['after_sha256']==row['expected_sha256']
    all_match=all(r['exact_after'] for r in rows)
    return dict(success=safe and all_match,snapshot=str(snapshot),source_root=str(source_root),
        allowed_root=str(allowed),snapshot_matches_freeze=True,restore_requested=bool(restore_line_endings),
        checked_count=len(rows),exact_before_count=exact,line_ending_only_count=endings,
        content_changed_count=changed,missing_count=missing,restored_count=restored,
        exact_after_count=sum(r['exact_after'] for r in rows),all_bytes_match=all_match,
        restoration_safe=safe,errors=errors,files=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,default=DEFAULT_SNAPSHOT)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--source-root',type=Path,default=HERE)
    parser.add_argument('--restore-line-endings',action='store_true')
    args=parser.parse_args()
    try:
        report=audit_source_bytes(args.snapshot,source_root=args.source_root,freeze=args.freeze,
                                  restore_line_endings=args.restore_line_endings)
    except (ValueError,RuntimeError,OSError) as error:
        print(json.dumps(dict(success=False,errors=[str(error)],restore_requested=args.restore_line_endings),ensure_ascii=False))
        raise SystemExit(2)
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
    if not report['success']:raise SystemExit(2)


if __name__=='__main__':main()
