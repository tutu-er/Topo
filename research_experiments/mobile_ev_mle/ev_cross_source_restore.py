"""Audit frozen EV source bytes; explicitly restore newline-only differences.

The source snapshot is evidence, not executable instructions. No target is
written unless the entire externally frozen batch matches frozen content after CRLF-to-LF
comparison. Restore writes the original snapshot bytes, including mixed endings.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re

HERE=Path(__file__).resolve().parent
DEFAULT_DIRECTORY=HERE/'evidence/ev_cross_all/v2'
ALLOWED_SOURCE_COUNTS=(31,33)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    path=Path(path);raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def audit_source_bytes(snapshot=None,*,source_root=HERE,freeze=None,restore_line_endings=False):
    """Read-only by default; restore only the complete validated newline batch."""
    source_root=Path(source_root).resolve();allowed=source_root.parent.resolve()
    if allowed.name!='research_experiments':
        raise ValueError('source_root must be a direct child of research_experiments')
    snapshot=Path(snapshot or DEFAULT_DIRECTORY/'source_snapshot.json.gz')
    freeze=Path(freeze or DEFAULT_DIRECTORY/'freeze.json')
    saved,frozen=load(snapshot),load(freeze)
    if saved.get('schema')!='source_bytes_base64':
        raise ValueError('unknown source snapshot schema')
    recorded_freeze_hash=saved.get('freeze_sha256')
    if recorded_freeze_hash is not None and (not isinstance(recorded_freeze_hash,str) or re.fullmatch('[0-9a-f]{64}',recorded_freeze_hash) is None):
        raise ValueError('snapshot requires a valid provenance freeze_sha256')
    hashes,sources=frozen.get('source_sha256'),saved.get('sources')
    if not isinstance(hashes,dict) or not isinstance(sources,dict) or len(hashes) not in ALLOWED_SOURCE_COUNTS or set(sources)!=set(hashes):
        raise ValueError('31 or 33 source keys must match the external cross-study freeze')
    targets=set();work=[];rows=[]
    for name,expected in hashes.items():
        if not isinstance(name,str) or not name:
            raise ValueError('nonempty relative source name required')
        windows=PureWindowsPath(name)
        if Path(name).is_absolute() or windows.drive or windows.root or ':' in name or '\0' in name:
            raise ValueError('absolute or special source path rejected: '+name)
        parts=name.split('/')
        # Allow the frozen sibling ../mobile_ev_probing files, while rejecting
        # redundant spellings, Windows aliases and extra parent traversals.
        if ('\\' in name or any(part in ('','.') for part in parts)
                or any(part=='..' for part in parts[1:])
                or parts[0]=='..' and len(parts)<3
                or any(part not in ('..',) and part.endswith((' ','.')) for part in parts)
                or any(PureWindowsPath(part).is_reserved() for part in parts if part!='..')):
            raise ValueError('noncanonical or aliased source path rejected: '+name)
        target=(source_root/name).resolve()
        try:
            target.relative_to(allowed)
        except ValueError as error:
            raise ValueError('source path escapes research_experiments: '+name) from error
        if target==allowed or target in targets:
            raise ValueError('directory or duplicate resolved target rejected: '+name)
        targets.add(target)
        if not isinstance(expected,str) or re.fullmatch('[0-9a-f]{64}',expected) is None:
            raise ValueError('invalid frozen SHA-256: '+name)
        record=sources[name]
        if not isinstance(record,dict) or record.get('sha256')!=expected:
            raise ValueError('snapshot SHA differs from external freeze: '+name)
        try:
            data=base64.b64decode(record.get('bytes_base64'),validate=True)
        except (ValueError,TypeError) as error:
            raise ValueError('invalid base64 source bytes: '+name) from error
        if digest(data)!=expected:
            raise ValueError('snapshot bytes do not match frozen SHA-256: '+name)
        current=target.read_bytes() if target.is_file() else None
        status='missing' if current is None else 'exact' if digest(current)==expected else 'line_endings_only' if current.replace(b'\r\n',b'\n')==data.replace(b'\r\n',b'\n') else 'content_changed'
        row=dict(name=name,path=str(target),expected_sha256=expected,status=status,
                 current_sha256=None if current is None else digest(current))
        work.append((target,data,current,row));rows.append(row)
    exact=sum(r['status']=='exact' for r in rows)
    endings=sum(r['status']=='line_endings_only' for r in rows)
    changed=sum(r['status']=='content_changed' for r in rows)
    missing=sum(r['status']=='missing' for r in rows)
    safe=not changed and not missing;restored=0;errors=[]
    if changed:errors.append('genuine content changes; whole-batch restore refused')
    if missing:errors.append('missing sources; whole-batch restore refused')
    if restore_line_endings and safe:
        # Recheck all files before the first write; no missing source is created.
        for target,data,current,row in work:
            if target.read_bytes()!=current:
                raise RuntimeError('source changed after preflight; restore stopped: '+row['name'])
        for target,data,current,row in work:
            if row['status']=='line_endings_only':
                if target.read_bytes()!=current:
                    raise RuntimeError('source changed before write; restore stopped: '+row['name'])
                target.write_bytes(data)
                if digest(target.read_bytes())!=row['expected_sha256']:
                    raise RuntimeError('post-restore SHA mismatch: '+row['name'])
                row['status']='restored_line_endings';restored+=1
    for target,data,current,row in work:
        row['after_sha256']=digest(target.read_bytes()) if target.is_file() else None
        row['exact_after']=row['after_sha256']==row['expected_sha256']
    all_match=all(r['exact_after'] for r in rows)
    actual_freeze_hash=digest(freeze.read_bytes())
    return dict(success=safe and all_match,snapshot=str(snapshot),freeze=str(freeze),
        snapshot_sha256=digest(snapshot.read_bytes()),freeze_sha256=digest(freeze.read_bytes()),
        snapshot_recorded_freeze_sha256=recorded_freeze_hash,
        freeze_provenance_bytes_match=None if recorded_freeze_hash is None else actual_freeze_hash==recorded_freeze_hash,
        freeze_hash_role='provenance only; external per-source hash map and decoded bytes are authoritative',
        source_root=str(source_root),allowed_root=str(allowed),snapshot_matches_freeze=True,
        restore_requested=bool(restore_line_endings),checked_count=len(rows),exact_before_count=exact,
        line_ending_only_count=endings,content_changed_count=changed,missing_count=missing,
        restored_count=restored,exact_after_count=sum(r['exact_after'] for r in rows),
        all_bytes_match=all_match,restoration_safe=safe,code_content_preserved=safe,
        comparison='CRLF replaced by LF for comparison; restore uses exact frozen bytes',
        scientific_fit_calls=0,scientific_selector_calls=0,errors=errors,files=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,default=DEFAULT_DIRECTORY/'source_snapshot.json.gz')
    parser.add_argument('--freeze',type=Path,default=DEFAULT_DIRECTORY/'freeze.json')
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
