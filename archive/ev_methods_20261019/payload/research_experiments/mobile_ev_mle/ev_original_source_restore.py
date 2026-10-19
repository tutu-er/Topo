"""Audit original-study source bytes; explicitly restore newline-only drift.

Snapshots are byte evidence, never executable instructions. The default is
read-only. No missing or genuinely changed source is overwritten, and the
whole batch is validated before any requested line-ending restoration.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json,os,re,stat
from pathlib import Path,PureWindowsPath

HERE=Path(__file__).resolve().parent
DEFAULT_DIRECTORY=HERE/'evidence/ev_original_domain/pilot_v2'


def digest(data):return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    out={}
    for name,value in pairs:
        if name in out:raise ValueError('duplicate JSON key: '+name)
        out[name]=value
    return out


def load(path):
    path=Path(path);raw=path.read_bytes()
    if path.suffix=='.gz':raw=gzip.decompress(raw)
    return json.loads(raw,object_pairs_hook=_unique_object)


def _component(part):
    if (not part or part in ('.','..') or part.endswith((' ','.')) or
            any(c in part for c in ('\\',':','\0','<','>','"','|','?','*')) or
            any(ord(c)<32 for c in part)):
        raise ValueError('noncanonical source path component: '+repr(part))
    stem=part.split('.')[0].rstrip(' .').upper()
    if stem in ('CON','PRN','AUX','NUL') or stem in {'COM'+str(i) for i in range(1,10)} or stem in {'LPT'+str(i) for i in range(1,10)}:
        raise ValueError('reserved Windows source path component: '+part)


def source_repository_path(name):
    """One canonical frozen name -> repository-relative source target."""
    if not isinstance(name,str) or not name:raise ValueError('nonempty relative source name required')
    windows=PureWindowsPath(name)
    if Path(name).is_absolute() or windows.drive or windows.root or '\\' in name or ':' in name:
        raise ValueError('absolute, Windows or aliased source path rejected: '+name)
    parts=name.split('/')
    if parts[:2]==['..','mobile_ev_probing'] and len(parts)>2:
        tail=parts[2:];prefix=['research_experiments','mobile_ev_probing']
    elif parts[0]=='rnj_wzzt_core' and len(parts)>1:
        tail=parts[1:];prefix=['rnj_wzzt_core']
    else:
        if parts[0].casefold() in ('rnj_wzzt_core','research_experiments'):
            raise ValueError('aliased repository-root source spelling rejected: '+name)
        tail=parts;prefix=['research_experiments','mobile_ev_mle']
    for part in tail:_component(part)
    return Path(*prefix,*tail)


def _linklike(path):
    if path.is_symlink():return True
    try:info=path.lstat()
    except FileNotFoundError:return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0))


def _target(root,relative):
    current=root
    for part in relative.parts:
        if current.is_dir():
            aliases=[p.name for p in current.iterdir() if p.name.casefold()==part.casefold()]
            if aliases and part not in aliases:raise ValueError('case alias rejected: '+str(relative))
        current=current/part
        if _linklike(current):raise ValueError('symlink/junction source target rejected: '+str(relative))
    resolved=current.resolve()
    try:resolved.relative_to(root)
    except ValueError as e:raise ValueError('source target escapes repository') from e
    if resolved==root:raise ValueError('source target is repository directory')
    return resolved


def audit_source_bytes(snapshot=None,*,freeze=None,source_root=HERE,repository_root=None,
                       restore_line_endings=False,expected_count=68):
    source_root=Path(source_root).resolve()
    repository_root=Path(repository_root).resolve() if repository_root is not None else source_root.parents[1]
    if source_root!=repository_root/'research_experiments'/'mobile_ev_mle':
        raise ValueError('source_root must be repository/research_experiments/mobile_ev_mle')
    if not repository_root.is_dir():raise ValueError('repository root does not exist')
    snapshot=Path(snapshot or DEFAULT_DIRECTORY/'source_snapshot.json.gz')
    freeze=Path(freeze or DEFAULT_DIRECTORY/'freeze.json')
    input_hashes=(digest(snapshot.read_bytes()),digest(freeze.read_bytes()))
    saved,frozen=load(snapshot),load(freeze);hashes=frozen.get('source_sha256')
    if (not isinstance(saved,dict) or not isinstance(hashes,dict) or len(hashes)!=expected_count
            or set(saved)!=set(hashes)):
        raise ValueError('snapshot keys must match the external original-study '+str(expected_count)+'-source freeze')
    work=[];rows=[];targets=set();identities=set()
    for name,expected in hashes.items():
        relative=source_repository_path(name);target=_target(repository_root,relative)
        key=str(target).casefold()
        if key in targets:raise ValueError('duplicate resolved source target: '+name)
        targets.add(key)
        if not isinstance(expected,str) or re.fullmatch('[0-9a-f]{64}',expected) is None:raise ValueError('invalid frozen SHA: '+name)
        item=saved[name]
        if not isinstance(item,dict) or item.get('sha256')!=expected:raise ValueError('snapshot/external freeze SHA mismatch: '+name)
        try:data=base64.b64decode(item['bytes_base64'],validate=True)
        except (ValueError,TypeError,KeyError) as e:raise ValueError('invalid snapshot source bytes: '+name) from e
        if digest(data)!=expected:raise ValueError('decoded snapshot bytes/hash mismatch: '+name)
        if target.is_file():
            info=target.stat();identity=(info.st_dev,info.st_ino)
            if info.st_ino and identity in identities:raise ValueError('hardlink/duplicate source object: '+name)
            identities.add(identity);current=target.read_bytes()
        else:identity=None;current=None
        status=('missing' if current is None else 'exact' if digest(current)==expected else
            'line_endings_only' if current.replace(b'\r\n',b'\n')==data.replace(b'\r\n',b'\n') else 'content_changed')
        row=dict(name=name,repository_path=relative.as_posix(),path=str(target),status=status,
            expected_sha256=expected,current_sha256=None if current is None else digest(current))
        rows.append(row);work.append((relative,target,data,current,identity,row))
    counts={label:sum(r['status']==label for r in rows) for label in ('exact','line_endings_only','content_changed','missing')}
    safe=not counts['content_changed'] and not counts['missing'];restored=0;errors=[]
    if counts['content_changed']:errors.append('genuine content changes; whole-batch restoration refused')
    if counts['missing']:errors.append('missing source; whole-batch restoration refused')
    if restore_line_endings and safe:
        if input_hashes!=(digest(snapshot.read_bytes()),digest(freeze.read_bytes())):
            raise RuntimeError('snapshot/freeze changed before whole-batch restoration')
        # Revalidate every path and every original byte before the first write.
        for relative,target,data,current,identity,row in work:
            if _target(repository_root,relative)!=target or target.read_bytes()!=current:
                raise RuntimeError('source changed after whole-batch preflight: '+row['name'])
            info=target.stat()
            if identity!=(info.st_dev,info.st_ino):raise RuntimeError('source object changed after preflight: '+row['name'])
        for relative,target,data,current,identity,row in work:
            if row['status']=='line_endings_only':
                if _target(repository_root,relative)!=target or target.read_bytes()!=current:
                    raise RuntimeError('source changed before line-ending write: '+row['name'])
                target.write_bytes(data)
                if digest(target.read_bytes())!=row['expected_sha256']:raise RuntimeError('post-restore hash mismatch')
                row['status']='restored_line_endings';restored+=1
    for relative,target,data,current,identity,row in work:
        if _target(repository_root,relative)!=target:raise RuntimeError('source path changed during audit')
        row['after_sha256']=digest(target.read_bytes()) if target.is_file() else None
        row['exact_after']=row['after_sha256']==row['expected_sha256']
    match=all(r['exact_after'] for r in rows)
    if input_hashes!=(digest(snapshot.read_bytes()),digest(freeze.read_bytes())):
        raise RuntimeError('snapshot/freeze changed during audit')
    return dict(success=safe and match,snapshot=str(snapshot),freeze=str(freeze),
        snapshot_sha256=input_hashes[0],freeze_sha256=input_hashes[1],evidence_inputs_unchanged=True,
        source_root=str(source_root),repository_root=str(repository_root),snapshot_matches_external_freeze=True,
        checked_count=len(rows),exact_before_count=counts['exact'],line_ending_only_count=counts['line_endings_only'],
        content_changed_count=counts['content_changed'],missing_count=counts['missing'],restored_count=restored,
        exact_after_count=sum(r['exact_after'] for r in rows),all_bytes_match=match,restoration_safe=safe,
        restore_requested=bool(restore_line_endings),scientific_fit_calls=0,scientific_selector_calls=0,
        comparison='only CRLF-to-LF equivalence; exact frozen bytes restored after full-batch preflight',errors=errors,files=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=DEFAULT_DIRECTORY/'source_snapshot.json.gz')
    p.add_argument('--freeze',type=Path,default=DEFAULT_DIRECTORY/'freeze.json')
    p.add_argument('--source-root',type=Path,default=HERE);p.add_argument('--repository-root',type=Path)
    p.add_argument('--expected-source-count',type=int,choices=(68,70),default=68,
        help='external freeze size: 68 for original-study runs, 70 for location ablation')
    p.add_argument('--restore-line-endings',action='store_true');p.add_argument('--report',type=Path)
    args=p.parse_args()
    try:result=audit_source_bytes(args.snapshot,freeze=args.freeze,source_root=args.source_root,
        repository_root=args.repository_root,restore_line_endings=args.restore_line_endings,
        expected_count=args.expected_source_count)
    except (ValueError,RuntimeError,OSError) as e:result=dict(success=False,restore_requested=args.restore_line_endings,errors=[str(e)])
    if args.report:
        with args.report.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,allow_nan=False))
    if not result['success']:raise SystemExit(2)


if __name__=='__main__':main()
