"""Restore the two preflight histories and their byte-exact science snapshot.

Only bytes and data are restored.  Restored Python sources are never imported
or executed.  The output repository root must not exist.  Snapshot paths follow
the original rules: rnj_wzzt_core/ is relative to the repository, all other
entries are relative to research_experiments/mobile_ev_mle, including legitimate
../ references that still resolve inside the fresh repository root.
"""
from __future__ import annotations

import argparse
import base64
import gzip
from hashlib import sha256
import json
from pathlib import Path,PureWindowsPath
import re

from . import ev_bayes_small_evidence as oldpack


ARCHIVES=('development_000.evidence.zip','development_001.evidence.zip')
SIDECARS={'preflight_freeze.json':'freeze.json',
          'preflight_source_snapshot.json.gz':'source_snapshot.json.gz'}


def _hash(data):return sha256(data).hexdigest()


def _read(path):
    payload=Path(path).read_bytes()
    return json.loads(gzip.decompress(payload) if str(path).endswith('.gz') else payload)


def _reject_links(path):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        junction=getattr(part,'is_junction',lambda:False)
        if part.is_symlink() or junction():raise ValueError('symbolic links/junctions are not allowed: '+str(part))


def _safe_source_path(output_root,name):
    if not isinstance(name,str) or not name or '\\' in name or '\x00' in name:
        raise ValueError('snapshot name must be a nonempty forward-slash relative path')
    win=PureWindowsPath(name)
    if win.is_absolute() or win.drive or name.startswith('/') or ':' in name:
        raise ValueError('absolute/drive/alternate-stream snapshot path rejected: '+name)
    for part in name.split('/'):
        if not part:raise ValueError('empty snapshot path component rejected')
        if part in ('.','..'):continue
        if part[-1] in (' ','.') or any(c in part for c in '<>"|?*'):
            raise ValueError('ambiguous Windows snapshot path component rejected: '+part)
        if re.fullmatch(r'(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])',part.split('.')[0]):
            raise ValueError('Windows device snapshot path rejected: '+part)
    root=Path(output_root).absolute()
    base=root if name.startswith('rnj_wzzt_core/') else root/'research_experiments'/'mobile_ev_mle'
    destination=(base/name).resolve()
    resolved_root=root.resolve()
    if destination==resolved_root or resolved_root not in destination.parents:
        raise ValueError('snapshot path escapes fresh repository root: '+name)
    _reject_links(destination)
    return destination


def _write(path,payload):
    path=Path(path);_reject_links(path)
    path.parent.mkdir(parents=True,exist_ok=True);_reject_links(path.parent)
    with path.open('xb') as handle:handle.write(payload)


def _archive_rows(audit):
    rows={}
    for row in audit.get('records',[]):
        packed=row['packed']
        name=str(packed['archive']).replace('\\','/').rsplit('/',1)[-1]
        if name not in ARCHIVES or name in rows:raise ValueError('archive audit must list the two exact preflight archives once')
        if not row.get('checked',{}).get('passed'):raise ValueError('original preflight archive verification did not pass')
        rows[name]=row
    if set(rows)!=set(ARCHIVES):raise ValueError('archive audit must contain both preflight histories')
    return rows


def restore_preflight(input_dir,output_root):
    source=Path(input_dir).absolute();target=Path(output_root).absolute()
    _reject_links(source);_reject_links(target)
    if target.exists():raise FileExistsError('restore requires a fresh repository root: '+str(target))
    source=source.resolve();target=target.resolve()
    required=(*SIDECARS,'archive_audit.json',*ARCHIVES)
    for name in required:
        path=source/name;_reject_links(path)
        if not path.is_file():raise FileNotFoundError(path)
    frozen=_read(source/'preflight_freeze.json');snapshot=_read(source/'preflight_source_snapshot.json.gz')
    hashes=snapshot.get('source_sha256');payloads=snapshot.get('source_base64')
    if not isinstance(hashes,dict) or not isinstance(payloads,dict) or not hashes:
        raise ValueError('nonempty snapshot source hashes and payloads required')
    if hashes!=frozen.get('source_sha256') or set(payloads)!=set(hashes):
        raise ValueError('freeze/snapshot source manifests disagree')
    # Resolve every output before creating anything.  Case-insensitive aliases
    # and file/directory prefix collisions are also rejected on POSIX hosts.
    planned=[];destinations={}
    for name,expected in hashes.items():
        if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
            raise ValueError('invalid source SHA256 '+str(name))
        destination=_safe_source_path(target,name);key=str(destination).casefold()
        if key in destinations:raise ValueError('duplicate resolved snapshot output '+name)
        decoded=base64.b64decode(payloads[name],validate=True)
        if _hash(decoded)!=expected:raise ValueError('snapshot source checksum mismatch '+name)
        destinations[key]=destination;planned.append((name,destination,decoded,expected))
    data_root=target/'outputs'/'preflight'
    sidecars=[(source/name,data_root/outname) for name,outname in SIDECARS.items()]
    data_paths=[data_root/'development'/('history_%03d.json.gz'%index) for index in range(2)]
    manifest_path=target/'restore_audit.json'
    all_paths=[p for _,p,_,_ in planned]+[p for _,p in sidecars]+data_paths+[manifest_path]
    keys=[str(p.resolve()).casefold() for p in all_paths]
    if len(keys)!=len(set(keys)):raise ValueError('snapshot output collides with restore data or manifest')
    keyset=set(keys)
    if any(str(parent).casefold() in keyset for p in all_paths for parent in p.parents):
        raise ValueError('snapshot output file/directory prefix collision')
    archive_audit=_archive_rows(_read(source/'archive_audit.json'))
    # Archive checks and exact data verification precede all filesystem writes.
    records=[]
    for index,name in enumerate(ARCHIVES):
        archive=source/name;raw=archive.read_bytes();packed=archive_audit[name]['packed']
        if _hash(raw)!=packed['archive_sha256']:raise ValueError('preflight archive checksum mismatch '+name)
        verified=oldpack.verify_record(archive)
        if not verified.get('passed'):raise ValueError('verify_record did not pass '+name)
        record,manifest=oldpack.unpack_record(archive)
        if record.get('policy_id')!=frozen.get('policy_id') or record.get('spec',{}).get('index')!=index or record.get('spec',{}).get('phase')!='development':
            raise ValueError('archive history/policy differs from corresponding preflight freeze')
        canonical=oldpack.canonical_json(record)
        if _hash(canonical)!=packed['canonical_json_sha256'] or verified['canonical_json_sha256']!=packed['canonical_json_sha256']:
            raise ValueError('preflight archive canonical content mismatch '+name)
        compressed=gzip.compress(canonical,compresslevel=1,mtime=0)
        decoded=json.loads(gzip.decompress(compressed));same,where=oldpack.exact_equal(record,decoded)
        if not same:raise ValueError('gzip data round trip changed a type or IEEE value: '+str(where))
        records.append(dict(index=index,name=name,record=record,payload=compressed,
            row=dict(archive=name,archive_sha256=_hash(raw),verify_record=verified,
                canonical_json_sha256=packed['canonical_json_sha256'],
                original_compressed_sha256=manifest['source']['original_compressed_sha256'],
                restored_compressed_sha256=_hash(compressed),types_and_float_bits_equal=True,
                gzip_compresslevel=1,gzip_mtime=0,
                restored_relative=str(data_paths[index].relative_to(target)).replace('\\','/'))))
    target.mkdir(parents=True,exist_ok=False)
    restored_sources=[]
    for name,destination,payload,expected in planned:
        _write(destination,payload)
        if _hash(destination.read_bytes())!=expected:raise ValueError('written source checksum mismatch '+name)
        restored_sources.append(dict(snapshot_name=name,relative=str(destination.relative_to(target)).replace('\\','/'),sha256=expected,bytes=len(payload)))
    for original,destination in sidecars:_write(destination,original.read_bytes())
    result_rows=[]
    for item,path in zip(records,data_paths):
        _write(path,item['payload'])
        read_back,provenance=oldpack.read_record(path);same,where=oldpack.exact_equal(item['record'],read_back)
        if not same or provenance['canonical_json_sha256']!=item['row']['canonical_json_sha256']:
            raise ValueError('restored disk record differs in types or float bits: '+str(where))
        result_rows.append(item['row'])
    result=dict(schema='ev_bayes_response_preflight_restore_v1',passed=True,policy_id=frozen['policy_id'],
        input_directory=str(source.resolve()),output_root=str(target.resolve()),source_count=len(restored_sources),
        sources=restored_sources,records=result_rows,
        sidecars=[dict(relative=str(p.relative_to(target)).replace('\\','/'),sha256=_hash(p.read_bytes())) for _,p in sidecars],
        source_snapshot_sha256=_hash((source/'preflight_source_snapshot.json.gz').read_bytes()),
        freeze_sha256=_hash((source/'preflight_freeze.json').read_bytes()),
        restore_tool_sha256=_hash(Path(__file__).read_bytes()),restored_sources_executed=False,
        path_scope='all resolved outputs strictly inside a fresh repository root; links and ambiguous paths rejected',
        compressed_scope='original gzip SHA retained as provenance; new gzip level1 bytes may differ while JSON types and float bits match')
    _write(manifest_path,oldpack.canonical_json(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output-root',required=True)
    args=parser.parse_args();result=restore_preflight(args.input,args.output_root)
    print(json.dumps(dict(passed=result['passed'],source_count=result['source_count'],record_count=len(result['records']),
        manifest=str(Path(result['output_root'])/'restore_audit.json'),restored_sources_executed=False),sort_keys=True))


if __name__=='__main__':main()
