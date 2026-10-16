"""Immutable merge of one physical bank and one formal EV phase.

Only the standard library is used. No AC, planner, calibration or evaluation
is run. Records are validated one at a time, then hard-linked where possible.
Sources must remain immutable because a hard link shares the original inode.
Snapshot payloads must match by decoded source bytes; gzip headers may differ.
All original metadata hashes and every source record SHA are retained.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil


SCHEMA='mobile_reliability_EV_v2'
MODES=('fixed_total','fixed_holdout','joint_total')
SEED_START={('n3','calibration'):101210001,('n3','evaluation'):101220001,
            ('n4','calibration'):101310001,('n4','evaluation'):101320001}
METADATA=('freeze.json','source_snapshot.json.gz','checkpoint.json','manifest.json',
          'delivery_manifest.json','merge_manifest.json')


def _require(ok,message):
    if not ok:raise ValueError(message)


def _read(path):
    data=Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if Path(path).suffix=='.gz' else data),data


def _encoded(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')


def _object_hash(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _write_new(path,data):
    with Path(path).open('xb') as stream:stream.write(data)


def _child(directory,name):
    _require(isinstance(name,str) and Path(name).name==name,'merge filename must be a basename')
    path=directory/name
    _require(path.resolve().parent==directory.resolve() and not path.is_symlink(),'merge filename escapes output directory')
    return path


def _snapshot(value,hashes):
    _require(value.get('source_sha256')==hashes,'snapshot source hashes differ from freeze')
    payload=value.get('source_base64')
    _require(isinstance(payload,dict) and set(payload)==set(hashes),'snapshot must retain every frozen source')
    digest=sha256()
    for name in sorted(hashes):
        source=base64.b64decode(payload[name],validate=True)
        _require(sha256(source).hexdigest()==hashes[name],'snapshot payload hash mismatch: '+name)
        label=name.encode('utf8');digest.update(len(label).to_bytes(8,'little'));digest.update(label)
        digest.update(len(source).to_bytes(8,'little'));digest.update(source)
    return digest.hexdigest()


def _freeze(directory):
    freeze,raw=_read(directory/'freeze.json');snapshot,snapshot_raw=_read(directory/'source_snapshot.json.gz')
    config=freeze.get('config');hashes=freeze.get('source_sha256')
    _require(isinstance(config,dict) and isinstance(hashes,dict) and hashes,'freeze config/source map required')
    _require(all(isinstance(k,str) and isinstance(v,str) and len(v)==64 for k,v in hashes.items()),'malformed frozen source hashes')
    policy=_object_hash(dict(config=config,source_sha256=hashes))
    _require(freeze.get('policy_id')==policy,'freeze policy identity does not bind config and sources')
    return freeze,raw,snapshot_raw,_snapshot(snapshot,hashes)


def _record(value,freeze,*,bank,phase,count):
    _require(value.get('schema')==SCHEMA,'record schema differs from declared EV experiment')
    physical=value.get('physical_bank',value.get('delivery_provenance',{}).get('physical_bank'))
    _require(physical==bank,'record physical bank differs from requested bank')
    _require(value.get('phase')==phase,'record phase differs from requested phase')
    index=value.get('index');seed=value.get('seed')
    _require(isinstance(index,int) and not isinstance(index,bool) and 0<=index<count,'record index outside predeclared 0..N-1')
    _require(isinstance(seed,int) and not isinstance(seed,bool) and seed==SEED_START[bank,phase]+index,
        'record seed differs from frozen formal index schedule')
    _require(value.get('case_id') in freeze['config'].get(bank+'_formal_cases',[]),'record case outside declared physical-bank mixture')
    fi=value.get('family_index')
    _require(isinstance(fi,int) and not isinstance(fi,bool) and 0<=fi<4,'record family index outside declared four structures')
    _require(value.get('policy_id')==freeze['policy_id'] and value.get('source_sha256')==freeze['source_sha256'],
        'record frozen policy/source identity differs')
    _require(value.get('source_unchanged') is True,'record must document unchanged scientific sources')
    _require(isinstance(value.get('status'),str) and value['status'],'record status required, including failures')
    _require(isinstance(value.get('candidate_miss'),bool),'record candidate-miss flag required')
    methods=value.get('methods');declared=freeze['config'].get('methods')
    _require(isinstance(declared,list) and declared and len(set(declared))==len(declared),'freeze must declare unique methods')
    _require(isinstance(methods,list),'record method slots required')
    actual=[m.get('name') for m in methods]
    _require(len(actual)==len(declared) and set(actual)==set(declared),'record missing, duplicate or extra method slots')
    totals=Counter(planning_failures=0,operation_failures=0,incomplete_inference_slots=0)
    for method in methods:
        modes=method.get('inference',{})
        _require(set(modes)==set(MODES),'record must retain all three inference modes')
        totals['planning_failures']+=bool(method.get('planning_failure'))
        totals['operation_failures']+=not method.get('operation',{}).get('success',False)
        for mode in modes.values():
            _require(isinstance(mode.get('complete'),bool) and isinstance(mode.get('losses'),dict),
                'inference complete bool/losses mapping required')
            totals['incomplete_inference_slots']+=not mode['complete']
    delivery=value.get('delivery_provenance')
    kind=('analysis_lite' if delivery.get('analysis_only') else 'compact') if isinstance(delivery,dict) else 'raw'
    return index,seed,kind,dict(totals)


def _link_or_copy(source,target):
    try:
        os.link(source,target)
        return 'hardlink',None
    except OSError as exc:
        if target.exists():raise
        with source.open('rb') as inp,target.open('xb') as out:shutil.copyfileobj(inp,out,1024*1024)
        return 'copy',type(exc).__name__+': '+str(exc)


def merge(sources,output,*,bank,phase,expected_count=None,expected_policy_id=None):
    _require((bank,phase) in SEED_START,'merge supports n3/n4 calibration/evaluation only')
    output=Path(output).resolve();directories=[Path(p).resolve() for p in sources]
    if output.exists():raise FileExistsError('merge output already exists; all existing files retained')
    _require(directories and len(set(directories))==len(directories),'unique nonempty source directories required')
    _require(output not in directories,'merge output cannot be a source')
    canonical=None;canonical_freeze=canonical_snapshot=None;snapshot_id=None;parts=[];rows=[];seen_indices=set();seen_seeds=set();kinds=set()
    for number,directory in enumerate(directories):
        _require(directory.is_dir(),'source directory missing: '+str(directory))
        freeze,freeze_raw,snapshot_raw,content_id=_freeze(directory)
        if canonical is None:
            canonical=freeze;canonical_freeze=freeze_raw;canonical_snapshot=snapshot_raw;snapshot_id=content_id
            count=freeze['config'].get(phase+'_histories_per_bank')
            _require(isinstance(count,int) and not isinstance(count,bool) and count>0,'positive frozen formal history count required')
            _require(expected_count is None or expected_count==count,'requested count differs from frozen formal count')
            _require(expected_policy_id is None or expected_policy_id==freeze['policy_id'],'unexpected frozen policy identity')
        else:
            for field in ('config','source_sha256','policy_id','environment'):
                _require(freeze.get(field)==canonical.get(field),'source freeze '+field+' differs across chunks')
            _require(content_id==snapshot_id,'decoded source snapshots differ across chunks')
        metadata=[]
        for name in METADATA:
            path=directory/name
            if not path.exists():continue
            _require(not path.is_symlink(),'metadata symlinks are not supported')
            data=path.read_bytes();metadata.append(dict(name=name,sha256=sha256(data).hexdigest(),bytes=len(data)))
        if (directory/'checkpoint.json').exists():
            checkpoint,_=_read(directory/'checkpoint.json')
            _require(checkpoint.get('phase',phase)==phase and checkpoint.get('bank',bank)==bank,'source checkpoint bank/phase differs')
        if (directory/'delivery_manifest.json').exists():
            from .ev_mobile_v2_delivery import verify
            verify(directory)
        paths=sorted({p for pattern in ('case_*','trial_*') for p in directory.glob(pattern) if p.is_file()})
        _require(paths,'source chunk contains no case/trial records')
        part=dict(directory=str(directory),metadata=metadata,source_snapshot_content_id=content_id,records=0)
        for path in paths:
            _require(not path.is_symlink() and (path.name.endswith('.json') or path.name.endswith('.json.gz')),
                'extra unsupported record file: '+path.name)
            record,data=_read(path);index,seed,kind,counts=_record(record,freeze,bank=bank,phase=phase,count=count)
            _require(index not in seen_indices,'duplicate record index: '+str(index))
            _require(seed not in seen_seeds,'duplicate independent history seed: '+str(seed))
            seen_indices.add(index);seen_seeds.add(seed);kinds.add(kind)
            extension='.json.gz' if path.suffix=='.gz' else '.json'
            rows.append(dict(index=index,seed=seed,case_id=record['case_id'],status=record['status'],
                candidate_miss=record['candidate_miss'],failure=record.get('failure'),failure_counts=counts,
                source_chunk=number,source_path=str(path),source_file=path.name,
                source_sha256=sha256(data).hexdigest(),source_bytes=len(data),evidence_kind=kind,
                output_file=f'case_{index:03d}_{seed}{extension}'))
            part['records']+=1
            del record,data
        parts.append(part)
    _require(seen_indices==set(range(count)),'missing or extra formal record indices; missing='+str(sorted(set(range(count))-seen_indices)))
    _require(len(kinds)==1,'do not mix raw, compact and analysis-lite chunks in one merge')
    rows.sort(key=lambda r:r['index']);status=Counter(r['status'] for r in rows);failures=Counter()
    for row in rows:failures.update(row['failure_counts'])
    manifest=dict(schema='ev_mobile_v2_merge_manifest_v1',physical_bank=bank,phase=phase,
        expected_count=count,case_count=len(rows),unique_seed_count=len(seen_seeds),
        index_range=[0,count-1],policy_id=canonical['policy_id'],source_sha256=canonical['source_sha256'],
        frozen_source_count=len(canonical['source_sha256']),source_snapshot_content_id=snapshot_id,
        evidence_kind=next(iter(kinds)),status_counts=dict(status),failed_record_slots=sum(v for k,v in status.items() if k!='complete'),
        candidate_miss_slots=sum(r['candidate_miss'] for r in rows),failure_counts=dict(failures),
        output_directory=str(output),sources=parts,entries=rows,
        merge_tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        selected_metadata_source=str(directories[0]),hardlinks_require_immutable_sources=True,
        source_snapshot_equality='decoded named source bytes; gzip headers may differ',
        scope='record identity, count and hash merge; no AC replay, statistical calibration or evaluation')
    output.mkdir(parents=True,exist_ok=False)
    try:
        _write_new(output/'freeze.json',canonical_freeze);_write_new(output/'source_snapshot.json.gz',canonical_snapshot)
        for number,part in enumerate(parts):
            for metadata in part['metadata']:
                name=metadata['name'];target=f'source_{number:03d}_{name}'
                data=(directories[number]/name).read_bytes()
                _require(sha256(data).hexdigest()==metadata['sha256'],'source metadata changed during merge')
                _write_new(output/target,data);metadata.update(preserved_file=target,copied_original_bytes=True)
        for row in rows:
            source=Path(row['source_path']);target=output/row['output_file']
            method,reason=_link_or_copy(source,target)
            actual=target.read_bytes();_require(len(actual)==row['source_bytes'] and sha256(actual).hexdigest()==row['source_sha256'],
                'source record changed during link/copy')
            row.update(materialization=method,copy_fallback_reason=reason,output_sha256=sha256(actual).hexdigest())
        manifest['materialization_counts']=dict(Counter(r['materialization'] for r in rows))
        manifest['manifest_id']=_object_hash(manifest)
        _write_new(output/'merge_manifest.json',_encoded(manifest))
        _write_new(output/'checkpoint.json',_encoded(dict(phase=phase,bank=bank,completed=count,planned=count,
            last_index=count-1,status='merged_complete',failed_record_slots=manifest['failed_record_slots'],
            planning_failures=failures['planning_failures'])))
    except Exception as exc:
        _write_new(output/'merge_incomplete.json',_encoded(dict(status='incomplete_merge',
            failure=type(exc).__name__+': '+str(exc),preserve_existing_files=True)))
        raise
    return manifest


def verify(directory,*,bank=None,phase=None,expected_count=None,expected_policy_id=None):
    """Read-only restart check. Case hashes bind the previously checked JSON.

    No large case is deserialized again; per-case bytes/hash, the complete
    source snapshot, original metadata and the exact index/seed schedule are
    rechecked. A partial merge has no complete manifest and is never resumed.
    """
    directory=Path(directory).resolve()
    _require(not (directory/'merge_incomplete.json').exists(),'incomplete merge retained; choose a fresh output')
    manifest,_=_read(directory/'merge_manifest.json')
    _require(manifest.get('schema')=='ev_mobile_v2_merge_manifest_v1','unknown merge manifest schema')
    _require(_object_hash({k:v for k,v in manifest.items() if k!='manifest_id'})==manifest.get('manifest_id'),'merge manifest hash mismatch')
    actual_bank,actual_phase=manifest['physical_bank'],manifest['phase'];count=manifest['expected_count']
    _require((actual_bank,actual_phase) in SEED_START,'unknown merged bank/phase')
    _require(bank is None or bank==actual_bank,'merged physical bank differs')
    _require(phase is None or phase==actual_phase,'merged phase differs')
    _require(expected_count is None or expected_count==count,'merged count differs')
    freeze,freeze_raw,snapshot_raw,content_id=_freeze(directory)
    _require(freeze['policy_id']==manifest['policy_id'] and freeze['source_sha256']==manifest['source_sha256'],'merged freeze identity differs')
    _require(expected_policy_id is None or expected_policy_id==freeze['policy_id'],'merged frozen policy differs')
    _require(content_id==manifest['source_snapshot_content_id'],'merged snapshot contents differ')
    _require(count==freeze['config'][actual_phase+'_histories_per_bank'],'merged count differs from frozen count')
    entries=manifest['entries'];_require(len(entries)==count and {r['index'] for r in entries}==set(range(count)),
        'merged indices missing, duplicate or extra')
    _require(manifest.get('case_count')==count and manifest.get('unique_seed_count')==count,
        'merged declared record/seed denominators differ')
    _require(len({r['seed'] for r in entries})==count,'merged independent seeds duplicate')
    statuses=Counter(r['status'] for r in entries)
    _require(dict(statuses)==manifest['status_counts'] and sum(v for k,v in statuses.items() if k!='complete')==manifest['failed_record_slots'],
        'merged failure/status denominators differ')
    names={r['output_file'] for r in entries}
    files={p.name for pattern in ('case_*','trial_*') for p in directory.glob(pattern) if p.is_file()}
    _require(names==files,'merged record file set differs from manifest')
    for row in entries:
        _require(row['seed']==SEED_START[actual_bank,actual_phase]+row['index'],'merged seed/index schedule differs')
        data=_child(directory,row['output_file']).read_bytes()
        _require(len(data)==row['source_bytes'] and sha256(data).hexdigest()==row['source_sha256']==row['output_sha256'],
            'merged record bytes/hash mismatch')
    for number,part in enumerate(manifest['sources']):
        for metadata in part['metadata']:
            data=_child(directory,metadata['preserved_file']).read_bytes()
            _require(len(data)==metadata['bytes'] and sha256(data).hexdigest()==metadata['sha256'],
                'preserved original metadata bytes/hash mismatch')
        if number==0:
            first={r['name']:r for r in part['metadata']}
            _require(sha256(freeze_raw).hexdigest()==first['freeze.json']['sha256']
                and sha256(snapshot_raw).hexdigest()==first['source_snapshot.json.gz']['sha256'],
                'selected metadata is not the first source original bytes')
    return dict(success=True,physical_bank=actual_bank,phase=actual_phase,case_count=count,
        policy_id=freeze['policy_id'],source_snapshot_content_id=content_id,manifest_id=manifest['manifest_id'],
        evidence_kind=manifest['evidence_kind'],failed_record_slots=manifest['failed_record_slots'],
        failure_counts=manifest['failure_counts'],verification_scope='case bytes/hash plus source/metadata/index/seed identity; no AC replay')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',nargs='+',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--bank',choices=('n3','n4'));parser.add_argument('--phase',choices=('calibration','evaluation'))
    parser.add_argument('--expected-count',type=int);parser.add_argument('--policy-id')
    parser.add_argument('--verify-directory',type=Path)
    args=parser.parse_args()
    if args.verify_directory is not None:
        value=verify(args.verify_directory,bank=args.bank,phase=args.phase,expected_count=args.expected_count,expected_policy_id=args.policy_id)
    else:
        if not all((args.directory,args.output,args.bank,args.phase)):parser.error('merge requires --directory --output --bank --phase')
        value=merge(args.directory,args.output,bank=args.bank,phase=args.phase,
            expected_count=args.expected_count,expected_policy_id=args.policy_id)
    print(json.dumps({k:v for k,v in value.items() if k not in ('sources','entries','source_sha256')},ensure_ascii=False))


if __name__=='__main__':main()
