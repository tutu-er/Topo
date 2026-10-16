"""Compact v4 evidence while preserving all realized AC/operation records.

Only repeated predictive-fit intermediates are summarized, with original
content hashes and full failed nodes retained.  Exact content-addressed pooling
then deduplicates the remaining record.  Restore supports the saved-record
analysis and physical auditor; it does not reconstruct omitted intermediates.
No parameter fit, optimizer or case generator is invoked by this module.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path


SCHEMA='mobile_v4_compact_evidence_v1'
REFERENCE_KEY='__v4_evidence_ref__'
FIT_INTERMEDIATES={'blocks','optimizer','optimizer_recomputed_score','feasible_witness'}
SUMMARY_FIELDS=('loss','normalizer','loglik','success','usable','status','message',
    'nfev','optimality','diagnostics','state','selected_origin','failure',
    'line_domain_feasible','source_box_feasible','physical_voltage_feasible')


def _bytes(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),
                      allow_nan=False).encode('utf-8')


def _hash(value):
    return sha256(_bytes(value)).hexdigest()


def _read(path):
    path=Path(path);payload=path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix=='.gz' else payload)


def _write(path,value):
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=_bytes(value)+b'\n'
    path.write_bytes(gzip.compress(payload,mtime=0) if path.suffix=='.gz' else payload)


def _path(tokens):
    return '/'+ '/'.join(str(token).replace('~','~0').replace('/','~1') for token in tokens)


def _failure_nodes(value,tokens):
    if isinstance(value,dict):
        failed=(value.get('success') is False or value.get('solver_success') is False
            or value.get('optimization_failure') is True or bool(value.get('optimizer_error'))
            or bool(value.get('failure')) or bool(value.get('failures')))
        if failed:
            return [dict(path=_path(tokens),complete_failed_node=deepcopy(value))]
        return [row for key,item in value.items() for row in _failure_nodes(item,tokens+(key,))]
    if isinstance(value,list):
        return [row for i,item in enumerate(value) for row in _failure_nodes(item,tokens+(i,))]
    return []


def _predictive_profiles(body):
    for family in ('aligned','guarded'):
        for key,profile in body.get('families',{}).get(family,{}).get('cache',{}).items():
            yield ('families',family,'cache',key),profile
    for key,profile in body.get('reporting_cache',{}).items():
        yield ('reporting_cache',key),profile


def compact_record(record,*,raw_sha256=None,raw_path=None,raw_bytes=None,pool_min_bytes=4096):
    """Keep final data exactly; summarize only four predictive-fit fields."""
    if not isinstance(record,dict) or pool_min_bytes<1:
        raise ValueError('record dictionary and positive pool threshold required')
    body=deepcopy(record);omissions=[];failures=[]
    for tokens,profile in _predictive_profiles(body):
        for i,reference in enumerate(profile.get('references',[])):
            for name,fit in reference.get('fits',{}).items():
                for field in sorted(FIT_INTERMEDIATES & set(fit)):
                    original=fit[field]
                    where=tokens+('references',i,'fits',name,field)
                    summary=({k:deepcopy(v) for k,v in original.items() if k in SUMMARY_FIELDS}
                             if isinstance(original,dict) and field!='blocks' else None)
                    omissions.append(dict(path=_path(where),path_tokens=list(where),
                        reason='repeated predictive explanatory-fit intermediate; final actual fits untouched',
                        content_sha256=_hash(original),canonical_bytes=len(_bytes(original)),
                        retained_summary_sha256=None if summary is None else _hash(summary)))
                    failures.extend(_failure_nodes(original,where))
                    if summary is None:
                        del fit[field]
                    else:
                        fit[field]=summary
    objects={}
    def intern(value):
        if isinstance(value,dict):
            if REFERENCE_KEY in value:
                raise ValueError('reserved evidence reference marker occurs in original data')
            encoded={key:intern(item) for key,item in value.items()}
        elif isinstance(value,list):
            encoded=[intern(item) for item in value]
        else:
            return value
        payload=_bytes(encoded)
        if len(payload)<pool_min_bytes:
            return encoded
        key=sha256(payload).hexdigest();objects.setdefault(key,encoded)
        return {REFERENCE_KEY:key}
    record_node=intern(body);failure_node=intern(failures)
    result=dict(schema=SCHEMA,record=record_node,objects=objects,omissions=omissions,
        failure_evidence=failure_node,source_record=dict(raw_path=raw_path,raw_sha256=raw_sha256,
            raw_compressed_bytes=raw_bytes,raw_object_sha256=_hash(record)),
        pool_min_bytes=int(pool_min_bytes),
        scope='complete observed/training/forecast/operation/realized AC evidence; predictive intermediates summarized',
        complete_raw_reconstruction=False)
    result['payload_sha256']=_hash(result)
    return result


def restore(compact):
    if compact.get('schema')!=SCHEMA or compact.get('payload_sha256')!=_hash(
            {k:v for k,v in compact.items() if k!='payload_sha256'}):
        raise ValueError('compact schema or payload hash mismatch')
    objects=compact['objects'];used=set();active=set()
    def expand(value):
        if isinstance(value,dict):
            if set(value)=={REFERENCE_KEY}:
                key=value[REFERENCE_KEY]
                if key in active or key not in objects:
                    raise ValueError('cyclic or missing content-addressed evidence')
                node=objects[key]
                if _hash(node)!=key:
                    raise ValueError('content-addressed evidence hash mismatch')
                used.add(key);active.add(key)
                result=expand(node);active.remove(key)
                return result
            return {key:expand(item) for key,item in value.items()}
        if isinstance(value,list):
            return [expand(item) for item in value]
        return value
    result=expand(compact['record']);failures=expand(compact['failure_evidence'])
    if used!=set(objects):
        raise ValueError('unreferenced evidence object')
    result['_evidence_provenance']=dict(schema=SCHEMA,source_record=compact['source_record'],
        omissions=compact['omissions'],failure_evidence=failures,
        complete_raw_reconstruction=False,scope=compact['scope'])
    return result


def validate_record(original,compact,*,physical_replay=True):
    from . import ev_mobile_v4_analysis as analysis
    restored=restore(compact)
    if compact['source_record']['raw_object_sha256']!=_hash(original):
        raise ValueError('source object differs from compact evidence provenance')
    for omitted in compact['omissions']:
        value=original
        for key in omitted['path_tokens']:
            value=value[key]
        if _hash(value)!=omitted['content_sha256']:
            raise ValueError('omission content hash differs from original')
    spec=original['spec'];index=spec['index']
    a=analysis.summarize_records([spec],{index:original})
    b=analysis.summarize_records([spec],{index:restored})
    exact_regions={name:_hash(original.get(name))==_hash(restored.get(name)) for name in
        ('actual','observed','observed_meter','future_meter','training_result','forecast',
         'operation_config','methods')}
    result=dict(analysis_equal=a==b,analysis_original_sha256=_hash(a),analysis_restored_sha256=_hash(b),
        exact_final_regions=exact_regions,omission_count=len(compact['omissions']),
        retained_failed_nodes=len(restored['_evidence_provenance']['failure_evidence']),
        complete_raw_reconstruction=False)
    if physical_replay:
        from . import ev_mobile_v4_audit as audit
        original_audit=audit.audit_record(original);restored_audit=audit.audit_record(restored)
        # This is the auditor's newly measured runtime, not a scientific output.
        stable=lambda value:{k:v for k,v in value.items() if k!='seconds'}
        result.update(physical_audit_equal=stable(original_audit)==stable(restored_audit),
            original_physical_success=original_audit['success'],
            restored_physical_success=restored_audit['success'],restored_physical_audit=restored_audit)
    result['success']=result['analysis_equal'] and all(exact_regions.values()) and result.get('physical_audit_equal',True)
    return result


def compact_file(path,output,*,physical_replay=True):
    path,output=Path(path),Path(output);payload=path.read_bytes();original=_read(path)
    compact=compact_record(original,raw_sha256=sha256(payload).hexdigest(),
        raw_path=str(path.resolve()),raw_bytes=len(payload))
    validation=validate_record(original,compact,physical_replay=physical_replay)
    if not validation['success']:
        raise ValueError('compact restoration changed analysis or physical audit')
    _write(output,compact)
    validation.update(input_path=str(path.resolve()),input_sha256=sha256(payload).hexdigest(),
        output_path=str(output.resolve()),output_sha256=sha256(output.read_bytes()).hexdigest(),
        raw_gzip_bytes=len(payload),compact_gzip_bytes=output.stat().st_size,
        object_count=len(compact['objects']),tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    return validation


def pack(directory,output,*,phases=('development','evaluation'),attachments=()):
    """Package complete requested phases, checking every source snapshot byte."""
    directory,output=Path(directory),Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not phases or len(set(phases))!=len(phases) or not set(phases)<={'development','evaluation'}:
        raise ValueError('distinct declared phases required')
    frozen=_read(directory/'freeze.json');snapshot=_read(directory/'source_snapshot.json.gz')
    if snapshot.get('source_sha256')!=frozen['source_sha256'] or set(snapshot['source_base64'])!=set(frozen['source_sha256']):
        raise ValueError('scientific source snapshot coverage differs from freeze')
    for name,expected in frozen['source_sha256'].items():
        if sha256(base64.b64decode(snapshot['source_base64'][name],validate=True)).hexdigest()!=expected:
            raise ValueError('scientific source snapshot hash mismatch: '+name)
    selected=[]
    for phase in phases:
        expected={f"{spec['index']:03d}.json.gz":spec for spec in frozen['specifications'][phase]}
        actual={path.name:path for path in (directory/phase).glob('*.json.gz')}
        if set(actual)!=set(expected):
            raise ValueError('requested phase not complete: '+phase)
        for name,spec in sorted(expected.items()):
            raw=_read(actual[name])
            if raw.get('policy_id')!=frozen['policy_id'] or raw['spec']!=spec:
                raise ValueError('record identity differs from frozen phase')
            selected.append((phase,actual[name]))
    output.mkdir(parents=True);files=[];records=[]
    def copy(path,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        payload=path.read_bytes();target.write_bytes(payload)
        files.append(dict(file=str(target.relative_to(output)),bytes=len(payload),sha256=sha256(payload).hexdigest()))
    for name in ('freeze.json','source_snapshot.json.gz'):
        copy(directory/name,output/name)
    for phase,path in selected:
        check=compact_file(path,output/phase/path.name,physical_replay=True)
        records.append(check)
        print('COMPACT',phase,path.name,check['raw_gzip_bytes'],check['compact_gzip_bytes'],flush=True)
    for attachment in attachments:
        path=Path(attachment)
        copy(path,output/'attachments'/path.name)
    manifest=dict(schema=SCHEMA,status='complete',policy_id=frozen['policy_id'],phases=list(phases),
        source_count=len(frozen['source_sha256']),source_snapshot_verified=True,
        raw_gzip_bytes=sum(row['raw_gzip_bytes'] for row in records),
        compact_gzip_bytes=sum(row['compact_gzip_bytes'] for row in records),
        record_count=len(records),records=records,files=files,
        all_restoration_checks_passed=all(row['success'] for row in records),
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    _write(output/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    one=sub.add_parser('compact');one.add_argument('--input',required=True);one.add_argument('--output',required=True)
    one.add_argument('--skip-physical-replay',action='store_true')
    many=sub.add_parser('pack');many.add_argument('--directory',required=True);many.add_argument('--output',required=True)
    many.add_argument('--phases',default='development,evaluation');many.add_argument('--attach',action='append',default=[])
    args=parser.parse_args()
    if args.mode=='compact':
        result=compact_file(args.input,args.output,physical_replay=not args.skip_physical_replay)
        _write(Path(args.output).with_name(Path(args.output).stem+'.validation.json'),result)
    else:
        result=pack(args.directory,args.output,phases=tuple(args.phases.split(',')),attachments=args.attach)
    print(json.dumps({k:result.get(k) for k in ('success','all_restoration_checks_passed',
        'raw_gzip_bytes','compact_gzip_bytes','record_count')}),flush=True)


if __name__=='__main__':
    main()
