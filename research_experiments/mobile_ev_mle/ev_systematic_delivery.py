"""Deliver complete inference replay data or explicitly labelled analysis records.

Raw records stay untouched. Compact records retain every top-level field except
selection's large proxy tables/arrays, whose scalar evidence is summarized and
bound to the raw file SHA. Compact delivery does not reconstruct full raw bytes.
Analysis-lite retains the analyzer's statistical projection, including complete
method rows; meters and fitted states remain only in raw/complete compact files.
Only the Python standard library is used; no fit or policy is run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

SCHEMA='systematic_compact_delivery_v1'
ANALYSIS_SCHEMA='systematic_analysis_lite_delivery_v1'
DROP_SCOPE=('selection frozen affine matrices, per-node source-QP tables and witnesses; '
    'planning AC response/Jacobian/source arrays; detailed exchange polynomial witnesses; '
    'complete originals remain in the referenced unchanged raw files')
KEEP_SCOPE=('all non-selection top-level fields exactly preserved, including identity/config/seeds/candidates/truth, '
    'training/validation/future meters and observations, complete training_result/history_prefits/future_records, '
    'all fixed/joint states/residuals/block losses/normalizers/optimizer failures/witnesses, methods and failures')
ANALYSIS_KEEP_SCOPE=('the complete analyzer.project_analysis_record projection: identity/phase/config/source, '
    'candidate/truth labels, complete method rows with inference losses/decisions inputs, operation/power, '
    'planning/deployment/inference timing, parameter errors, failures, common costs and checks')
ANALYSIS_DROP_SCOPE=('meter observations, training/validation/future physical fit records and all fitted '
    'states/residuals/witnesses; forecast and all selection proxy evidence are not included; '
    'unchanged raw and separate complete compact records retain that evidence')
METADATA=('freeze.json','source_snapshot.json.gz','manifest.json','checkpoint.json')
LARGE_KEYS={'tables','background','nodes','witnesses','evaluations','fit_results','frozen_pairs',
    'context','model','models','scenario_models','selected_operation_x','operation_x','x',
    'B','A0','At','G','state_jacobian','jac','residual','reference_values','competitor_values'}
PROFILE_KEYS={'profile','prediction','selected_profile','selected_prediction','raw_profile','raw_prediction',
    'gradient_profile','candidate_profile','candidate_prediction'}
EXACT_SELECTION_KEYS={'planning_actual_seconds','dependencies','planning_seconds','setup_seconds',
    'context_seconds','common_cost','power_kw','selected_power_kw','power','selected_power'}


def _encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')


def _object_hash(value):return hashlib.sha256(_encoded(value)).hexdigest()
def _bytes_hash(value):return hashlib.sha256(value).hexdigest()
def _read(path):
    path=Path(path);data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def _write_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(data)


def _write_json_new(path,value):
    data=_encoded(value)
    if Path(path).suffix=='.gz':data=gzip.compress(data,compresslevel=9,mtime=0)
    _write_new(path,data)


def _retained(record):return {k:v for k,v in record.items() if k not in ('selection','delivery_provenance')}


def _child(directory,name):
    if not isinstance(name,str) or Path(name).name!=name:raise ValueError('mapping filename must be a basename')
    path=Path(directory)/name
    if path.resolve().parent!=Path(directory).resolve() or path.is_symlink():
        raise ValueError('mapping escapes delivery directory or refers to a symlink')
    return path


def _small(value,max_bytes=12000):
    try:return len(_encoded(value))<=max_bytes
    except (ValueError,TypeError):return False


def _failure_evidence(value,path='selection'):
    """Keep every reported failure/rejection/stop reason with its original path."""
    result=[]
    if isinstance(value,dict):
        if value.get('success') is False or value.get('feasible') is False:
            result.append(dict(path=path,kind='unsuccessful_record',
                details={k:deepcopy(value[k]) for k in ('success','feasible','status','message','failure','failures')
                    if k in value}))
        for key,child in value.items():
            target=path+'/'+str(key)
            if ('failure' in key or key in ('rejection_reason','stop_reason')) and child not in (None,False,[],{}):
                result.append(dict(path=target,kind='reported_reason',value=deepcopy(child)))
            elif isinstance(child,(dict,list)):
                result.extend(_failure_evidence(child,target))
    elif isinstance(value,list):
        for index,child in enumerate(value):
            if isinstance(child,(dict,list)):result.extend(_failure_evidence(child,path+'/'+str(index)))
    return result


def _profile_summary(value):
    if value is None:return None
    if not isinstance(value,dict):return deepcopy(value) if _small(value) else dict(omitted=True)
    keep={k:deepcopy(v) for k,v in value.items() if k not in
        ('pairs','fit_results','reference_values','competitor_values','residual','gradient_per_kw')
        and k not in LARGE_KEYS and _small(v,3000)}
    if isinstance(value.get('pairs'),list):
        pairs=[]
        fields=('a','b','name','success','half_SSE','added','joint','primal_upper','tangent_lower_numeric',
            'q_primal_upper','q_tangent_lower','numerical_gap','optimality','status','message','failure',
            'seconds','source_kkt_violation','scaled_source_kkt_violation','terminal_voltage_margin','total_nfev')
        for pair in value['pairs']:
            pairs.append({k:deepcopy(pair[k]) for k in fields if isinstance(pair,dict) and k in pair})
        keep.update(pair_count=len(pairs),pairs=pairs)
    keep['planning_arrays_omitted']=True
    return keep


def _selection_item(value,depth=0):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,list):
        return [_selection_item(v,depth+1) for v in value] if len(value)<=200 else dict(
            omitted=True,item_count=len(value),scope='full list retained in raw file')
    if not isinstance(value,dict):raise ValueError('selection must contain JSON-compatible data')
    result={};omitted=[]
    for key,child in value.items():
        if key in EXACT_SELECTION_KEYS:result[key]=deepcopy(child)
        elif key in PROFILE_KEYS or key=='action_QP':result[key]=_profile_summary(child)
        elif key in LARGE_KEYS:omitted.append(key)
        elif key in ('grids','iterations','initial_candidates','initials'):
            result[key]=_selection_item(child,depth+1)
        elif key in ('raw','exchange','MILP','table'):
            result[key]=_selection_item(child,depth+1)
        elif isinstance(child,(str,int,float,bool)) or child is None:result[key]=child
        elif _small(child):result[key]=deepcopy(child)
        elif isinstance(child,dict) and depth<6:result[key]=_selection_item(child,depth+1)
        else:omitted.append(key)
    if omitted:result['delivery_omitted_keys']=sorted(omitted)
    return result


def compact_record(record,*,raw_name,raw_sha256,raw_bytes,raw_path=None,tool_sha256=None):
    if 'delivery_provenance' in record:raise ValueError('input is already a compact delivery record')
    result=dict(record)
    if 'selection' in record:
        selection=record['selection']
        if selection is not None and not isinstance(selection,dict):raise ValueError('selection must be a mapping or None')
        result['selection']=_selection_item(selection) if selection is not None else dict(source_selection_available=False)
        result['selection']['delivery_failure_evidence']=_failure_evidence(selection)
        if isinstance(selection,dict) and 'frozen_pairs' in selection:
            result['selection']['frozen_pair_summaries']=[dict(name=row.get('name'),
                residual_rows=len(row.get('c',[])),control_columns=len(row.get('B',[[]])[0]) if row.get('B') else 0,
                source_columns=len(row.get('A0',[[]])[0]) if row.get('A0') else 0,
                full_pair_payload_sha256=_object_hash(row)) for row in selection['frozen_pairs']]
    result['delivery_provenance']=dict(schema=SCHEMA,raw_file=raw_name,raw_sha256=raw_sha256,
        raw_bytes=int(raw_bytes),raw_path=raw_path,tool_sha256=tool_sha256,
        retained_payload_sha256=_object_hash(_retained(record)),preserved_scope=KEEP_SCOPE,
        summarized_scope=DROP_SCOPE,full_raw_byte_reconstruction_supported=False)
    if _retained(result)!=_retained(record):raise AssertionError('inference replay fields changed during compaction')
    return result


def analysis_record(record,*,raw_name,raw_sha256,raw_bytes,raw_path=None,tool_sha256=None):
    """Project statistical data only; this file is not complete physical evidence."""
    from . import ev_systematic_analyze as analyzer
    if 'delivery_provenance' in record:raise ValueError('input is already a compact or analysis-lite delivery record')
    analyzer._validate_record(record,None)
    result=deepcopy(analyzer.project_analysis_record(record))
    result['delivery_provenance']=dict(schema=ANALYSIS_SCHEMA,raw_file=raw_name,raw_sha256=raw_sha256,
        raw_bytes=int(raw_bytes),raw_path=raw_path,tool_sha256=tool_sha256,phase=record.get('phase'),
        source_id=record.get('source_id',record.get('checks',{}).get('source_id')),
        retained_payload_sha256=_object_hash(_retained(result)),preserved_scope=ANALYSIS_KEEP_SCOPE,
        summarized_scope=ANALYSIS_DROP_SCOPE,full_meter_and_fit_evidence_included=False,
        analysis_only=True,full_raw_byte_reconstruction_supported=False)
    return result


def _deliver(directory,output,*,analysis_only=False):
    directory,output=Path(directory).resolve(),Path(output).resolve()
    if directory==output:raise ValueError('compact output must differ from raw directory')
    if output.exists() and any(output.iterdir()):raise FileExistsError('output directory is not empty; existing files retained')
    paths=sorted({p for pattern in ('case_*.json','case_*.json.gz','trial_*.json','trial_*.json.gz')
        for p in directory.glob(pattern)})
    if not paths:raise ValueError('no raw case/trial files found')
    output.mkdir(parents=True,exist_ok=True);tool=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    schema=ANALYSIS_SCHEMA if analysis_only else SCHEMA
    keep_scope=ANALYSIS_KEEP_SCOPE if analysis_only else KEEP_SCOPE
    drop_scope=ANALYSIS_DROP_SCOPE if analysis_only else DROP_SCOPE
    manifest=dict(schema=schema,raw_directory=str(directory),compact_directory=str(output),tool_sha256=tool,
        tool_file=Path(__file__).name,preserved_scope=keep_scope,summarized_scope=drop_scope,
        full_raw_byte_reconstruction_supported=False,entries=[],metadata=[])
    if analysis_only:manifest.update(analysis_only=True,full_meter_and_fit_evidence_included=False)
    for name in METADATA:
        source=directory/name
        if source.exists():
            data=source.read_bytes();_write_new(output/name,data)
            manifest['metadata'].append(dict(name=name,sha256=_bytes_hash(data),bytes=len(data),copied_original_bytes=True))
    for path in paths:
        data=path.read_bytes();record=json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)
        raw_hash=_bytes_hash(data)
        projector=analysis_record if analysis_only else compact_record
        packed=projector(record,raw_name=path.name,raw_sha256=raw_hash,raw_bytes=len(data),
            raw_path=str(path),tool_sha256=tool)
        target=output/path.name;_write_json_new(target,packed);compact_data=target.read_bytes()
        manifest['entries'].append(dict(raw_file=path.name,raw_sha256=raw_hash,raw_bytes=len(data),
            compact_file=target.name,compact_sha256=_bytes_hash(compact_data),compact_bytes=len(compact_data),
            retained_payload_sha256=packed['delivery_provenance']['retained_payload_sha256']))
    manifest.update(raw_case_bytes=sum(r['raw_bytes'] for r in manifest['entries']),
        compact_case_bytes=sum(r['compact_bytes'] for r in manifest['entries']),case_count=len(paths))
    manifest['case_byte_ratio']=manifest['compact_case_bytes']/manifest['raw_case_bytes']
    manifest['manifest_id']=_object_hash(manifest)
    _write_json_new(output/'delivery_manifest.json',manifest)
    verify(output,raw_directory=directory)
    return manifest


def compact(directory,output):
    return _deliver(directory,output)


def analysis(directory,output):
    return _deliver(directory,output,analysis_only=True)


def verify(directory,*,raw_directory=None):
    directory=Path(directory).resolve();manifest=_read(directory/'delivery_manifest.json')
    if manifest.get('schema') not in (SCHEMA,ANALYSIS_SCHEMA):raise ValueError('unknown delivery schema')
    analysis_only=manifest['schema']==ANALYSIS_SCHEMA
    if _object_hash({k:v for k,v in manifest.items() if k!='manifest_id'})!=manifest.get('manifest_id'):
        raise ValueError('delivery manifest hash mismatch')
    checked=0;retained=0
    for row in manifest['metadata']:
        data=_child(directory,row['name']).read_bytes()
        if len(data)!=row['bytes'] or _bytes_hash(data)!=row['sha256']:raise ValueError('metadata bytes/hash mismatch')
    for row in manifest['entries']:
        path=_child(directory,row['compact_file'])
        data=path.read_bytes()
        if len(data)!=row['compact_bytes'] or _bytes_hash(data)!=row['compact_sha256']:
            raise ValueError('compact file bytes/hash mismatch')
        record=_read(path);provenance=record.get('delivery_provenance',{})
        if provenance.get('schema')!=manifest['schema']:raise ValueError('record/manifest delivery schema mismatch')
        if analysis_only and (provenance.get('analysis_only') is not True or
                provenance.get('full_meter_and_fit_evidence_included') is not False):
            raise ValueError('analysis-lite evidence scope is missing')
        if any(provenance.get(k)!=row.get(k) for k in ('raw_sha256','raw_bytes','retained_payload_sha256')):
            raise ValueError('raw reference/mapping mismatch')
        if provenance.get('raw_file')!=row['raw_file'] or provenance.get('tool_sha256')!=manifest['tool_sha256']:
            raise ValueError('delivery provenance differs from manifest')
        if _object_hash(_retained(record))!=row['retained_payload_sha256']:
            raise ValueError('retained delivery payload hash mismatch')
        if raw_directory is not None:
            raw_path=_child(Path(raw_directory),row['raw_file']);raw_data=raw_path.read_bytes()
            if len(raw_data)!=row['raw_bytes'] or _bytes_hash(raw_data)!=row['raw_sha256']:
                raise ValueError('original raw file bytes/hash mismatch')
            original=json.loads(gzip.decompress(raw_data) if raw_path.suffix=='.gz' else raw_data)
            if analysis_only:
                from . import ev_systematic_analyze as analyzer
                original=analyzer.project_analysis_record(original)
            if _retained(record)!=_retained(original):raise ValueError('retained raw/projected fields differ')
            retained+=1
        checked+=1
    return dict(success=True,checked_cases=checked,exact_retained_payload_cases=retained,
        checked_metadata=len(manifest['metadata']),manifest_id=manifest['manifest_id'],
        analysis_only=analysis_only,full_meter_and_fit_evidence_included=not analysis_only,
        full_raw_byte_reconstruction_supported=False)


def bundle(directory,output,*,max_bytes=20*1024*1024):
    directory,output=Path(directory).resolve(),Path(output).resolve();verify(directory)
    if max_bytes<2048:raise ValueError('archive limit must be at least 2048 bytes')
    if output.exists() and any(output.iterdir()):raise FileExistsError('bundle output is not empty')
    delivery=_read(directory/'delivery_manifest.json')
    names={row['compact_file'] for row in delivery['entries']}|{row['name'] for row in delivery['metadata']}|{'delivery_manifest.json'}
    files=sorted(_child(directory,name) for name in names)
    if any(path.is_symlink() for path in files):raise ValueError('symlink inputs are unsupported')
    output.mkdir(parents=True,exist_ok=True);groups=[];current=[];size=0
    for path in files:
        estimate=path.stat().st_size+2048
        if current and size+estimate+16384>max_bytes:groups.append(current);current=[];size=0
        current.append(path);size+=estimate
    if current:groups.append(current)
    archives=[]
    while groups:
        group=groups.pop(0);name=f'compact_part_{len(archives)+1:03d}.tar.xz';target=output/name
        # Build in memory so a file-size check precedes every immutable write.
        import io
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:xz',preset=6) as tar:
            for path in group:tar.add(path,arcname=path.name,recursive=False)
        data=buffer.getvalue()
        if len(data)>max_bytes:
            if len(group)==1:raise ValueError('one compact file alone exceeds archive limit: '+group[0].name)
            split=len(group)//2;groups=[group[:split],group[split:],*groups];continue
        _write_new(target,data)
        archives.append(dict(file=name,bytes=len(data),sha256=_bytes_hash(data),
            members=[dict(file=p.name,bytes=p.stat().st_size,sha256=_bytes_hash(p.read_bytes())) for p in group]))
    analysis_only=delivery['schema']==ANALYSIS_SCHEMA
    manifest=dict(schema='systematic_analysis_lite_bundle_v1' if analysis_only else 'systematic_compact_bundle_v1',
        max_bytes=int(max_bytes),archives=archives,analysis_only=analysis_only,
        delivery_manifest_sha256=_bytes_hash((directory/'delivery_manifest.json').read_bytes()),
        restore_scope=('archives reproduce analysis-lite files only; meters and all fitted states are absent'
            if analysis_only else 'archives reproduce compact files only; full raw records are separately retained'))
    manifest['manifest_id']=_object_hash(manifest);_write_json_new(output/'bundle_manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='mode',required=True)
    for name in ('compact','analysis','bundle'):
        sub=commands.add_parser(name);sub.add_argument('--directory',type=Path,required=True);sub.add_argument('--output',type=Path,required=True)
        if name=='bundle':sub.add_argument('--max-mb',type=float,default=20.)
    sub=commands.add_parser('verify');sub.add_argument('--directory',type=Path,required=True);sub.add_argument('--raw-directory',type=Path)
    args=parser.parse_args()
    if args.mode=='compact':result=compact(args.directory,args.output)
    elif args.mode=='analysis':result=analysis(args.directory,args.output)
    elif args.mode=='verify':result=verify(args.directory,raw_directory=args.raw_directory)
    else:result=bundle(args.directory,args.output,max_bytes=int(args.max_mb*1024*1024))
    print(json.dumps({k:v for k,v in result.items() if k not in ('entries','metadata','archives')},ensure_ascii=False))


if __name__=='__main__':main()
