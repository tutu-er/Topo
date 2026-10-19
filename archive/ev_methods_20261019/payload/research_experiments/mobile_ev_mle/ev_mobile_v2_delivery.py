"""Eleven-method validation over the established immutable delivery format.

The old compactor, hash verifier and archive builder are reused without edits.
Only method validation and analysis projection dispatch differ.  Packaging
schemas remain compatible; manifests explicitly declare mobile-v2 and eleven
methods. Analysis-lite excludes meters and all fitted states. Full compact
retains every non-selection field exactly, while planning arrays remain raw.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path

from . import ev_systematic_delivery as base
from . import ev_systematic_analyze as analyzer


METHODS=('none','constant','highlow','square_fixed','square_mobile','scalar_fixed',
    'scalar_mobile','cap_fixed','cap_mobile','jointgrad_fixed','jointgrad_mobile')
MODES=('fixed_total','fixed_holdout','joint_total')
SCHEMA=base.SCHEMA
ANALYSIS_SCHEMA=base.ANALYSIS_SCHEMA
FAMILY='mobile_reliability_EV_v2'


def _validate(record,methods=METHODS):
    required=('case_id','seed','phase','true_name','candidate_miss','status','methods')
    if any(k not in record for k in required):raise ValueError('mobile record missing required fields')
    if record.get('source_unchanged') is False:raise ValueError('record reports scientific source drift')
    actual=[r.get('name') for r in record['methods']]
    if len(actual)!=len(methods) or len(set(actual))!=len(actual) or set(actual)!=set(methods):
        raise ValueError('record must retain the eleven declared mobile-v2 methods')
    for method in record['methods']:
        if set(method.get('inference',{}))!=set(MODES):raise ValueError('all three inference modes required')
        for mode in method['inference'].values():
            if not isinstance(mode.get('losses'),dict) or not isinstance(mode.get('complete'),bool):
                raise ValueError('losses mapping and explicit completeness bool required')
            value=mode.get('M')
            if value is not None and (not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value)):
                raise ValueError('margin must be finite or null')


def project_analysis_record(record):
    """Use the same small projection consumed by ev_reliability_v2."""
    _validate(record)
    return deepcopy(analyzer.project_analysis_record(record))


def analysis_record(record,*,raw_name,raw_sha256,raw_bytes,raw_path=None,tool_sha256=None):
    if 'delivery_provenance' in record:raise ValueError('input is already a compact or analysis-lite delivery record')
    result=project_analysis_record(record)
    result['delivery_provenance']=dict(schema=ANALYSIS_SCHEMA,raw_file=raw_name,
        raw_sha256=raw_sha256,raw_bytes=int(raw_bytes),raw_path=raw_path,tool_sha256=tool_sha256,
        phase=record.get('phase'),source_id=record.get('source_id',record.get('checks',{}).get('source_id')),
        physical_bank=record.get('physical_bank'),experiment_family=FAMILY,declared_methods=list(METHODS),
        retained_payload_sha256=base._object_hash(base._retained(result)),
        preserved_scope=base.ANALYSIS_KEEP_SCOPE,summarized_scope=base.ANALYSIS_DROP_SCOPE,
        full_meter_and_fit_evidence_included=False,analysis_only=True,
        full_raw_byte_reconstruction_supported=False)
    return result


def _deliver(directory,output,*,analysis_only=False):
    directory,output=Path(directory).resolve(),Path(output).resolve()
    if directory==output:raise ValueError('delivery must differ from raw directory')
    if output.exists() and any(output.iterdir()):raise FileExistsError('existing delivery directory retained')
    paths=sorted({p for pattern in ('case_*.json','case_*.json.gz','trial_*.json','trial_*.json.gz') for p in directory.glob(pattern)})
    if not paths:raise ValueError('no raw case/trial records')
    output.mkdir(parents=True,exist_ok=True);tool=sha256(Path(__file__).read_bytes()).hexdigest()
    legacy=sha256(Path(base.__file__).read_bytes()).hexdigest();schema=ANALYSIS_SCHEMA if analysis_only else SCHEMA
    manifest=dict(schema=schema,experiment_family=FAMILY,declared_methods=list(METHODS),
        raw_directory=str(directory),compact_directory=str(output),tool_sha256=tool,tool_file=Path(__file__).name,
        reused_compaction_tool_sha256=legacy,preserved_scope=base.ANALYSIS_KEEP_SCOPE if analysis_only else base.KEEP_SCOPE,
        summarized_scope=base.ANALYSIS_DROP_SCOPE if analysis_only else base.DROP_SCOPE,
        full_raw_byte_reconstruction_supported=False,entries=[],metadata=[])
    if analysis_only:manifest.update(analysis_only=True,full_meter_and_fit_evidence_included=False)
    for name in base.METADATA:
        path=directory/name
        if not path.exists():continue
        data=path.read_bytes();base._write_new(output/name,data)
        manifest['metadata'].append(dict(name=name,sha256=base._bytes_hash(data),bytes=len(data),copied_original_bytes=True))
        if name=='freeze.json':
            declared=json.loads(data).get('config',{}).get('methods')
            if declared is not None and tuple(declared)!=METHODS:raise ValueError('freeze declares different mobile methods')
    banks=set()
    for path in paths:
        data=path.read_bytes();record=json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)
        _validate(record)
        if 'delivery_provenance' in record:raise ValueError('input is already a compact or analysis-lite delivery record')
        options=dict(raw_name=path.name,raw_sha256=base._bytes_hash(data),raw_bytes=len(data),raw_path=str(path),tool_sha256=tool)
        packed=analysis_record(record,**options) if analysis_only else base.compact_record(record,**options)
        packed['delivery_provenance'].update(experiment_family=FAMILY,declared_methods=list(METHODS),
            physical_bank=record.get('physical_bank'),reused_compaction_tool_sha256=legacy)
        if record.get('physical_bank') is not None:banks.add(record['physical_bank'])
        target=output/path.name;base._write_json_new(target,packed);saved=target.read_bytes()
        manifest['entries'].append(dict(raw_file=path.name,raw_sha256=options['raw_sha256'],raw_bytes=len(data),
            compact_file=path.name,compact_sha256=base._bytes_hash(saved),compact_bytes=len(saved),
            retained_payload_sha256=packed['delivery_provenance']['retained_payload_sha256']))
    manifest.update(physical_banks=sorted(banks),case_count=len(paths),
        raw_case_bytes=sum(r['raw_bytes'] for r in manifest['entries']),
        compact_case_bytes=sum(r['compact_bytes'] for r in manifest['entries']))
    manifest['case_byte_ratio']=manifest['compact_case_bytes']/manifest['raw_case_bytes']
    manifest['manifest_id']=base._object_hash(manifest);base._write_json_new(output/'delivery_manifest.json',manifest)
    verify(output,raw_directory=directory)
    return manifest


def analysis(directory,output):return _deliver(directory,output,analysis_only=True)
def compact(directory,output):return _deliver(directory,output)


def verify(directory,*,raw_directory=None):
    directory=Path(directory);manifest=base._read(directory/'delivery_manifest.json')
    if manifest.get('experiment_family')!=FAMILY or tuple(manifest.get('declared_methods',()))!=METHODS:
        raise ValueError('mobile-v2 delivery method/family declaration differs')
    for row in manifest['entries']:
        record=base._read(base._child(directory,row['compact_file']));_validate(record)
        provenance=record.get('delivery_provenance',{})
        if provenance.get('experiment_family')!=FAMILY or tuple(provenance.get('declared_methods',()))!=METHODS:
            raise ValueError('mobile-v2 record method/family declaration differs')
        if raw_directory is not None:
            raw=base._read(base._child(Path(raw_directory),row['raw_file']));_validate(raw)
            if provenance.get('physical_bank')!=raw.get('physical_bank'):raise ValueError('raw physical-bank label differs')
    return dict(base.verify(directory,raw_directory=raw_directory),declared_methods=list(METHODS),
        experiment_family=FAMILY,method_count=len(METHODS))


def bundle(directory,output,*,max_bytes=20*1024*1024):
    verify(directory)
    return base.bundle(directory,output,max_bytes=max_bytes)


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
