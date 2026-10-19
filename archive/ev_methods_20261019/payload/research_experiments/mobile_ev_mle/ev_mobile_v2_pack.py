"""Package the declared EV-v2 evidence after every computation has finished.

This is a sequence of existing delivery calls, not a new experiment framework.
All raw paths and temporary work stay below --raw-root (the D-stage outputs).
The requested evidence directory contains bundles/manifests only. Full raw
gzip files are split as exact bytes when necessary; no fields are discarded.
--plan prints the declarations without inspecting any data directory.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from . import ev_mobile_v2_delivery as delivery
from . import ev_mobile_v2_merge as merging


POLICY='35e7a11ba65a5366930888272f516c3165355f5dbe20de69919734dfd45e23e3'
MAX_BYTES=20*1024*1024
PLAN=(
    ('mobile_v2_fullpilot_n3','n3','pilot',tuple(range(3))),
    ('mobile_v2_fullpilot_n4','n4','pilot',(0,)),
    ('mobile_v2_dev_n3_part0','n3','development',tuple(range(10))),
    ('mobile_v2_dev_n3_part1','n3','development',tuple(range(10,20))),
    ('mobile_v2_dev_n4','n4','development',tuple(range(8))),
    ('mobile_v2_guard_dev_n3_part0','n3','development',tuple(range(10))),
    ('mobile_v2_guard_dev_n3_part1','n3','development',tuple(range(10,20))),
    ('mobile_v2_guard_dev_n4','n4','development',tuple(range(8))),
    ('mobile_v2_guard60_cost_n3','n3','development',(8,)),
    ('mobile_v2_guard60_cost_n4','n4','development',(0,5,7)),
    ('mobile_v2_formal_cal_n3','n3','calibration',tuple(range(96))),
    ('mobile_v2_formal_cal_n4','n4','calibration',tuple(range(96))),
    ('mobile_v2_formal_eval_n3','n3','evaluation',tuple(range(48))),
    ('mobile_v2_formal_eval_n4','n4','evaluation',tuple(range(48))))


def _require(ok,message):
    if not ok:raise ValueError(message)


def _under(path,root):
    path=Path(path).resolve();_require(path.is_relative_to(root.resolve()),'path outside declared D-stage outputs boundary')
    return path


def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);delivery.base._write_json_new(path,value)


def _copy(source,target):
    data=Path(source).read_bytes();target.parent.mkdir(parents=True,exist_ok=True)
    delivery.base._write_new(target,data)
    return dict(file=str(target),sha256=sha256(data).hexdigest(),bytes=len(data),copied_original_bytes=True)


def _raw_bundle(source,output,*,max_bytes):
    """Exact original gzip/metadata bytes, with ordered chunks and whole hashes."""
    _require(not output.exists(),'raw bundle output already exists');output.mkdir(parents=True)
    rows=[]
    for number,path in enumerate(sorted(p for p in source.iterdir() if p.is_file())):
        digest=sha256();parts=[];offset=0
        with path.open('rb') as stream:
            while True:
                data=stream.read(max_bytes)
                if not data:break
                name=f'raw_{number:03d}_part_{len(parts):03d}.bin'
                delivery.base._write_new(output/name,data);digest.update(data)
                parts.append(dict(file=name,offset=offset,bytes=len(data),sha256=sha256(data).hexdigest()));offset+=len(data)
        rows.append(dict(original_file=path.name,bytes=offset,sha256=digest.hexdigest(),parts=parts))
    value=dict(schema='ev_mobile_v2_exact_raw_bundle_v1',max_bytes=max_bytes,files=rows,
        source_directory=str(source),scope='all original selected case gzip bytes and full metadata; B planning arrays retained',
        restore='concatenate each file parts in listed order, then verify whole bytes and SHA256 before opening')
    value['manifest_id']=delivery.base._object_hash(value);_write(output/'raw_bundle_manifest.json',value)
    verify_raw_bundle(output)
    return value


def verify_raw_bundle(directory):
    directory=Path(directory);value=delivery.base._read(directory/'raw_bundle_manifest.json')
    _require(delivery.base._object_hash({k:v for k,v in value.items() if k!='manifest_id'})==value.get('manifest_id'),'raw bundle manifest hash')
    for row in value['files']:
        digest=sha256();offset=0
        for part in row['parts']:
            data=delivery.base._child(directory,part['file']).read_bytes()
            _require(part['offset']==offset and len(data)==part['bytes']<=value['max_bytes'] and sha256(data).hexdigest()==part['sha256'],
                'raw chunk order/bytes/hash mismatch')
            digest.update(data);offset+=len(data)
        _require(offset==row['bytes'] and digest.hexdigest()==row['sha256'],'whole original raw bytes/hash mismatch')
    return dict(success=True,files=len(value['files']),parts=sum(len(r['parts']) for r in value['files']))


def _combine(group,selections,work,infos):
    """Form a small representative input directory, preserving every record byte."""
    target=work/group;_require(not target.exists(),'representative source already exists');target.mkdir(parents=True)
    first=infos[selections[0][0]];selected=[]
    delivery.base._write_new(target/'freeze.json',first['freeze_bytes'])
    delivery.base._write_new(target/'source_snapshot.json.gz',first['snapshot_bytes'])
    used=set()
    for logical,index in selections:
        info=infos[logical]
        _require(info['freeze']==first['freeze'] and info['snapshot_id']==first['snapshot_id'],
            'representative group mixes frozen policies/source snapshots')
        candidates=[r for r in info['records'] if index is None or r['index']==index]
        _require(candidates if index is None else len(candidates)==1,'representative index absent or duplicate')
        for row in candidates:
            name=row['raw_file'];_require(name not in used,'representative filename collision');used.add(name)
            source=info['source']/name;method,reason=merging._link_or_copy(source,target/name)
            data=(target/name).read_bytes();_require(len(data)==row['raw_bytes'] and sha256(data).hexdigest()==row['raw_sha256'],
                'representative original bytes/hash changed')
            selected.append(dict(logical_source=logical,source_path=str(source),file=name,index=row['index'],
                physical_bank=row['physical_bank'],phase=row['phase'],bytes=len(data),sha256=row['raw_sha256'],
                materialization=method,copy_fallback_reason=reason))
    manifest=dict(config=first['freeze']['config'],policy_id=first['freeze']['policy_id'],
        source_sha256=first['freeze']['source_sha256'],representative_group=group,records=selected,
        scope='mixed phases are retained per record; this directory is not a calibration/evaluation sample')
    _write(target/'manifest.json',manifest)
    return target,selected


def declared_plan():
    return dict(logical_directories=len(PLAN),experiment_slots=sum(len(p[3]) for p in PLAN),
        sources=[dict(directory=p[0],bank=p[1],phase=p[2],indices=list(p[3]),count=len(p[3])) for p in PLAN],
        compact_representatives=dict(orig30=4,guard60=10),complete_raw_records=2,max_package_bytes=MAX_BYTES,
        scope='declarations only; this function reads no source directories')


def pack(raw_root,output,*,max_bytes=MAX_BYTES):
    raw_root=Path(raw_root).resolve();output=_under(output,raw_root);work=output.with_name(output.name+'_work')
    _require(2048<=max_bytes<=MAX_BYTES,'package limit must be between 2048 bytes and 20 MiB')
    if output.exists() or work.exists():raise FileExistsError('evidence or work directory already exists; preserve it and choose fresh names')
    for name,_,_,indices in PLAN:
        source=_under(raw_root/name,raw_root)
        _require(source.is_dir(),'required logical source missing: '+name)
        files=list(source.glob('case_*.json.gz'))
        _require(len(files)==len(indices),'logical source record count differs: '+name)
    output.mkdir(parents=True);work.mkdir(parents=True);infos={};data=[];groups=[]
    try:
        for name,bank,phase,indices in PLAN:
            source=raw_root/name;lite=work/'analysis'/name
            manifest=delivery.analysis(source,lite)
            frozen,freeze_bytes,snapshot_bytes,snapshot_id=merging._freeze(source)
            if name.startswith('mobile_v2_formal_'):_require(frozen['policy_id']==POLICY,'formal policy differs from frozen guard60')
            records=[]
            for entry in manifest['entries']:
                row=delivery.base._read(lite/entry['compact_file'])
                _require(row['phase']==phase and row['policy_id']==frozen['policy_id'] and row['source_sha256']==frozen['source_sha256'],
                    'record phase/policy/source differs from logical source')
                _require(row['delivery_provenance']['physical_bank']==bank,'record physical bank differs from logical source')
                records.append(dict(entry,index=row['index'],phase=phase,physical_bank=bank))
            _require(len(records)==len(indices) and {r['index'] for r in records}==set(indices),'logical source missing/duplicate/extra indices')
            infos[name]=dict(source=source,freeze=frozen,freeze_bytes=freeze_bytes,snapshot_bytes=snapshot_bytes,
                snapshot_id=snapshot_id,records=records)
            bundled=delivery.bundle(lite,output/'bundles'/'analysis'/name,max_bytes=max_bytes)
            _copy(lite/'delivery_manifest.json',output/'manifests'/(name+'_analysis_manifest.json'))
            metadata=[]
            for extra in ('merge_manifest.json','orchestration_complete.json'):
                if (source/extra).exists():metadata.append(_copy(source/extra,output/'manifests'/(name+'_'+extra)))
            data.append(dict(logical_source=name,physical_bank=bank,phase=phase,case_count=len(records),
                raw_directory=str(source),raw_case_bytes=manifest['raw_case_bytes'],
                source_policy_id=frozen['policy_id'],snapshot_content_id=snapshot_id,
                delivery_manifest_id=manifest['manifest_id'],bundle_manifest=bundled,additional_metadata=metadata))
        original=[('mobile_v2_fullpilot_n3',None),('mobile_v2_fullpilot_n4',None)]
        formal=[(f'mobile_v2_formal_{phase}_{bank}',index) for bank in ('n3','n4')
            for phase,indices in (('cal',(0,)),('eval',(0,12,24,36))) for index in indices]
        for group,selections in (('representatives_orig30',original),('representatives_guard60',formal)):
            source,selected=_combine(group,selections,work,infos);compact=work/'compact'/group
            manifest=delivery.compact(source,compact);bundled=delivery.bundle(compact,output/'bundles'/group,max_bytes=max_bytes)
            _copy(compact/'delivery_manifest.json',output/'manifests'/(group+'_compact_manifest.json'))
            groups.append(dict(group=group,records=selected,case_count=len(selected),delivery_manifest_id=manifest['manifest_id'],bundle_manifest=bundled))
        source,selected=_combine('selected_complete_raw',[(f'mobile_v2_formal_eval_{bank}',0) for bank in ('n3','n4')],work,infos)
        raw_bundle=_raw_bundle(source,output/'bundles'/'selected_complete_raw',max_bytes=max_bytes)
        index=dict(schema='ev_mobile_v2_evidence_pack_v1',experiment_slots=sum(d['case_count'] for d in data),
            logical_directories=len(data),max_package_bytes=max_bytes,raw_root=str(raw_root),work_directory=str(work),
            datasets=data,compact_groups=groups,complete_raw=dict(records=selected,bundle=raw_bundle),
            pack_tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            reused_delivery_tool_sha256=sha256(Path(delivery.__file__).read_bytes()).hexdigest(),
            packages=[])
        for path in sorted((output/'bundles').rglob('*')):
            if path.is_file() and path.suffix in ('.xz','.bin'):
                value=path.read_bytes();_require(len(value)<=max_bytes,'package exceeds declared maximum')
                index['packages'].append(dict(file=str(path.relative_to(output)).replace('\\','/'),bytes=len(value),sha256=sha256(value).hexdigest()))
        _require(index['experiment_slots']==352 and index['logical_directories']==14,'declared delivery denominator differs')
        index['manifest_id']=delivery.base._object_hash(index);_write(output/'pack_manifest.json',index)
        text=('EV-v2 evidence\n\n'
            '352 experiment slots from 14 logical directories are included as analysis-lite. Development pairs and reruns are slots, not additional independent histories.\n\n'
            'Analysis-lite preserves method scores, failures, operations, timings and parameter errors. It excludes measurements and fitted states.\n\n'
            'The 14 preselected compact records retain all actual measurements and fixed/joint fit states. Selection/planning arrays are summarized. Their original phases are unchanged.\n\n'
            'Both banks evaluation index 0 are also included as exact original gzip-byte parts, with every B planning array. Restore each original file by concatenating the parts listed in raw_bundle_manifest.json, then verify its whole SHA256.\n\n'
            'Each tar.xz bundle reproduces the corresponding delivery directory; its bundle_manifest.json lists member hashes. Every binary package is at most 20 MiB.\n\n'
            'All full raw records remain unchanged under the D-stage outputs root recorded in pack_manifest.json. Analysis-lite and compact do not recreate full raw bytes. Complete source snapshots remain in every logical delivery.\n')
        delivery.base._write_new(output/'README.md',text.encode('utf8'))
        return index
    except Exception as exc:
        _write(output/'pack_incomplete.json',dict(status='incomplete',failure=type(exc).__name__+': '+str(exc),preserve_all_files=True))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw-root',type=Path,
        default=Path(__file__).resolve().parents[2]/'outputs')
    parser.add_argument('--output',type=Path);parser.add_argument('--plan',action='store_true')
    args=parser.parse_args()
    if args.plan:value=declared_plan()
    else:
        if args.output is None:parser.error('--output is required unless --plan is used')
        value=pack(args.raw_root,args.output)
        value={k:value[k] for k in ('experiment_slots','logical_directories','manifest_id','work_directory')}
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
